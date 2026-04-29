#include "expert-cache.cuh"
#include <cinttypes>

// ── LRU helpers (caller must hold cache->mtx) ──────────────────────────

static void lru_unlink(ggml_expert_cache * cache, int idx) {
    ggml_expert_cache_slot & s = cache->slots[idx];

    if (s.prev >= 0) {
        cache->slots[s.prev].next = s.next;
    } else {
        cache->lru_head = s.next;
    }

    if (s.next >= 0) {
        cache->slots[s.next].prev = s.prev;
    } else {
        cache->lru_tail = s.prev;
    }

    s.prev = -1;
    s.next = -1;
}

static void lru_push_front(ggml_expert_cache * cache, int idx) {
    ggml_expert_cache_slot & s = cache->slots[idx];

    s.prev = -1;
    s.next = cache->lru_head;

    if (cache->lru_head >= 0) {
        cache->slots[cache->lru_head].prev = idx;
    }

    cache->lru_head = idx;

    if (cache->lru_tail < 0) {
        cache->lru_tail = idx;
    }
}

// ── Init / Free ─────────────────────────────────────────────────────────

ggml_expert_cache * ggml_expert_cache_init(size_t total_size_bytes, size_t slot_size_bytes, int device) {
    GGML_ASSERT(slot_size_bytes > 0);

    int n_slots = (int)(total_size_bytes / slot_size_bytes);
    if (n_slots < 1) {
        n_slots = 1;
    }

    size_t pool_bytes = (size_t)n_slots * slot_size_bytes;

    ggml_cuda_set_device(device);

    void * pool = nullptr;
    cudaError_t err = cudaMalloc(&pool, pool_bytes);
    if (err != cudaSuccess) {
        GGML_LOG_WARN("%s: expert cache allocation failed (%.1f MiB), cache disabled\n", __func__, (double)pool_bytes / (1024.0 * 1024.0));
        cudaGetLastError(); // clear error
        
        return nullptr;
    }

    ggml_expert_cache * cache = new ggml_expert_cache();
    cache->pool       = pool;
    cache->n_slots    = n_slots;
    cache->slot_size  = slot_size_bytes;
    cache->total_size = pool_bytes;
    cache->device     = device;
    cache->lru_head   = 0;
    cache->lru_tail   = n_slots - 1;
    cache->hits       = 0;
    cache->misses     = 0;
    cache->h2d_copies = 0;
    cache->h2d_bytes  = 0;
    cache->d2d_copies = 0;
    cache->d2d_bytes  = 0;
    cache->skipped_h2d_due_to_hit = 0;
    cache->staging_buf  = nullptr;
    cache->staging_size = 0;

    // allocate staging from cache budget: staging = slot_size * 128 (one full expert tensor)
    size_t staging_want = slot_size_bytes * 128;
    if (staging_want > pool_bytes / 3) staging_want = pool_bytes / 3; // cap at 1/3 of budget
    {
        cudaError_t se = cudaMalloc(&cache->staging_buf, staging_want);
        if (se == cudaSuccess) {
            cache->staging_size = staging_want;
        } else {
            cudaGetLastError();
            cache->staging_buf = nullptr;
            cache->staging_size = 0;
        }
    }


    cache->slots = new ggml_expert_cache_slot[n_slots];

    for (int i = 0; i < n_slots; i++) {
        cache->slots[i].data       = (char *)pool + (size_t)i * slot_size_bytes;
        cache->slots[i].tensor_ptr = nullptr;
        cache->slots[i].expert_idx = -1;
        cache->slots[i].size       = 0;
        cache->slots[i].prev       = i - 1;
        cache->slots[i].next       = (i + 1 < n_slots) ? i + 1 : -1;
    }

    GGML_LOG_INFO("%s: expert cache arena initialized on device %d — %d slots, %.1f MiB total\n",
                  __func__, device, n_slots, (double)pool_bytes / (1024.0 * 1024.0));

    return cache;
}

void ggml_expert_cache_free(ggml_expert_cache * cache) {
    if (!cache) {
        return;
    }

    int64_t total = cache->hits + cache->misses;
    double hit_rate = total > 0 ? 100.0 * (double)cache->hits / (double)total : 0.0;

    GGML_LOG_INFO("%s: expert cache stats (device %d) - hits: %" PRId64 ", misses: %" PRId64 ", hit rate: %.1f%%, h2d copies: %" PRId64 ", h2d bytes: %" PRId64 ", d2d copies: %" PRId64 ", d2d bytes: %" PRId64 ", skipped h2d: %" PRId64 "\n",
                  __func__, cache->device, cache->hits, cache->misses, hit_rate,
                  cache->h2d_copies, cache->h2d_bytes,
                  cache->d2d_copies, cache->d2d_bytes,
                  cache->skipped_h2d_due_to_hit);

    if (cache->staging_buf) { CUDA_CHECK(cudaFree(cache->staging_buf)); }
    CUDA_CHECK(cudaFree(cache->pool));
    delete[] cache->slots;
    delete cache;
}

// ── Lookup / Insert ─────────────────────────────────────────────────────

void * ggml_expert_cache_lookup(
        ggml_expert_cache *               cache,
        const ggml_expert_cache_key_base & key_base,
        int64_t                           expert_idx,
        bool *                            was_hit) {
    GGML_ASSERT(cache != nullptr);

    std::lock_guard<std::mutex> lock(cache->mtx);

    ggml_expert_cache_key key{key_base, expert_idx};
    auto it = cache->slot_map.find(key);
    if (it == cache->slot_map.end()) {
        cache->misses++;
        if (was_hit) {
            *was_hit = false;
        }
        return nullptr;
    }

    const int idx = it->second;
    cache->hits++;
    if (was_hit) {
        *was_hit = true;
    }

    // promote to MRU
    lru_unlink(cache, idx);
    lru_push_front(cache, idx);

    return cache->slots[idx].data;
}


bool ggml_expert_cache_copy_hits(
        ggml_expert_cache *               cache,
        const ggml_expert_cache_key_base & key_base,
        void *                            dst_data,
        int64_t                           n_expert,
        size_t                            expert_size,
        const ggml_bitset_t *             used,
        cudaStream_t                      stream) {
    GGML_ASSERT(cache != nullptr);
    GGML_ASSERT(expert_size <= cache->slot_size);

    std::lock_guard<std::mutex> lock(cache->mtx);

    std::vector<int> hit_slots;
    hit_slots.reserve((size_t) n_expert);

    // preflight: only take the D2D path when all required experts are present
    for (int64_t id = 0; id < n_expert; ++id) {
        if (!ggml_bitset_get(used, id)) {
            continue;
        }

        ggml_expert_cache_key key{key_base, id};
        auto it = cache->slot_map.find(key);
        if (it == cache->slot_map.end()) {
            cache->misses++;
            return false;
        }

        hit_slots.push_back(it->second);
    }

    int64_t hit_i = 0;
    for (int64_t id = 0; id < n_expert; ++id) {
        if (!ggml_bitset_get(used, id)) {
            continue;
        }

        const int idx = hit_slots[(size_t) hit_i++];
        cache->hits++;
        lru_unlink(cache, idx);
        lru_push_front(cache, idx);

        CUDA_CHECK(cudaMemcpyAsync(
            (char *) dst_data + id * (int64_t) expert_size,
            cache->slots[idx].data,
            expert_size,
            cudaMemcpyDeviceToDevice,
            stream));

        cache->d2d_copies += 1;
        cache->d2d_bytes  += (int64_t) expert_size;
        cache->skipped_h2d_due_to_hit += 1;
    }

    return true;
}


void * ggml_expert_cache_get(
        ggml_expert_cache *               cache,
        const ggml_expert_cache_key_base & key_base,
        int64_t                           expert_idx,
        const void *                      src_data,
        size_t                            expert_size,
        cudaStream_t                      stream,
        bool *                            was_hit,
        bool                              count_h2d) {
    GGML_ASSERT(cache != nullptr);
    GGML_ASSERT(expert_size <= cache->slot_size);

    std::lock_guard<std::mutex> lock(cache->mtx);

    ggml_expert_cache_key key{key_base, expert_idx};

    auto it = cache->slot_map.find(key);
    if (it != cache->slot_map.end()) {
        const int idx = it->second;
        cache->hits++;
        if (was_hit) {
            *was_hit = true;
        }

        lru_unlink(cache, idx);
        lru_push_front(cache, idx);

        return cache->slots[idx].data;
    }

    cache->misses++;
    if (was_hit) {
        *was_hit = false;
    }

    const int victim = cache->lru_tail;
    GGML_ASSERT(victim >= 0);

    ggml_expert_cache_slot & slot = cache->slots[victim];

    if (slot.tensor_ptr != nullptr) {
        for (auto old = cache->slot_map.begin(); old != cache->slot_map.end(); ++old) {
            if (old->second == victim) {
                cache->slot_map.erase(old);
                break;
            }
        }
    }

    lru_unlink(cache, victim);
    lru_push_front(cache, victim);

    CUDA_CHECK(cudaMemcpyAsync(slot.data, src_data, expert_size, cudaMemcpyDefault, stream));
    if (count_h2d) {
        cache->h2d_copies += 1;
        cache->h2d_bytes  += (int64_t) expert_size;
    } else {
        cache->d2d_copies += 1;
        cache->d2d_bytes  += (int64_t) expert_size;
    }

    slot.tensor_ptr = (void *) key_base.source_tensor_id;
    slot.expert_idx = expert_idx;
    slot.size       = expert_size;

    cache->slot_map[key] = victim;

    return slot.data;
}

