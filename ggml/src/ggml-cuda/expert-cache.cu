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
    cache->lru_head   = 0;
    cache->lru_tail   = n_slots - 1;
    cache->hits       = 0;
    cache->misses     = 0;

    cache->slots = new ggml_expert_cache_slot[n_slots];

    for (int i = 0; i < n_slots; i++) {
        cache->slots[i].data       = (char *)pool + (size_t)i * slot_size_bytes;
        cache->slots[i].tensor_ptr = nullptr;
        cache->slots[i].expert_idx = -1;
        cache->slots[i].size       = 0;
        cache->slots[i].prev       = i - 1;
        cache->slots[i].next       = (i + 1 < n_slots) ? i + 1 : -1;
    }

    GGML_LOG_INFO("%s: expert cache initialized — %d slots, %.1f MiB total\n",
                  __func__, n_slots, (double)pool_bytes / (1024.0 * 1024.0));

    return cache;
}

void ggml_expert_cache_free(ggml_expert_cache * cache) {
    if (!cache) {
        return;
    }

    int64_t total = cache->hits + cache->misses;
    double hit_rate = total > 0 ? 100.0 * (double)cache->hits / (double)total : 0.0;

    GGML_LOG_INFO("%s: expert cache stats - hits: %" PRId64 ", misses: %" PRId64 ", hit rate: %.1f%%\n",
                  __func__, cache->hits, cache->misses, hit_rate);

    CUDA_CHECK(cudaFree(cache->pool));
    delete[] cache->slots;
    delete cache;
}

// ── Lookup / Insert ─────────────────────────────────────────────────────

void * ggml_expert_cache_get(
        ggml_expert_cache * cache,
        void *              tensor_ptr,
        int64_t             expert_idx,
        const void *        src_data,
        size_t              expert_size,
        cudaStream_t        stream) {
    GGML_ASSERT(cache != nullptr);
    GGML_ASSERT(expert_size <= cache->slot_size);

    std::lock_guard<std::mutex> lock(cache->mtx);

    ggml_expert_cache_key key{tensor_ptr, expert_idx};

    // ── Cache hit ───────────────────────────────────────────────────────
    auto it = cache->slot_map.find(key);
    if (it != cache->slot_map.end()) {
        int idx = it->second;
        cache->hits++;

        // promote to MRU
        lru_unlink(cache, idx);
        lru_push_front(cache, idx);

        return cache->slots[idx].data;
    }

    // ── Cache miss — evict LRU tail ─────────────────────────────────────
    cache->misses++;

    int victim = cache->lru_tail;
    GGML_ASSERT(victim >= 0);

    ggml_expert_cache_slot & slot = cache->slots[victim];

    // remove old mapping if slot was occupied
    if (slot.tensor_ptr != nullptr) {
        ggml_expert_cache_key old_key{slot.tensor_ptr, slot.expert_idx};
        cache->slot_map.erase(old_key);
    }

    // unlink victim from LRU tail, push to front
    lru_unlink(cache, victim);
    lru_push_front(cache, victim);

    // copy expert data from CPU to GPU
    CUDA_CHECK(cudaMemcpyAsync(slot.data, src_data, expert_size, cudaMemcpyHostToDevice, stream));

    // update slot metadata
    slot.tensor_ptr = tensor_ptr;
    slot.expert_idx = expert_idx;
    slot.size       = expert_size;

    // insert new mapping
    cache->slot_map[key] = victim;

    return slot.data;
}
