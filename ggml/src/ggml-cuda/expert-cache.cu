#include "expert-cache.cuh"
#include <cinttypes>
#include <cstring>

// ── Per-partition LRU helpers (caller must hold cache->mtx) ─────────

static void lru_unlink(ggml_expert_cache * cache, int idx) {
    ggml_expert_cache_slot & s = cache->slots[idx];
    int part = s.partition;
    ggml_expert_cache_partition & p = cache->partitions[part];

    if (s.prev >= 0) {
        cache->slots[s.prev].next = s.next;
    } else {
        p.lru_head = s.next;
    }

    if (s.next >= 0) {
        cache->slots[s.next].prev = s.prev;
    } else {
        p.lru_tail = s.prev;
    }

    s.prev = -1;
    s.next = -1;
}

static void lru_push_front(ggml_expert_cache * cache, int idx) {
    ggml_expert_cache_slot & s = cache->slots[idx];
    int part = s.partition;
    ggml_expert_cache_partition & p = cache->partitions[part];

    s.prev = -1;
    s.next = p.lru_head;

    if (p.lru_head >= 0) {
        cache->slots[p.lru_head].prev = idx;
    }

    p.lru_head = idx;

    if (p.lru_tail < 0) {
        p.lru_tail = idx;
    }
}

// ── Init / Free ─────────────────────────────────────────────────────

ggml_expert_cache * ggml_expert_cache_init(size_t total_size_bytes, size_t slot_size_bytes, int device) {
    GGML_ASSERT(slot_size_bytes > 0);

    // Reserve 1/3 of budget for staging, rest for cache slots
    size_t staging_budget = total_size_bytes / 3;
    size_t max_staging = slot_size_bytes * 128; // one full expert tensor
    if (staging_budget > max_staging) staging_budget = max_staging;
    size_t cache_budget = total_size_bytes - staging_budget;

    int n_slots = (int)(cache_budget / slot_size_bytes);
    if (n_slots < 1) n_slots = 1;

    size_t pool_bytes = (size_t)n_slots * slot_size_bytes;

    ggml_cuda_set_device(device);

    void * pool = nullptr;
    cudaError_t err = cudaMalloc(&pool, pool_bytes);
    if (err != cudaSuccess) {
        GGML_LOG_WARN("%s: cache alloc failed (%.1f MiB)\n", __func__, pool_bytes / (1024.0 * 1024.0));
        cudaGetLastError();
        return nullptr;
    }

    ggml_expert_cache * cache = new ggml_expert_cache();
    cache->pool       = pool;
    cache->n_slots    = n_slots;
    cache->slot_size  = slot_size_bytes;
    cache->total_size = pool_bytes;
    cache->hits       = 0;
    cache->misses     = 0;

    // per-partition config: target 10-16 slots per partition
    // for 235B: 282 tensors, want ~10 slots each = 2820 slots
    cache->slots_per_partition = 10;
    if (cache->slots_per_partition > n_slots) cache->slots_per_partition = n_slots;
    cache->max_partitions = n_slots / cache->slots_per_partition;
    cache->n_partitions = 0;
    cache->partitions.reserve(cache->max_partitions);

    cache->slots = new ggml_expert_cache_slot[n_slots];
    for (int i = 0; i < n_slots; i++) {
        cache->slots[i].data       = (char *)pool + (size_t)i * slot_size_bytes;
        cache->slots[i].tensor_ptr = nullptr;
        cache->slots[i].expert_idx = -1;
        cache->slots[i].size       = 0;
        cache->slots[i].partition  = -1; // unassigned until partition created
        cache->slots[i].prev       = -1;
        cache->slots[i].next       = -1;
    }

    // staging buffer
    cache->staging_buf = nullptr;
    cache->staging_size = 0;
    if (staging_budget > 0) {
        cudaError_t se = cudaMalloc(&cache->staging_buf, staging_budget);
        if (se == cudaSuccess) {
            cache->staging_size = staging_budget;
        } else {
            cudaGetLastError();
        }
    }

    GGML_LOG_INFO("%s: expert cache: %d slots (%.1f MiB), %d slots/partition, max %d partitions, staging %.1f MiB\n",
                  __func__, n_slots, pool_bytes / (1024.0 * 1024.0),
                  cache->slots_per_partition, cache->max_partitions,
                  cache->staging_size / (1024.0 * 1024.0));

    return cache;
}

void ggml_expert_cache_free(ggml_expert_cache * cache) {
    if (!cache) return;

    int64_t total = cache->hits + cache->misses;
    double hit_rate = total > 0 ? 100.0 * (double)cache->hits / (double)total : 0.0;

    GGML_LOG_INFO("%s: hits: %" PRId64 ", misses: %" PRId64 ", hit rate: %.1f%%, partitions: %d\n",
                  __func__, cache->hits, cache->misses, hit_rate, cache->n_partitions);

    if (cache->staging_buf) { CUDA_CHECK(cudaFree(cache->staging_buf)); }
    CUDA_CHECK(cudaFree(cache->pool));
    delete[] cache->slots;
    delete cache;
}

// ── Get or create partition for a tensor_ptr ────────────────────────

static int get_or_create_partition(ggml_expert_cache * cache, void * tensor_ptr) {
    auto it = cache->tensor_to_partition.find(tensor_ptr);
    if (it != cache->tensor_to_partition.end()) {
        return it->second;
    }

    // create new partition
    if (cache->n_partitions >= cache->max_partitions) {
        return -1; // no more partitions available
    }

    int part_id = cache->n_partitions++;
    int start = part_id * cache->slots_per_partition;
    int n = cache->slots_per_partition;

    ggml_expert_cache_partition p;
    p.start_slot = start;
    p.n_slots = n;
    p.lru_head = start;
    p.lru_tail = start + n - 1;

    // init LRU chain for this partition
    for (int i = start; i < start + n; i++) {
        cache->slots[i].partition = part_id;
        cache->slots[i].prev = (i > start) ? i - 1 : -1;
        cache->slots[i].next = (i < start + n - 1) ? i + 1 : -1;
    }

    cache->partitions.push_back(p);
    cache->tensor_to_partition[tensor_ptr] = part_id;

    return part_id;
}

// ── Lookup / Insert ─────────────────────────────────────────────────

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

    // cache hit
    auto it = cache->slot_map.find(key);
    if (it != cache->slot_map.end()) {
        int idx = it->second;
        cache->hits++;
        lru_unlink(cache, idx);
        lru_push_front(cache, idx);
        return cache->slots[idx].data;
    }

    // cache miss — get or create partition for this tensor
    cache->misses++;

    int part_id = get_or_create_partition(cache, tensor_ptr);
    if (part_id < 0) {
        // no partition available — do direct H2D to a temp slot (slot 0 as fallback)
        // this should be rare
        CUDA_CHECK(cudaMemcpyAsync(cache->slots[0].data, src_data, expert_size, cudaMemcpyHostToDevice, stream));
        return cache->slots[0].data;
    }

    ggml_expert_cache_partition & p = cache->partitions[part_id];

    // evict LRU tail of THIS partition
    int victim = p.lru_tail;
    GGML_ASSERT(victim >= 0);

    ggml_expert_cache_slot & slot = cache->slots[victim];

    // remove old mapping
    if (slot.tensor_ptr != nullptr) {
        ggml_expert_cache_key old_key{slot.tensor_ptr, slot.expert_idx};
        cache->slot_map.erase(old_key);
    }

    lru_unlink(cache, victim);
    lru_push_front(cache, victim);

    // H2D copy
    CUDA_CHECK(cudaMemcpyAsync(slot.data, src_data, expert_size, cudaMemcpyHostToDevice, stream));

    slot.tensor_ptr = tensor_ptr;
    slot.expert_idx = expert_idx;
    slot.size       = expert_size;

    cache->slot_map[key] = victim;

    return slot.data;
}
