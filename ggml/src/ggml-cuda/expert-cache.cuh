#pragma once

#include "common.cuh"
#include <cstdint>
#include <unordered_map>
#include <mutex>

struct ggml_expert_cache_slot {
    void *   data;         // GPU memory for this slot
    void *   tensor_ptr;   // which src0 tensor this caches (nullptr = empty)
    int64_t  expert_idx;   // which expert index within that tensor
    size_t   size;         // actual bytes used in this slot (may be < slot_size)

    // intrusive doubly-linked list for LRU
    int prev;  // index of previous slot in LRU order (-1 = head)
    int next;  // index of next slot in LRU order (-1 = tail)
};

struct ggml_expert_cache_key {
    void *  tensor_ptr;
    int64_t expert_idx;

    bool operator==(const ggml_expert_cache_key & other) const {
        return tensor_ptr == other.tensor_ptr && expert_idx == other.expert_idx;
    }
};

struct ggml_expert_cache_key_hash {
    size_t operator()(const ggml_expert_cache_key & k) const {
        size_t h1 = std::hash<void *>{}(k.tensor_ptr);
        size_t h2 = std::hash<int64_t>{}(k.expert_idx);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct ggml_expert_cache {
    ggml_expert_cache_slot * slots;     // array of N slots
    void *                   pool;      // single GPU allocation backing all slots
    int                      n_slots;
    size_t                   slot_size; // bytes per slot (largest expert slice)
    size_t                   total_size;

    int lru_head; // most recently used
    int lru_tail; // least recently used (eviction candidate)

    std::unordered_map<ggml_expert_cache_key, int, ggml_expert_cache_key_hash> slot_map;

    std::mutex mtx;

    int64_t hits;
    int64_t misses;

    int64_t h2d_expert_copies;
    int64_t h2d_bytes;
    int64_t d2d_expert_copies;
    int64_t d2d_bytes;
    int64_t skipped_h2d_due_to_hit;

    bool    debug_enabled;

    void *  staging_buf;
    size_t  staging_size;
};

ggml_expert_cache * ggml_expert_cache_init(size_t total_size_bytes, size_t slot_size_bytes, int device);
void                ggml_expert_cache_free(ggml_expert_cache * cache);

void * ggml_expert_cache_get(
    ggml_expert_cache * cache,
    void *              tensor_ptr,
    int64_t             expert_idx,
    const void *        src_data,
    size_t              expert_size,
    cudaStream_t        stream,
    bool *              was_hit = nullptr);
