#pragma once

#include "common.cuh"
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <vector>

struct ggml_expert_cache_slot {
    void *   data;         // GPU memory for this slot
    void *   tensor_ptr;   // which src0 tensor this caches (nullptr = empty)
    int64_t  expert_idx;   // which expert index within that tensor
    size_t   size;         // actual bytes used in this slot
    int      partition;    // which partition this slot belongs to

    // intrusive doubly-linked list for per-partition LRU
    int prev;
    int next;
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

struct ggml_expert_cache_partition {
    int lru_head;  // most recently used slot index (-1 = empty)
    int lru_tail;  // least recently used slot index (-1 = empty)
    int n_slots;   // number of slots in this partition
    int start_slot; // first slot index in the global array
};

struct ggml_expert_cache {
    ggml_expert_cache_slot * slots;     // array of all slots
    void *                   pool;      // single GPU allocation
    int                      n_slots;
    size_t                   slot_size;
    size_t                   total_size;

    // per-tensor-ptr partitions
    std::vector<ggml_expert_cache_partition> partitions;
    std::unordered_map<void *, int> tensor_to_partition; // tensor_ptr -> partition index
    int n_partitions;           // number of partitions created
    int max_partitions;         // max partitions (= n_slots / slots_per_partition)
    int slots_per_partition;    // slots allocated per partition

    // global lookup
    std::unordered_map<ggml_expert_cache_key, int, ggml_expert_cache_key_hash> slot_map;

    std::mutex mtx;

    int64_t hits;
    int64_t misses;

    // staging buffer
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
    cudaStream_t        stream);
