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

// Scheduler-derived cache-key context (deterministic across lookup/insert):
// - source_tensor_id: source tensor identity at scheduler boundary (canonical base tensor)
// - backend_id: scheduler split backend id / device route
// - type + ne[] + nb[]: dtype and full layout shape/strides
// - expert_size: bytes per expert slice
struct ggml_expert_cache_key_base {
    uint32_t  version;          // key schema version for scheduler<->backend ABI safety
    uint32_t  reserved;         // reserved for future flags
    uintptr_t source_tensor_id;
    uint32_t  backend_id;
    uint32_t  type;
    uint64_t  ne[4];
    uint64_t  nb[4];
    uint64_t  expert_size;
};

struct ggml_expert_cache_key {
    ggml_expert_cache_key_base base;
    int64_t                    expert_idx;

    bool operator==(const ggml_expert_cache_key & other) const {
        return base.version         == other.base.version &&
               base.source_tensor_id == other.base.source_tensor_id &&
               base.backend_id      == other.base.backend_id &&
               base.type            == other.base.type &&
               base.expert_size     == other.base.expert_size &&
               base.ne[0]           == other.base.ne[0] &&
               base.ne[1]           == other.base.ne[1] &&
               base.ne[2]           == other.base.ne[2] &&
               base.ne[3]           == other.base.ne[3] &&
               base.nb[0]           == other.base.nb[0] &&
               base.nb[1]           == other.base.nb[1] &&
               base.nb[2]           == other.base.nb[2] &&
               base.nb[3]           == other.base.nb[3] &&
               expert_idx           == other.expert_idx;
    }
};

struct ggml_expert_cache_key_hash {
    size_t operator()(const ggml_expert_cache_key & k) const {
        size_t h = std::hash<uint32_t>{}(k.base.version);
        h ^= std::hash<uintptr_t>{}(k.base.source_tensor_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);

        h ^= std::hash<uint32_t>{}(k.base.backend_id) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.base.type)       + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.expert_size)+ 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.ne[0])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.ne[1])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.ne[2])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.ne[3])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.nb[0])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.nb[1])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.nb[2])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(k.base.nb[3])      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.expert_idx)       + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);

        return h;
    }
};

struct ggml_expert_cache {
    ggml_expert_cache_slot * slots;     // array of N slots
    void *                   pool;      // single GPU allocation backing all slots
    int                      n_slots;
    size_t                   slot_size; // bytes per slot (largest expert slice)
    size_t                   total_size;
    int                      device;

    int lru_head; // most recently used
    int lru_tail; // least recently used (eviction candidate)

    std::unordered_map<ggml_expert_cache_key, int, ggml_expert_cache_key_hash> slot_map;

    std::mutex mtx;

    int64_t hits;
    int64_t misses;
    int64_t h2d_copies;
    int64_t h2d_bytes;
    int64_t d2d_copies;
    int64_t d2d_bytes;
    int64_t skipped_h2d_due_to_hit;

    void *  staging_buf;
    size_t  staging_size;
};

ggml_expert_cache * ggml_expert_cache_init(size_t total_size_bytes, size_t slot_size_bytes, int device);
void                ggml_expert_cache_free(ggml_expert_cache * cache);

void * ggml_expert_cache_get(
    ggml_expert_cache *               cache,
    const ggml_expert_cache_key_base & key_base,
    int64_t                           expert_idx,
    const void *                      src_data,
    size_t                            expert_size,
    cudaStream_t                      stream,
    bool *                            was_hit = nullptr);
