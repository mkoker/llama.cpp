#pragma once

#include <stddef.h>
#include <stdint.h>

#define GGML_EXPERT_CACHE_KEY_VERSION 1u

// Shared scheduler/backend cache-key ABI. Keep POD-only: ggml-backend.cpp
// serializes this across the backend proc-address boundary and ggml-cuda consumes it.
struct ggml_expert_cache_key_base {
    uint32_t  version;
    uint32_t  reserved;
    uintptr_t source_tensor_id;
    uint32_t  backend_id;
    uint32_t  type;
    uint64_t  ne[4];
    uint64_t  nb[4];
    uint64_t  expert_size;
};

static_assert(offsetof(ggml_expert_cache_key_base, version) == 0, "expert-cache key ABI drift: version");
static_assert(offsetof(ggml_expert_cache_key_base, reserved) == 4, "expert-cache key ABI drift: reserved");
static_assert(offsetof(ggml_expert_cache_key_base, source_tensor_id) == 8, "expert-cache key ABI drift: source_tensor_id");
static_assert(offsetof(ggml_expert_cache_key_base, backend_id) == 16, "expert-cache key ABI drift: backend_id");
static_assert(offsetof(ggml_expert_cache_key_base, type) == 20, "expert-cache key ABI drift: type");
static_assert(offsetof(ggml_expert_cache_key_base, ne) == 24, "expert-cache key ABI drift: ne");
static_assert(offsetof(ggml_expert_cache_key_base, nb) == 56, "expert-cache key ABI drift: nb");
static_assert(offsetof(ggml_expert_cache_key_base, expert_size) == 88, "expert-cache key ABI drift: expert_size");
static_assert(sizeof(ggml_expert_cache_key_base) == 96, "expert-cache key ABI drift: size");
