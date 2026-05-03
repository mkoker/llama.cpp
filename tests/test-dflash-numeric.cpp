#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

[[noreturn]] static void fail(const char * msg) {
    std::fprintf(stderr, "test-dflash-numeric: FAIL: %s\n", msg);
    std::exit(1);
}

static uint16_t read_u16_le(FILE * f) {
    uint8_t b[2];
    if (std::fread(b, 1, sizeof(b), f) != sizeof(b)) {
        fail("short read while reading u16");
    }
    return (uint16_t) b[0] | ((uint16_t) b[1] << 8);
}

static uint32_t read_u32_le(FILE * f) {
    uint8_t b[4];
    if (std::fread(b, 1, sizeof(b), f) != sizeof(b)) {
        fail("short read while reading u32");
    }
    return (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
}

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t) v << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

struct ref_sample {
    std::string name;
    std::vector<uint32_t> indices;
    std::vector<uint16_t> bf16_values;
};

static std::vector<ref_sample> load_reference(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        fail("missing reference file; run scripts/generate-dflash-numeric-ref.py");
    }

    char magic[8];
    if (std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) || std::memcmp(magic, "DFLREF1\0", 8) != 0) {
        fail("bad reference magic");
    }

    const uint32_t n_tensors = read_u32_le(f);
    std::vector<ref_sample> refs;
    refs.reserve(n_tensors);

    for (uint32_t i = 0; i < n_tensors; ++i) {
        const uint16_t name_len = read_u16_le(f);
        std::string name(name_len, '\0');
        if (std::fread(name.data(), 1, name.size(), f) != name.size()) {
            fail("short read while reading tensor name");
        }
        const uint32_t n = read_u32_le(f);
        ref_sample s;
        s.name = std::move(name);
        s.indices.resize(n);
        s.bf16_values.resize(n);
        for (uint32_t j = 0; j < n; ++j) {
            s.indices[j] = read_u32_le(f);
            s.bf16_values[j] = read_u16_le(f);
        }
        refs.push_back(std::move(s));
    }

    if (std::fclose(f) != 0) {
        fail("failed closing reference file");
    }
    if (refs.empty()) {
        fail("empty reference file");
    }
    return refs;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s DFLASH_DRAFT.gguf [REFERENCE.bin]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * ref_path = argc >= 3 ? argv[2] : "/mnt/nvme/dflash-refs/qwen3-coder-next-dflash-bf16.samples.bin";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fail("failed to load DFlash draft model");
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);
    if (n_layer != 8) {
        fail("unexpected DFlash layer count");
    }
    if (n_embd != 2048) {
        fail("unexpected DFlash embedding size");
    }
    llama_model_free(model);

    gguf_init_params params = {};
    params.no_alloc = true;
    gguf_context * gguf = gguf_init_from_file(model_path, params);
    if (!gguf) {
        fail("failed to open GGUF metadata");
    }

    FILE * model_file = std::fopen(model_path, "rb");
    if (!model_file) {
        gguf_free(gguf);
        fail("failed to open GGUF data");
    }

    const std::vector<ref_sample> refs = load_reference(ref_path);
    double max_abs_err = 0.0;
    size_t n_checked = 0;

    for (const ref_sample & ref : refs) {
        const int tensor_id = gguf_find_tensor(gguf, ref.name.c_str());
        if (tensor_id < 0) {
            fail("reference tensor not found in GGUF");
        }
        if (gguf_get_tensor_type(gguf, tensor_id) != GGML_TYPE_BF16) {
            fail("reference tensor is not BF16 in GGUF");
        }

        const size_t tensor_size = gguf_get_tensor_size(gguf, tensor_id);
        if (tensor_size % sizeof(uint16_t) != 0) {
            fail("BF16 tensor byte size is not element-aligned");
        }
        const size_t ne = tensor_size / sizeof(uint16_t);

        const size_t tensor_data = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_id);
        for (size_t i = 0; i < ref.indices.size(); ++i) {
            if ((size_t) ref.indices[i] >= ne) {
                fail("reference sample index out of tensor bounds");
            }
            if (std::fseek(model_file, (long) (tensor_data + (size_t) ref.indices[i] * sizeof(uint16_t)), SEEK_SET) != 0) {
                fail("failed seeking GGUF tensor data");
            }
            const uint16_t got_bf16 = read_u16_le(model_file);
            const float got = bf16_to_f32(got_bf16);
            const float exp = bf16_to_f32(ref.bf16_values[i]);
            max_abs_err = std::max(max_abs_err, (double) std::fabs(got - exp));
            ++n_checked;
        }
    }

    std::fclose(model_file);
    gguf_free(gguf);
    llama_backend_free();

    if (n_checked < 256) {
        fail("too few numeric samples checked");
    }
    if (max_abs_err > 1e-2) {
        std::fprintf(stderr, "test-dflash-numeric: FAIL max_abs_err=%.8f checked=%zu\n", max_abs_err, n_checked);
        return 1;
    }

    std::fprintf(stderr, "test-dflash-numeric: PASS PyTorch-safetensors numeric reference n_layer=%d n_embd=%d checked=%zu max_abs_err=%.8f\n", n_layer, n_embd, n_checked, max_abs_err);
    return 0;
}
