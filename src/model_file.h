#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// a tensor inside a memory-mapped gguf, viewed as rows of ne[0] values.
struct tensor_ref {
    std::string     name;
    ggml_type       type;
    int64_t         ne[4];
    const uint8_t * data;

    int64_t n_rows() const { return ne[1] * ne[2] * ne[3]; }

    size_t row_size() const { return ggml_row_size(type, ne[0]); }

    int n_dims() const { return ne[3] > 1 ? 4 : ne[2] > 1 ? 3 : ne[1] > 1 ? 2 : 1; }

    // dequantize one row with ggml's own reference routines.
    void dequantize_row(int64_t row, float * out) const;
};

// a (possibly split) gguf model, mapped read-only. metadata comes from shard 1.
struct model_file {
    std::string             path;
    std::vector<tensor_ref> tensors;

    explicit model_file(const std::string & path);
    ~model_file();

    const tensor_ref * find(const std::string & name) const;
    std::string        get_str(const std::string & key, const std::string & def = "") const;
    int64_t            get_int(const std::string & key, int64_t def = -1) const;

    std::string arch() const { return get_str("general.architecture"); }

    int64_t n_embd() const { return get_int(arch() + ".embedding_length"); }

    int64_t n_layer() const { return get_int(arch() + ".block_count"); }

  private:
    struct shard;
    std::vector<std::unique_ptr<shard>> shards;
};
