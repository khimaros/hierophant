#include "model_file.h"

#include "llama.h"
#include "util.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

static const char * KV_SPLIT_COUNT = "split.count";

struct model_file::shard {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    void *         map  = MAP_FAILED;
    size_t         size = 0;

    explicit shard(const std::string & path) {
        gguf_init_params params = { /* .no_alloc = */ true, /* .ctx = */ &meta };
        gguf                    = gguf_init_from_file(path.c_str(), params);
        if (!gguf) {
            fatal("failed to read gguf '%s'", path.c_str());
        }
        int         fd = open(path.c_str(), O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            fatal("failed to open '%s'", path.c_str());
        }
        size = st.st_size;
        map  = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (map == MAP_FAILED) {
            fatal("failed to mmap '%s'", path.c_str());
        }
    }

    ~shard() {
        if (map != MAP_FAILED) {
            munmap(map, size);
        }
        gguf_free(gguf);
        ggml_free(meta);
    }
};

void tensor_ref::dequantize_row(int64_t row, float * out) const {
    const uint8_t * src = data + row * row_size();
    if (type == GGML_TYPE_F32) {
        memcpy(out, src, ne[0] * sizeof(float));
        return;
    }
    const auto * traits = ggml_get_type_traits(type);
    if (!traits->to_float) {
        fatal("tensor '%s' has type %s, which ggml cannot dequantize", name.c_str(), ggml_type_name(type));
    }
    traits->to_float(src, out, ne[0]);
}

static std::vector<std::string> shard_paths(const std::string & path, gguf_context * first) {
    const int64_t kid     = gguf_find_key(first, KV_SPLIT_COUNT);
    const int     n_split = kid < 0 ? 1 : (int) gguf_get_val_u16(first, kid);
    if (n_split <= 1) {
        return { path };
    }
    char prefix[4096];
    if (!llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, n_split)) {
        fatal("'%s' declares %d splits but is not named like shard 1", path.c_str(), n_split);
    }
    std::vector<std::string> paths;
    for (int i = 0; i < n_split; i++) {
        char p[4096];
        llama_split_path(p, sizeof(p), prefix, i, n_split);
        paths.emplace_back(p);
    }
    return paths;
}

model_file::model_file(const std::string & path) : path(path) {
    shards.emplace_back(new shard(path));
    const auto paths = shard_paths(path, shards[0]->gguf);
    for (size_t i = 1; i < paths.size(); i++) {
        shards.emplace_back(new shard(paths[i]));
    }
    for (const auto & s : shards) {
        const size_t data_offset = gguf_get_data_offset(s->gguf);
        for (int64_t i = 0; i < gguf_get_n_tensors(s->gguf); i++) {
            const char *        name = gguf_get_tensor_name(s->gguf, i);
            const ggml_tensor * t    = ggml_get_tensor(s->meta, name);
            tensor_ref          ref;
            ref.name = name;
            ref.type = t->type;
            memcpy(ref.ne, t->ne, sizeof(ref.ne));
            ref.data = (const uint8_t *) s->map + data_offset + gguf_get_tensor_offset(s->gguf, i);
            if (ref.data + ggml_nbytes(t) > (const uint8_t *) s->map + s->size) {
                fatal("tensor '%s' extends past the end of its file", name);
            }
            tensors.push_back(ref);
        }
    }
}

model_file::~model_file() = default;

const tensor_ref * model_file::find(const std::string & name) const {
    for (const auto & t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

std::string model_file::get_str(const std::string & key, const std::string & def) const {
    const int64_t kid = gguf_find_key(shards[0]->gguf, key.c_str());
    if (kid < 0 || gguf_get_kv_type(shards[0]->gguf, kid) != GGUF_TYPE_STRING) {
        return def;
    }
    return gguf_get_val_str(shards[0]->gguf, kid);
}

int64_t model_file::get_int(const std::string & key, int64_t def) const {
    const gguf_context * g   = shards[0]->gguf;
    const int64_t        kid = gguf_find_key(g, key.c_str());
    if (kid < 0) {
        return def;
    }
    switch (gguf_get_kv_type(g, kid)) {
        case GGUF_TYPE_UINT8:
            return gguf_get_val_u8(g, kid);
        case GGUF_TYPE_INT8:
            return gguf_get_val_i8(g, kid);
        case GGUF_TYPE_UINT16:
            return gguf_get_val_u16(g, kid);
        case GGUF_TYPE_INT16:
            return gguf_get_val_i16(g, kid);
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(g, kid);
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(g, kid);
        case GGUF_TYPE_UINT64:
            return (int64_t) gguf_get_val_u64(g, kid);
        case GGUF_TYPE_INT64:
            return gguf_get_val_i64(g, kid);
        default:
            return def;
    }
}
