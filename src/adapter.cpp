// build a LoRA that removes refusal directions from every residual writer W of
// a gguf: W' = W - w Q (Q^T W) for orthonormal Q, so A = Q^T W and B = -w Q.
// with one direction this is rank 1. A is computed from the dequantized
// weights of this exact file, so the edit is exact for it at runtime scale 1.

#include "directions.h"
#include "hf_cache.h"
#include "model_file.h"
#include "sites.h"
#include "util.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>

static const char * KV_GENERAL_TYPE     = "general.type";
static const char * KV_GENERAL_ARCH     = "general.architecture";
static const char * KV_GENERAL_NAME     = "general.name";
static const char * KV_ADAPTER_TYPE     = "adapter.type";
static const char * KV_LORA_ALPHA       = "adapter.lora.alpha";
static const char * KV_DIRECTION        = "hierophant.direction_index";
static const char * KV_DIRECTIONS       = "hierophant.direction_indices";
static const char * KV_CURVE            = "hierophant.curve";
static const char * KV_SITES            = "hierophant.sites";
static const char * KV_TARGETS          = "hierophant.targets";
static const char * KV_PROJECT_HARMLESS = "hierophant.project_harmless";
static const char * KV_HARMLESS_PCS     = "hierophant.harmless_pcs";
static const char * KV_POSITIONS        = "hierophant.positions";
static const char * KV_THINK            = "hierophant.think";
static const char * SPLIT_BY_KIND       = "kind";
static const char * KV_REBASED_FROM     = "hierophant.rebased_from";
static const char * KV_SPLIT_GROUP      = "hierophant.split_group";
static const char * KV_SPLIT_LAYER      = "hierophant.split_layer";
static const char * KV_BASE_REPO        = "hierophant.base_repo";
static const char * KV_BASE_FILE        = "hierophant.base_file";
static const char * KV_HARMFUL          = "hierophant.harmful_set";
static const char * KV_N_HARMFUL        = "hierophant.n_harmful";
static const char * KV_HARMLESS         = "hierophant.harmless_set";
static const char * KV_N_HARMLESS       = "hierophant.n_harmless";
static const char * KV_RESIDUAL         = "hierophant.residual_set";
static const char * KV_N_RESIDUAL       = "hierophant.n_residual";
static const char * KV_RESIDUAL_LORA    = "hierophant.residual_lora";
static const char * KV_RESIDUAL_SCALE   = "hierophant.residual_lora_scale";
static const char * KV_VERSION          = "hierophant.llama_cpp";
static const char * LORA_A_SUFFIX       = ".lora_a";
static const char * LORA_B_SUFFIX       = ".lora_b";
static const double PER_LAYER           = -1.0;
static const char * OUTPUT_PLACEHOLDER  = "{}";
static const char * PER_LAYER_NAME      = "layer";
static const char   LIST_SEPARATOR      = ',';
static const char   RANK_SEPARATOR      = '+';
static const char * GROUPS[]            = { GROUP_ATTN, GROUP_MLP, GROUP_EMBD };
// a direction left with less than this norm after removing the earlier ones
// adds nothing new and becomes a zero column.
static const double DEGENERATE          = 1e-4;

// heretic's per-layer weight: max_weight at max_pos, falling linearly to
// min_weight at min_dist layers away, and no edit beyond that.
struct curve {
    double max_weight = 1.0;
    double max_pos    = 0.0;
    double min_weight = 1.0;
    double min_dist   = std::numeric_limits<double>::infinity();

    static curve parse(const std::string & s) {
        const auto p = split(s, LIST_SEPARATOR);
        if (p.size() != 4) {
            fatal("--curve wants max_weight,max_pos,min_weight,min_dist, got '%s'", s.c_str());
        }
        return { std::stod(p[0]), std::stod(p[1]), std::stod(p[2]), std::stod(p[3]) };
    }

    // shortest round-trip form, so a curve chosen by the search survives exactly.
    std::string str() const {
        std::string out;
        for (double v : { max_weight, max_pos, min_weight, min_dist }) {
            char buf[32];
            out += (out.empty() ? "" : ",") + std::string(buf, std::to_chars(buf, buf + sizeof(buf), v).ptr);
        }
        return out;
    }

    // weight at a layer, or nan where the curve does not reach.
    double at(int layer) const {
        const double d = std::fabs(layer - max_pos);
        if (d > min_dist) {
            return std::nan("");
        }
        return std::isinf(min_dist) ? max_weight : max_weight + (d / min_dist) * (min_weight - max_weight);
    }
};

// a curve per site group. "--curve SPEC" sets every group and "--curve
// GROUP=SPEC" one of them, in order; --weight then flattens them all.
// "--curve NAME=SPEC" names a group or a tensor kind; a kind's curve beats
// its group's.
struct curves {
    std::map<std::string, curve> by_group;
    std::map<std::string, curve> by_kind;

    static curves parse(const args & a) {
        curves out;
        for (const char * g : GROUPS) {
            out.by_group[g] = curve{};
        }
        for (const auto & s : a.all("--curve")) {
            const size_t eq = s.find('=');
            if (eq == std::string::npos) {
                for (auto & entry : out.by_group) {
                    entry.second = curve::parse(s);
                }
                continue;
            }
            const std::string name                                        = s.substr(0, eq);
            (out.by_group.count(name) ? out.by_group : out.by_kind)[name] = curve::parse(s.substr(eq + 1));
        }
        if (a.has("--weight")) {
            for (auto * m : { &out.by_group, &out.by_kind }) {
                for (auto & entry : *m) {
                    entry.second.max_weight = entry.second.min_weight = a.f64("--weight", 1.0);
                }
            }
        }
        return out;
    }

    // every kind named in a curve must be one this model has.
    void check(const std::vector<site> & sites) const {
        for (const auto & entry : by_kind) {
            if (std::none_of(sites.begin(), sites.end(), [&](const site & s) { return s.kind == entry.first; })) {
                fatal("--curve name '%s' is neither a group (attn, mlp or embd) nor a site kind of this model",
                      entry.first.c_str());
            }
        }
    }

    double at(const site & s) const {
        auto it = by_kind.find(s.kind);
        return (it != by_kind.end() ? it->second : by_group.at(group_of(s.kind))).at(std::max(s.layer, 0));
    }

    // one spec when every group agrees and no kind has its own, else "attn=...;mlp=...;embd=...;KIND=...".
    std::string str() const {
        std::string out;
        bool        same = by_kind.empty();
        for (const char * g : GROUPS) {
            same &= by_group.at(g).str() == by_group.at(GROUP_ATTN).str();
            out += std::string(out.empty() ? "" : ";") + g + "=" + by_group.at(g).str();
        }
        for (const auto & entry : by_kind) {
            out += ";" + entry.first + "=" + entry.second.str();
        }
        return same ? by_group.at(GROUP_ATTN).str() : out;
    }
};

using vecs = std::vector<std::vector<float>>;

// r^T W for each 2D slice of a tensor (one d_in vector per expert) and each
// direction, dequantizing every row once however many directions there are.
static vecs project_slices(const tensor_ref & t, const vecs & rs, int n_threads) {
    const int64_t d_in = t.ne[0], d_out = t.ne[1], n_slices = t.ne[2];
    const size_t  k = rs.size();
    vecs          out(k, std::vector<float>(n_slices * d_in));
    for (int64_t s = 0; s < n_slices; s++) {
        std::vector<std::vector<double>> acc(n_threads, std::vector<double>(k * d_in, 0.0));
        parallel_for(d_out, n_threads, [&](int64_t b, int64_t e, int w) {
            std::vector<float> row(d_in);
            for (int64_t i = b; i < e; i++) {
                t.dequantize_row(s * d_out + i, row.data());
                for (size_t d = 0; d < k; d++) {
                    double *     a  = &acc[w][d * d_in];
                    const double ri = rs[d][i];
                    for (int64_t j = 0; j < d_in; j++) {
                        a[j] += ri * row[j];
                    }
                }
            }
        });
        for (size_t d = 0; d < k; d++) {
            for (int64_t j = 0; j < d_in; j++) {
                double v = 0;
                for (const auto & a : acc) {
                    v += a[d * d_in + j];
                }
                out[d][s * d_in + j] = (float) v;
            }
        }
    }
    return out;
}

// r . e_t for every row of an embedding table and each direction.
static vecs project_rows(const tensor_ref & t, const vecs & rs, int n_threads) {
    const int64_t n_embd = t.ne[0];
    vecs          out(rs.size(), std::vector<float>(t.n_rows()));
    parallel_for(t.n_rows(), n_threads, [&](int64_t b, int64_t e, int) {
        std::vector<float> row(n_embd);
        for (int64_t i = b; i < e; i++) {
            t.dequantize_row(i, row.data());
            for (size_t d = 0; d < rs.size(); d++) {
                double v = 0;
                for (int64_t j = 0; j < n_embd; j++) {
                    v += (double) rs[d][j] * row[j];
                }
                out[d][i] = (float) v;
            }
        }
    });
    return out;
}

// gram-schmidt. a direction already in the span of the earlier ones becomes a
// zero column, so every site of an adapter keeps the same rank.
static vecs orthonormal(vecs qs) {
    for (size_t c = 0; c < qs.size(); c++) {
        auto & q = qs[c];
        for (size_t p = 0; p < c; p++) {
            double d = 0;
            for (size_t i = 0; i < q.size(); i++) {
                d += (double) q[i] * qs[p][i];
            }
            for (size_t i = 0; i < q.size(); i++) {
                q[i] = (float) (q[i] - d * qs[p][i]);
            }
        }
        double n = 0;
        for (float x : q) {
            n += (double) x * x;
        }
        n = std::sqrt(n);
        for (float & x : q) {
            x = n < DEGENERATE ? 0.0f : (float) (x / n);
        }
    }
    return qs;
}

struct lora_pair {
    std::string        name; // base tensor name
    int64_t            rank;
    int64_t            ne_a[3], ne_b[3];
    std::vector<float> a, b;
};

// one site's pair for k directions, in the layouts llama.cpp expects (see
// llama-adapter.cpp): A [n_in, k, n_slices] and B [k, n_out, n_slices], except
// that an embedding is a lookup, so its A is [k, n_vocab].
static lora_pair assemble(const site & s, const vecs & q, const vecs & as, double weight) {
    const tensor_ref & t        = *s.tensor;
    const bool         embd     = s.kind == KIND_TOKEN_EMBD;
    const int64_t      n_slices = embd ? 1 : t.ne[2];
    const int64_t      n_in     = embd ? t.ne[1] : t.ne[0];
    const int64_t      n_out    = embd ? t.ne[0] : t.ne[1];
    const int64_t      k        = (int64_t) q.size();
    lora_pair          p{
        t.name, k, { embd ? k : n_in, embd ? n_in : k, n_slices },
          { k, n_out, n_slices },
          {},
          {}
    };
    p.a.resize(n_in * k * n_slices);
    p.b.resize(n_out * k * n_slices);
    for (int64_t e = 0; e < n_slices; e++) {
        for (int64_t c = 0; c < k; c++) {
            for (int64_t j = 0; j < n_in; j++) {
                p.a[embd ? j * k + c : (e * k + c) * n_in + j] = as[c][e * n_in + j];
            }
            for (int64_t i = 0; i < n_out; i++) {
                p.b[(e * n_out + i) * k + c] = (float) (-weight * q[c][i]);
            }
        }
    }
    return p;
}

// a pair for each output, from that output's orthonormal directions.
static std::vector<lora_pair> build_pairs(const site & s, const std::vector<vecs> & qs, double weight, int n_threads) {
    vecs flat;
    for (const auto & q : qs) {
        flat.insert(flat.end(), q.begin(), q.end());
    }
    const bool embd = s.kind == KIND_TOKEN_EMBD;
    const vecs as   = embd ? project_rows(*s.tensor, flat, n_threads) : project_slices(*s.tensor, flat, n_threads);
    std::vector<lora_pair> pairs;
    size_t                 first = 0;
    for (const auto & q : qs) {
        pairs.push_back(assemble(s, q, vecs(as.begin() + first, as.begin() + first + q.size()), weight));
        first += q.size();
    }
    return pairs;
}

static ggml_tensor * new_tensor(ggml_context *             ctx,
                                const std::string &        name,
                                const int64_t              ne[3],
                                const std::vector<float> & data,
                                ggml_type                  type) {
    ggml_tensor * t =
        ne[2] > 1 ? ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]) : ggml_new_tensor_2d(ctx, type, ne[0], ne[1]);
    ggml_set_name(t, name.c_str());
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(data.data(), (ggml_fp16_t *) t->data, (int64_t) data.size());
    } else {
        memcpy(t->data, data.data(), data.size() * sizeof(float));
    }
    return t;
}

// an abliteration adapter read back: each site's B is -w Q, so it carries the
// directions and the weight; only A depends on the base weights.
struct source_adapter {
    gguf_context *              kv   = nullptr;
    int64_t                     rank = 0;
    std::map<std::string, vecs> b; // columns of the first slice of each lora_b, by base tensor name
    std::string                 path;

    explicit source_adapter(const std::string & path) : path(path) {
        ggml_context *   ctx    = nullptr;
        gguf_init_params params = { false, &ctx };
        kv                      = gguf_init_from_file(path.c_str(), params);
        if (!kv) {
            fatal("failed to read adapter '%s'", path.c_str());
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            const std::string name = t->name;
            if (ends_with(name, LORA_B_SUFFIX)) {
                b[name.substr(0, name.size() - strlen(LORA_B_SUFFIX))] = columns(t);
            }
        }
        ggml_free(ctx);
    }

    ~source_adapter() { gguf_free(kv); }

    vecs columns(const ggml_tensor * t) {
        const int64_t k = t->ne[0], n = t->ne[1];
        if (rank && k != rank) {
            fatal("'%s' mixes ranks %lld and %lld", path.c_str(), (long long) rank, (long long) k);
        }
        rank = k;
        std::vector<float> v(k * n);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, v.data(), k * n);
        } else if (t->type == GGML_TYPE_F32) {
            memcpy(v.data(), t->data, k * n * sizeof(float));
        } else {
            fatal("'%s' in '%s' is %s; expected f32 or f16", t->name, path.c_str(), ggml_type_name(t->type));
        }
        vecs out(k, std::vector<float>(n));
        for (int64_t i = 0; i < n; i++) {
            for (int64_t c = 0; c < k; c++) {
                out[c][i] = v[i * k + c];
            }
        }
        return out;
    }

    // the site's directions and weight, or false if the adapter skips it.
    bool site(const std::string & name, vecs & q, double & w) const {
        auto it = b.find(name);
        if (it == b.end()) {
            return false;
        }
        w = 0;
        for (const auto & col : it->second) {
            double n = 0;
            for (float x : col) {
                n += (double) x * x;
            }
            w = std::max(w, std::sqrt(n));
        }
        if (w == 0) {
            return false;
        }
        q = it->second;
        for (auto & col : q) {
            for (float & x : col) {
                x = (float) (-x / w);
            }
        }
        return true;
    }
};

// one output adapter: its name in -o, and a direction index per removed
// direction ("33", "33+20" or "layer").
struct output_spec {
    std::string         name;
    std::vector<double> indices;
};

// what every output records about how it was made.
struct provenance {
    const model_file &     model;
    const model_source &   source;
    const directions *     dirs;
    const directions *     residual;
    const source_adapter * from;
    const curves &         c;
};

static std::string targets_of(const std::vector<site> & used) {
    std::vector<std::string> kinds;
    std::string              out;
    for (const auto & s : used) {
        if (std::find(kinds.begin(), kinds.end(), s.kind) == kinds.end()) {
            kinds.push_back(s.kind);
            out += (out.empty() ? "" : ",") + s.kind;
        }
    }
    return out;
}

// how the directions were captured and chosen.
static void directions_metadata(gguf_context * g, const directions & d, const output_spec & o, const curves & c) {
    gguf_set_val_f32(g, KV_DIRECTION, (float) o.indices[0]);
    gguf_set_val_str(g, KV_DIRECTIONS, o.name.c_str());
    gguf_set_val_str(g, KV_CURVE, c.str().c_str());
    gguf_set_val_bool(g, KV_PROJECT_HARMLESS, d.project_harmless);
    gguf_set_val_u32(g, KV_HARMLESS_PCS, (uint32_t) d.harmless_pcs);
    gguf_set_val_u32(g, KV_POSITIONS, (uint32_t) d.positions);
    gguf_set_val_bool(g, KV_THINK, d.think);
    gguf_set_val_str(g, KV_HARMFUL, d.harmful_set.c_str());
    gguf_set_val_u32(g, KV_N_HARMFUL, (uint32_t) d.n_harmful);
    gguf_set_val_str(g, KV_HARMLESS, d.harmless_set.c_str());
    gguf_set_val_u32(g, KV_N_HARMLESS, (uint32_t) d.n_harmless);
}

// llama.cpp's adapter keys, plus enough provenance to describe the adapter
// without the files it was built from. alpha = rank makes the runtime lora
// scale the global strength. a rebased adapter keeps its source's keys.
static gguf_context *
adapter_metadata(const provenance & pv, const output_spec & o, const std::vector<site> & used, int64_t rank) {
    gguf_context * g = gguf_init_empty();
    if (pv.from) {
        gguf_set_kv(g, pv.from->kv);
        gguf_set_val_str(g, KV_REBASED_FROM, basename_of(pv.from->path).c_str());
    }
    gguf_set_val_str(g, KV_GENERAL_TYPE, "adapter");
    gguf_set_val_str(g, KV_GENERAL_ARCH, pv.model.arch().c_str());
    gguf_set_val_str(g, KV_GENERAL_NAME, "hierophant abliteration");
    gguf_set_val_str(g, KV_ADAPTER_TYPE, "lora");
    gguf_set_val_f32(g, KV_LORA_ALPHA, (float) rank);
    gguf_set_val_u32(g, KV_SITES, (uint32_t) used.size());
    gguf_set_val_str(g, KV_TARGETS, targets_of(used).c_str());
    gguf_set_val_str(g, KV_BASE_REPO, pv.source.repo.c_str());
    gguf_set_val_str(g, KV_BASE_FILE,
                     pv.source.repo.empty() ? basename_of(pv.model.path).c_str() : pv.source.file.c_str());
    gguf_set_val_str(g, KV_VERSION, LLAMA_CPP_COMMIT);
    if (pv.dirs) {
        directions_metadata(g, *pv.dirs, o, pv.c);
    }
    if (pv.residual) {
        gguf_set_val_str(g, KV_RESIDUAL, pv.residual->harmful_set.c_str());
        gguf_set_val_u32(g, KV_N_RESIDUAL, (uint32_t) pv.residual->n_harmful);
        gguf_set_val_str(g, KV_RESIDUAL_LORA, pv.residual->lora.c_str());
        gguf_set_val_f32(g, KV_RESIDUAL_SCALE, (float) pv.residual->lora_scale);
    }
    return g;
}

static void
write_adapter(const std::string & path, gguf_context * g, const std::vector<lora_pair> & pairs, ggml_type type) {
    size_t bytes = 0;
    for (const auto & p : pairs) {
        bytes += (p.a.size() + p.b.size()) * ggml_type_size(type) + 2 * GGML_MEM_ALIGN;
    }
    ggml_init_params params = { 2 * pairs.size() * ggml_tensor_overhead() + bytes, nullptr, false };
    ggml_context *   ctx    = ggml_init(params);
    for (const auto & p : pairs) {
        gguf_add_tensor(g, new_tensor(ctx, p.name + LORA_A_SUFFIX, p.ne_a, p.a, type));
        gguf_add_tensor(g, new_tensor(ctx, p.name + LORA_B_SUFFIX, p.ne_b, p.b, type));
    }
    if (!gguf_write_to_file(g, path.c_str(), false)) {
        fatal("failed to write '%s'", path.c_str());
    }
    gguf_free(g);
    ggml_free(ctx);
}

static ggml_type parse_type(const std::string & s) {
    if (s == "f32") {
        return GGML_TYPE_F32;
    }
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    fatal("--type must be f32 or f16, got '%s'", s.c_str());
}

// -o with its "{}" replaced by name, which several outputs require.
static std::string output_path(const std::string & tmpl, const std::string & name, bool required) {
    const size_t at = tmpl.find(OUTPUT_PLACEHOLDER);
    if (at == std::string::npos) {
        if (required) {
            fatal("-o needs a %s placeholder when several adapters are written", OUTPUT_PLACEHOLDER);
        }
        return tmpl;
    }
    return tmpl.substr(0, at) + name + tmpl.substr(at + strlen(OUTPUT_PLACEHOLDER));
}

// "--direction-index 20,layer,26.5+20"; no list means each layer's own direction.
static std::vector<output_spec> parse_outputs(const std::string & list) {
    std::vector<output_spec> out;
    for (const auto & entry : split(list, LIST_SEPARATOR)) {
        output_spec o{ entry, {} };
        for (const auto & s : split(entry, RANK_SEPARATOR)) {
            o.indices.push_back(s == PER_LAYER_NAME ? PER_LAYER : std::stod(s));
        }
        out.push_back(o);
    }
    return out.empty() ?
               std::vector<output_spec>{
                   { PER_LAYER_NAME, { PER_LAYER } }
    } :
               out;
}

// directions for one site, per output, and the site's weight; false to skip it.
using site_plan = std::function<bool(const site &, std::vector<vecs> &, double &)>;

// a residual's direction at an output's first index joins that output's last.
static site_plan plan_from_directions(const directions &               dirs,
                                      const directions *               residual,
                                      const std::vector<output_spec> & outs,
                                      const curves &                   c) {
    std::vector<vecs> globals; // an empty vector where an index is per-layer
    for (const auto & o : outs) {
        vecs g;
        for (double index : o.indices) {
            g.push_back(index == PER_LAYER ? std::vector<float>{} : dirs.interpolated(index));
        }
        if (residual) {
            g.push_back(residual->interpolated(o.indices[0]));
        }
        globals.push_back(g);
    }
    return [&dirs, c, globals](const site & s, std::vector<vecs> & qs, double & w) {
        const int layer = std::max(s.layer, 0);
        w               = c.at(s);
        if (std::isnan(w) || w == 0) {
            return false;
        }
        if (layer >= dirs.n_layer) {
            fprintf(stderr, "warning: %s is past the last captured layer, skipped\n", s.tensor->name.c_str());
            return false;
        }
        qs = globals;
        for (auto & q : qs) {
            for (auto & r : q) {
                if (r.empty()) {
                    r = dirs.refusal(layer);
                }
            }
            q = orthonormal(q);
        }
        return true;
    };
}

static site_plan plan_from_adapter(const source_adapter & from, int64_t n_embd) {
    return [&from, n_embd](const site & s, std::vector<vecs> & qs, double & w) {
        vecs q;
        if (!from.site(s.tensor->name, q, w)) {
            return false;
        }
        if ((int64_t) q[0].size() != n_embd) {
            fatal("%s in '%s' has %zu values, the model has n_embd %lld", s.tensor->name.c_str(), from.path.c_str(),
                  q[0].size(), (long long) n_embd);
        }
        qs = { q };
        return true;
    };
}

// one adapter per (group, layer) at weight 1, so a search can weight each per
// request: "{}" becomes "attn-12", "mlp-12" or "embd". by kind, the parts are
// "attn_output-12", "ffn_down_exps-12", "token_embd" and so on.
static void write_split(const std::string &            tmpl,
                        bool                           by_kind,
                        const provenance &             pv,
                        const output_spec &            o,
                        const std::vector<site> &      used,
                        const std::vector<lora_pair> & pairs,
                        ggml_type                      type) {
    std::map<std::string, std::vector<size_t>> parts;
    const auto                                 group = [&](const site & s) {
        return by_kind ? s.kind : std::string(group_of(s.kind));
    };
    for (size_t i = 0; i < used.size(); i++) {
        const std::string g = group(used[i]);
        parts[used[i].layer < 0 ? g : g + "-" + std::to_string(used[i].layer)].push_back(i);
    }
    for (const auto & [name, members] : parts) {
        std::vector<site>      sites;
        std::vector<lora_pair> ps;
        for (size_t i : members) {
            sites.push_back(used[i]);
            ps.push_back(pairs[i]);
        }
        gguf_context * g = adapter_metadata(pv, o, sites, pairs[0].rank);
        gguf_set_val_str(g, KV_SPLIT_GROUP, group(sites[0]).c_str());
        gguf_set_val_i32(g, KV_SPLIT_LAYER, sites[0].layer);
        write_adapter(output_path(tmpl, name, true), g, ps, type);
    }
    fprintf(stderr, "wrote %zu split adapters (%zu sites)\n", parts.size(), used.size());
}

// every site's pairs, one list per output, and the sites that were used.
static std::vector<std::vector<lora_pair>> build_all(const site_report & report,
                                                     const site_plan &   plan,
                                                     size_t              n_outputs,
                                                     int                 n_threads,
                                                     const args &        a,
                                                     std::vector<site> & used) {
    std::vector<std::vector<lora_pair>> pairs(n_outputs);
    progress                            p(a, "sites", report.sites.size());
    for (size_t i = 0; i < report.sites.size(); i++) {
        const site &      s = report.sites[i];
        std::vector<vecs> qs;
        double            w;
        if (plan(s, qs, w)) {
            auto built = build_pairs(s, qs, w, n_threads);
            for (size_t d = 0; d < built.size(); d++) {
                pairs[d].push_back(std::move(built[d]));
            }
            used.push_back(s);
            fprintf(stderr, "site %zu/%zu: %s %s weight %.3f%s", used.size(), report.sites.size(),
                    s.tensor->name.c_str(), ggml_type_name(s.tensor->type), w, p.eol().c_str());
        }
        p.update(i + 1);
    }
    if (used.empty()) {
        fatal("no sites to ablate");
    }
    return pairs;
}

static std::unique_ptr<directions> load_directions(const args & a, const model_file & model, const std::string & path) {
    std::unique_ptr<directions> dirs(new directions(directions::load(path)));
    dirs->project_harmless = !a.has("--no-project-harmless");
    dirs->harmless_pcs     = a.i32("--harmless-pcs", 0);
    if (dirs->harmless_pcs > 0 && dirs->harmless_rows.empty()) {
        fatal("--harmless-pcs needs directions captured with --keep-rows");
    }
    if (dirs->arch != model.arch() || dirs->n_embd != model.n_embd()) {
        fatal("directions are for %s (n_embd %lld), model is %s (n_embd %lld)", dirs->arch.c_str(),
              (long long) dirs->n_embd, model.arch().c_str(), (long long) model.n_embd());
    }
    return dirs;
}

// options that cannot go together, refused before any work.
static void check_options(const args & a, const std::vector<output_spec> & outs, bool rebase) {
    if (a.has("--split") && (outs.size() > 1 || a.has("--curve") || a.has("--weight"))) {
        fatal("--split takes one direction index and no --curve or --weight; the parts are weighted at runtime");
    }
    const auto per_layer = [](const output_spec & o) {
        return o.indices[0] == PER_LAYER;
    };
    if (a.has("--residual") && (rebase || std::any_of(outs.begin(), outs.end(), per_layer))) {
        fatal("--residual needs --direction-index: its direction is taken at the first index of each output");
    }
    if (a.has("--split-by") && a.str("--split-by") != SPLIT_BY_KIND && a.str("--split-by") != "group") {
        fatal("--split-by must be group or kind, got '%s'", a.str("--split-by").c_str());
    }
}

int cmd_adapter(int argc, char ** argv) {
    // --project-harmless is the default and still accepted
    const args   a(argc, argv, { "--project-harmless", "--no-project-harmless", "--split" });
    const bool   rebase = a.has("--from-adapter"), split_out = a.has("--split");
    const auto   outs = rebase ? std::vector<output_spec>(1) : parse_outputs(a.str("--direction-index"));
    const curves c    = curves::parse(a);
    check_options(a, outs, rebase);
    const std::string  tmpl   = a.need("-o");
    const model_source source = resolve_model(a);
    const model_file   model(source.path);
    const int          n_threads = a.i32("-t", default_threads());
    const auto         report    = find_sites(model, split(a.str("--targets"), LIST_SEPARATOR));
    for (const auto & w : report.warnings) {
        fprintf(stderr, "warning: %s\n", w.c_str());
    }
    c.check(report.sites);

    std::unique_ptr<directions>     dirs, residual;
    std::unique_ptr<source_adapter> from;
    if (rebase) {
        from.reset(new source_adapter(a.str("--from-adapter")));
    } else {
        dirs = load_directions(a, model, a.need("-d"));
    }
    if (a.has("--residual")) {
        residual = load_directions(a, model, a.str("--residual"));
    }
    const site_plan plan =
        rebase ? plan_from_adapter(*from, model.n_embd()) : plan_from_directions(*dirs, residual.get(), outs, c);
    std::vector<site> used;
    const auto        pairs = build_all(report, plan, outs.size(), n_threads, a, used);
    if (from && used.size() != from->b.size()) {
        fprintf(stderr, "warning: '%s' has %zu sites, %zu of them found in this model\n", from->path.c_str(),
                from->b.size(), used.size());
    }
    const provenance pv{ model, source, dirs.get(), residual.get(), from.get(), c };
    const ggml_type  type = parse_type(a.str("--type", "f32"));
    if (split_out) {
        write_split(tmpl, a.str("--split-by") == SPLIT_BY_KIND, pv, outs[0], used, pairs[0], type);
        return 0;
    }
    for (size_t d = 0; d < outs.size(); d++) {
        const std::string path = output_path(tmpl, outs[d].name, outs.size() > 1);
        write_adapter(path, adapter_metadata(pv, outs[d], used, pairs[d][0].rank), pairs[d], type);
        fprintf(stderr, "wrote %s (%zu sites, rank %lld)\n", path.c_str(), used.size(), (long long) pairs[d][0].rank);
    }
    return 0;
}
