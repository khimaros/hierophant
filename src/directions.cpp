#include "directions.h"

#include "ggml.h"
#include "gguf.h"
#include "util.h"

#include <cmath>
#include <cstring>

static const char * KV_KIND       = "hierophant.kind";
static const char * KV_ARCH       = "hierophant.arch";
static const char * KV_N_HARMFUL  = "hierophant.n_harmful";
static const char * KV_N_HARMLESS = "hierophant.n_harmless";
static const char * KV_HARMFUL    = "hierophant.harmful_set";
static const char * KV_HARMLESS   = "hierophant.harmless_set";
static const char * KV_POSITIONS  = "hierophant.positions";
static const char * KV_THINK      = "hierophant.think";
static const char * KV_LORA       = "hierophant.capture_lora";
static const char * KV_LORA_SCALE = "hierophant.capture_lora_scale";
static const char * KIND          = "directions";
static const char * T_HARMFUL     = "harmful_mean";
static const char * T_HARMLESS    = "harmless_mean";
static const char * T_ROWS        = "harmless_rows";
static const int    PCA_ITERS     = 200;
// a basis vector left with less than this norm after removing the earlier
// ones adds nothing and is dropped.
static const double NEGLIGIBLE    = 1e-6;

void normalize(std::vector<float> & v) {
    double n = 0;
    for (float x : v) {
        n += (double) x * x;
    }
    n = std::sqrt(n);
    if (n == 0) {
        fatal("cannot normalize a zero vector");
    }
    for (float & x : v) {
        x = (float) (x / n);
    }
}

void directions::save(const std::string & path) const {
    const size_t bytes =
        2 * n_layer * n_embd * sizeof(float) + harmless_rows.size() * sizeof(ggml_fp16_t) + 3 * GGML_MEM_ALIGN;
    ggml_init_params params = { 3 * ggml_tensor_overhead() + bytes, nullptr, false };
    ggml_context *   ctx    = ggml_init(params);
    gguf_context *   g      = gguf_init_empty();
    gguf_set_val_str(g, KV_KIND, KIND);
    gguf_set_val_str(g, KV_ARCH, arch.c_str());
    gguf_set_val_u32(g, KV_N_HARMFUL, (uint32_t) n_harmful);
    gguf_set_val_u32(g, KV_N_HARMLESS, (uint32_t) n_harmless);
    gguf_set_val_str(g, KV_HARMFUL, harmful_set.c_str());
    gguf_set_val_str(g, KV_HARMLESS, harmless_set.c_str());
    gguf_set_val_u32(g, KV_POSITIONS, (uint32_t) positions);
    gguf_set_val_bool(g, KV_THINK, think);
    gguf_set_val_str(g, KV_LORA, lora.c_str());
    gguf_set_val_f32(g, KV_LORA_SCALE, (float) lora_scale);
    if (!harmless_rows.empty()) {
        // f16 halves what is the bulk of the file; the rows only steer which directions to avoid.
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, n_harmless, n_layer);
        ggml_set_name(t, T_ROWS);
        ggml_fp32_to_fp16_row(harmless_rows.data(), (ggml_fp16_t *) t->data, (int64_t) harmless_rows.size());
        gguf_add_tensor(g, t);
    }
    for (auto [name, src] : {
             std::pair{ T_HARMFUL,  &harmful  },
             std::pair{ T_HARMLESS, &harmless }
    }) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_layer);
        ggml_set_name(t, name);
        memcpy(t->data, src->data(), ggml_nbytes(t));
        gguf_add_tensor(g, t);
    }
    if (!gguf_write_to_file(g, path.c_str(), false)) {
        fatal("failed to write '%s'", path.c_str());
    }
    gguf_free(g);
    ggml_free(ctx);
}

directions directions::load(const std::string & path) {
    ggml_context *   ctx    = nullptr;
    gguf_init_params params = { false, &ctx };
    gguf_context *   g      = gguf_init_from_file(path.c_str(), params);
    if (!g) {
        fatal("failed to read '%s'", path.c_str());
    }
    const auto str = [&](const char * k) {
        const int64_t id = gguf_find_key(g, k);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id));
    };
    const auto u32 = [&](const char * k) {
        const int64_t id = gguf_find_key(g, k);
        return id < 0 ? 0 : (int64_t) gguf_get_val_u32(g, id);
    };
    if (str(KV_KIND) != KIND) {
        fatal("'%s' is not a hierophant directions file", path.c_str());
    }
    directions d;
    d.arch         = str(KV_ARCH);
    d.n_harmful    = u32(KV_N_HARMFUL);
    d.n_harmless   = u32(KV_N_HARMLESS);
    d.harmful_set  = str(KV_HARMFUL);
    d.harmless_set = str(KV_HARMLESS);
    d.positions    = std::max<int64_t>(1, u32(KV_POSITIONS));
    d.think        = gguf_find_key(g, KV_THINK) >= 0 && gguf_get_val_bool(g, gguf_find_key(g, KV_THINK));
    d.lora         = str(KV_LORA);
    d.lora_scale   = gguf_find_key(g, KV_LORA_SCALE) < 0 ? 0.0 : gguf_get_val_f32(g, gguf_find_key(g, KV_LORA_SCALE));
    if (const ggml_tensor * rows = ggml_get_tensor(ctx, T_ROWS)) {
        d.harmless_rows.resize(ggml_nelements(rows));
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) rows->data, d.harmless_rows.data(), ggml_nelements(rows));
    }
    for (auto [name, dst] : {
             std::pair{ T_HARMFUL,  &d.harmful  },
             std::pair{ T_HARMLESS, &d.harmless }
    }) {
        const ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t || t->type != GGML_TYPE_F32) {
            fatal("'%s' has no f32 tensor '%s'", path.c_str(), name);
        }
        d.n_embd  = t->ne[0];
        d.n_layer = t->ne[1];
        dst->assign((const float *) t->data, (const float *) t->data + ggml_nelements(t));
    }
    gguf_free(g);
    ggml_free(ctx);
    return d;
}

static double dot(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) {
        s += (double) a[i] * b[i];
    }
    return s;
}

// remove from v its components along an orthonormal basis.
static void project_out(std::vector<float> & v, const std::vector<std::vector<float>> & basis) {
    for (const auto & b : basis) {
        const double c = dot(v, b);
        for (size_t i = 0; i < v.size(); i++) {
            v[i] = (float) (v[i] - c * b[i]);
        }
    }
}

// extend an orthonormal basis by v, unless v adds nothing new.
static void extend(std::vector<std::vector<float>> & basis, std::vector<float> v) {
    const double before = std::sqrt(dot(v, v));
    project_out(v, basis);
    if (before > 0 && std::sqrt(dot(v, v)) > NEGLIGIBLE * before) {
        normalize(v);
        basis.push_back(v);
    }
}

// the next eigenvector of a symmetric n x n matrix, orthogonal to the ones
// already found, by power iteration from a fixed start.
static std::vector<double>
next_eigenvector(const std::vector<double> & gram, int64_t n, const std::vector<std::vector<double>> & found) {
    std::vector<double> v(n);
    for (int64_t i = 0; i < n; i++) {
        v[i] = std::sin(1.0 + i * (found.size() + 1.618));
    }
    for (int it = 0; it < PCA_ITERS; it++) {
        std::vector<double> w(n, 0.0);
        for (int64_t a = 0; a < n; a++) {
            for (int64_t b = 0; b < n; b++) {
                w[a] += gram[a * n + b] * v[b];
            }
        }
        for (const auto & e : found) {
            double p = 0;
            for (int64_t i = 0; i < n; i++) {
                p += w[i] * e[i];
            }
            for (int64_t i = 0; i < n; i++) {
                w[i] -= p * e[i];
            }
        }
        double norm = 0;
        for (double y : w) {
            norm += y * y;
        }
        if (norm == 0) {
            break;
        }
        for (int64_t i = 0; i < n; i++) {
            v[i] = w[i] / std::sqrt(norm);
        }
    }
    return v;
}

// top k principal directions of n rows of width d, through the n x n gram
// matrix of the centred rows (n is the prompt count, far below d); each
// eigenvector v maps back to the direction X^T v.
static std::vector<std::vector<float>> principal_components(const float * rows, int64_t n, int64_t d, int k) {
    std::vector<double> x(n * d), mean(d, 0.0), gram(n * n, 0.0);
    for (int64_t i = 0; i < n * d; i++) {
        mean[i % d] += rows[i] / n;
    }
    for (int64_t i = 0; i < n * d; i++) {
        x[i] = rows[i] - mean[i % d];
    }
    for (int64_t a = 0; a < n; a++) {
        for (int64_t b = 0; b < n; b++) {
            for (int64_t j = 0; j < d; j++) {
                gram[a * n + b] += x[a * d + j] * x[b * d + j];
            }
        }
    }
    std::vector<std::vector<double>> eig;
    std::vector<std::vector<float>>  out;
    for (int c = 0; c < std::min<int64_t>(k, n); c++) {
        eig.push_back(next_eigenvector(gram, n, eig));
        std::vector<float> pc(d, 0.0f);
        for (int64_t j = 0; j < d; j++) {
            for (int64_t i = 0; i < n; i++) {
                pc[j] += (float) (eig.back()[i] * x[i * d + j]);
            }
        }
        out.push_back(pc);
    }
    return out;
}

const std::vector<std::vector<float>> & directions::protected_basis(int64_t layer) const {
    auto it = protected_cache.find(layer);
    if (it != protected_cache.end()) {
        return it->second;
    }
    std::vector<std::vector<float>> basis;
    if (project_harmless) {
        extend(basis, std::vector<float>(harmless.begin() + layer * n_embd, harmless.begin() + (layer + 1) * n_embd));
    }
    if (harmless_pcs > 0) {
        if (harmless_rows.empty()) {
            fatal("--harmless-pcs needs directions captured with --keep-rows");
        }
        const float * rows = &harmless_rows[layer * n_harmless * n_embd];
        for (auto & pc : principal_components(rows, n_harmless, n_embd, harmless_pcs)) {
            extend(basis, pc);
        }
    }
    return protected_cache[layer] = basis;
}

std::vector<float> directions::refusal(int64_t layer) const {
    if (layer < 0 || layer >= n_layer) {
        fatal("layer %lld is outside [0, %lld)", (long long) layer, (long long) n_layer);
    }
    std::vector<float> r(n_embd);
    for (int64_t i = 0; i < n_embd; i++) {
        r[i] = harmful[layer * n_embd + i] - harmless[layer * n_embd + i];
    }
    normalize(r);
    const auto & basis = protected_basis(layer);
    if (!basis.empty()) {
        project_out(r, basis);
        normalize(r);
    }
    return r;
}

std::vector<float> directions::interpolated(double index) const {
    const int64_t lo = (int64_t) std::floor(index);
    const double  t  = index - lo;
    if (t == 0) {
        return refusal(lo);
    }
    auto       r  = refusal(lo);
    const auto hi = refusal(lo + 1);
    for (int64_t i = 0; i < n_embd; i++) {
        r[i] = (float) (r[i] + t * (hi[i] - r[i]));
    }
    normalize(r);
    return r;
}
