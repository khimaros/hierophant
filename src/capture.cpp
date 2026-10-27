// run harmful and harmless prompts through the model and record the mean
// residual stream after every block at the last prompt token, which is the
// position that produces the first generated token.

#include "chat.h"
#include "common.h"
#include "directions.h"
#include "ggml-backend.h"
#include "hf_cache.h"
#include "llama.h"
#include "util.h"

#include <cstring>
#include <fstream>

// names llama.cpp gives each block's output. hyper-connection archs (qwen4exp,
// deepseek4) use l_last for the [n_embd, n_streams, n_tokens] wide residual.
static const char * LAYER_OUTPUT_PREFIXES[] = { "l_out-", "l_last-" };
static const int    DEFAULT_N_CTX           = 4096;
static const int    PROGRESS_EVERY          = 32;

struct capture_state {
    int64_t            n_embd    = 0;
    int64_t            n_layer   = 0;
    int64_t            positions = 1; // how many final tokens are averaged
    std::vector<float> last;          // [n_layer][n_embd], the final tokens of the latest ubatch
    std::vector<bool>  seen;
    std::vector<float> slab;
    bool               warned = false;
};

// layer index of an exact block output name; views such as "l_last-46 (reshaped)" are not outputs.
static int64_t layer_of(const char * name) {
    for (const char * prefix : LAYER_OUTPUT_PREFIXES) {
        const size_t n = strlen(prefix);
        if (strncmp(name, prefix, n) == 0) {
            char *        end = nullptr;
            const int64_t il  = strtoll(name + n, &end, 10);
            return end != name + n && *end == '\0' ? il : -1;
        }
    }
    return -1;
}

// keep the mean of the final tokens' rows (the last one by default; the
// template's closing tokens with --positions); for a wide residual, also the
// mean over its streams, which is what llama.cpp itself reports as the layer input.
static bool on_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto *        s  = (capture_state *) user_data;
    const int64_t il = layer_of(t->name);
    if (il < 0) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (il >= s->n_layer) {
        return true;
    }
    const bool    wide      = t->ne[2] > 1;
    const int64_t n_streams = wide ? t->ne[1] : 1;
    const size_t  tok_nb    = wide ? t->nb[2] : t->nb[1];
    const int64_t n_tokens  = wide ? t->ne[2] : t->ne[1];
    if (t->type != GGML_TYPE_F32 || t->ne[0] != s->n_embd || t->nb[0] != sizeof(float) ||
        (wide && t->nb[1] != s->n_embd * sizeof(float))) {
        if (!s->warned) {
            fprintf(stderr, "warning: %s is not a contiguous f32 residual of width n_embd, not captured\n", t->name);
            s->warned = true;
        }
        return true;
    }
    const int64_t       n_pos = std::min(s->positions, n_tokens);
    std::vector<double> acc(s->n_embd, 0.0);
    s->slab.resize(n_streams * s->n_embd);
    for (int64_t p = n_tokens - n_pos; p < n_tokens; p++) {
        ggml_backend_tensor_get(t, s->slab.data(), p * tok_nb, s->slab.size() * sizeof(float));
        for (int64_t k = 0; k < n_streams; k++) {
            for (int64_t i = 0; i < s->n_embd; i++) {
                acc[i] += s->slab[k * s->n_embd + i];
            }
        }
    }
    float * out = &s->last[il * s->n_embd];
    for (int64_t i = 0; i < s->n_embd; i++) {
        out[i] = (float) (acc[i] / (n_streams * n_pos));
    }
    s->seen[il] = true;
    return true;
}

// one prompt per line; "\n" and "\\" are the only escapes.
static void read_prompts(const std::string & path, std::vector<std::string> & out) {
    std::ifstream f(path);
    if (!f) {
        fatal("failed to open '%s'", path.c_str());
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        std::string p;
        for (size_t i = 0; i < line.size(); i++) {
            if (line[i] == '\\' && i + 1 < line.size()) {
                const char c = line[++i];
                p += c == 'n' ? '\n' : c;
            } else {
                p += line[i];
            }
        }
        out.push_back(p);
    }
}

// the prompts of a comma-separated list of files, at most limit in all, pooled as the
// scripts' load_prompts does: each file's next prompt in turn, so a limit mixes the sets.
static std::vector<std::string> load_prompts(const std::string & paths, int limit) {
    std::vector<std::vector<std::string>> sets;
    size_t                                longest = 0;
    for (const auto & path : split(paths, ',')) {
        read_prompts(path, sets.emplace_back());
        longest = std::max(longest, sets.back().size());
    }
    std::vector<std::string> out;
    for (size_t i = 0; i < longest; i++) {
        for (const auto & s : sets) {
            if (i < s.size() && (limit <= 0 || (int) out.size() < limit)) {
                out.push_back(s[i]);
            }
        }
    }
    if (out.empty()) {
        fatal("no prompts in '%s'", paths.c_str());
    }
    return out;
}

static std::string basenames_of(const std::string & paths) {
    std::string out;
    for (const auto & path : split(paths, ',')) {
        out += (out.empty() ? "" : ",") + basename_of(path);
    }
    return out;
}

struct prompt_runner {
    llama_model *             model;
    llama_context *           ctx;
    common_chat_templates_ptr tmpls;
    std::string               system;
    bool                      think;
    capture_state &           state;
    const args &              a;

    std::vector<llama_token> tokenize(const std::string & prompt) const {
        common_chat_templates_inputs in;
        const auto                   add = [&](const char * role, const std::string & content) {
            common_chat_msg m;
            m.role    = role;
            m.content = content;
            in.messages.push_back(m);
        };
        if (!system.empty()) {
            add("system", system);
        }
        add("user", prompt);
        in.add_generation_prompt = true;
        in.enable_thinking       = think;
        const std::string text   = common_chat_templates_apply(tmpls.get(), in).prompt.str();
        return common_tokenize(llama_model_get_vocab(model), text, true, true);
    }

    // sum of the captured residuals over all prompts, [n_layer][n_embd]; with
    // rows, each prompt's own as well, [n_layer][n_prompts][n_embd].
    std::vector<double>
    run(const std::vector<std::string> & prompts, const char * label, std::vector<float> * rows = nullptr) const {
        const int64_t       n = (int64_t) prompts.size(), n_embd = state.n_embd;
        std::vector<double> sum(state.n_layer * n_embd, 0.0);
        if (rows) {
            rows->assign(state.n_layer * n * n_embd, 0.0f);
        }
        progress p(a, label, prompts.size(), PROGRESS_EVERY, "prompts");
        for (size_t i = 0; i < prompts.size(); i++) {
            auto tokens = tokenize(prompts[i]);
            if ((int64_t) tokens.size() > (int64_t) llama_n_ctx(ctx)) {
                fatal("%s prompt %zu has %zu tokens, more than the context size", label, i, tokens.size());
            }
            std::fill(state.seen.begin(), state.seen.end(), false);
            llama_memory_clear(llama_get_memory(ctx), true);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) != 0) {
                fatal("decode failed on %s prompt %zu", label, i);
            }
            for (int64_t l = 0; l < state.n_layer; l++) {
                if (!state.seen[l]) {
                    fatal("layer %lld output was not observed; this arch may not name it l_out or l_last",
                          (long long) l);
                }
            }
            for (size_t j = 0; j < sum.size(); j++) {
                sum[j] += state.last[j];
            }
            for (int64_t l = 0; rows && l < state.n_layer; l++) {
                std::copy_n(&state.last[l * n_embd], n_embd, &(*rows)[(l * n + (int64_t) i) * n_embd]);
            }
            p.update(i + 1);
        }
        return sum;
    }
};

static std::vector<float> mean_of(const std::vector<double> & sum, size_t n) {
    std::vector<float> out(sum.size());
    for (size_t i = 0; i < sum.size(); i++) {
        out[i] = (float) (sum[i] / n);
    }
    return out;
}

// a context whose eval callback fills state, with --lora applied if given.
static llama_context * open_context(const args & a, llama_model * model, capture_state & state) {
    state.n_embd    = llama_model_n_embd(model);
    state.n_layer   = llama_model_n_layer(model);
    state.positions = a.i32("--positions", 1);
    if (state.positions < 1) {
        fatal("--positions must be at least 1");
    }
    state.last.assign(state.n_layer * state.n_embd, 0.0f);
    state.seen.assign(state.n_layer, false);

    auto cparams              = llama_context_default_params();
    cparams.n_ctx             = a.i32("-c", DEFAULT_N_CTX);
    cparams.n_batch           = cparams.n_ctx;
    cparams.n_threads         = a.i32("-t", default_threads());
    cparams.n_threads_batch   = cparams.n_threads;
    cparams.cb_eval           = on_eval;
    cparams.cb_eval_user_data = &state;
    llama_context * ctx       = llama_init_from_model(model, cparams);
    if (!ctx) {
        fatal("failed to create context");
    }

    if (a.has("--lora")) {
        llama_adapter_lora * lora  = llama_adapter_lora_init(model, a.str("--lora").c_str());
        float                scale = (float) a.f64("--lora-scale", 1.0);
        if (!lora || llama_set_adapters_lora(ctx, &lora, 1, &scale) != 0) {
            fatal("failed to apply lora '%s'", a.str("--lora").c_str());
        }
    }
    return ctx;
}

int cmd_capture(int argc, char ** argv) {
    const args a(argc, argv, { "--think", "--keep-rows" });
    const int  limit    = a.i32("--limit", 0);
    const auto harmful  = load_prompts(a.need("--harmful"), limit);
    const auto harmless = load_prompts(a.need("--harmless"), limit);
    const auto out_path = a.need("-o");

    llama_backend_init();
    auto mparams         = llama_model_default_params();
    mparams.n_gpu_layers = a.i32("-ngl", mparams.n_gpu_layers);
    llama_model * model  = llama_model_load_from_file(resolve_model(a).path.c_str(), mparams);
    if (!model) {
        fatal("failed to load model");
    }
    capture_state       state;
    llama_context *     ctx = open_context(a, model, state);
    const prompt_runner runner{ model, ctx, common_chat_templates_init(model, ""), a.str("--system"), a.has("--think"),
                                state, a };

    char arch[128] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));

    directions d;
    d.arch         = arch;
    d.n_embd       = state.n_embd;
    d.n_layer      = state.n_layer;
    d.n_harmful    = (int64_t) harmful.size();
    d.n_harmless   = (int64_t) harmless.size();
    d.harmful_set  = basenames_of(a.str("--harmful"));
    d.harmless_set = basenames_of(a.str("--harmless"));
    d.positions    = state.positions;
    d.think        = a.has("--think");
    d.lora         = a.has("--lora") ? basename_of(a.str("--lora")) : "";
    d.lora_scale   = a.has("--lora") ? a.f64("--lora-scale", 1.0) : 0.0;
    d.harmful      = mean_of(runner.run(harmful, "harmful"), harmful.size());
    d.harmless =
        mean_of(runner.run(harmless, "harmless", a.has("--keep-rows") ? &d.harmless_rows : nullptr), harmless.size());
    d.save(out_path);
    fprintf(stderr, "wrote %s (%lld layers, %lld harmful, %lld harmless)\n", out_path.c_str(), (long long) d.n_layer,
            (long long) d.n_harmful, (long long) d.n_harmless);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
