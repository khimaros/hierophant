// kl divergence of the next-token distribution from the base model's over every
// position of a text, as llama-perplexity --kl-divergence measures it, for several
// adapters and scales from one model load. the base's log-probabilities for every
// chunk are kept in memory rather than in a file, so that each run applies its
// adapter once: every change of adapters re-plans the compute graph.

#include "common.h"
#include "hf_cache.h"
#include "llama.h"
#include "util.h"

#include <fstream>
#include <iterator>

static const int   DEFAULT_N_CTX = 512;
static const float MIN_BASE_LOGP = -16.0f; // base tokens less likely than this are left out, as llama-perplexity does

struct kl_run {
    std::string          path;
    float                scale   = 0.0f;
    llama_adapter_lora * adapter = nullptr;
    double               sum     = 0.0;
    size_t               same = 0, count = 0;
};

// what each chunk is scored on: as in llama-perplexity, the positions of its second half
// that predict a token of it
struct chunk_rows {
    int                first, n_rows, n_vocab;
    std::vector<float> base; // [chunk][n_rows][n_vocab] log-probabilities
    std::vector<int>   base_top;

    size_t row(int chunk, int r) const { return (size_t) chunk * n_rows + r; }
};

// log-probabilities of one row of logits; returns its most likely token
static int log_softmax(const float * logits, int n, float * out) {
    int imax = 0;
    for (int i = 1; i < n; i++) {
        imax = logits[i] > logits[imax] ? i : imax;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp(logits[i] - logits[imax]);
    }
    const float lse = logits[imax] + (float) log(sum);
    for (int i = 0; i < n; i++) {
        out[i] = logits[i] - lse;
    }
    return imax;
}

static void decode_chunk(llama_context * ctx, llama_batch & batch, const llama_token * tokens, int n_ctx, int first) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    batch.n_tokens            = n_ctx;
    for (int i = 0; i < n_ctx; i++) {
        batch.token[i]     = i == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i >= first;
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    if (llama_decode(ctx, batch) != 0) {
        fatal("decode failed");
    }
}

// the chunk's scored rows of logits. llama_get_logits_ith is not a plain read: its first
// call after a decode puts rows back in batch order (output_reorder) when the batch was
// split out of order, as a hybrid model's is, so it is called here and not from workers.
static std::vector<const float *> rows_of(llama_context * ctx, const chunk_rows & c) {
    std::vector<const float *> rows(c.n_rows);
    for (int r = 0; r < c.n_rows; r++) {
        rows[r] = llama_get_logits_ith(ctx, c.first + r);
    }
    return rows;
}

static void keep_base(llama_context * ctx, chunk_rows & c, int chunk) {
    const auto rows = rows_of(ctx, c);
    for (int r = 0; r < c.n_rows; r++) {
        const size_t i = c.row(chunk, r);
        c.base_top[i]  = log_softmax(rows[r], c.n_vocab, &c.base[i * c.n_vocab]);
    }
}

// add the chunk just decoded, compared row by row with the base's, to run
static void compare(llama_context * ctx, const chunk_rows & c, int chunk, kl_run & run, int n_threads) {
    const auto          rows = rows_of(ctx, c);
    std::vector<double> sums(n_threads, 0.0);
    std::vector<size_t> same(n_threads, 0);
    parallel_for(c.n_rows, n_threads, [&](int64_t b, int64_t e, int w) {
        std::vector<float> cur(c.n_vocab);
        for (int64_t r = b; r < e; r++) {
            const int     top  = log_softmax(rows[r], c.n_vocab, cur.data());
            const size_t  at   = c.row(chunk, (int) r);
            const float * base = &c.base[at * c.n_vocab];
            for (int i = 0; i < c.n_vocab; i++) {
                if (base[i] > MIN_BASE_LOGP) {
                    sums[w] += exp(base[i]) * (base[i] - cur[i]);
                }
            }
            same[w] += top == c.base_top[at];
        }
    });
    for (int w = 0; w < n_threads; w++) {
        run.sum += sums[w];
        run.same += same[w];
    }
    run.count += c.n_rows;
}

static std::vector<kl_run> runs_of(const args & a, llama_model * model) {
    const auto paths = a.all("--lora"), scales = a.all("--scale");
    if (paths.size() != scales.size() || paths.empty()) {
        fatal("give each --lora ADAPTER a --scale S after it");
    }
    std::map<std::string, llama_adapter_lora *> loaded;
    std::vector<kl_run>                         runs(paths.size());
    for (size_t i = 0; i < paths.size(); i++) {
        runs[i].path  = paths[i];
        runs[i].scale = std::stof(scales[i]);
        auto *& lora  = loaded[paths[i]];
        lora          = lora ? lora : llama_adapter_lora_init(model, paths[i].c_str());
        if (!lora) {
            fatal("failed to load lora '%s'", paths[i].c_str());
        }
        runs[i].adapter = lora;
    }
    return runs;
}

static std::vector<llama_token> text_tokens(const args & a, llama_model * model) {
    std::ifstream f(a.need("--text"));
    if (!f) {
        fatal("failed to open '%s'", a.str("--text").c_str());
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!text.empty() && text.back() == '\n') {
        text.pop_back(); // as llama-perplexity's -f reads it
    }
    return common_tokenize(llama_model_get_vocab(model), text, true);
}

static void print_runs(const std::vector<kl_run> & runs) {
    for (const auto & r : runs) {
        const double kld  = r.scale == 0.0f ? 0.0 : r.sum / r.count;
        const double same = r.scale == 0.0f ? 1.0 : (double) r.same / r.count;
        printf("{\"lora\": \"%s\", \"scale\": %g, \"kld\": %.9g, \"same_top\": %.9g}\n", r.path.c_str(), r.scale, kld,
               same);
    }
    fflush(stdout);
}

// llama.cpp's warnings and errors only: its debug lines (every graph it plans) would run
// to megabytes over a few dozen runs
static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static llama_context * open_kl_context(llama_model * model, int n_ctx, int n_threads) {
    auto cparams  = llama_context_default_params();
    cparams.n_ctx = cparams.n_batch = cparams.n_ubatch = n_ctx;
    cparams.n_threads = cparams.n_threads_batch = n_threads;
    llama_context * ctx                         = llama_init_from_model(model, cparams);
    if (!ctx) {
        fatal("failed to create context");
    }
    return ctx;
}

// the base over every chunk, then each run over every chunk with its adapter applied once
static void measure(llama_context *                  ctx,
                    const std::vector<llama_token> & tokens,
                    int                              n_chunk,
                    int                              n_ctx,
                    std::vector<kl_run> &            runs,
                    const args &                     a) {
    const int  n_threads = a.i32("-t", default_threads());
    chunk_rows c{
        n_ctx / 2, n_ctx - 1 - n_ctx / 2, llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx))), {}, {}
    };
    c.base.resize(c.row(n_chunk, 0) * c.n_vocab);
    c.base_top.resize(c.row(n_chunk, 0));
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    const auto  n_on  = (size_t) std::count_if(runs.begin(), runs.end(), [](const kl_run & r) { return r.scale != 0; });
    progress    p(a, "kl", (size_t) n_chunk * (1 + n_on));
    size_t      done = 0;
    for (int k = 0; k < n_chunk; k++) {
        decode_chunk(ctx, batch, &tokens[(size_t) k * n_ctx], n_ctx, c.first);
        keep_base(ctx, c, k);
        p.update(++done);
    }
    for (auto & r : runs) {
        if (r.scale == 0.0f) {
            continue;
        }
        if (llama_set_adapters_lora(ctx, &r.adapter, 1, &r.scale) != 0) {
            fatal("failed to apply lora '%s'", r.path.c_str());
        }
        for (int k = 0; k < n_chunk; k++) {
            decode_chunk(ctx, batch, &tokens[(size_t) k * n_ctx], n_ctx, c.first);
            compare(ctx, c, k, r, n_threads);
            p.update(++done);
        }
    }
    llama_batch_free(batch);
}

int cmd_kl(int argc, char ** argv) {
    const args a(argc, argv, {});
    const int  n_ctx = a.i32("-c", DEFAULT_N_CTX);
    llama_log_set(quiet_log, nullptr);
    llama_model * model   = load_model(a);
    auto          runs    = runs_of(a, model);
    const auto    tokens  = text_tokens(a, model);
    const int     n_max   = (int) (tokens.size() / n_ctx);
    const int     n_chunk = a.i32("--chunks", -1) < 0 ? n_max : std::min(a.i32("--chunks", -1), n_max);
    if (n_chunk < 1) {
        fatal("'%s' has %zu tokens, fewer than one chunk of %d", a.str("--text").c_str(), tokens.size(), n_ctx);
    }
    llama_context * ctx = open_kl_context(model, n_ctx, a.i32("-t", default_threads()));
    measure(ctx, tokens, n_chunk, n_ctx, runs, a);
    print_runs(runs);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
