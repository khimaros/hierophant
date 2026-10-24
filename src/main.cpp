#include "hf_cache.h"
#include "model_file.h"
#include "sites.h"
#include "util.h"

#include <cstring>

int cmd_capture(int argc, char ** argv);
int cmd_adapter(int argc, char ** argv);

static const char * USAGE = R"(usage: hierophant <command> [options]

commands:
  info     MODEL [--targets K,K]
           list the residual writers that would be ablated, and why an
           ablation of this arch may be inexact
  capture  MODEL --harmful FILE[,FILE] --harmless FILE[,FILE] -o DIRECTIONS
           [-ngl N] [-c N] [-t N] [--limit N] [--system TEXT] [--think]
           [--positions N] [--keep-rows] [--lora ADAPTER [--lora-scale S]]
           mean residual after every block at the first generated position
           (or over the last N prompt tokens); --keep-rows also stores each
           harmless prompt's residuals, for --harmless-pcs
  adapter  MODEL -d DIRECTIONS -o ADAPTER [--direction-index F[+F],...]
           [--weight W] [--curve [NAME=]MAX_W,MAX_POS,MIN_W,MIN_DIST]...
           [--targets K,K] [--type f32|f16] [--no-project-harmless]
           [--harmless-pcs K] [--split [--split-by group|kind]] [-t N]
  adapter  MODEL --from-adapter ADAPTER -o ADAPTER
           abliteration lora computed from MODEL's own weights. F+F removes
           both directions (rank 2); NAME is a group (attn, mlp, embd) or a
           tensor kind; --split writes one weight-1 adapter per group (or
           kind) and layer for the search
  resolve  MODEL
           print the local path MODEL refers to
  version  print the llama.cpp commit this build links

MODEL is either -m PATH or, as in llama-server, -hf REPO[:QUANT] [-hff FILE]
[-hft TOKEN] [--offline]: QUANT defaults to Q4_K_M, a split model loads from
shard 1, and missing files are downloaded into the huggingface cache unless
--offline is given. HF_TOKEN and LLAMA_ARG_OFFLINE work as in llama-server.

prompt files hold one prompt per line, with \n for embedded newlines.
)";

static int cmd_resolve(int argc, char ** argv) {
    printf("%s\n", resolve_model(args(argc, argv, {})).path.c_str());
    return 0;
}

static int cmd_info(int argc, char ** argv) {
    const args       a(argc, argv, {});
    const model_file model(resolve_model(a).path);
    const auto       report = find_sites(model, split(a.str("--targets"), ','));
    printf("arch %s, n_embd %lld, n_layer %lld\n", model.arch().c_str(), (long long) model.n_embd(),
           (long long) model.n_layer());
    for (const auto & s : report.sites) {
        const auto & t = *s.tensor;
        printf("site %-40s %-8s [%lld, %lld, %lld]\n", t.name.c_str(), ggml_type_name(t.type), (long long) t.ne[0],
               (long long) t.ne[1], (long long) t.ne[2]);
    }
    printf("%zu sites\n", report.sites.size());
    for (const auto & k : report.unclassified) {
        printf("unclassified %s: outputs n_embd values but is not a known writer; pass it in --targets to ablate it\n",
               k.c_str());
    }
    for (const auto & w : report.warnings) {
        printf("warning: %s\n", w.c_str());
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fputs(USAGE, stderr);
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "info") {
        return cmd_info(argc - 2, argv + 2);
    }
    if (cmd == "capture") {
        return cmd_capture(argc - 2, argv + 2);
    }
    if (cmd == "adapter") {
        return cmd_adapter(argc - 2, argv + 2);
    }
    if (cmd == "resolve") {
        return cmd_resolve(argc - 2, argv + 2);
    }
    if (cmd == "version") {
        printf("llama.cpp %s\n", LLAMA_CPP_COMMIT);
        return 0;
    }
    fputs(USAGE, stderr);
    return cmd == "help" || cmd == "--help" || cmd == "-h" ? 0 : 1;
}
