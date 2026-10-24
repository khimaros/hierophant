#include "sites.h"

#include "util.h"

#include <cstring>
#include <map>
#include <set>

static const char * BLOCK_PREFIX  = "blk.";
static const char * WEIGHT_SUFFIX = ".weight";
static const char * OUTPUT_HEAD   = "output.weight";

// per-block matrices whose output is added to the residual stream. with
// hyper-connections each stream adds a per-token scalar multiple of the
// block output, so ablating these still keeps every stream clean.
static const std::set<std::string> WRITER_KINDS = {
    "attn_output", "attn_output_b",   "ffn_down",          "ffn_down_exps", "ffn_down_shexp",     "ffn_down_chexps",
    "ssm_out",     "time_mix_output", "channel_mix_value", "ple_value",     "shortconv.out_proj",
};

static const std::set<std::string> ATTN_KINDS = {
    "attn_output", "attn_output_b", "ssm_out", "time_mix_output", "shortconv.out_proj",
};

// norms applied to a writer's output before the residual add. they rescale per
// channel, so removing a direction before them does not remove it after.
static const std::set<std::string> POST_NORM_KINDS = {
    "post_attention_norm", "post_ffw_norm", "post_ffw_norm_1", "post_ffw_norm_2",
    "attn_post_norm",      "ffn_post_norm", "post_norm",       "layer_output_norm",
};

// qwen4exp adds silu(conv(rms_norm(gated ple value))) beside the gated value
// itself; the norm and the nonlinearity bring the direction back.
static const char * PLE_CONV_KIND = "ple_conv1d";

// split "blk.12.ffn_down.weight" into (12, "ffn_down", "weight").
static bool parse_block_name(const std::string & name, int & layer, std::string & kind, std::string & leaf) {
    const size_t p = strlen(BLOCK_PREFIX);
    if (name.rfind(BLOCK_PREFIX, 0) != 0) {
        return false;
    }
    const size_t a = name.find('.', p);
    const size_t b = name.rfind('.');
    if (a == std::string::npos || b <= a) {
        return false;
    }
    layer = std::stoi(name.substr(p, a - p));
    kind  = name.substr(a + 1, b - a - 1);
    leaf  = name.substr(b + 1);
    return true;
}

const char * group_of(const std::string & kind) {
    if (kind == KIND_TOKEN_EMBD) {
        return GROUP_EMBD;
    }
    return ATTN_KINDS.count(kind) ? GROUP_ATTN : GROUP_MLP;
}

site_report find_sites(const model_file & model, const std::vector<std::string> & kinds) {
    // an explicitly requested kind counts as a writer, so new archs need no code change.
    const std::set<std::string> wanted(kinds.begin(), kinds.end());
    const auto                  is_writer = [&](const std::string & k) {
        return WRITER_KINDS.count(k) || wanted.count(k);
    };
    const auto want = [&](const std::string & k) {
        return wanted.empty() || wanted.count(k);
    };
    const int64_t n_embd = model.n_embd();

    site_report                r;
    std::map<std::string, int> post_norms;
    std::set<std::string>      unclassified, biased, found;
    std::vector<int>           ple_conv_layers;

    for (const auto & t : model.tensors) {
        int         layer;
        std::string kind, leaf;
        if (!parse_block_name(t.name, layer, kind, leaf)) {
            continue;
        }
        if (kind == PLE_CONV_KIND && leaf == "weight") {
            ple_conv_layers.push_back(layer);
        }
        if (POST_NORM_KINDS.count(kind) && leaf == "weight") {
            post_norms[kind]++;
        }
        if (is_writer(kind) && want(kind) && leaf == "bias") {
            biased.insert(kind);
        }
        if (leaf != "weight" || t.n_dims() < 2) {
            continue;
        }
        if (!is_writer(kind)) {
            if (t.ne[1] == n_embd && t.ne[0] != n_embd) {
                unclassified.insert(kind);
            }
            continue;
        }
        if (!want(kind)) {
            continue;
        }
        if (t.ne[1] != n_embd) {
            r.warnings.push_back(t.name + " does not output n_embd values, skipped");
            continue;
        }
        found.insert(kind);
        r.sites.push_back({ &t, kind, layer });
    }

    const tensor_ref * embd = model.find(std::string(KIND_TOKEN_EMBD) + WEIGHT_SUFFIX);
    if (embd && want(KIND_TOKEN_EMBD)) {
        found.insert(KIND_TOKEN_EMBD);
        if (!model.find(OUTPUT_HEAD)) {
            r.warnings.push_back(
                "token_embd is tied to the output head, where llama.cpp would also apply the adapter; skipped");
        } else {
            r.sites.insert(r.sites.begin(), { embd, KIND_TOKEN_EMBD, -1 });
        }
    }
    for (const auto & [kind, n] : post_norms) {
        r.warnings.push_back(kind + " follows a writer in " + std::to_string(n) +
                             " layers; ablation there is approximate");
    }
    for (const auto & kind : biased) {
        r.warnings.push_back(kind + " has a bias, whose refusal component a LoRA cannot remove");
    }
    for (int layer : ple_conv_layers) {
        r.warnings.push_back("layer " + std::to_string(layer) +
                             " adds a normalized conv of its ple value, which a LoRA cannot clean; approximate there");
    }
    for (const auto & k : wanted) {
        if (!found.count(k)) {
            r.warnings.push_back("no tensor of kind '" + k + "' found");
        }
    }
    r.unclassified.assign(unclassified.begin(), unclassified.end());
    return r;
}
