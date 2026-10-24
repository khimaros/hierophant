#pragma once

#include "model_file.h"

#include <string>
#include <vector>

static const char * KIND_TOKEN_EMBD = "token_embd";
static const char * GROUP_ATTN      = "attn";
static const char * GROUP_MLP       = "mlp";
static const char * GROUP_EMBD      = "embd";

// a matrix that writes into the residual stream, i.e. an ablation site.
struct site {
    const tensor_ref * tensor;
    std::string        kind;  // tensor name without the block prefix, e.g. ffn_down_exps
    int                layer; // -1 for token_embd
};

struct site_report {
    std::vector<site>        sites;
    std::vector<std::string> warnings;
    std::vector<std::string> unclassified; // kinds that look like writers but are not known to be
};

// enumerate the residual writers of a model, restricted to `kinds` unless empty.
site_report find_sites(const model_file & model, const std::vector<std::string> & kinds);

// the component a kind belongs to, as heretic tunes them: attention (and the
// token mixers that replace it), the embedding, and mlp for everything else.
const char * group_of(const std::string & kind);
