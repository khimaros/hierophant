#pragma once

#include "util.h"

#include <string>

// a base model named either by local path (-m) or as llama-server's -hf names
// one (--hf-repo REPO[:QUANT] [--hf-file FILE] [--hf-token T] [--offline]),
// downloaded into the huggingface cache unless it is there or --offline is given.
struct model_source {
    std::string path;
    std::string repo; // without the quant; empty for a plain path
    std::string file; // path inside the repo
};

model_source resolve_model(const args & a);
