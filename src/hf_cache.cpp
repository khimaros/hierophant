#include "hf_cache.h"

#include "arg.h"
#include "common.h"
#include "download.h"

static const char * TOKEN_ENV   = "HF_TOKEN";
static const char * OFFLINE_ENV = "LLAMA_ARG_OFFLINE";

static std::string env(const char * name) {
    const char * v = getenv(name);
    return v ? v : "";
}

// llama-server's -hf path, through the same calls: plan from the hub (from the
// cache when offline or unreachable), download what is missing, load shard 1.
static model_source resolve_hf(const args & a) {
    common_params params;
    params.model.hf_repo = a.str("--hf-repo");
    params.model.hf_file = a.str("--hf-file");
    params.hf_token      = a.str("--hf-token", env(TOKEN_ENV));
    params.offline       = a.has("--offline") || !env(OFFLINE_ENV).empty();
    std::string repo;
    try {
        repo = common_download_split_repo_tag(params.model.hf_repo).first;
    } catch (const std::exception & e) {
        fatal("--hf-repo %s: %s", params.model.hf_repo.c_str(), e.what());
    }
    auto handler = common_models_handler_init(params, LLAMA_EXAMPLE_COMMON);
    common_models_handler_apply(handler, params);
    if (params.model.path.empty()) {
        fatal("%s%s%s not found%s", params.model.hf_repo.c_str(), params.model.hf_file.empty() ? "" : " ",
              params.model.hf_file.c_str(), params.offline ? " in the cache (--offline)" : "");
    }
    return { params.model.path, repo, handler.plan.primary.path };
}

model_source resolve_model(const args & a) {
    return a.has("--hf-repo") ? resolve_hf(a) : model_source{ a.need("-m"), "", "" };
}
