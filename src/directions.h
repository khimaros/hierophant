#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// mean residual stream after each layer, at the first generated position, for
// harmful and harmless prompts. row l is the output of block l.
struct directions {
    std::string        arch;
    int64_t            n_embd     = 0;
    int64_t            n_layer    = 0;
    int64_t            n_harmful  = 0;
    int64_t            n_harmless = 0;
    int64_t            positions  = 1;     // final tokens averaged per prompt
    bool               think      = false; // captured with the template's thinking on
    std::string        harmful_set;        // prompt file names, for provenance
    std::string        harmless_set;
    std::string        lora;               // adapter applied while capturing, for a residual direction
    double             lora_scale = 0;
    std::vector<float> harmful;            // [n_layer][n_embd]
    std::vector<float> harmless;           // [n_layer][n_embd]
    std::vector<float> harmless_rows;      // [n_layer][n_harmless][n_embd], with --keep-rows

    void              save(const std::string & path) const;
    static directions load(const std::string & path);

    // remove the harmless mean's component from each direction, so ablating
    // it leaves the average harmless activation alone (AtomicChat's recipe).
    // on qwen3.8 it cut refusals at scale 1.25 from 0.50 to 0.16 and halved kl.
    bool project_harmless = true;

    // also remove the top principal components of the harmless rows, the
    // directions harmless prompts vary along most. needs harmless_rows.
    int harmless_pcs = 0;

    // unit difference of means for one layer.
    std::vector<float> refusal(int64_t layer) const;

    // heretic-style fractional index: lerp between adjacent layers, renormalized.
    std::vector<float> interpolated(double index) const;

  private:
    // orthonormal directions a refusal direction is kept orthogonal to, per layer.
    mutable std::map<int64_t, std::vector<std::vector<float>>> protected_cache;

    const std::vector<std::vector<float>> & protected_basis(int64_t layer) const;
};

void normalize(std::vector<float> & v);
