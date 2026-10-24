# requirements

hierophant builds abliteration (refusal removal) LoRA adapters directly from a
quantized gguf, so a decensored variant costs one small adapter file instead of
a second copy of the weights.

## adapters

- R1. an adapter is a gguf LoRA that llama.cpp loads with `--lora` or
  `--lora-scaled`, with no patches to llama.cpp. it is rank 1 by default,
  and rank k when k directions are removed together.
- R2. the factors are computed from the dequantized weights of the exact gguf
  that will be served, so at runtime scale 1 the edit is exact for that file.
  an adapter is only valid for the base whose sha256 it records.
- R3. every matrix that writes into the residual stream is a site: attention
  and linear-attention outputs, dense and expert down projections, shared
  experts, and `token_embd` when it is not tied to the output head. writers
  are enumerated per arch rather than hardcoded to two names, and a kind the
  tool does not know can be added from the command line.
- R4. sites where the ablation cannot be exact (post-norms, biases,
  hyper-connections, tied embeddings) are reported, not silently accepted.
- R5. the runtime lora scale is the global strength (alpha = rank). per-layer
  weights are baked into B.

## directions

- R6. directions are differences of means of the residual stream after each
  block, at the first generated position, over harmful and harmless prompts
  formatted with the model's own chat template, with the harmless mean's
  component removed by default. on request: the mean over the last few
  prompt tokens, the top principal components of the harmless residuals
  removed as well, and only the prompts the base model treats as labelled.
- R7. a single global direction (fractional layer index, as in heretic) and a
  per-layer direction are both supported, with heretic's per-layer weight
  curve, set separately for attention, mlp and embedding writers, or for
  any tensor kind.

## evaluation

- R8. an adapter is scored on refusals (heretic's markers by default, or a
  model as judge; harmful test split) and first-token kl divergence against
  the base (harmless test split), at several runtime scales from one
  llama-server load, beside the log-probability of the base model's own
  replies under the adapter, a continuous measure of the same thing.
  full-sequence kl over a text is available on request.
- R9. the prompts that pick the direction are never the ones that score it.
  this holds for translated and filtered sets too (a split derives into the
  same split), extra scoring sets leave out every prompt of the train and
  test splits, and a search's chosen curve can be scored on test prompts it
  did not tune on.
- R13. weight curves and the direction index can be searched without writing
  an adapter per trial, the leaders can be re-scored on other prompts, and
  the search reports the flags that build its best trial exactly.

## output contract (agreed with aimbot)

- R10. each adapter has a sidecar `<adapter>.json` with: base repo, file and
  sha256 (shard 1 if split); kind; rank; recommended scale; sites; arch;
  targets; curve (one, or one per group); directions provenance and position
  (index, or a list of indices at rank k); eval (refusals base vs adapted
  with n, set and judge, kld mean and top1_same with set, and full-sequence
  kld when measured); tool; llama_cpp commit.

## deployment

- R11. the base can be named as llama-server names it (`--hf-repo
  REPO[:QUANT]`, `--hf-file`), including split models, and is resolved as
  llama-server resolves it: missing files are downloaded into the
  huggingface cache, and `--offline` uses only the cache.
- R12. the llama.cpp commit is pinned, and the build that writes an adapter
  is the build that validates it.
