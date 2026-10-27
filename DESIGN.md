# design

## the edit

abliteration removes a unit refusal direction r from everything written into
the residual stream: for each writer W, W' = W - w r (r^T W). the change is
rank 1, so it is a LoRA with A = r^T W and B = -w r. llama.cpp computes
W x + scale * B (A x) (with alpha = rank, `scale` is the runtime value), so at
scale 1 and w = 1 the stream never gets a component along r from that writer.

removing k directions at once uses their orthonormal basis Q (gram-schmidt,
in the order given): W' = W - w Q (Q^T W), so A = Q^T W and B = -w Q, rank
k, with alpha = k. a direction inside the span of the earlier ones becomes a
zero column rather than a lower rank, because llama.cpp scales each site by
alpha / rank and every site of an adapter must share one rank.

A is computed from this file's dequantized weights, using ggml's own
`to_float` for each row. that dequantizer is the one llama.cpp's CPU backend
uses, so the edit is exact in weight space for the served file. the remaining
runtime error comes from llama.cpp quantizing activations for the base matmul
(q8 for quantized weights, f16 for f16 weights), which the f32 lora path does
not. the e2e tests measure it: about 1e-5 relative on an f16 model.

## layouts (llama-adapter.cpp, llama-graph.cpp)

| site               | base ne                  | lora_a ne             | lora_b ne              |
|--------------------|--------------------------|-----------------------|------------------------|
| 2d writer          | [d_in, n_embd]           | [d_in, k]             | [k, n_embd]            |
| expert writer (3d) | [d_in, n_embd, n_expert] | [d_in, k, n_expert]   | [k, n_embd, n_expert]  |
| token_embd         | [n_embd, n_vocab]        | [k, n_vocab]          | [k, n_embd]            |

`token_embd` is a lookup, so llama.cpp gathers rows of A by token id and does
not transpose B. its A row t is r . e_t. when the embedding is tied to the
output head (no `output.weight`), llama.cpp would apply the same adapter to
the lm head and fail, so it is skipped and the stream keeps the embedding's
small r component.

## components

- `src/model_file` maps every shard of a gguf read-only and dequantizes rows.
- `src/hf_cache` resolves `--hf-repo REPO[:QUANT]`/`--hf-file` through
  llama-server's own calls (`common_models_handler_init` and `_apply`), so
  quant matching, shard choice, the cache location and downloading behave
  exactly as `llama-server -hf` does. the scripts resolve through
  `hierophant resolve` and hand llama-server the resolved path.
- `src/sites` enumerates residual writers by tensor kind and reports what
  makes an arch inexact. explicitly requested kinds count as writers, so a new
  arch needs a flag, not a code change.
- `src/capture` runs prompts through libllama with an eval callback on each
  `l_out-N` tensor and keeps the last token's row, which is the position that
  yields the first generated token, or the mean of the last N rows with
  `--positions`. `llama-cvector-generator --method mean` is
  not used: it averages over every padded position of paired prompts and
  drops the last layer.
- `src/directions` stores per-layer harmful and harmless means and derives the
  unit difference, per layer or interpolated at a fractional index. by
  default the harmless mean's component is projected out first: the raw
  difference also carries part of what every prompt shares, and ablating
  that disturbs harmless replies (kl) without removing more refusal.
- `--from-adapter` reads each site's direction and weight back from an
  existing rank-1 adapter (B = -w r) and recomputes only A, which is what
  ties an adapter to one base file.
- `src/adapter` turns directions plus weight curves into lora pairs and
  writes the gguf with provenance in `hierophant.*` keys. each site belongs
  to a group (`attn`, `mlp`, `embd`, from `group_of` in `src/sites`) with its
  own curve, as heretic tunes attention and mlp apart.
- `--split` writes the same pairs at weight 1 as one adapter per group and
  layer. since the lora delta is linear in the runtime scale, loading them
  all and scaling part (g, l) by curve_g(l) is the merged adapter with those
  curves, so `scripts/hierophant_search.py` tries a curve per request, with
  no file writes or server restarts. it mirrors the c++ curve formula and
  prints the `--curve` flags (in round-trip float form) that rebuild the
  winner exactly.
- `scripts/hierophant_eval.py` starts the pinned llama-server with
  `--lora-init-without-apply` and sends each request with its own
  `"lora":[{"id":0,"scale":s}]`. the server clears its cache when the lora
  set changes. it writes the sidecar. a refusal is judged by heretic's
  markers, by the base model itself on the same server (all adapters at
  scale 0) or by a model at another endpoint. with `--think` the reasoning
  is generated but only the answer is judged.
- the eval's request cache is an append-only json-lines file: one line per
  outcome (reply, verdict, first-token distribution, reply prefix, reply
  logp, wikitext kl run, file sha256), keyed by a hash of the stage, the
  options that stage depends on (`CACHE_STAGES`), what is applied ("base",
  or each adapter's sha256 and scale) and the prompt. appending one line
  per outcome makes an interrupt cost only the requests in flight, and a
  line cut off mid-write is skipped on reading. keying adapters by content
  rather than path lets an incumbent copied into a later round, or a
  rebuilt identical adapter, keep its results. llama-server starts on the
  first request the cache cannot answer. the base's logits for the
  wikitext kl are not kept (every position's whole vocabulary); they are
  recomputed when some kl run is missing.
- `reply_logp` measures refusal without sampling: the base's first K reply
  tokens (thinking off) are scored under the adapter one at a time, each
  request prefilling the earlier tokens as the assistant's reply, so the
  server returns the next token's log-probability. llama-server has no
  prompt log-probs, but with prompt caching each request only processes one
  new token. at scale 0 it reproduces the generation's own log-probabilities
  (`tests/test_eval.py` checks this).
- with `--keep-rows`, `--harmless-pcs K` finds the harmless rows' top
  principal components per layer through the n x n gram matrix of the
  centred rows (n prompts, far below n_embd) and keeps the refusal
  direction orthogonal to them and to the harmless mean. the basis is
  cached per layer.
- `scripts/filter_prompts.py` keeps the training prompts whose base reply
  the judge labels as their set expects. `fetch_datasets.Manifest` records
  every derived set (filtered or translated) with its source's split.
- the search loads split sets for several direction indices on one server;
  the index is a choice each trial makes, kept from the best trial during
  refinement most of the time. `--rescore` re-scores the leaders on a
  second slice of prompts, since a minimum over many noisy trials
  overstates the winner.
- `scripts/hierophant_pipeline.py` only sequences the other tools. its
  prompt budget: the train split picks directions, harmful/harmless-test
  0..N tune the search and N..2N re-score its leaders, and the extra sets
  (jailbreakbench, llm-lat, both free of the test and train prompts)
  confirm, so no stage scores on prompts an earlier one chose with. a step
  is done when its marker exists; config.json pins what the markers mean,
  and AFFECTS maps each option to the first step that reads it (unlisted
  ones to filter), so a change drops only the markers it invalidates.
  `--screen` scores jbb+lat from 300 on, past every full confirmation.
  the thinking check follows the confirmation so it can run each adapter at
  its picked scale, where it would be served.
- `scripts/hierophant_auto.py` is the default flow: a beam search over
  recipes (index, harmless pcs, residual, weight) with successive halving,
  built on the pipeline's `Run`, filter and capture. its design choices:
  - one capture with `--keep-rows` serves every pcs count, and a uniform
    weight equals a runtime scale, so a candidate's sweep over scales is a
    sweep over weights without rebuilding.
  - candidates are compared at one point, their refusals interpolated to
    where the wikitext kl reaches the budget, rather than at a fixed scale:
    a stronger direction reaches fewer refusals at a lower scale without
    being better. the scale found there is baked in as the weight.
  - selection (jbb+lat 300..) and promotion (jbb+lat 0..300, leader beside
    the incumbent on the same prompts) never share prompts, so the
    winner's curse of picking the best of many noisy scores stays out of
    the reported number.
  - best.gguf and best.json are each replaced with one rename, so ctrl-c
    never leaves a half-written best; state.json is written after each
    round, and an unfinished round is redone from what it had finished:
    each adapter build leaves a `.built` stamp of its arguments (with the
    mtimes of its input files), and every eval shares the run's request
    cache.
  - convergence counts rounds at full prompts in which no different recipe
    took over; refining the best's own weight does not count as progress.
  - proposals come cheapest first (index, the nearest pcs counts,
    residual, a residual member's index), so a capped round still moves
    the layer. on qwen3.8, residual and every-pcs proposals filled each
    round's slots, the layer stopped moving after round 1, and pcs
    variants of residual members each needed their own filter and capture
    (an hour of builds in one round) while losing. pcs steps one count at
    a time: a count that wins joins the beam and its neighbours follow.
  - the beam keeps each shape's (pcs, residual) best, then near ties: on
    qwen3.8 a beam of neighbouring indices of one recipe crowded out the
    pcs and residual variants, which were then never refined. a near tie
    is within the standard error of a difference at the round's prompts;
    round 1 starts at 128 prompts, since at 32 the noise (+-0.05) dwarfed
    what candidates differ by and the step halving locked onto it.
  - a candidate whose shape changed is swept wide again: adding a
    residual raises the kl per unit weight, so its budget point fell below
    the narrow sweep around the parent's weight and it was compared at a
    lower kl than its rivals.
  - promotion is decided on the same `--report-prompts` every round, and
    pooled prompt files alternate, so a report's number means the same
    from round to round; growing it with the round, in file order, made an
    unchanged best's refusals rise as llm-lat prompts joined jailbreakbench.
  - promotion compares the worse of thinking-off and short-thinking
    refusals. short thinking is a reasoning budget, which llama-server's
    sampler enforces from the think tags it detects and so works whatever
    the template, plus `low` effort where the template takes it. templates
    disagree on levels (qwen3.8 low/medium/xhigh, mistral none/high, glm
    maps all but high to max) and some raise on others, so `auto` sends
    one probe request and keeps the template default on an error. selection
    stays thinking off: it is most of the generation.
- with thinking on, a reply cut off before its answer is judged by the end
  of its reasoning: an abliterated model often runs out of tokens while
  still planning how to comply, which an empty-answer-is-a-refusal rule
  counted as refusing. `unfinished` keeps the budget-exhaustion rate apart.
- a residual direction is the difference of means taken with an adapter
  applied, on only the prompts it still refuses: what drives the refusals
  one direction leaves. `adapter --residual` appends it to each output's
  directions before gram-schmidt, so the rank-2 edit removes both exactly
  and the second column is the part of the residual the first lacks.
- the scale in a sidecar is the fewest refusals within a kl budget, on the
  full-sequence kl when measured: refusal rates flatten while kl keeps
  climbing, and the least-refusing scale alone picked q38-v2 at 2.6x its
  wikitext kl at scale 1.
- `scripts/translate_prompts.py` reuses the eval's server and markers to
  translate prompt files line by line. a translation is dropped when it is
  empty, unchanged, over 4x its source (an answer, not a translation) or
  carries a refusal marker its source lacks. each output names its source
  in the manifest and inherits its split, so `describe_set` reports it as
  that dataset and split plus a language.
- `--kld-text` hands full-sequence kl to `llama-perplexity`: one run saves
  the base logits, then one run per adapter and scale with `--lora-scaled`.
  it runs after the server stops, so the two never hold the model at once.
- `--progress` bars (`scripts/progress.py`, `progress` in `src/util.h`, one
  format) end each redraw in a carriage return and the finished bar in a
  newline. the pipeline's `Run.sh` reads its children's output as bytes and
  cuts it after either, so it draws redraws in place and logs only lines;
  text-mode pipes would turn every redraw into a line. the eval counts a
  request when it finishes rather than in order, and a llama-perplexity
  chunk from the `[N]` or table row it prints. a `Run` may hold a bar over
  several commands (auto: a round's builds): the children's bars named in
  `rollup` are parsed back (`progress.COUNT`) and advance it by the share
  of the unit under way, so it moves per site without a bar per build.
  an outer bar (a round's stages, an eval's whole sweep) is opened with
  `with` and never drawn alone, which would leave a redraw for the next
  printed line to overwrite: it prefixes every bar drawn while it is open,
  the children's included (`Run.redraw`), and closes on a line. so the
  prefix nests across processes without a terminal protocol, and the rollup
  regex skips it. only a terminal's lines are cut to its width; a pipe gets
  the whole line for the parent to parse.
- `scripts/gguf_lite.py` reads gguf metadata and f32/f16 tensors with the
  standard library, for the sidecar and for tests that check the math
  independently of the c++ code.

## kl divergence

the server returns the top-k logprobs of the full pre-sampling softmax. kl is
taken over the base's top-k; a token missing from the adapted top-k gets the
adapted smallest listed logprob, which overstates its probability and so
understates that token's term. with k = 50 the missing mass is small on
instruct models.

llama-server sends no logprobs when its one token is half a utf-8
character or trimmed as a stop word; that prompt is left out of the
first-token kl and of reply_logp rather than given a guessed distribution.

## build

cmake fetches llama.cpp at the pinned commit (or uses
`FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`), builds `hierophant` against `llama`,
`llama-common` and `ggml`, and builds `llama-server` and `llama-perplexity`
from the same tree, so validation runs on the build that wrote the adapter.
https comes from llama.cpp's pinned boringssl (`LLAMA_BUILD_BORINGSSL`)
rather than the host's openssl, so the build needs no ssl dev package and
links the same library everywhere.
