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
(Q8 for quantized weights, F16 for F16 weights), which the F32 LoRA path does
not. the E2E tests measure it: about 1e-5 relative on an F16 model.

## layouts (llama-adapter.cpp, llama-graph.cpp)

| site               | base ne                  | lora_a ne             | lora_b ne              |
|--------------------|--------------------------|-----------------------|------------------------|
| 2d writer          | [d_in, n_embd]           | [d_in, k]             | [k, n_embd]            |
| expert writer (3d) | [d_in, n_embd, n_expert] | [d_in, k, n_expert]   | [k, n_embd, n_expert]  |
| token_embd         | [n_embd, n_vocab]        | [k, n_vocab]          | [k, n_embd]            |

`token_embd` is a lookup, so llama.cpp gathers rows of A by token ID and does
not transpose B. its A row t is r . e_t. when the embedding is tied to the
output head (no `output.weight`), llama.cpp would apply the same adapter to
the LM head and fail, so it is skipped and the stream keeps the embedding's
small r component.

## components

- `src/model_file` maps every shard of a GGUF read-only and dequantizes rows.
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
  that disturbs harmless replies (KL) without removing more refusal.
- `--from-adapter` reads each site's direction and weight back from an
  existing rank-1 adapter (B = -w r) and recomputes only A, which is what
  ties an adapter to one base file.
- `src/adapter` turns directions plus weight curves into LoRA pairs and
  writes the GGUF with provenance in `hierophant.*` keys. each site belongs
  to a group (`attn`, `mlp`, `embd`, from `group_of` in `src/sites`) with its
  own curve, as heretic tunes attention and MLP apart.
- `--split` writes the same pairs at weight 1 as one adapter per group and
  layer. since the LoRA delta is linear in the runtime scale, loading them
  all and scaling part (g, l) by curve_g(l) is the merged adapter with those
  curves, so `scripts/hierophant_search.py` tries a curve per request, with
  no file writes or server restarts. it mirrors the C++ curve formula and
  prints the `--curve` flags (in round-trip float form) that rebuild the
  winner exactly.
- `scripts/hierophant_eval.py` starts the pinned llama-server with
  `--lora-init-without-apply` and sends each request with its own
  `"lora":[{"id":0,"scale":s}]`. the server clears its cache when the LoRA
  set changes. it writes the sidecar. a refusal is judged by heretic's
  markers, by the base model itself on the same server (all adapters at
  scale 0) or by a model at another endpoint. with `--think` the reasoning
  is generated but only the answer is judged.
- the eval's request cache is an append-only JSON-lines file: one line per
  outcome (reply, verdict, first-token distribution, reply prefix, reply
  logp, WikiText KL run, file SHA-256), keyed by a hash of the stage, the
  options that stage depends on (`CACHE_STAGES`), what is applied ("base",
  or each adapter's SHA-256 and scale) and the prompt. appending one line
  per outcome makes an interrupt cost only the requests in flight, and a
  line cut off mid-write is skipped on reading. keying adapters by content
  rather than path lets an incumbent copied into a later round, or a
  rebuilt identical adapter, keep its results. llama-server starts on the
  first request the cache cannot answer. the base's logits for the
  WikiText KL are not kept (every position's whole vocabulary); they are
  recomputed when some KL run is missing.
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
  (JailbreakBench, LLM-LAT, both free of the test and train prompts)
  confirm, so no stage scores on prompts an earlier one chose with. a step
  is done when its marker exists; config.json pins what the markers mean,
  and AFFECTS maps each option to the first step that reads it (unlisted
  ones to filter), so a change drops only the markers it invalidates.
  `--screen` scores JBB + LAT from 300 on, past every full confirmation.
  the thinking check follows the confirmation so it can run each adapter at
  its picked scale, where it would be served.
- `scripts/hierophant_auto.py` is the default flow: successive halving over
  recipes (index, harmless PCs, residual, weight), then a beam search around
  the survivors with the index step halving each round,
  built on the pipeline's `Run`, filter and capture. its design choices:
  - one capture with `--keep-rows` serves every PC count, and a uniform
    weight equals a runtime scale, so a candidate's sweep over scales is a
    sweep over weights without rebuilding.
  - candidates are compared at one point, their refusals interpolated to
    where the KL reaches the budget, rather than at a fixed scale: a
    stronger direction reaches fewer refusals at a lower scale without
    being better. the scale found there is baked in as the weight. the KL
    is the larger of the first-token and WikiText KL: on qwen3.8, harmless
    PCs cut the first-token KL from 0.100 to 0.026 at equal refusals while
    the WikiText KL barely moved, so with the WikiText KL alone PCs only
    ever lost. the eval's own pick still uses the WikiText KL alone, which
    varies less between prompt slices.
  - every eval runs `--at-budget`, since the ranking reads only the budget
    point, and selection adds `--logp-tokens 0`, since it ranks by refusals
    alone. on qwen3.8 at 300 prompts a sweep point took 14 minutes: reply
    logp 7.5, replies 5, WikiText KL 0.8 and first tokens 0.3. the report
    keeps reply_logp for the sidecar of the adapter it promotes.
  - `--screen-layers` (opt in: checked on qwen3.8 against round 1's 16 full
    scores, its rank correlation was about 0.37, mostly from putting L29
    last; it ranked L34 38th and L33-pcs8 36th of 60 and preferred late
    layers, which lower the base's first token without lowering refusals)
    scores every layer of the grid's range, with
    each first PC count, on one-token requests alone: reply_logp from the
    base's first reply token, where the KL reaches the budget, ranked lowest
    first. this is the refusal score arditi et al. pick directions by. the
    PC counts are screened too because they interact with the layer (on
    qwen3.8 8 PCs helped L33 and hurt L34). the best `--screen-keep` (16
    pairs) are round 1.
    `--screen-only` reruns the screen in a finished run's directory beside
    its round 1 refusals, which is how the ranking gets checked.
  - selection (JBB + LAT 300..) and promotion (JBB + LAT 0..300, leader beside
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
  - proposals go member by member in rank order, each member's cheapest
    first (index, the nearest PC counts, residual; a residual member only
    moves its index). on qwen3.8, residual and every-PC-count proposals
    filled each round's slots, the layer stopped moving after round 1, and
    PC variants of residual members each needed their own filter and
    capture (an hour of builds in one round) while losing. ordering by kind
    across members fixed that, then starved the other way: four plain
    members within noise took 4 of 9 slots and their 8 index moves left no
    room for any PC count or residual. by member, the cap cuts the last
    members' moves, and the leader's shapes are tried every round. the PC
    count steps one count at a time: a count that wins joins the beam and
    its neighbours follow.
  - the beam keeps each shape's (PCs, residual) best, then near ties: on
    qwen3.8 a beam of neighbouring indices of one recipe crowded out the
    PC and residual variants, which were then never refined. a shape's
    best takes its slot before any near tie of another: taken in rank
    order, L34 and L38-res filled the last slots after round 2 and
    L38-pcs4, so PCs with a residual, never came up. a near tie
    is within the standard error of a difference at the round's prompts.
  - the search opens with successive halving over the recipe itself: round
    1 is every middle and late layer crossed with `--first-pcs` (0 and 8),
    and each next round keeps the better half by rank (and anything within
    a standard error of its last), until the survivors fit the beam or the prompts stop doubling (only
    more prompts split a tie; a model refusing nothing would halve
    forever). once at most 8 survive, each one's residual joins at the
    weight it was found at, since each needs a filter and a capture (about
    10 minutes each on qwen3.8). on qwen3.8 the beam alone settled the
    layer on plain directions first (L16.5 0.098; L31.5 and L34 cut by
    0.005 past a standard error), so PCs and residuals were only ever tried
    on L16.5, which a residual hurt; by hand, L33 with 8 PCs and a residual
    reached 0.093 against the search's 0.113. which recipe helps depends on
    the layer (8 PCs helped L33 and hurt L34), so the layer is not chosen
    before the recipe. keeping half by rank rather than by a noise cut lets
    round 1 start at 32 prompts: a near tie survives to be told apart on
    twice as many. a cut at exactly half was itself noise: on qwen3.8 round
    3, L33 with 8 PCs missed it by 2 of 128 prompts (it had ranked first on
    the same prompts in an earlier run), so its residual, the manual
    recipe, was never built. the margin is one standard error of the cut's
    own rate, not of a difference, so the survivors still halve roughly.
  - round 1 has every layer, not a grid: a grid 5 layers apart missed L33
    on qwen3.8 (adjacent layers' directions have a cosine near 0.82), and
    halving does not move layers. moving the leaders' layers while halving
    was tried and dropped once every layer was in: it added 8 candidates a
    round, and halving barely shrank. refinement moves them afterwards.
  - a candidate whose shape changed is swept wide again: adding a
    residual raises the KL per unit weight, so its budget point fell below
    the narrow sweep around the parent's weight and it was compared at a
    lower KL than its rivals.
  - promotion is decided on the same `--report-prompts` every round, and
    pooled prompt files alternate, so a report's number means the same
    from round to round; growing it with the round, in file order, made an
    unchanged best's refusals rise as LLM-LAT prompts joined JailbreakBench.
  - promotion is decided on thinking-off refusals (all 300 report
    prompts); short thinking only vetoes a leader worse than the best by
    more than the standard error of a difference on its prompts. comparing
    the worse of the two modes let short thinking on 64 prompts decide
    ties: on qwen3.8 one adapter scored 0.047 and then 0.094 on the same
    prompts, and the 0.094 against 0.078 kept it out. `--think-prompts`
    is 128 for the same reason. short thinking is a reasoning budget, which llama-server's
    sampler enforces from the think tags it detects and so works whatever
    the template, plus `low` effort where the template takes it. templates
    disagree on levels (qwen3.8 low/medium/xhigh, mistral none/high, GLM
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
- the scale in a sidecar is the fewest refusals within a KL budget, on the
  full-sequence KL when measured: refusal rates flatten while KL keeps
  climbing, and the least-refusing scale alone picked q38-v2 at 2.6x its
  WikiText KL at scale 1.
- `scripts/translate_prompts.py` reuses the eval's server and markers to
  translate prompt files line by line. a translation is dropped when it is
  empty, unchanged, over 4x its source (an answer, not a translation) or
  carries a refusal marker its source lacks. each output names its source
  in the manifest and inherits its split, so `describe_set` reports it as
  that dataset and split plus a language.
- `--kld-text` hands full-sequence KL to `hierophant kl` (`src/kl.cpp`),
  which measures what `llama-perplexity --kl-divergence` does (the second
  half of each chunk, base tokens below log-probability -16 left out) for
  every uncached adapter and scale from one model load. llama-perplexity
  took a process, and a load of the model, per run: on qwen3.8 about 40 s
  each, beside seconds of compute. the base's log-probabilities for every
  chunk stay in memory (about 1.2 GB for 8 chunks of a 150k vocabulary)
  rather than in a file, and the chunks loop inside the runs, because each
  change of adapters re-plans llama.cpp's compute graph. it keeps them in 32
  bits where llama-perplexity's base file keeps 16, and agrees with it to 5
  digits; the cache key is the same, so earlier runs' values are reused.
  it runs after the server stops, so the two never hold the model at once.
  the rows of logits are fetched on one thread before the workers compare
  them: `llama_get_logits_ith` reorders the output rows on its first call
  after a decode when the batch was split out of order (a hybrid model's
  is), and calling it from the workers corrupted the heap on qwen3.8.
- `--at-budget` measures before it generates: the first-token KL at every
  scale and the KL over the text, then `budget_scale` (where the larger of
  the two first reaches `--max-kld`, interpolated, rounded so that a rerun
  asks for the same scale), then the rest of `score` there alone. a ranking
  at the budget reads that one point, so a wide sweep of 6 scales replies
  once instead of 6 times. the server is stopped for each KL run and
  started again after. `--logp-tokens 0` drops reply_logp (a request per
  token, the slowest stage) and `--no-replies` drops the replies, for a
  screen on one-token requests alone.
- `--progress` bars (`scripts/progress.py`, `progress` in `src/util.h`, one
  format) end each redraw in a carriage return and the finished bar in a
  newline. the pipeline's `Run.sh` reads its children's output as bytes and
  cuts it after either, so it draws redraws in place and logs only lines;
  text-mode pipes would turn every redraw into a line. the eval counts a
  request when it finishes rather than in order, and a KL chunk from the
  bar `hierophant kl` draws, reading only its latest line: matching the
  whole of its log at every read was quadratic, fell behind the child,
  and the full pipe stopped both. a `Run` may hold a bar over
  several commands (auto: a round's builds): the children's bars named in
  `rollup` are parsed back (`progress.COUNT`) and advance it by the share
  of the unit under way, so it moves per site without a bar per build.
  every child bar, rolled up or not (a residual's filter, its capture),
  names the build under way and its step in the detail (`L38-res replies`).
  one bar is drawn at a time: with several, each starting over at 0%, the
  share on screen jumped around. an outer bar (an eval's whole sweep, a
  round) is opened with `with`; a bar opened inside it in the same process
  is not drawn but moves it (`follow`, never back within a unit) and
  finishes one of its units when it closes, or the outer bar is told of
  its stages. across processes, `Run.redraw` feeds a child's drawn bar to
  the open outer bar as its stage's share, parsed back from its percent,
  so the units must be one source each (the report's build is a stage of
  its own). `entry` wraps stdout and stderr in `Above`, so a printed line
  clears the bar, goes above it and redraws it. only a terminal's lines
  are cut to its width or cleared; a pipe gets the whole line for the
  parent to parse.
- ctrl-c, kill and a closed terminal all end a script the same way
  (`entry`): SIGINT, SIGTERM and SIGHUP raise KeyboardInterrupt, later
  ones are ignored so they cannot cut the cleanup short, every child is
  terminated and waited for (killed after 30 s), and it exits 130. so
  llama-server, which holds the model's memory, never outlives the command
  that started it: `Run.sh` stops its child, the eval drops its queued
  requests (`request_pool`) and stops its server. llama-server finishes
  the replies it is generating before it honours a SIGTERM, and exits at
  once on a second, so it gets that second one after 2 s. waiting out the
  first took the whole 30 s, the same 30 the automated run gives the eval,
  which it could kill before the eval killed its server.
- `scripts/hierophant_publish.py` stages, and with `--upload` uploads, one
  Hugging Face repo per model holding one adapter per base quant, since an
  adapter is exact only for the file its factors came from (R2). the model
  and quant are read from the sidecar's base repo and file (the repo less
  `-GGUF`; the file less the model and any shard suffix), so the names
  follow the base's own. the README is rebuilt from every sidecar staged in
  the directory, and each row scores the scale its serve line gives (1 when
  the pick was 0, as `best.txt` serves it), read from the sweep. uploading
  is a separate, explicit step, because it publishes.
- `scripts/gguf_lite.py` reads GGUF metadata and F32/F16 tensors with the
  standard library, for the sidecar and for tests that check the math
  independently of the C++ code.

## KL divergence

the server returns the top-k logprobs of the full pre-sampling softmax. KL is
taken over the base's top-k; a token missing from the adapted top-k gets the
adapted smallest listed logprob, which overstates its probability and so
understates that token's term. with k = 50 the missing mass is small on
instruct models.

llama-server sends no logprobs when its one token is half a UTF-8
character or trimmed as a stop word; that prompt is left out of the
first-token KL and of reply_logp rather than given a guessed distribution.

## build

cmake fetches llama.cpp at the pinned commit (or uses
`FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`), builds `hierophant` against `llama`,
`llama-common` and `ggml`, and builds `llama-server` and `llama-perplexity`
(which the tests check `hierophant kl` against) from the same tree, so
validation runs on the build that wrote the adapter.
HTTPS comes from llama.cpp's pinned boringssl (`LLAMA_BUILD_BORINGSSL`)
rather than the host's openssl, so the build needs no SSL dev package and
links the same library everywhere.
