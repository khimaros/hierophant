# hierophant

abliteration LoRA adapters made directly from a quantized gguf. instead of
storing a second, decensored copy of a model, keep the gguf you already serve
and a rank-1 or rank-2 adapter computed from its own weights.

## build

```
make CMAKE_ARGS=-DGGML_VULKAN=on  # assuming vulkan backend
```

this produces `build/bin/hierophant` and the `build/bin/llama-server` and
`build/bin/llama-perplexity` it is validated against. the first build also
fetches and builds the boringssl version llama.cpp pins, for downloading
models over https.

## use

```
make datasets             # once: the prompt sets it scores with
scripts/hierophant_auto.py -ngl 99 --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
```

that is the whole flow. it downloads the model if needed, finds the layers
and recipe that remove the most refusals within a kl budget, and keeps
refining until the result stops improving. the best adapter so far is
always at `lora/<org>--<repo>--<quant>/best.gguf`, so you can stop it with
ctrl-c at any point and use what it found:

```
llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  --lora-scaled lora/unsloth--Qwen3-4B-Instruct-2507-GGUF--Q4_K_M/best.gguf:1
```

the adapter's strength is built in, so its scale is close to 1; `best.txt`
gives the exact serve line. rerun the same command to continue: every reply,
verdict and kl run is kept in `cache.jsonl` as it arrives and every finished
adapter build is kept, so only the requests in flight at ctrl-c are repeated.
the cache also spares later rounds what earlier ones scored (the base, and an
incumbent kept from the last round). add `--status` to print the current
best and each round's without starting anything.

a run looks like this (illustrative numbers, close to Qwen3.8-Flash-Next's
by hand; each command it runs also prints one line naming its log):

```
== prepare
== round 1: which layer (13 candidates, 32 prompts each)
candidate                refusals@kl  weight      kl
L34                            0.190    1.41  0.1000
L31                            0.220    1.48  0.1000
...
>> new best: L34 (round 1)  refusals 0.220 (0.250 with short thinking), kl 0.1000
== round 2: refine L34, L31, L36 (9 candidates, 64 prompts each)
>> new best: L34-pcs8 (round 2)  refusals 0.160 (0.190 with short thinking), kl 0.1000
== round 3: refine L34-pcs8, L34, L32.75 (9 candidates, 128 prompts each)
>> new best: L34-pcs8-res (round 3)  refusals 0.070 (0.094 with short thinking), kl 0.1000
...
>> best unchanged for 2 rounds at full prompts: converged
== final check: L34-pcs8-res with thinking on (50 prompts)
best weights: lora/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS/best.gguf
```

how it decides:

- round 1 builds one adapter per layer over the middle and late layers and
  sweeps each over scales 0.5 to 2. every candidate is scored by its
  refusals where its kl over wikitext reaches the budget (`--max-kld`,
  default 0.1), interpolated between the scales around it, and carries the
  weight that reaches it into the next round.
- each later round keeps the leaders (`--beam`, 3) and tries their
  neighbours, one change at a time: a residual direction (what the
  adapter still refuses, removed as well), keeping the direction clear of
  4, 8 or 16 principal components of the harmless prompts, and a nearby
  layer, half as far each round down to a quarter layer. the prompts double
  each round, from `--first-prompts` (32) up to all 402 selection prompts.
- candidates are chosen on jailbreakbench + llm-lat from prompt 300 on. the
  round's leader is then scored beside the current best on the first 300,
  which nothing was chosen on, and replaces `best.gguf` only if it refuses
  less there. the leader being the best's own recipe at a new weight
  refines it in place.
- promotion also scores both on the first `--think-prompts` (64) of those
  with short thinking: `low` reasoning effort unless the model's chat
  template rejects that level, and at most `--think-budget` (512) reasoning
  tokens before llama-server forces the answer. the leader must refuse less
  in its worse mode, so a recipe that only holds with thinking off cannot
  take over. models name their levels differently and some shorten thinking
  through another knob entirely; `--think-effort LEVEL` and
  `--think-kwargs '{"think_fast": true}'` set those.
- it stops when two rounds in a row at full prompts bring no new recipe,
  or after `--rounds N`, and then checks the best with thinking on as it
  would be served (the template's default effort, no budget), whose
  replies can run long; that check is reported, not chosen on.

the directory holds:

- `best.gguf`, `best.json`: the adapter and its sidecar (base sha256,
  recipe, held-out scores).
- `best.txt`: what the adapter is, its numbers and how to serve it.
- `history.txt`: the best after each round.
- `round-N/`: every candidate of that round, its scores and `round.json`.
- `logs/`: every command's output.

other options: `--out DIR` instead of `lora/...`, `-m model.gguf` for a
local file, `--judge self` to have the model judge refusals instead of
matching phrases, `--indices 30,33` to start from given layers instead of a
grid, `--train-data OTHER/data` to reuse another run's filtered prompts,
`--max-candidates` (9) per round, `--no-think-check`, `--verbose` to stream
every command and its own progress (each eval stage, capture, builds, kl
runs) instead of one line per command, `--progress` to draw a progress
bar in place for each unit of work (see below), and `-ngl`/`-np` as for
llama-server.

`--progress` works on every command: `hierophant capture` counts prompts,
`hierophant adapter` sites, the eval, filter and search scripts each request
as it finishes, and the eval each llama-perplexity chunk:

```
base replies [#########---------------] 12/32 38% 0:41 eta 1:08
```

the pipeline and the automated run pass it on to every command they run.
a bar is redrawn in place on the terminal, and `logs/` keep only its
finished line. the automated run draws one bar for a round's builds,
advanced by every site of every build, rather than a bar per build.

## step by step

the automated run is built from these commands, which can also be run by
hand:

```
# sites that will be ablated, and caveats
build/bin/hierophant info --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
build/bin/hierophant capture --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  --harmful data/harmful-train.txt --harmless data/harmless-train.txt -o dirs.gguf
build/bin/hierophant adapter --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  -d dirs.gguf -o qwen3-4b-abliterated.gguf --direction-index 20
scripts/hierophant_eval.py --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  --lora qwen3-4b-abliterated.gguf --scales 0.5,1,1.25,1.5
```

the prompt sets are not in the repo. `make datasets` fetches them from the
huggingface hub and refuses any whose rows no longer hash as
`data/manifest.json` pins (`scripts/fetch_datasets.py data --more
--update` takes the new rows). check their terms before redistributing:

| file | source | license |
|---|---|---|
| harmful-train, harmful-test | mlabonne/harmful_behaviors (advbench) | none declared |
| harmless-train, harmless-test | mlabonne/harmless_alpaca (stanford alpaca) | none declared; alpaca is cc-by-nc-4.0 |
| harmful-jbb | JailbreakBench/JBB-Behaviors | mit |
| harmful-lat | LLM-LAT/harmful-dataset | none declared |
| wikitext-test | Salesforce/wikitext | cc-by-sa-3.0, gfdl |

the base is named as llama-server names it, through llama.cpp's own code:
`-hf`/`--hf-repo REPO[:QUANT]` (QUANT defaults to Q4_K_M, case-insensitive),
or `-hff`/`--hf-file FILE` for one file of the repo. a split model loads from
shard 1. missing files are downloaded into the huggingface cache
(`LLAMA_CACHE`, `HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`),
with `-hft`/`--hf-token` or `HF_TOKEN` for gated repos; `--offline` (or
`LLAMA_ARG_OFFLINE`) uses only the cache. `-m PATH` takes a file instead.

the eval prints refusal rate and kl divergence at each scale and writes
`qwen3-4b-abliterated.json` beside the adapter, recording the base sha256,
the recommended scale and the scores. the recommended scale has the fewest
refusals whose kl stays within `--max-kld` (default 0.1; the `--kld-text`
kl when measured, else the first-token kl). repeat `--lora` to score several
adapters on one server load; an adapter hierophant did not build (a
published one, as a baseline) is scored the same way but gets no sidecar.
serve it with:

```
llama-server -m <base.gguf> --lora-scaled qwen3-4b-abliterated.gguf:1.25
```

an adapter is only valid on the base whose sha256 its sidecar records.

### options

- `--direction-index F` uses one direction for every layer: the refusal
  direction after block F, interpolated between blocks when F is fractional.
  without it, or with `layer`, each layer uses its own direction. a list
  such as `--direction-index layer,26,32 -o out-{}.gguf` writes one adapter
  per entry from a single pass over the weights.
- `--direction-index 33+20` removes both directions at once, as a rank-2
  adapter (any number can be joined with `+`). a direction already inside
  the span of the earlier ones adds a zero column.
- `scripts/projection.py BASE ADAPTED --adapter A` compares a capture with
  one taken through adapter A (`capture --lora A`) and shows how much of A's
  direction is left at each layer, reading the direction from A itself.
- directions have the harmless mean's component removed before they are
  ablated, as AtomicChat's card describes; on Qwen3.8-Flash-Next this cut
  refusals at scale 1.25 from 50% to 16% and halved the kl.
  `--no-project-harmless` uses the plain difference of means.
- `--from-adapter X.gguf` (instead of `-d`) rebases an existing rank-1
  abliteration adapter onto this gguf: each site keeps X's direction and
  weight, read from its B, and A is recomputed from this file's weights.
  use it to carry a published adapter exactly onto the quant you serve.
- `--weight W` scales every site; `--curve MAX_W,MAX_POS,MIN_W,MIN_DIST` is
  heretic's per-layer curve and drops layers beyond MIN_DIST.
  `--curve GROUP=...` sets the curve of one group, `attn` (attention and the
  token mixers that replace it), `mlp` or `embd`, so `--curve 1,33,0.5,15
  --curve attn=1.2,33,0.8,10` weights attention and mlp writers apart.
  `--curve KIND=...` sets one tensor kind (`ffn_down_shexp`, `ssm_out`, ...)
  and beats its group's curve.
- `--harmless-pcs K` also keeps the direction orthogonal to the top K
  principal components of the harmless prompts' residuals, the directions
  harmless prompts vary along most. it needs a capture with `--keep-rows`.
- `--targets K,K` limits the sites to these tensor kinds (e.g.
  `attn_output,ffn_down_exps`). a kind `info` lists as unclassified can be
  added here.
- `--type f16` halves the adapter size.
- `--split -o parts/s-{}.gguf` writes one weight-1 adapter per group and
  layer (`s-attn-12.gguf`, `s-mlp-12.gguf`, `s-embd.gguf`) for the search;
  `--split-by kind` splits by tensor kind instead (`s-ffn_down_shexp-12.gguf`).
- `capture --lora A` captures with an adapter applied, which shows how much
  of the direction is left.
- `--residual R.gguf` adds a second direction to each output: R's direction
  at the output's first index, with R captured under an adapter A
  (`capture --lora A`) from the prompts A still refuses
  (`filter_prompts.py --lora A`). the result is rank 2, aimed at what one
  direction leaves behind. the sidecar records it under
  `directions.residual`.
- `capture --positions N` averages the last N prompt tokens (the chat
  template's closing tokens) rather than only the last, for a steadier
  direction. `capture --keep-rows` stores each harmless prompt's residuals
  as well (n_harmless * n_layer * n_embd * 2 bytes).
- `capture --think` captures with the template's thinking on; models served
  with reasoning may refuse in a different place.
- `scripts/filter_prompts.py -m model.gguf --harmful data/harmful-train.txt
  --harmless data/harmless-train.txt` keeps only the harmful prompts the base model
  refuses and the harmless ones it answers, judged as the eval judges
  (`--judge`), into `data/*.filtered.txt`; capture from those.
- `capture --harmful a.txt,b.txt` pools several prompt files; `--limit`
  caps the total. `scripts/fetch_datasets.py --extra
  NAME=DATASET:SPLIT:N[:COLUMN[:CONFIG]]` fetches another huggingface
  dataset (a multilingual one, say) into `data/NAME.txt`.
- `scripts/fetch_datasets.py --more` fetches more harmful prompts for
  scoring: `harmful-jbb` (jailbreakbench) and `harmful-lat` (llm-lat),
  without any prompt `harmful-train` or `harmful-test` already has.
- the directions file depends only on the arch, so a capture on a
  higher-precision file of the same model can build an adapter for a quant.

### eval options

- `--judge self` asks the served base model (adapters off) whether each reply
  refuses, instead of heretic's markers; `--judge URL --judge-model NAME`
  asks a model behind an openai-compatible endpoint. every reply and its
  verdict is kept in `--results`, so the judge can be checked.
- `--think` generates the refusal replies with thinking on and judges only
  the answer after it. a reply cut off before its answer is reported as
  `unfinished` and judged by the end of its reasoning, so one still working
  out how to help is not a refusal; raise `--max-tokens` to see fewer. kl is
  always measured on the answer's first token with thinking off.
- with `--think`, `--reasoning-budget N` ends the reasoning after N tokens
  (llama-server forces the end tag, after `--reasoning-budget-message`), so
  every reply reaches an answer; `--max-tokens` must leave room for it.
  `--reasoning-effort LEVEL` sets the template's `reasoning_effort`, and
  `auto` asks for `low` unless the template rejects it. `--think-kwargs
  JSON` adds chat template kwargs for models with other knobs. the sidecar
  records all three.
- every request's outcome is appended to `--cache` (default: the
  `--results` path plus `.cache.jsonl`) as it arrives, keyed by the
  prompt, the adapter's sha256 and scale, and only the options that shape
  it. a rerun after ctrl-c repeats only what was in flight and starts
  llama-server only if something is missing; a changed judge re-judges the
  kept replies. `filter_prompts.py` keeps one in its `--out-dir`.
- every point reports `loops`: the share of replies (reasoning and answer)
  in which some 8-word passage recurs 4 times, a model going round in
  circles rather than deliberating at length.
- every point also reports `reply_logp`: the adapted model's mean
  log-probability of the base model's own first reply tokens
  (`--logp-tokens`, 4) on the harmful prompts. the base mostly refuses, so
  this falls as refusal weakens, smoothly where the refusal rate moves in
  steps of 1/n.
- `--offset N` skips the first N prompts of each set. a search that tuned
  on `--limit 50` is confirmed on the other half with `--offset 50`.
  `--harmless-offset` gives the harmless set its own offset.
- `--kld-text data/wikitext-test.txt` also runs `llama-perplexity` with the
  adapter at each scale and records its mean kl over every position of the
  text (`--kld-ctx`, `--kld-chunks`), the measure model cards usually
  publish. the base logits it saves take n_vocab * ctx * 2 bytes per chunk
  under `TMPDIR`.

## pipeline

`scripts/hierophant_pipeline.py` runs the whole recipe for one model into
one output directory: filter the training prompts, capture, split at each
direction index, search, build the winner, confirm it on prompts the search
never saw (jailbreakbench and llm-lat, with full-sequence kl over wikitext),
which picks its scale under `--confirm-max-kld` and writes its sidecar, and
check it with thinking on at that scale (`--think-scales` to override). the
summary marks each adapter's picked scale with `*` and shows the thinking
check's refusals, unfinished replies and loops beside it.

```
scripts/hierophant_pipeline.py -ngl 99 --hf-repo unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS \
  --name q38-v2 --indices 32,33,34 --uniform --trials 18 --startup 9 --ref q38-33p.gguf
```

without `--out`, the run goes into `lora/<org>--<repo>--<quant>` under the
current directory (here `lora/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS`).

each finished step leaves a marker, so rerunning the same command resumes
after an interruption; `--from search` (or any step) reruns from there. a
changed option without `--from` reruns from the first step that reads it
(`--indices` from split, `--confirm-scales` from confirm), so outputs of
two configurations never mix. every command's output is also in
`OUT/logs/<step>.log`, and `OUT/summary.json` puts the winner beside the
`--ref` adapters (whose sidecars go under `OUT/refs`, not beside the
originals). `--positions`, `--harmless-pcs`, `--no-filter`,
`--split-by kind` and `--think-capture` change the direction recipe; run
each variant into its own `--out` to compare them. the search defaults suit
Qwen3.8-Flash-Next (edits reach nearly every layer).

- `--screen` compares recipes in a fraction of the time. it skips split and
  search and builds every `--indices` entry at weight 1 as
  `OUT/NAME-<index>.gguf`. a uniform weight is the same as a runtime scale,
  so the sweep over `0.8..1.6` covers what a uniform search would. all of
  them are scored on 60 jbb+lat prompts past the 300 a full confirmation
  uses, with 4 chunks of wikitext kl, then 16 thinking replies of up to 2048
  tokens each at the picked scale. any of these can still be set.
- `--residual SCALE` adds `OUT/NAME-res.gguf`: the first built adapter plus
  the direction of what it still refuses with it applied at SCALE (the
  scale you would serve it at; 1 for a searched winner, whose weight is
  built in).
- `--train-data OTHER/data` reuses another run's filtered training prompts,
  so a variant skips the filter step.
- `--targets K,K` limits the edited site kinds, e.g. to leave out
  `token_embd` and `ple_value`, which touch every generated token.

```
scripts/hierophant_pipeline.py -ngl 99 --hf-repo unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS \
  --screen --out lora/q38-s-pcs8 --name s --indices 33,32 --harmless-pcs 8 --residual 1.4 \
  --train-data lora/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS/data
```

compare screens at equal `kld_seq` rather than at equal scale: a stronger
direction reaches fewer refusals at a lower scale without being better.
with 60 prompts, differences under about 0.08 in refusals are noise.

## qwen3.8-flash-next

the best recipe found for UD-IQ4_XS is direction 33 with
`--harmless-pcs 8` plus `--residual 1.4`, served at scale 1 (the command
above, `OUT/s-res.gguf`). on the screen slice:

| adapter | refusals | first-token kl | wikitext kl |
|---|---|---|---|
| pcs8 + residual, scale 1 | 0.067 | 0.026 | 0.065 |
| residual without pcs, scale 1.2 | 0.067 | 0.100 | 0.070 |
| direction 33 alone, scale 1.4 | 0.133 | 0.060 | 0.081 |
| AtomicChat-style uniform 33, scale 1.25 (run1, jbb+lat 0..300) | 0.283 | 0.076 | 0.067 |

what did not help there: direction 32 (ties 33), joining a second layer's
direction (`33+20`, `33+40`), `--positions 4`, leaving out the embedding
and output tensors with `--targets`, searched curves (they tie a uniform
weight on held-out prompts), and `--think-capture` (about 1.0 refusals up
to scale 1.6). with thinking on, the adapted model complies but often
reasons past 4096 tokens before answering, so serve it with a generous
token budget.

## search

heretic tunes a weight curve per component. `scripts/hierophant_search.py`
does the same without writing a file per trial: llama-server loads the split
adapters unapplied, and each trial is one set of per-request scales, the
curves' weights at each layer.

```
build/bin/hierophant adapter -m model.gguf -d dirs.gguf --direction-index 33 --split -o parts/s-{}.gguf
scripts/hierophant_search.py -m model.gguf --split-dir parts --limit 50 --trials 60 --max-kld 0.05 \
  --results search.json
```

it samples heretic's ranges at random for `--startup` trials, then refines
around the best: the fewest refusals within `--max-kld`, or with
`--objective logp` the lowest `reply_logp`, which tells close trials apart
on far fewer prompts. `--rescore K --rescore-offset O --rescore-limit N`
scores the K leaders again on other prompts and picks the best of those,
so the winner is not just the luckiest draw. the end prints the flags that
build the winner as one adapter, and `--results` is rewritten after every
trial, so a stopped search keeps what it found. confirm the winner on a
third slice of prompts, or on `harmful-jbb`/`harmful-lat`, before trusting
it.

repeat `--split-dir` with split sets built at different direction indices
to make the index a searched choice as well; a set split `--split-by kind`
searches a curve per tensor kind. `--uniform` searches one flat weight for
every site instead (from `--max-weight`), a one-dimensional search per
index: on Qwen3.8-Flash-Next, searched curves only tied the uniform
adapter once scored on held-out prompts.

`--position`, `--max-weight` and `--distance` widen or narrow the space.
heretic's reach (`--distance 0,0.6`) can miss models whose refusal is
written early: on Qwen3.8-Flash-Next only edits covering nearly every layer
moved refusals, so search it with `--distance 0.5,1 --position 0.5,0.9
--max-weight 1,1.8`.

## translated prompts

`scripts/translate_prompts.py` translates prompt sets with a model, so
directions can be captured from several languages and adapters scored in
them. it starts the pinned llama-server itself, optionally with an
abliterated adapter, since an aligned model often refuses to translate a
harmful prompt; `--url URL --url-model NAME` uses a running endpoint.

```
scripts/translate_prompts.py -m model.gguf --lora q38-33p.gguf --lora-scale 1.25 -ngl 99 -np 8 \
  --languages zh,ru,es,de,ja,ar,hi,sw \
  data/harmful-train.txt data/harmless-train.txt data/harmful-test.txt data/harmless-test.txt
build/bin/hierophant capture -m model.gguf --harmful data/harmful-train.txt,data/harmful-train.zh.txt,... ...
scripts/hierophant_eval.py -m model.gguf --lora A.gguf --harmful data/harmful-test.zh.txt \
  --harmless data/harmless-test.zh.txt --judge self
```

`data/harmful-test.txt` becomes `data/harmful-test.zh.txt`: train and test
stay apart in every language. replies that refuse, answer the prompt instead
of translating it, or come back unchanged are dropped and listed in
`harmful-test.zh.dropped.json`; the manifest records the source's sha256,
the language and the translator. heretic's markers are english, so score
other languages with `--judge`.

## test

```
make test-e2e             # downloads two tiny pinned models into models/
make precommit
```
