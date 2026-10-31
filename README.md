# hierophant

abliteration LoRA adapters made directly from a quantized GGUF. instead of
storing a second, decensored copy of a model, keep the GGUF you already serve
and a rank-1 or rank-2 adapter computed from its own weights.

## build

```
# build with llama.cpp defaults
make

# build with llama.cpp vulkan backend
make CMAKE_ARGS=-DGGML_VULKAN=on
```

this produces `build/bin/hierophant-core` (the C++ tools) and the
`build/bin/llama-server` it is validated against, `build/bin/llama-cli` for
`hierophant chat`, and `build/bin/llama-perplexity`, which the tests check
`hierophant core kl` against. the `hierophant` script at the top of the repo is the one command to
run; it can be symlinked onto your PATH. the first build also
fetches and builds the boringssl version llama.cpp pins, for downloading
models over HTTPS.

## use

```
# download harmful/harmless prompts
make datasets

# optimize a LoRA adapter automatically for an HF model
./hierophant build unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS --progress -ngl 99
```

that is the whole flow. it downloads the model if needed, finds the layers
and recipe that remove the most refusals within a KL budget, and keeps
refining until the result stops improving. the best adapter so far is
always at `out/<org>--<repo>--<quant>/best.gguf`, so you can stop it with
ctrl-c at any point and use what it found. ctrl-c, kill or a closed
terminal stops every server and tool it started before it returns, so the
model's memory is free when the prompt comes back:

```
llama-server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS -ngl 99 \
  --lora-scaled out/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS/best.gguf:1
```

the adapter's strength is built in, so its scale is close to 1; `best.txt`
gives the exact serve line, and "serving" below shows how to load it
unapplied and turn it on per request. rerun the same command to continue:
every reply, verdict and KL run is kept in `cache.jsonl` as it arrives and every finished
adapter build is kept, so only the requests in flight at ctrl-c are repeated.
the cache also spares later rounds what earlier ones scored (the base, and an
incumbent kept from the last round). `./hierophant status MODEL` prints the
current best and each round's without starting anything.

a run looks like this (LFM3-Omni-2B's, shortened; each command it runs also
prints one line naming its log):

```
== build: searching
== prepare
== round 1: which layer and pcs (32 candidates, 32 prompts each)
candidate                refusals@kl  weight      kl
L16-pcs8                       0.156    1.13  0.0433
L14-pcs8                       0.156     1.1  0.0434
...
>> new best: L16-pcs8 (round 1)  refusals 0.103 (0.109 with short thinking, no reasoning seen), kl 0.0486
== round 2: the better half of round 1 (21 candidates, 64 prompts each)
>> new best: L9-pcs8 (round 2)  refusals 0.097 (0.109 with short thinking, no reasoning seen), kl 0.0494
== round 3: the better half of round 2 (12 candidates, 128 prompts each)
>> new best: L14 (round 3)  refusals 0.080 (0.117 with short thinking, no reasoning seen), kl 0.0492
== round 4: the better half of round 3 (11 candidates, 256 prompts each)
== round 5: the better half of round 4 (16 candidates, 402 prompts each)
== round 6: refine L14-pcs8-res, L14, L15-pcs8, L16-res (9 candidates, 402 prompts each)
>> best unchanged for 2 rounds at full prompts: converged
== final check: L14 with thinking on (50 prompts)
best weights: out/LiquidAI--LFM3-Omni-2B-GGUF--Q8_0/best.gguf
>> accepted L14 for Q8_0 in configs/LiquidAI--LFM3-Omni-2B-GGUF.toml
```

how it decides:

- round 1 builds an adapter for every middle and late layer (14 to 43 of
  48) with each `--first-pcs` count (0 and 8) and sweeps each over scales
  0.5 to 2. every layer, because neighbouring layers differ: a grid 5
  layers apart missed L33 on Qwen3.8-Flash-Next.
  every candidate is scored by its
  refusals where its KL reaches the budget (`--max-kld`, default 0.05),
  and carries the weight that reaches it into the next round. the KL is the
  larger of the first-token KL and the KL over WikiText. the KL is measured
  at every scale first, then at the scale predicted to reach the budget,
  stepping down until it is measured within it, and replies are generated
  only there, the one point the ranking reads; selection also leaves out the reply logp, which only
  the report keeps.
- `--screen-layers` first screens every layer of that range with each
  `--first-pcs` count on the first token alone (how unlikely the base's own
  first reply token becomes where the KL reaches the budget, on 64 prompts)
  and scores only the best `--screen-keep` (16) in full. it is cheap but
  off by default: on Qwen3.8-Flash-Next it ranked the best two candidates
  36th and 38th of 60, preferring late layers that change the first token
  without lowering refusals. `./hierophant screen MODEL` runs the screen
  alone in an existing run's directory (`OUT/screen`), printing round 1's refusals
  beside its ranking, and leaves the run as it was.
- the next rounds halve: each keeps the better half of the last by rank,
  however close the scores, and anything within a standard error of the
  last of it, which its prompts cannot tell apart, at the weights they were
  found to reach the budget at. once at most 8 are left, or at the latest
  before the last round, which keeps only the best 8, each survivor's
  residual direction joins them (what the adapter still refuses, removed as
  well), which needs a filter and a capture of its own. a residual that
  lands more than 10% from the weight it was captured under is captured
  again there and scored once more in that round, since its first score
  came from a capture made too strong. which recipe helps
  depends on the layer, so no layer is settled before its recipes are
  tried. halving stops once the survivors fit the beam, or at full
  prompts, where more prompts can no longer split a tie.
- each round after that keeps up to `--beam` (4) leaders: the best of each
  shape (harmless PC count, residual or not), so a recipe change is not
  crowded out by neighbouring layers of one, then more of a shape while
  they are within the noise of the leader. it tries their
  neighbours, one change at a time, member by member from the leader down
  and cheapest first within each, up to `--max-candidates` (which counts
  the leaders, scored again on the round's prompts), so the cap cuts the
  last members' moves rather than the leader's: a nearby layer, half as far each round down to a
  quarter layer; the next count either side of 0, 4, 8 or 16 principal
  components of the harmless prompts to keep the direction clear of; then
  a residual direction for each (what the adapter still refuses, removed
  as well), which needs a filter and a capture of its own. a leader with a
  residual only moves its layer. a changed
  shape moves the weight the budget allows, so it is swept from 0.5 to 2
  again rather than around its parent's weight. the prompts double each
  round, from `--first-prompts` (32) up to all 402 selection prompts.
- candidates are chosen on JailbreakBench + LLM-LAT from prompt 300 on. the
  round's leader is then scored beside the current best on the first
  `--report-prompts` (300), which nothing was chosen on, the same ones
  every round, and replaces `best.gguf` only if it refuses less there. the
  leader being the best's own recipe at a new weight refines it in place.
  pooled prompt files are taken a prompt from each in turn, so the first
  300 mix both sets (all 89 of JailbreakBench and 211 of LLM-LAT).
- promotion also scores both on the first `--think-prompts` (128) of those
  with short thinking: `low` reasoning effort unless the model's chat
  template rejects that level, and at most `--think-budget` (512) reasoning
  tokens before llama-server forces the answer. the leader must refuse less
  with thinking off, and not more with short thinking beyond what its fewer
  prompts can tell apart, so a recipe that only holds with thinking off
  cannot take over. models name their levels differently and some shorten thinking
  through another knob entirely; `--think-effort LEVEL` and
  `--think-kwargs '{"think_fast": true}'` set those.
- it stops when two rounds in a row at full prompts bring no new recipe,
  or after `--rounds N`, and then checks the best with thinking on as it
  would be served (the template's default effort, no budget), whose
  replies run to full length. that check is reported, not chosen on: it
  counts refusals by markers, comparable with the report, and by the base
  as judge, which also counts warnings and deflection and so reads far
  higher on an adapter's long replies. it also counts loops, replies that
  keep repeating a passage, which only full-length replies show, and how
  many reasoned at all: a model that never does gets `no reasoning seen`
  beside its short thinking score, which is then just another plain reply.

the directory holds:

- `best.gguf`, `best.json`: the adapter and its sidecar (base SHA-256,
  recipe, held-out scores, the final check).
- `best.txt`: what the adapter is, its numbers, the llama.cpp commit that
  built it, and how to serve it.
- `history.txt`: the best after each round.
- `round-N/`: each round's scores and `round.json`. its adapters are deleted
  once it is recorded, unless `--keep-rounds`.
- `logs/`: every command's output.
- `dirs.gguf`, `data/`, `cache.jsonl`, `residual/`: what any recipe of this
  base can reuse, shared with the named runs in `runs/`.

other options: `--out DIR` instead of `out/...`, `-m model.gguf` for a
local file, `--judge self` to have the model judge refusals instead of
matching phrases, `--indices 30,33` to start from given layers instead of a
grid, `--train-data OTHER/data` to reuse another run's filtered prompts,
`--max-candidates` (9) per refining round, `--no-think-check`, `--verbose` to stream
every command and its own progress (each eval stage, capture, builds, KL
runs) instead of one line per command, `--progress` to draw a progress
bar in place for each unit of work (see below), and `-ngl`/`-np` as for
llama-server.

`--progress` works on every command: `hierophant core capture` counts prompts,
`hierophant core adapter` sites, `eval`, `data filter` and `curves` each request
as it finishes, and `hierophant core kl` each chunk of each run:

```
base replies [#########---------------] 12/32 38% 0:41 eta 1:08
```

`hierophant build` passes it on to every command it runs.
one bar is redrawn in place on the terminal, below what is printed, and
`logs/` keep only finished lines. the eval draws one bar for its whole
sweep, a unit per request stage (replies, verdicts, ...) at each scale and
per KL run, moved by the requests and chunks of the one under way. the
automated run draws one bar per round, a unit per stage (builds, each
selection sweep, the report's build, the report, its short thinking),
moved by the stage's builds or its eval's sweep, and names what runs:

```
round 2 select-wide [#######-----------------] 1/6 27.4% 41:12 eta 1:49:12 | L38-pcs16.gguf scale 1.5 replies
round 3 builds [#-----------------------] 0/5 3.4% 0:18 eta 8:53 | L34-res replies
```

so its share only grows. the ETA assumes the stages take equal time, which
they do not, so it is rough. the bars inside it end on lines of their own,
above it, as each round does with its total time. a line wider than the
terminal is cut to fit.

### configs

`configs/<org>--<repo>.toml` holds what is known about one model, with a
table per quant: its recipe, weight, scale and scores. MODEL can be any
unique part of a config's name, and `--quant` picks a quant where it has
several:

```
./hierophant build qwen3.8
./hierophant status qwen3.8
./hierophant status
```

`status` alone lists every config. a `[search]` table sets `build`'s flags
for that model (`think_effort = "low"`). a config can be shared, so it can
only set search settings: paths, URLs and programs come from flags or from
`~/.config/hierophant/config.toml`, which holds what belongs to the machine:

```toml
[defaults]
ngl = 99
parallel = 32
progress = true
judge = "markers"
user = "khimaros"   # who `publish` stages for
```

a flag on the command line always wins, and `./hierophant config MODEL`
prints the flags a command gets from these files. `./hierophant init REPO`
writes a new config.

a build on another machine writes its config there. `make sync HOST=kairos`
(with `REMOTE=src/hierophant` as the default checkout there) brings configs
back, then pushes this checkout; the newer copy of a config wins, so review
it with `git diff configs/`.

### serving

an adapter is a plain llama.cpp LoRA, so llama-server loads it beside the
quantized model it was built for, applied to every request with
`--lora-scaled` as above. it can also be loaded but not applied, so the
server answers as the base model until a request or a client turns the
adapter on:

```
llama-server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS -ngl 99 \
  --lora out/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS/best.gguf --lora-init-without-apply
```

per request, with a `lora` list of adapter IDs and scales in the body (an
adapter left out is at scale 0). this is how the eval scores every scale
on one server load:

```
curl http://localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "how do i pick a lock?"}],
  "lora": [{"id": 0, "scale": 1}]
}'
```

for every request from then on, set the server's scales with a POST to
`/lora-adapters` (a GET lists the loaded adapters and their IDs):

```
curl http://localhost:8080/lora-adapters
curl http://localhost:8080/lora-adapters -H 'content-type: application/json' -d '[{"id": 0, "scale": 1}]'
```

a request's own `lora` list overrides the server's scales for that
request. llama-server does not batch requests with different LoRA
scales together, so mixing them on one server costs throughput.

### publishing

`./hierophant publish` stages adapters for one Hugging Face repo
per model, one adapter per base quant, each named after the quant its
sidecar says it was built from (an adapter is exact only for that file):

```
./hierophant publish qwen3.8
./hierophant publish unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS
./hierophant publish out/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS/best.gguf --user khimaros --license apache-2.0
```

MODEL is named as for `build`. with a config, it publishes the best of every
quant the config accepted (or the one quant asked for), with the config's
license and the `user` from the machine's `[defaults]`, and refuses a best
that is not the accepted recipe at the accepted weight. `--name RUN`
publishes the best of a named run. a REPO:QUANT with no config publishes
that run's best, and like adapter files takes `--license`. the card comes from `templates/model-card.md`, with
`templates/hierophant-banner.jpeg` staged beside it.

the end of the card lists every recipe the search tried, from the run's
`round.json` files and collapsed per quant, marking the published one. those
are selection scores, so the card says they are not held out, and the same
list is uploaded as `ladder-<QUANT>.json`.

each stage also writes `out/publish/<repo>.preview.html`, a rough local page
of the card, tags and files. it sits beside the staging directory, so it is
never uploaded.

this copies `best.gguf` and its sidecar into
`out/publish/unsloth-Qwen3.8-Flash-Next-hierophant-LoRA-GGUF/` as
`Qwen3.8-Flash-Next-hierophant-UD-IQ4_XS.gguf` and `.json` (the quant last,
as GGUF files name it), and writes its
`README.md` from every sidecar staged there: the hub metadata (`base_model`,
`base_model_relation: adapter`, the license, tags), a link here, the
`hf download` and llama-server lines for each quant, a table of each one's
base SHA-256, scale, recipe, held-out refusals and KL, and how it was made.
staging another quant rewrites the table. nothing is uploaded until the same
command is run with `--upload`, which runs `hf upload` on the directory; it
needs `hf auth login` first. an upload then joins the repo to the `hierophant`
collection in its namespace, created the first time: `--collection TITLE`
or `--collection NAMESPACE/SLUG` names another, and `--collection ""` joins
none. `--license` is the base model's own, `--repo`
names a repo other than `USER/<org>-<Model>-hierophant-LoRA-GGUF`, which
names the base's uploader since an adapter is exact for its files, and `--quant`
gives the quant when the base file's name does not say it.

llama.cpp has no Hugging Face download for adapters, so users fetch one
with `hf download` and pass its path to `--lora`, as the README it writes
shows.

## step by step

the automated run is built from these commands, which can also be run by
hand:

```
# sites that will be ablated, and caveats
./hierophant core info --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
./hierophant core capture --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  --harmful data/harmful-train.txt --harmless data/harmless-train.txt -o dirs.gguf
./hierophant core adapter --hf-repo unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  -d dirs.gguf -o qwen3-4b-abliterated.gguf --direction-index 20
./hierophant eval unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  --lora qwen3-4b-abliterated.gguf --scales 0.5,1,1.25,1.5
```

the prompt sets are not in the repo. `make datasets` fetches them from the
huggingface hub and refuses any whose rows no longer hash as
`data/manifest.json` pins (`./hierophant data fetch data --more
--update` takes the new rows). check their terms before redistributing:

| file | source | license |
|---|---|---|
| harmful-train, harmful-test | mlabonne/harmful_behaviors (advbench) | none declared |
| harmless-train, harmless-test | mlabonne/harmless_alpaca (stanford alpaca) | none declared; alpaca is CC-BY-NC-4.0 |
| harmful-jbb | JailbreakBench/JBB-Behaviors | MIT |
| harmful-lat | LLM-LAT/harmful-dataset | none declared |
| wikitext-test | Salesforce/wikitext | CC-BY-SA-3.0, GFDL |

the base is named as llama-server names it, through llama.cpp's own code:
`-hf`/`--hf-repo REPO[:QUANT]` (QUANT defaults to Q4_K_M, case-insensitive),
or `-hff`/`--hf-file FILE` for one file of the repo. a split model loads from
shard 1. missing files are downloaded into the huggingface cache
(`LLAMA_CACHE`, `HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`),
with `-hft`/`--hf-token` or `HF_TOKEN` for gated repos; `--offline` (or
`LLAMA_ARG_OFFLINE`) uses only the cache. `-m PATH` takes a file instead.

the eval prints refusal rate and KL divergence at each scale and writes
`qwen3-4b-abliterated.json` beside the adapter, recording the base SHA-256,
the recommended scale and the scores. the recommended scale has the fewest
refusals whose KL stays within `--max-kld` (default 0.05; the `--kld-text`
KL when measured, else the first-token KL). repeat `--lora` to score several
adapters on one server load; an adapter hierophant did not build (a
published one, as a baseline) is scored the same way but gets no sidecar.
serve it with:

```
llama-server -m <base.gguf> --lora-scaled qwen3-4b-abliterated.gguf:1.25
```

an adapter is only valid on the base whose SHA-256 its sidecar records.

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
  refusals at scale 1.25 from 50% to 16% and halved the KL.
  `--no-project-harmless` uses the plain difference of means.
- `--from-adapter X.gguf` (instead of `-d`) rebases an existing rank-1
  abliteration adapter onto this GGUF: each site keeps X's direction and
  weight, read from its B, and A is recomputed from this file's weights.
  use it to carry a published adapter exactly onto the quant you serve.
- `--weight W` scales every site; `--curve MAX_W,MAX_POS,MIN_W,MIN_DIST` is
  heretic's per-layer curve and drops layers beyond MIN_DIST.
  `--curve GROUP=...` sets the curve of one group, `attn` (attention and the
  token mixers that replace it), `mlp` or `embd`, so `--curve 1,33,0.5,15
  --curve attn=1.2,33,0.8,10` weights attention and MLP writers apart.
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
  (`hierophant data filter --lora A`). the result is rank 2, aimed at what one
  direction leaves behind. the sidecar records it under
  `directions.residual`.
- `capture --positions N` averages the last N prompt tokens (the chat
  template's closing tokens) rather than only the last, for a steadier
  direction. `capture --keep-rows` stores each harmless prompt's residuals
  as well (n_harmless * n_layer * n_embd * 2 bytes).
- `capture --think` captures with the template's thinking on; models served
  with reasoning may refuse in a different place.
- `./hierophant data filter -m model.gguf --harmful data/harmful-train.txt
  --harmless data/harmless-train.txt` keeps only the harmful prompts the base model
  refuses and the harmless ones it answers, judged as the eval judges
  (`--judge`), into `data/*.filtered.txt`; capture from those.
- `capture --harmful a.txt,b.txt` pools several prompt files, a prompt
  from each in turn (as every command does); `--limit` caps the total.
  `./hierophant data fetch --extra
  NAME=DATASET:SPLIT:N[:COLUMN[:CONFIG]]` fetches another huggingface
  dataset (a multilingual one, say) into `data/NAME.txt`.
- `./hierophant data fetch --more` fetches more harmful prompts for
  scoring: `harmful-jbb` (JailbreakBench) and `harmful-lat` (LLM-LAT),
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
  out how to help is not a refusal; raise `--max-tokens` to see fewer. KL is
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
  prompt, the adapter's SHA-256 and scale, and only the options that shape
  it. a rerun after ctrl-c repeats only what was in flight and starts
  llama-server only if something is missing; a changed judge re-judges the
  kept replies. `hierophant data filter` keeps one in its `--out-dir`.
- every point reports `loops`: the share of replies (reasoning and answer)
  in which some 8-word passage recurs 4 times, a model going round in
  circles rather than deliberating at length.
- every point also reports `reply_logp`: the adapted model's mean
  log-probability of the base model's own first reply tokens
  (`--logp-tokens`, 4) on the harmful prompts. the base mostly refuses, so
  this falls as refusal weakens, smoothly where the refusal rate moves in
  steps of 1/n. `--logp-tokens 0` leaves it out; it is the slowest stage, a
  request per token.
- `--offset N` skips the first N prompts of each set. a search that tuned
  on `--limit 50` is confirmed on the other half with `--offset 50`.
  `--harmless-offset` gives the harmless set its own offset.
- `--kld-text data/wikitext-test.txt` also records each adapter's mean KL
  over every position of the text at each scale (`--kld-ctx`,
  `--kld-chunks`), the measure model cards usually publish: `hierophant
  kl`, which measures what `llama-perplexity --kl-divergence` does, every
  run from one model load. it keeps the base's log-probabilities in memory,
  n_vocab * ctx * 2 bytes per chunk.
- `--at-budget` measures the KL at every scale first and replies only at
  the scale where the larger of the first-token and text KL reaches
  `--max-kld`, measured there first and stepped down if it is over;
  `--no-replies` scores the first token alone. the automated run uses both.

`hierophant core kl` also runs on its own:

```
./hierophant core kl -m model.gguf --text data/wikitext-test.txt -c 512 --chunks 8 --lora a.gguf --scale 1 --lora a.gguf --scale 1.5
```

it prints one JSON line per run: `{"lora": "a.gguf", "scale": 1, "kld": 0.072, "same_top": 0.79}`.

## runs

`./hierophant build MODEL` is the only flow. it decides what to do from the
model's config and says so on its first line:

- no accepted recipe: it searches, and a search that converges fills the
  config. a search stopped early (ctrl-c, `--rounds`) changes nothing.
- an accepted recipe: it builds that one candidate for this quant, with its
  residual capture and a held-out eval, then records the numbers in the
  config. an accepted recipe on another quant counts, though whether it
  carries over is what that eval measures.
- `--search` searches anyway, and `--recipe L33-pcs8-res` builds any recipe.
  a search never replaces an accepted recipe: it prints the winner, and
  `./hierophant accept MODEL [RECIPE]` writes it over. `--no-accept` leaves
  the config alone.

without `--out`, a run goes into `out/<org>--<repo>--<quant>` under the
current directory. `--name RUN` keeps a run apart in `OUT/runs/RUN`, reusing
`OUT`'s directions, cache and residuals, and any command that reads a run
takes it too.

```
./hierophant compare qwen3.8 --lora other.gguf     # beside the best, on the report's prompts
./hierophant accept qwen3.8 L31-pcs8-res           # any recipe a report scored
./hierophant verify qwen3.8                        # rebuild from nothing, beside the recorded scores
./hierophant serve qwen3.8 -- --port 8080          # the rest goes to llama-server
./hierophant chat qwen3.8                          # the same adapter in llama-cli
./hierophant prune qwen3.8                         # list what no round will read; --yes deletes it
```

a finished round's adapters are deleted as it is recorded, since no later
round reads them and their scores stay in `round.json` and `cache.jsonl`;
`--keep-rounds` keeps them. `prune` also removes the residuals of candidates
no run holds, which are captured again if they come back. `verify` works in
`OUT/runs/verify` with an empty cache, so it measures whether the recipe
reproduces, including on another machine.

compare runs at an equal KL rather than an equal scale: a stronger direction
reaches fewer refusals at a lower scale without being better. at 300 prompts,
differences under about 0.02 in refusals are noise.

## qwen3.8-flash-next

the best recipe found by hand for UD-IQ4_XS is direction 33 with 8 harmless
PCs and a residual captured under the adapter at 1.4, served at scale 1.
on a 60-prompt screen:

| adapter | refusals | first-token KL | WikiText KL |
|---|---|---|---|
| pcs8 + residual, scale 1 | 0.067 | 0.026 | 0.065 |
| residual without pcs, scale 1.2 | 0.067 | 0.100 | 0.070 |
| direction 33 alone, scale 1.4 | 0.133 | 0.060 | 0.081 |
| AtomicChat-style uniform 33, scale 1.25 (run1, JBB + LAT 0..300) | 0.283 | 0.076 | 0.067 |

what did not help there: direction 32 (ties 33), joining a second layer's
direction (`33+20`, `33+40`), `--positions 4`, leaving out the embedding
and output tensors with `--targets`, searched curves (they tie a uniform
weight on held-out prompts), and `--think-capture` (about 1.0 refusals up
to scale 1.6). with thinking on, the adapted model complies but often
reasons past 4096 tokens before answering, so serve it with a generous
token budget.

## search

heretic tunes a weight curve per component. `./hierophant curves`
does the same without writing a file per trial: llama-server loads the split
adapters unapplied, and each trial is one set of per-request scales, the
curves' weights at each layer.

```
./hierophant core adapter -m model.gguf -d dirs.gguf --direction-index 33 --split -o parts/s-{}.gguf
./hierophant curves model.gguf --split-dir parts --limit 50 --trials 60 --max-kld 0.05 \
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

`./hierophant data translate` translates prompt sets with a model, so
directions can be captured from several languages and adapters scored in
them. it starts the built llama-server itself, optionally with an
abliterated adapter, since an aligned model often refuses to translate a
harmful prompt; `--url URL --url-model NAME` uses a running endpoint.

```
./hierophant data translate -m model.gguf --lora q38-33p.gguf --lora-scale 1.25 -ngl 99 -np 8 \
  --languages zh,ru,es,de,ja,ar,hi,sw \
  data/harmful-train.txt data/harmless-train.txt data/harmful-test.txt data/harmless-test.txt
./hierophant core capture -m model.gguf --harmful data/harmful-train.txt,data/harmful-train.zh.txt,... ...
./hierophant eval model.gguf --lora A.gguf --harmful data/harmful-test.zh.txt \
  --harmless data/harmless-test.zh.txt --judge self
```

`data/harmful-test.txt` becomes `data/harmful-test.zh.txt`: train and test
stay apart in every language. replies that refuse, answer the prompt instead
of translating it, or come back unchanged are dropped and listed in
`harmful-test.zh.dropped.json`; the manifest records the source's SHA-256,
the language and the translator. heretic's markers are english, so score
other languages with `--judge`.

## test

```
make test-e2e             # downloads two tiny pinned models into models/
make precommit
```
