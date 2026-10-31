---
base_model: {base_repo}
base_model_relation: adapter
{license}
tags:
{tags}
---

![hierophant]({banner})

# {model} hierophant LoRA

refusal-removal (abliteration) LoRA adapters for [{base_repo}]({hub}/{base_repo}), one per
quant, made with [hierophant]({github}). each is a small file loaded beside the base model,
not a second copy of its weights.

## use

pick the adapter for the quant you serve: an adapter is exact only for the base file it
was built from.

```
{usage}
```

a lower scale (e.g. `:0.8`) keeps the model closer to the base; `:0` turns it off.
llama-server can also load an adapter without applying it (`--lora-init-without-apply`)
and apply it per request, so one server answers both ways.

## adapters

| quant | adapter | base file SHA-256 | llama.cpp | scale | recipe | refusals (base -> adapted) | first-token KL | WikiText KL | loops |
|---|---|---|---|---|---|---|---|---|---|
{rows}

refusals are counted with thinking off on held-out prompts the recipe was not chosen
on; each adapter's JSON sidecar names them, the judge and every setting. KL is measured
against the base at the scale given. loops is the share of full-length replies to those
harmful prompts that keep repeating a passage, which short replies and KL do not show.

## approach

hierophant runs harmful and harmless prompts through the quantized model itself and
takes the difference of their mean residual streams at the first generated token as the
refusal direction. the adapter removes that direction from every matrix that writes into
the residual stream, computed from that quant's own dequantized weights, so at scale 1
the edit is exact for that file. which layer's direction, how strongly, and whether to
keep it clear of the harmless prompts' principal components or add a second, residual
direction were chosen by an automated search, ranked by refusals on prompts it was not
tuned on at a fixed KL budget. see [hierophant]({github}) for the method and the tools.

## limitations

- only the listed base files: on another quant the edit is approximate and unmeasured.
- with thinking on, replies can reason at length before answering; serve with a generous
  token budget.

{ladder}