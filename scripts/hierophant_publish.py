#!/usr/bin/env python3
"""stage abliteration adapters for a hugging face repo, and upload them when asked.

one repo per model and one adapter per base quant, each named after the quant it was
built from (an adapter is exact only for that file) and staged beside its sidecar.
the README is written from every sidecar staged so far, so publishing another quant
rewrites its table. nothing leaves the machine without --upload.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hierophant_eval import entry

GITHUB = "https://github.com/khimaros/hierophant"
HUB = "https://huggingface.co"
NAME_TAG = "-hierophant-"  # <Model>-hierophant-<QUANT>.gguf: the quant last, as GGUF files end
REPO_SUFFIX = "-hierophant-LoRA-GGUF"  # <user>/<Model>-hierophant-LoRA-GGUF
GGUF_SUFFIX = re.compile(r"-gguf$", re.IGNORECASE)  # a quant repo's name: <Model>-GGUF
SHARD = re.compile(r"-\d{5}-of-\d{5}$")  # a split file's shard: -00001-of-00003
TAGS = ("gguf", "lora", "llama.cpp", "hierophant", "abliteration")
STAGE = Path("lora") / "publish"  # under the current directory, one subdirectory per repo


def names_of(base, quant=""):
    """(model, quant) of a sidecar's base: the model from its repo without -GGUF, the
    quant from its file once the model and any shard suffix are taken off."""
    model = GGUF_SUFFIX.sub("", base["repo"].split("/")[-1])
    stem = SHARD.sub("", Path(base["file"]).stem)
    if not quant:
        if not stem.lower().startswith(model.lower() + "-"):
            sys.exit(f"error: cannot tell the quant of {base['file']} from {base['repo']}; give --quant")
        quant = stem[len(model) + 1:]
    return model, quant


def staged(out, model):
    """[(quant, file stem, sidecar)] of every adapter staged in out, by quant."""
    found = []
    for side in sorted(out.glob(f"{model}{NAME_TAG}*.json")):
        found.append((side.stem.removeprefix(f"{model}{NAME_TAG}"), side.stem, json.loads(side.read_text())))
    return found


def recipe_of(d):
    """a sidecar's directions as a short recipe: L34, 8 harmless PCs, a residual."""
    index = d.get("index")
    where = "per-layer" if index is None else "+".join(f"L{i:g}" for i in (index if isinstance(index, list)
                                                                            else [index]))
    parts = [where]
    if d.get("harmless_pcs"):
        parts.append(f"{d['harmless_pcs']} harmless PCs")
    if "residual" in d:
        parts.append("a residual")
    return ", ".join(parts)


def served(s):
    """the scale an adapter is served at: its sidecar's, or 1 when no scale refused less than
    the base within the budget, as best.txt serves it."""
    return s["scale"] or 1


def scores_at(s):
    """the sweep point at the served scale, so the card's numbers are the ones its serve line
    gives; the sidecar's pick when the sweep has no replies there."""
    e = s["eval"]
    point = next((p for p in e["sweep"] if p["scale"] == served(s) and "refusals" in p), None)
    if point:
        return point
    return {"refusals": e["refusals"]["adapted"], "kld": e["kld"]["mean"],
            **({"kld_seq": e["kld_seq"]["mean"]} if "kld_seq" in e else {})}


def row_of(quant, stem, s):
    r, at = s["eval"]["refusals"], scores_at(s)
    seq = f"{at['kld_seq']:.3f}" if "kld_seq" in at else "-"
    return (f"| {quant} | {stem}.gguf | {s['base']['sha256']} | {served(s):g} | {recipe_of(s['directions'])} "
            f"| {r['base']:.2f} -> {at['refusals']:.2f} (n={r['n']}) | {at['kld']:.3f} | {seq} |")


def usage_of(repo, base_repo, entries):
    lines = []
    for quant, stem, s in entries:
        lines += [f"# {quant}", f"hf download {repo} {stem}.gguf --local-dir .",
                  f"llama-server -hf {base_repo}:{quant} --lora-scaled {stem}.gguf:{served(s):g}", ""]
    return "\n".join(lines).rstrip()


def card(repo, base_repo, license, entries):
    """the repo's README.md: hub metadata, how to use each adapter, its scores and how it was made."""
    model = repo.split("/")[-1].removesuffix(REPO_SUFFIX)
    tags = "\n".join(f"- {t}" for t in TAGS)
    rows = "\n".join(row_of(*x) for x in entries)
    return f"""---
base_model: {base_repo}
base_model_relation: adapter
license: {license}
tags:
{tags}
---

# {model} hierophant LoRA

refusal-removal (abliteration) LoRA adapters for [{base_repo}]({HUB}/{base_repo}), one per
quant, made with [hierophant]({GITHUB}). each is a small file loaded beside the base model,
not a second copy of its weights.

## use

pick the adapter for the quant you serve: an adapter is exact only for the base file it
was built from.

```
{usage_of(repo, base_repo, entries)}
```

a lower scale (e.g. `:0.8`) keeps the model closer to the base; `:0` turns it off.
llama-server can also load an adapter without applying it (`--lora-init-without-apply`)
and apply it per request, so one server answers both ways.

## adapters

| quant | adapter | base file SHA-256 | scale | recipe | refusals (base -> adapted) | first-token KL | WikiText KL |
|---|---|---|---|---|---|---|---|
{rows}

refusals are counted with thinking off on held-out prompts the recipe was not chosen
on; each adapter's JSON sidecar names them, the judge and every setting. KL is measured
against the base at the scale given.

## approach

hierophant runs harmful and harmless prompts through the quantized model itself and
takes the difference of their mean residual streams at the first generated token as the
refusal direction. the adapter removes that direction from every matrix that writes into
the residual stream, computed from that quant's own dequantized weights, so at scale 1
the edit is exact for that file. which layer's direction, how strongly, and whether to
keep it clear of the harmless prompts' principal components or add a second, residual
direction were chosen by an automated search, ranked by refusals on prompts it was not
tuned on at a fixed KL budget. see [hierophant]({GITHUB}) for the method and the tools.

## limitations

- only the listed base files: on another quant the edit is approximate and unmeasured.
- with thinking on, replies can reason at length before answering; serve with a generous
  token budget.
"""


def stage(a):
    """copy each adapter and its sidecar into the repo's staging directory under its
    quant's name; returns (repo, base repo, directory, model)."""
    repo = base_repo = out = model = None
    for lora in map(Path, a.adapters):
        side = json.loads(lora.with_suffix(".json").read_text())
        this_model, quant = names_of(side["base"], a.quant)
        if model and (this_model, side["base"]["repo"]) != (model, base_repo):
            sys.exit(f"error: {lora} is for {side['base']['repo']}, not {base_repo}; publish it on its own")
        model, base_repo = this_model, side["base"]["repo"]
        repo = a.repo or f"{a.user}/{model}{REPO_SUFFIX}"
        out = Path(a.out) if a.out else STAGE / repo.split("/")[-1]
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{model}{NAME_TAG}{quant}"
        shutil.copy(lora, out / f"{stem}.gguf")
        shutil.copy(lora.with_suffix(".json"), out / f"{stem}.json")
        print(f"staged {out / stem}.gguf", flush=True)
    return repo, base_repo, out, model


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("adapters", nargs="+", help="adapter .gguf files, each with its sidecar .json beside it")
    p.add_argument("--license", required=True, help="the base model's license, as its own card names it")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--user", help=f"hub user or org: the repo is USER/<Model>{REPO_SUFFIX}")
    who.add_argument("--repo", help="the hub repo, instead of one named after the model")
    p.add_argument("--quant", default="", help="the base quant, when its file name does not say it")
    p.add_argument("--out", help=f"staging directory (default {STAGE}/<repo name>)")
    p.add_argument("--upload", action="store_true", help="upload the staging directory with hf upload")
    p.add_argument("--hf", default="hf", help="the huggingface hub cli")
    return p.parse_args(argv)


def main(argv):
    a = parse_args(argv)
    repo, base_repo, out, model = stage(a)
    (out / "README.md").write_text(card(repo, base_repo, a.license, staged(out, model)))
    print(f"wrote {out / 'README.md'}", flush=True)
    cmd = [str(a.hf), "upload", repo, str(out), ".", "--repo-type", "model"]
    if not a.upload:
        print(f"to publish: {shlex.join(cmd)}")
        return
    if subprocess.run(cmd, check=False).returncode != 0:
        sys.exit(f"error: {shlex.join(cmd)} failed")
    print(f"published {HUB}/{repo}")


if __name__ == "__main__":
    entry(main)
