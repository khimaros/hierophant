#!/usr/bin/env python3
"""stage abliteration adapters for a hugging face repo, and upload them when asked.

one repo per model and one adapter per base quant, each named after the quant it was
built from (an adapter is exact only for that file) and staged beside its sidecar.
the README is written from every sidecar staged so far, so publishing another quant
rewrites its table. nothing leaves the machine without --upload, which the machine's
config can pass.
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
from hierophant_config import built_recipe, recipe_name
from hierophant_eval import entry
from hierophant_preview import write_preview

GITHUB = "https://github.com/khimaros/hierophant"
HUB = "https://huggingface.co"
NAME_TAG = "-hierophant-"  # <Model>-hierophant-<QUANT>.gguf: the quant last, as GGUF files end
REPO_SUFFIX = "-hierophant-LoRA-GGUF"  # <user>/<org>-<Model>-hierophant-LoRA-GGUF
GGUF_SUFFIX = re.compile(r"-gguf$", re.IGNORECASE)  # a quant repo's name: <Model>-GGUF
SHARD = re.compile(r"-\d{5}-of-\d{5}$")  # a split file's shard: -00001-of-00003
TAGS = ("gguf", "lora", "llama.cpp", "hierophant", "abliteration")
STAGE = Path("out") / "publish"  # under the current directory, one subdirectory per repo
COLLECTION = "hierophant"  # the hub collection every uploaded repo joins
# at most 150 characters; the hub only reads it when the collection is first created
COLLECTION_ABOUT = "abliteration LoRA adapters for GGUF models"
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
CARD = TEMPLATES / "model-card.md"  # {placeholders} filled in by card
BANNER = TEMPLATES / "hierophant-banner.jpeg"  # staged beside each card
HUB_LICENSES = ("apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "gpl-3.0")  # those models here use
SHA_CHARS = 12  # of the base's SHA-256 in the card's table; its sidecar has all of it
COMMIT_CHARS = 9  # of the llama.cpp commit there, as git abbreviates it
LADDER = "ladder-"  # ladder-<QUANT>.json: every recipe a quant's search tried, kept apart from the sidecars
LADDER_HEAD = "| recipe | last round | prompts | refusals | weight | KL |\n|---|---|---|---|---|---|"
LADDER_INTRO = """## every recipe tried

each recipe at the last round it reached, furthest first. these are the search's selection
scores, not held out: a round's leader is flattered by being picked on those same prompts,
and recipes that stopped early were scored on fewer of them. only the table above is
held-out."""


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


def repo_of(user, base_repo, model):
    """the hub repo of a base's adapters. it names the base's uploader, as a config's name
    does, since an adapter is exact for one uploader's files; the hub forbids "--"."""
    return f"{user}/{base_repo.partition('/')[0]}-{model}{REPO_SUFFIX}"


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
    final = s["eval"].get("final")
    loops = f"{final['loops']:.2f} (n={final['n']})" if final else "-"
    return (f"| {quant} | {stem}.gguf | {s['base']['sha256'][:SHA_CHARS]} | {s.get('llama_cpp', '-')[:COMMIT_CHARS]} "
            f"| {served(s):g} | {recipe_of(s['directions'])} "
            f"| {r['base']:.2f} -> {at['refusals']:.2f} (n={r['n']}) | {at['kld']:.3f} | {seq} | {loops} |")


def usage_of(repo, base_repo, entries):
    lines = []
    for quant, stem, s in entries:
        lines += [f"# {quant}", f"hf download {repo} {stem}.gguf --local-dir .",
                  f"llama-server -hf {base_repo}:{quant} --lora-scaled {stem}.gguf:{served(s):g}", ""]
    return "\n".join(lines).rstrip()


def round_of(path):
    return int(path.parent.name.removeprefix("round-"))


def ladder_of(run):
    """every candidate of a run's rounds at the last round it reached, furthest first and
    best first within a round."""
    last = {}
    for path in sorted(run.glob("round-*/round.json"), key=round_of):
        r = json.loads(path.read_text())
        for x in r["candidates"]:
            name = recipe_name(x["cand"])
            last[name] = {"recipe": name, "round": round_of(path), "prompts": r["n"], "refusals": x["refusals"],
                          "weight": x["cand"]["weight"], "kl": x["kl"]}
    return sorted(last.values(), key=lambda x: (-x["round"], x["refusals"], x["kl"]))


def ladder_section(out):
    """every recipe tried, one collapsed table per staged quant, for the end of the card."""
    tables = []
    for path in sorted(out.glob(f"{LADDER}*.json")):
        ladder = json.loads(path.read_text())
        rows = "\n".join(f"| {x['recipe']}{' (published)' if x['published'] else ''} | {x['round']} | {x['prompts']} "
                         f"| {x['refusals']:.3f} | {x['weight']:g} | {x['kl']:.3f} |" for x in ladder)
        quant = path.stem.removeprefix(LADDER)
        tables.append(f"<details><summary>{quant}: {len(ladder)} recipes</summary>\n\n{LADDER_HEAD}\n{rows}\n\n</details>")
    return "\n\n".join([LADDER_INTRO, *tables]) + "\n" if tables else ""


def license_lines(license, base_repo):
    """the card's license metadata: the hub's field only takes its own identifiers, so a
    model's own license is "other", by name, linking to the base that carries its text."""
    if license in HUB_LICENSES:
        return f"license: {license}"
    return f"license: other\nlicense_name: {license}\nlicense_link: {HUB}/{base_repo}"


def card(repo, base_repo, license, entries, ladder=""):
    """the repo's README.md: hub metadata, how to use each adapter, its scores and how it was made."""
    model = GGUF_SUFFIX.sub("", base_repo.split("/")[-1])
    return CARD.read_text().format(
        model=model, base_repo=base_repo, license=license_lines(license, base_repo), hub=HUB, github=GITHUB,
        banner=BANNER.name,
        tags="\n".join(f"- {t}" for t in TAGS), rows="\n".join(row_of(*x) for x in entries),
        usage=usage_of(repo, base_repo, entries), ladder=ladder)


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
        repo = a.repo or repo_of(a.user, base_repo, model)
        out = Path(a.out) if a.out else STAGE / repo.split("/")[-1]
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(BANNER, out / BANNER.name)
        stem = f"{model}{NAME_TAG}{quant}"
        shutil.copy(lora, out / f"{stem}.gguf")
        shutil.copy(lora.with_suffix(".json"), out / f"{stem}.json")
        published = built_recipe(side["directions"])
        ladder = [{**x, "published": x["recipe"] == published} for x in ladder_of(lora.parent)]
        if ladder:
            (out / f"{LADDER}{quant}.json").write_text(json.dumps(ladder, indent=2) + "\n")
        print(f"staged {out / stem}.gguf", flush=True)
    return repo, base_repo, out, model


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("adapters", nargs="+", help="adapter .gguf files, each with its sidecar .json beside it")
    p.add_argument("--license", required=True, help="the base model's license, as its own card names it")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--user", help=f"hub user or org: the repo is USER/<org>-<Model>{REPO_SUFFIX}")
    who.add_argument("--repo", help="the hub repo, instead of one named after the model")
    p.add_argument("--quant", default="", help="the base quant, when its file name does not say it")
    p.add_argument("--out", help=f"staging directory (default {STAGE}/<repo name>)")
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=False,
                   help="upload the staging directory with hf upload; --no-upload stages only, whatever "
                        "the machine's config says")
    p.add_argument("--collection", default=COLLECTION,
                   help="the hub collection an upload joins: a title in the repo's namespace, created if "
                        "missing, or a NAMESPACE/SLUG; empty for none")
    p.add_argument("--hf", default="hf", help="the huggingface hub cli")
    return p.parse_args(argv)


def collect(a, repo):
    """add an uploaded repo to its collection, creating the collection the first time."""
    slug = a.collection
    if "/" not in slug:
        made = [str(a.hf), "collections", "create", slug, "--namespace", repo.partition("/")[0],
                "--description", COLLECTION_ABOUT, "--exists-ok", "--format", "quiet"]
        p = subprocess.run(made, stdout=subprocess.PIPE, text=True, check=False)
        if p.returncode != 0:
            sys.exit(f"error: {shlex.join(made)} failed; the upload itself succeeded")
        slug = p.stdout.strip()
    add = [str(a.hf), "collections", "add-item", slug, repo, "model", "--exists-ok"]
    if subprocess.run(add, check=False).returncode != 0:
        sys.exit(f"error: {shlex.join(add)} failed; the upload itself succeeded")
    print(f"in collection {HUB}/collections/{slug}")


def main(argv):
    a = parse_args(argv)
    repo, base_repo, out, model = stage(a)
    (out / "README.md").write_text(card(repo, base_repo, a.license, staged(out, model), ladder_section(out)))
    print(f"wrote {out / 'README.md'}\npreview: {write_preview(repo, out)}", flush=True)
    cmd = [str(a.hf), "upload", repo, str(out), ".", "--repo-type", "model"]
    if not a.upload:
        print(f"to publish: {shlex.join(cmd)}")
        return
    if subprocess.run(cmd, check=False).returncode != 0:
        sys.exit(f"error: {shlex.join(cmd)} failed")
    print(f"published {HUB}/{repo}")
    if a.collection:
        collect(a, repo)


if __name__ == "__main__":
    entry(main)
