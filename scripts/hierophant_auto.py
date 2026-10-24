#!/usr/bin/env python3
"""find the best abliteration adapter for a model, round by round, until it stops improving.

prepare filters the training prompts and captures every layer's directions
once. each round builds candidates (a direction index, harmless principal
components to avoid, optionally a residual direction, and a weight), scores
them on a slice of jailbreakbench + llm-lat over a sweep of scales, and ranks
them by their refusals where their kl reaches the budget. the leader is then
scored beside the current best on held-out prompts and replaces OUT/best.gguf
only if it wins there. later rounds try fewer candidates, near the leaders, on
more prompts. ctrl-c at any time leaves the best usable; rerunning redoes the
unfinished round from its finished builds and every request already answered.
"""

import argparse
import json
import os
import shutil
import sys
from itertools import pairwise
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from gguf_lite import Reader
from hierophant_eval import MAX_KLD, add_model_args, load_prompts
from hierophant_pipeline import (
    BIN,
    DATA,
    MIN_RESIDUAL,
    PROGRESS_HELP,
    Run,
    confirm_sets,
    default_out,
    loras_of,
    seed_manifest,
    step_capture,
    step_filter,
)
from progress import Bar

SELECT_OFFSET = 300  # candidates are chosen on jbb+lat from here; the first 300 report the best
REPORT_MAX = 300
FIRST_PROMPTS = 32  # doubled every round, up to all of the selection prompts
GRID = (0.3, 0.9, 0.05)  # round 1 indices: from, to and step as fractions of the depth
WIDE_SCALES = "0.5,0.75,1,1.25,1.5,2"  # round 1, before any weight is known
NARROW_SCALES = "0.85,1,1.15"  # later rounds, around the weight each candidate carries
PCS = (0, 4, 8, 16)  # harmless principal components a direction may be kept orthogonal to
BEAM = 3
MAX_CANDIDATES = 9
MIN_STEP = 0.25  # finest index step, in layers
RESIDUAL_FROM = 2  # first round that may add a residual direction
RESIDUAL_LEADERS = 2  # beam members that may get one
RESIDUAL_REUSE = 0.1  # a residual captured within this relative weight of a candidate's is reused
MIN_SCALE = 0.05  # below this the budget allows no edit; the weight is left as it was
PATIENCE = 2  # rounds at full prompts without a new best before stopping
SELECT_CHUNKS, REPORT_CHUNKS = 4, 8  # wikitext kl chunks
THINK_LIMIT, THINK_MAX_TOKENS = 50, 4096  # the final check, as the model would be served
# promotion also scores both adapters with thinking on, kept short and cheap: a low effort
# where the template takes it and a reasoning budget, after which the answer is forced
THINK_PROMPTS, THINK_BUDGET, THINK_EFFORT = 64, 512, "auto"
BUDGET_MESSAGE = "\n\ntime to answer.\n"
BUILT_SUFFIX = ".built"  # beside an adapter's output name: the arguments of its finished build
SHARED_CACHE = "cache.jsonl"  # every eval request of the run, by what determines its outcome


def key(c):
    """a candidate's name, which is also its file stem: L33, L32.5-pcs8, L33-pcs8-res."""
    return f"L{c['index']:g}" + (f"-pcs{c['pcs']}" if c["pcs"] else "") + ("-res" if c["residual"] else "")


def at_budget(points, budget):
    """{refusals, scale, kl} where a sweep's kl reaches the budget, interpolated between
    the scales around it, or its fewest refusals within the budget when nothing crosses it.
    the wikitext kl is the measure when there is one."""
    pts = sorted(({"scale": p["scale"], "refusals": p["refusals"], "kl": p.get("kld_seq", p["kld"])}
                  for p in points), key=lambda p: p["scale"])
    options = [p for p in pts if p["kl"] <= budget]
    for a, b in pairwise(pts):
        if a["kl"] <= budget < b["kl"]:
            t = (budget - a["kl"]) / (b["kl"] - a["kl"])
            options.append({"scale": a["scale"] + t * (b["scale"] - a["scale"]),
                            "refusals": a["refusals"] + t * (b["refusals"] - a["refusals"]), "kl": budget})
    return min(options, key=lambda p: (p["refusals"], p["kl"]))


def grid(n_layer):
    """round 1's indices, spread over the middle and late layers, and the step between them."""
    lo, hi, step = GRID
    fractions = [lo + k * step for k in range(round((hi - lo) / step) + 1)]
    return sorted({min(max(round(n_layer * f), 1), n_layer - 1) for f in fractions}), n_layer * step


def proposals(beam, step, n_layer, residual_for):
    """each beam member's neighbours, varying one thing, in the order they paid off on
    qwen3.8 for when the candidates are capped: a residual (for the first residual_for),
    the next harmless pcs counts, then the index by +-step (to a quarter layer)."""
    out = []
    for i, c in enumerate(beam):
        if i < residual_for and not c["residual"]:
            out.append({**c, "residual": True})
        k = PCS.index(c["pcs"])
        out += [{**c, "pcs": p} for p in PCS[max(k - 1, 0):k + 2] if p != c["pcs"]]
        for d in (-step, step):
            index = round((c["index"] + d) * 4) / 4
            if 1 <= index <= n_layer - 1 and index != c["index"]:
                out.append({**c, "index": index})
    return out


def worst(x):
    return max(x["refusals"], x.get("think", 0))


def beats(lead, best):
    """whether the leader refuses less than the best in its worse mode, thinking off or on,
    so a recipe that only holds with thinking off cannot take over."""
    return worst(lead) < worst(best)


def without_think(x):
    """x without short-thinking scores, which a promotion measures again (or, without
    --think-prompts, not at all)."""
    return {k: v for k, v in x.items() if not k.startswith("think")}


def unchanged_after(unchanged, outcome, n, pool):
    """rounds at full prompts in a row without a different recipe taking the lead."""
    return 0 if outcome == "new" or n < pool else unchanged + 1


def first_state(a, n_layer):
    indices, step = ([float(i) for i in a.indices.split(",")], 1.0) if a.indices else grid(n_layer)
    return {"round": 0, "step": step, "candidates": [{"index": i, "pcs": 0, "residual": False, "weight": 1.0}
                                                     for i in indices],
            "beam": [], "seen": [], "best": None, "unchanged": 0}


def next_candidates(st, n_layer, a, rnd):
    """the beam, then its untried neighbours, up to --max-candidates."""
    step = max(st["step"] / 2, MIN_STEP)
    residual_for = RESIDUAL_LEADERS if rnd >= RESIDUAL_FROM else 0
    out, names = list(st["beam"]), {key(c) for c in st["beam"]}
    for c in proposals(st["beam"], step, n_layer, residual_for):
        if len(out) >= a.max_candidates:
            break
        if key(c) not in names and key(c) not in st["seen"]:
            out.append(c)
            names.add(key(c))
    return out, step


class Auto(Run):
    """a pipeline run that builds, scores and promotes candidates round by round."""

    echo = False
    rollup = ("sites",)  # each build's sites advance the round's builds bar

    def build(self, d, cands):
        """one adapter per candidate in d, each named after it; those differing only in
        index share one pass over the weights."""
        label = "-".join(d.relative_to(self.out).parts)
        groups = {}
        for c in cands:
            if not c["residual"]:
                groups.setdefault((c["weight"], c["pcs"]), []).append(c)
        self.bar = Bar(f"{label} builds", len(groups) + sum(c["residual"] for c in cands), sys.stdout) \
            if self.progress else None
        for (weight, pcs), group in groups.items():
            suffix = f"-pcs{pcs}" if pcs else ""
            self.make_adapter(label, ",".join(f"{c['index']:g}" for c in group), weight, pcs, d / f"L{{}}{suffix}.gguf")
        kept = []
        for c in cands:
            resid = self.residual_dirs(c) if c["residual"] else None
            if resid:
                self.make_adapter(label, f"{c['index']:g}", c["weight"], c["pcs"], d / f"{key(c)}.gguf",
                                  "--residual", resid)
            elif c["residual"]:
                self.built_one()
                continue
            kept.append(c)
        if self.bar:
            self.bar.close()
            self.bar = None
        return kept

    def built_one(self):
        if self.bar:
            self.bar.update()

    def make_adapter(self, step, indices, weight, pcs, out, *extra):
        """build unless this exact build already finished: a rerun of an interrupted round
        keeps what it had built."""
        args = [self.dirs, "--direction-index", indices, "--weight", weight, *(["--harmless-pcs", pcs] if pcs else []),
                "--type", self.a.type, *extra]
        # an input file rewritten since (a recaptured residual) makes it a different build
        stamp = json.dumps([f"{x}@{Path(x).stat().st_mtime_ns}" if Path(str(x)).is_file() else str(x) for x in args])
        done = Path(f"{out}{BUILT_SUFFIX}")
        if not (done.exists() and done.read_text() == stamp):
            done.unlink(missing_ok=True)
            self.hierophant(step, "adapter", "-d", *args, "-o", out)
            done.write_text(stamp)
        self.built_one()

    def residual_dirs(self, c):
        """the directions of what c without its residual still refuses, captured with it
        applied; None when too few prompts are left. reused across nearby weights."""
        base = {**c, "residual": False}
        d = self.out / "residual" / key(base)
        done = d / "weight.json"
        if done.exists() and abs(json.loads(done.read_text())["weight"] / c["weight"] - 1) <= RESIDUAL_REUSE:
            return d / "dirs.gguf" if (d / "dirs.gguf").exists() else None
        # kept, not cleared: its filter's cache and finished builds survive an interrupt
        for stale in (done, d / "dirs.gguf"):
            stale.unlink(missing_ok=True)
        seed_manifest(d)
        lora, (harmful, harmless) = d / "under.gguf", self.train_sets()
        self.make_adapter("residual", f"{c['index']:g}", c["weight"], c["pcs"], lora)
        under = ["--lora", lora, "--lora-scale", 1]
        self.script("residual", "filter_prompts.py", *self.serve(), "--judge", self.a.judge, "--harmful", harmful,
                    *under, "--out-dir", d, "--server-log", self.logs / "server-residual.log")
        still = d / f"{harmful.stem}.filtered.txt"
        n = len(still.read_text().splitlines())
        if n >= MIN_RESIDUAL:
            self.hierophant("residual", "capture", "--harmful", still, "--harmless", harmless, *self.capture_flags(),
                            *under, "-o", d / "dirs.gguf")
        else:
            print(f"only {n} prompts still refused under {key(base)}; no residual for it")
        done.write_text(json.dumps({"weight": c["weight"]}) + "\n")
        return d / "dirs.gguf" if n >= MIN_RESIDUAL else None

    def prepare(self):
        print("== prepare", flush=True)
        for step, fn in (("filter", step_filter), ("capture", step_capture)):
            if (self.done / step).exists():
                print(f"{step}: done", flush=True)
                continue
            fn(self)
            (self.done / step).touch()

    def round(self, st, n_layer, pool):
        """build, score on the selection prompts, rank, then report the leader."""
        rnd = st["round"] + 1
        d = self.out / f"round-{rnd}"
        d.mkdir(parents=True, exist_ok=True)
        n = min(pool, self.a.first_prompts * 2 ** (rnd - 1))
        cands, step = (st["candidates"], st["step"]) if rnd == 1 else next_candidates(st, n_layer, self.a, rnd)
        what = "which layer" if rnd == 1 else f"refine {', '.join(key(c) for c in st['beam'])}"
        print(f"== round {rnd}: {what} ({len(cands)} candidates, {n} prompts each)", flush=True)
        cands = self.build(d, cands)
        sweeps = self.score(d.name, [d / f"{key(c)}.gguf" for c in cands], n, SELECT_OFFSET,
                            WIDE_SCALES if rnd == 1 else NARROW_SCALES, SELECT_CHUNKS, d / "select.json")
        ranked = rank(cands, sweeps, self.a.max_kld)
        print_ranked(ranked)
        best, outcome = self.report(d, ranked[0], st["best"], min(n, REPORT_MAX), rnd)
        new = {"round": rnd, "n": n, "step": step, "candidates": [], "beam": [x["cand"] for x in ranked[:self.a.beam]],
               "seen": sorted({*st["seen"], *(key(c) for c in cands)}), "best": best,
               "unchanged": unchanged_after(st["unchanged"], outcome, n, pool)}
        (d / "round.json").write_text(json.dumps({"n": n, "candidates": ranked, "outcome": outcome, "best": best},
                                                 indent=2) + "\n")
        self.finish(new, outcome)
        return new

    def report(self, d, leader, best, n, rnd):
        """the leader, rebuilt at the weight that reaches the budget, beside the current best
        on n held-out prompts, thinking off and with short thinking. returns the best from now
        on and the outcome: "new" when a different recipe takes over, "refined" when the best's
        own recipe is rebuilt at its new weight, "kept" when the best stays as it was."""
        rd = d / "report"
        rd.mkdir(exist_ok=True)
        mine = self.build(rd, [leader["cand"]])
        refined = bool(best and mine and best["key"] == key(mine[0]))
        loras = [rd / f"{key(c)}.gguf" for c in mine]
        if best and not refined:
            shutil.copy(self.out / "best.gguf", rd / "incumbent.gguf")
            loras.append(rd / "incumbent.gguf")
        if not loras:
            return best, "kept"
        sweeps = self.score(f"{d.name}-report", loras, n, 0, NARROW_SCALES, REPORT_CHUNKS, rd / "report.json",
                            sidecars=True)
        base = next(p["refusals"] for p in next(iter(sweeps.values())) if p["scale"] == 0)
        think = self.think_score(d.name, rd, loras, min(n, self.a.think_prompts))
        if best and not refined:
            best = {**without_think(best), **at_budget(sweeps["incumbent.gguf"], self.a.max_kld), "n": n,
                    "base": base, **think.get("incumbent.gguf", {})}
        if not mine:
            return best, "kept"
        c = mine[0]
        lead = {**at_budget(sweeps[f"{key(c)}.gguf"], self.a.max_kld), "key": key(c), "cand": c, "round": rnd, "n": n,
                "base": base, **think.get(f"{key(c)}.gguf", {})}
        if best and not refined and not beats(lead, best):
            return best, "kept"
        self.promote(rd / f"{key(c)}.gguf")
        return ({**lead, "round": best["round"]} if refined else lead), "refined" if refined else "new"

    def think_score(self, label, rd, loras, n):
        """{adapter file name: {think, think_base, think_n}}: refusals with thinking on, each
        adapter at its own weight (scale 1), on the first n report prompts. the reasoning is
        kept short, which is not how the model is served; the final check covers that."""
        if not n:
            return {}
        a = self.a
        capped = a.think_budget >= 0
        budget = ["--reasoning-budget", a.think_budget, "--reasoning-budget-message", BUDGET_MESSAGE] if capped else []
        self.eval(f"{label}-think",
                    *self.serve(a.think_budget + a.max_tokens if capped else a.think_max_tokens), *loras_of(loras),
                    "--harmful", confirm_sets(), "--limit", n, "--harmless-offset", 0, "--scales", 1, "--think",
                    "--reasoning-effort", a.think_effort, *budget, "--think-kwargs", json.dumps(a.think_kwargs),
                    "--judge", a.judge, "--no-sidecar", "--results", rd / "think.json",
                    "--server-log", self.logs / f"server-{label}-think.log")
        sweeps = json.loads((rd / "think.json").read_text())
        at = {name: {p["scale"]: p["refusals"] for p in s} for name, s in sweeps.items()}
        return {name: {"think": r[1], "think_base": r[0], "think_n": n} for name, r in at.items()}

    def promote(self, lora):
        """the adapter and its sidecar become best.*, each replaced in one step."""
        for ext in (".gguf", ".json"):
            tmp = self.out / f"best{ext}.tmp"
            shutil.copy(lora.with_suffix(ext), tmp)
            os.replace(tmp, self.out / f"best{ext}")

    def finish(self, st, outcome):
        b = st["best"]
        verb = {"new": "new best", "refined": "best refined", "kept": "best unchanged"}[outcome]
        thinking = f" ({b['think']:.3f} with short thinking)" if "think" in b else ""
        line = f"{b['key']} (round {b['round']})  refusals {b['refusals']:.3f}{thinking}, kl {b['kl']:.4f}"
        print(f">> {verb}: {line}\n   {self.out / 'best.gguf'}", flush=True)
        with open(self.out / "history.txt", "a") as f:
            f.write(f"round {st['round']} ({st['n']} prompts): {verb}: {line}\n")
        self.write_best(st)
        tmp = self.out / "state.json.tmp"
        tmp.write_text(json.dumps(st, indent=2) + "\n")
        os.replace(tmp, self.out / "state.json")

    def write_best(self, st):
        b, think = st["best"], st.get("think")
        scale = json.loads((self.out / "best.json").read_text())["scale"]
        lines = [f"{b['key']} at weight {b['cand']['weight']:g}, found in round {b['round']}",
                 (f"held-out: refusals {b['refusals']:.3f} (base {b['base']:.3f}) at kl {b['kl']:.4f}, "
                  f"{b['n']} prompts")]
        if "think" in b:
            lines.append(f"held-out, short thinking: refusals {b['think']:.3f} (base {b['think_base']:.3f}), "
                         f"{b['think_n']} prompts")
        if think:
            lines.append(f"thinking on: refusals {think['refusals']:.3f}, unfinished {think['unfinished']:.3f}, "
                         f"loops {think['loops']:.3f}, {think['n']} prompts")
        lora = (self.out / "best.gguf").absolute()
        if not scale:
            lines.append("no scale of it refused less than the base within the kl budget; served at 1 anyway")
        lines += [f"serve: llama-server {serve_model(self.a)} --lora-scaled {lora}:{scale or 1:g}",
                  f"sidecar: {lora.with_suffix('.json')}"]
        (self.out / "best.txt").write_text("\n".join(lines) + "\n")

    def final_check(self, st):
        """the best with thinking on, once per best: replies that never finish show up here."""
        b = st["best"]
        if not self.a.think_check or (st.get("think") or {}).get("for") == [b["key"], b["cand"]["weight"]]:
            return
        print(f"== final check: {b['key']} with thinking on ({self.a.think_limit} prompts)", flush=True)
        d = self.out / "final"
        d.mkdir(exist_ok=True)
        shutil.copy(self.out / "best.gguf", d / "best.gguf")
        scale = json.loads((self.out / "best.json").read_text())["scale"] or 1
        self.eval("final", *self.serve(self.a.think_max_tokens), "--lora", d / "best.gguf",
                    "--harmful", confirm_sets(), "--limit", self.a.think_limit, "--harmless-offset", 0,
                    "--scales", f"{scale:g}", "--think", "--judge", "self", "--no-sidecar",
                    "--results", d / "think.json", "--server-log", self.logs / "server-final.log")
        p = next(p for p in json.loads((d / "think.json").read_text())["best.gguf"] if p["scale"] != 0)
        st["think"] = {"for": [b["key"], b["cand"]["weight"]], "n": self.a.think_limit,
                       **{k: p[k] for k in ("refusals", "unfinished", "loops")}}
        self.write_best(st)
        (self.out / "state.json").write_text(json.dumps(st, indent=2) + "\n")

    def eval(self, label, *args):
        """an eval sharing the run's cache: an adapter file, prompt and setting already scored
        by an earlier round (the base, or an incumbent kept since) is not scored again."""
        self.script(label, "hierophant_eval.py", "--cache", self.out / SHARED_CACHE, *args)

    def load_state(self):
        path = self.out / "state.json"
        return json.loads(path.read_text()) if path.exists() else None

    def score(self, label, loras, n, offset, scales, chunks, results, sidecars=False):
        """{adapter file name: sweep} on n jbb+lat prompts from offset."""
        a = self.a
        self.eval(label, *self.serve(), *loras_of(loras), "--harmful", confirm_sets(),
                    "--limit", n, "--offset", offset, "--harmless-offset", 0, "--scales", scales,
                    "--kld-text", a.kld_text, "--kld-ctx", a.kld_ctx, "--kld-chunks", chunks, "--max-kld", a.max_kld,
                    "--judge", a.judge, "--perplexity", a.perplexity, *([] if sidecars else ["--no-sidecar"]),
                    "--results", results, "--server-log", self.logs / f"server-{label}.log")
        return json.loads(results.read_text())


def rank(cands, sweeps, budget):
    """candidates by refusals at the kl budget, each carrying the weight that reaches it."""
    out = []
    for c in cands:
        at = at_budget(sweeps[f"{key(c)}.gguf"], budget)
        weight = round(c["weight"] * at["scale"], 3) if at["scale"] >= MIN_SCALE else c["weight"]
        out.append({**at, "cand": {**c, "weight": weight}})
    return sorted(out, key=lambda x: (x["refusals"], x["kl"]))


def print_ranked(ranked):
    print(f"{'candidate':24} {'refusals@kl':>11} {'weight':>7} {'kl':>7}")
    for x in ranked:
        print(f"{key(x['cand']):24} {x['refusals']:11.3f} {x['cand']['weight']:7.3g} {x['kl']:7.4f}")


def serve_model(a):
    if not a.hf_repo:
        return f"-m {Path(a.model).absolute()}"
    return f"-hf {a.hf_repo}" + (f" -hff {a.hf_file}" if a.hf_file else "")


def stopped(st, a):
    if st["unchanged"] >= PATIENCE:
        print(f">> best unchanged for {PATIENCE} rounds at full prompts: converged")
        return True
    return bool(a.rounds) and st["round"] >= a.rounds


def status(out):
    for name in ("best.txt", "history.txt"):
        path = out / name
        print(path.read_text() if path.exists() else f"no {name} in {out} yet")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p)
    p.add_argument("--out", help="output directory (default lora/<model>, named after -m or --hf-repo)")
    p.add_argument("--status", action="store_true", help="print the best so far and each round's, and exit")
    p.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (default: once converged)")
    p.add_argument("--max-kld", type=float, default=MAX_KLD, help="kl budget, over --kld-text")
    p.add_argument("--judge", default="markers",
                   help="markers (heretic's), self (the served base model) or an openai-compatible base url")
    p.add_argument("--indices", help="round 1's direction indices, instead of a grid over the depth")
    p.add_argument("--train-data", help="filtered training sets from another run's OUT/data")
    g = p.add_argument_group("effort")
    g.add_argument("--first-prompts", type=int, default=FIRST_PROMPTS, help="round 1's prompts, doubled each round")
    g.add_argument("--beam", type=int, default=BEAM, help="leaders each round refines")
    g.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    g.add_argument("--train-limit", type=int, default=0)
    g.add_argument("--think-prompts", type=int, default=THINK_PROMPTS,
                   help="prompts each promotion also scores with short thinking (0: thinking off only)")
    g.add_argument("--think-budget", type=int, default=THINK_BUDGET,
                   help="reasoning tokens before the answer is forced, in promotion (-1: no limit)")
    g.add_argument("--think-effort", default=THINK_EFFORT,
                   help="reasoning_effort in promotion: a level the model's template names, auto for low "
                        "unless it rejects that, or empty for the template default")
    g.add_argument("--think-kwargs", type=json.loads, default={},
                   help="more chat template kwargs as json for thinking replies, for a model's own knobs")
    g.add_argument("--no-think-check", dest="think_check", action="store_false",
                   help="skip the final check, which runs the best as served: template default effort, no budget")
    g.add_argument("--think-limit", type=int, default=THINK_LIMIT)
    g.add_argument("--think-max-tokens", type=int, default=THINK_MAX_TOKENS)
    g = p.add_argument_group("running")
    g.add_argument("-v", "--verbose", action="store_true",
                   help="stream every command and its own progress, not one line per command")
    g.add_argument("--progress", action="store_true", help=PROGRESS_HELP)
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--kld-text", default=DATA / "wikitext-test.txt")
    g.add_argument("--kld-ctx", type=int, default=512)
    g.add_argument("--type", choices=("f16", "f32"), default="f16")
    g.add_argument("-ngl", "--ngl", type=int, default=-1)
    g.add_argument("-np", "--parallel", type=int, default=4)
    g.add_argument("--server", default=BIN / "llama-server")
    g.add_argument("--perplexity", default=BIN / "llama-perplexity")
    g.add_argument("--hierophant", default=BIN / "hierophant")
    a = p.parse_args(argv)
    if not a.model and not a.hf_repo:
        p.error("give -m or --hf-repo")
    a.out = a.out or default_out(a)
    # what the pipeline's prepare steps read: one capture that every pcs count can use
    vars(a).update(no_filter=False, positions=1, harmless_pcs=max(PCS), think_capture=False, name="best")
    return a


def main(argv):
    a = parse_args(argv)
    if a.status:
        return status(Path(a.out).expanduser())
    r = Auto(a)
    r.echo = a.verbose
    r.prepare()
    n_layer = Reader(r.dirs).tensors["harmful_mean"]["ne"][1]
    pool = len(load_prompts(confirm_sets())) - SELECT_OFFSET
    st = r.load_state() or first_state(a, n_layer)
    for k in range(1, st["round"] + 1):
        print(f"== round {k}: done", flush=True)
    while not stopped(st, a):
        st = r.round(st, n_layer, pool)
    if st["best"]:
        r.final_check(st)
        print(f"best weights: {r.out / 'best.gguf'}\n" + (r.out / "best.txt").read_text(), end="", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
