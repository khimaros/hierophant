#!/usr/bin/env python3
"""find the best abliteration adapter for a model, round by round, until it stops improving.

prepare filters the training prompts and captures every layer's directions
once. each round builds candidates (a direction index, harmless principal
components to avoid, optionally a residual direction, and a weight), scores
them on a slice of jailbreakbench + llm-lat over a sweep of scales, and ranks
them by their refusals where their kl reaches the budget. the leader is then
scored beside the current best on held-out prompts and replaces OUT/best.gguf
only if it wins there. round 1 tries every middle and late layer with each first
pcs count; the next rounds keep the better half of the last (and anything within
noise of it), adding residuals
once few are left, until a beam is left; later rounds refine that beam's
neighbours. every round doubles the prompts. ctrl-c at any time leaves the best usable; rerunning redoes the
unfinished round from its finished builds and every request already answered.
"""

import argparse
import json
import math
import os
import shutil
import sys
from itertools import pairwise
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from gguf_lite import Reader
from hierophant_config import accept as config_accept
from hierophant_config import accepted_recipe, load_configs, parse_recipe
from hierophant_eval import MAX_KLD, add_model_args, entry, has_marker, kl_of, load_prompts
from hierophant_run import (
    BIN,
    DATA,
    MIN_RESIDUAL,
    PROGRESS_HELP,
    RESUME_HINT,
    Run,
    confirm_sets,
    default_out,
    loras_of,
    seed_manifest,
    step_capture,
    step_filter,
)
from progress import Bar, Quiet, outer

SELECT_OFFSET = 300  # candidates are chosen on jbb+lat from here; the first 300 report the best
REPORT_PROMPTS = 300  # every promotion is decided on the same held-out prompts
# doubled every round, up to all of the selection prompts. at 15% refusals 32 prompts carry
# +-0.06, more than candidates differ by, which halving survives by keeping half of them
FIRST_PROMPTS = 32
GRID = (0.3, 0.9)  # round 1 tries every layer from and to these fractions of the depth
FIRST_PCS = "0,8"  # round 1 tries every index with each of these harmless pcs counts
RESIDUAL_SURVIVORS = 8  # halving adds residuals once this few survive: each needs a capture
WIDE_SCALES = "0.5,0.75,1,1.25,1.5,2"  # before a candidate's weight is known
NARROW_SCALES = "0.85,1,1.15"  # around the weight a candidate carries
PCS = (0, 4, 8, 16)  # harmless principal components a direction may be kept orthogonal to
BEAM = 4
MAX_CANDIDATES = 9
MIN_STEP = 0.25  # finest index step, in layers
RESIDUAL_REUSE = 0.1  # a residual captured within this relative weight of a candidate's is reused
MIN_SCALE = 0.05  # below this the budget allows no edit; the weight is left as it was
PATIENCE = 2  # rounds at full prompts without a new best before stopping
SELECT_CHUNKS, REPORT_CHUNKS = 4, 8  # wikitext kl chunks
THINK_LIMIT, THINK_MAX_TOKENS = 50, 4096  # the final check, as the model would be served
# promotion also scores both adapters with thinking on, kept short and cheap: a low effort
# where the template takes it and a reasoning budget, after which the answer is forced
THINK_PROMPTS, THINK_BUDGET, THINK_EFFORT = 128, 512, "auto"
BUDGET_MESSAGE = "\n\ntime to answer.\n"
BUILT_SUFFIX = ".built"  # beside an adapter's output name: the arguments of its finished build
SHARED_CACHE = "cache.jsonl"  # every eval request of the run, by what determines its outcome
GIB = 1 << 30
# selection ranks by refusals alone, so it leaves out the reply logp, its slowest stage
# (a request per reply token); the report keeps it for the sidecar of what it promotes
SELECT_ONLY = ("--logp-tokens", 0)
# --screen-layers: every layer of the grid's range with each first pcs count, ranked by how unlikely the base's own
# first reply token becomes where the kl reaches the budget: one-token requests, no replies
SCREEN_ONLY = ("--no-replies", "--logp-tokens", 1)
SCREEN_SCALES = "0.5,1,1.5,2"
SCREEN_PROMPTS = 64
SCREEN_KEEP = 16  # (layer, pcs) pairs: as many as 8 layers would give round 1


def key(c):
    """a candidate's name, which is also its file stem: L33, L32.5-pcs8, L33-pcs8-res."""
    return f"L{c['index']:g}" + (f"-pcs{c['pcs']}" if c["pcs"] else "") + ("-res" if c["residual"] else "")


def at_budget(points, budget, metric="refusals"):
    """{metric, scale, kl} where a sweep's kl (kl_of) reaches the budget, interpolated
    between the scales around it, or its least metric within the budget when nothing
    crosses it. a point without the metric (one --at-budget measured the kl of alone)
    only placed the budget."""
    pts = sorted(({"scale": p["scale"], metric: p[metric], "kl": kl_of(p)} for p in points if metric in p),
                 key=lambda p: p["scale"])
    options = [p for p in pts if p["kl"] <= budget]
    for a, b in pairwise(pts):
        if a["kl"] <= budget < b["kl"]:
            t = (budget - a["kl"]) / (b["kl"] - a["kl"])
            options.append({"scale": a["scale"] + t * (b["scale"] - a["scale"]),
                            metric: a[metric] + t * (b[metric] - a[metric]), "kl": budget})
    return min(options, key=lambda p: (p[metric], p["kl"]))


def grid(n_layer):
    """round 1's indices, every middle and late layer, and the step between them: neighbouring
    layers' directions differ (a cosine near 0.82 on qwen3.8), and a sparser grid missed L33."""
    lo, hi = GRID
    return list(range(max(round(n_layer * lo), 1), min(round(n_layer * hi), n_layer - 1) + 1)), 1.0


def proposals(beam, step, n_layer, residual):
    """the beam members' neighbours, each varying one thing, member by member in rank order
    so that a cap cuts the lower-ranked members' moves, never the leader's shapes; within a
    member cheapest first: the index by +-step (to a quarter layer), the next harmless pcs
    count either side, then a residual (with residual), which needs a filter and a capture
    of its own. a residual member only moves its index, and a pcs count that wins joins the
    beam, so further counts are reached a step at a time. a changed shape moves where the
    kl budget is reached, so it is swept wide rather than around the weight it inherits."""

    def moves(c):
        out = [{**c, "index": i} for i in (round((c["index"] + d) * 4) / 4 for d in (-step, step))
               if 1 <= i <= n_layer - 1 and i != c["index"]]
        if c["residual"]:
            return out
        out += [{**c, "pcs": PCS[j], "wide": True}
                for j in (PCS.index(c["pcs"]) - 1, PCS.index(c["pcs"]) + 1) if 0 <= j < len(PCS)]
        return out + ([{**c, "residual": True, "wide": True}] if residual else [])

    return [m for c in beam for m in moves(c)]


def shape(c):
    return c["pcs"], c["residual"]


def stderr(p, n):
    """the standard error of a refusal rate p on n prompts, floored at one refusal."""
    p = max(p, 1 / n)
    return math.sqrt(p * (1 - p) / n)


def select_beam(ranked, n, size):
    """the leaders the next round refines, in rank order: each shape's best first, so a
    recipe change is not crowded out by neighbouring indices of one, then others of a shape
    while within the noise of the leader (the standard error of a difference on n prompts)."""
    lead = ranked[0]["refusals"]
    noise = math.sqrt(2) * stderr(lead, n)
    firsts = {}
    for i, x in enumerate(ranked):
        firsts.setdefault(shape(x["cand"]), i)
    picked = sorted(firsts.values())[:size]
    near = [i for i, x in enumerate(ranked) if i not in picked and x["refusals"] <= lead + noise]
    return [ranked[i]["cand"] for i in sorted(picked + near[:size - len(picked)])]


def halve(ranked, n, size, seen=(), last=False):
    """(candidates, beam) after a halving round on n prompts: its better half by rank and
    anything within a standard error of the last of it, which the prompts cannot tell from
    it, each at the weight it was found to reach the budget at. once at most
    RESIDUAL_SURVIVORS are left, each one's residual variant joins them, unless seen before.
    before the last round only that many go on: near ties could otherwise keep more than
    that all the way, and no residual would be measured at all. once the survivors fit in
    size, no candidates and them as the beam."""
    cut = ranked[math.ceil(len(ranked) / 2) - 1]["refusals"]
    keep = [x["cand"] for x in ranked if x["refusals"] <= cut + stderr(cut, n)]
    keep = keep[:RESIDUAL_SURVIVORS] if last else keep
    if len(keep) <= size:
        return [], keep
    residuals = [{**c, "residual": True, "wide": True} for c in keep if not c["residual"]]
    return keep + [r for r in residuals if len(keep) <= RESIDUAL_SURVIVORS and key(r) not in seen], []


def plan(ranked, halving, n, pool, seen, a):
    """(candidates, beam) for the next round: halving goes on while the prompts can still
    double, since only more prompts split a tie; then the beam is chosen."""
    if not halving or n >= pool:
        return [], select_beam(ranked, n, a.beam)
    return halve(ranked, n, a.beam, seen, last=n * 2 >= pool)


def beats(lead, best):
    """whether the leader refuses less than the best with thinking off, on all the report's
    prompts, and not more with short thinking beyond the standard error of a difference on
    its fewer prompts: there, one prompt either way decided ties, so it guards against a
    recipe that only holds with thinking off rather than ranking."""
    if lead["refusals"] >= best["refusals"]:
        return False
    if "think" not in lead or "think" not in best:
        return True
    noise = math.sqrt(2) * stderr(max(lead["think"], best["think"]), min(lead["think_n"], best["think_n"]))
    return lead["think"] - best["think"] <= noise


def without_think(x):
    """x without short-thinking scores, which a promotion measures again (or, without
    --think-prompts, not at all)."""
    return {k: v for k, v in x.items() if not k.startswith("think")}


def unchanged_after(unchanged, outcome, n, pool):
    """rounds at full prompts in a row without a different recipe taking the lead."""
    return 0 if outcome == "new" or n < pool else unchanged + 1


def first_state(a, n_layer):
    indices, step = ([float(i) for i in a.indices.split(",")], 1.0) if a.indices else grid(n_layer)
    cands = [{**parse_recipe(a.recipe), "weight": a.recipe_weight, "wide": True}] if a.recipe else \
        crossed(indices, a.first_pcs)
    return {"round": 0, "step": step, "candidates": cands, "beam": [], "seen": [], "best": None, "unchanged": 0,
            "halving": True}


def crossed(indices, pcs):
    """round 1's candidates: every index with every pcs count, plain and swept wide."""
    return [{"index": float(i), "pcs": p, "residual": False, "weight": 1.0, "wide": True}
            for i in indices for p in pcs]


def next_candidates(st, n_layer, a):
    """the beam, then its untried neighbours, up to --max-candidates."""
    step = max(st["step"] / 2, MIN_STEP)
    out, names = list(st["beam"]), {key(c) for c in st["beam"]}
    for c in proposals(st["beam"], step, n_layer, residual=True):
        if len(out) >= a.max_candidates:
            break
        if key(c) not in names and key(c) not in st["seen"]:
            out.append(c)
            names.add(key(c))
    return out, step


class Auto(Run):
    """a run that builds, scores and promotes candidates round by round."""

    echo = False
    rollup = ("sites",)  # each build's sites advance the round's builds bar
    round_bar = Quiet()  # the round's stage, before every bar drawn in it

    def __init__(self, a):
        """a named run keeps its own rounds and best in OUT/runs/NAME; what any recipe of
        this base can reuse (the directions, the cache, the residuals) stays in OUT."""
        super().__init__(a)
        self.root, self.out = self.out, run_out(a)
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def build(self, d, cands):
        """one adapter per candidate in d, each named after it; those differing only in
        index share one pass over the weights."""
        label = "-".join(d.relative_to(self.out).parts)
        groups = {}
        for c in cands:
            if not c["residual"]:
                groups.setdefault((c["weight"], c["pcs"]), []).append(c)
        first = next(iter(groups.values()), None) or [c for c in cands if c["residual"]][:1]
        self.bar = Bar(f"{label} builds", len(groups) + sum(c["residual"] for c in cands), sys.stdout,
                       detail=",".join(key(c) for c in first)) if self.progress else None
        for (weight, pcs), group in groups.items():
            suffix = f"-pcs{pcs}" if pcs else ""
            self.doing = ",".join(key(c) for c in group)
            self.make_adapter(label, ",".join(f"{c['index']:g}" for c in group), weight, pcs, d / f"L{{}}{suffix}.gguf")
        kept = []
        for c in cands:
            self.doing = key(c)
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
        d = self.root / "residual" / key(base)
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
        halving = st.get("halving")
        cands, step = (st["candidates"], st["step"]) if halving else next_candidates(st, n_layer, self.a)
        screening = rnd == 1 and self.a.screen_layers and not self.a.indices
        what = ("which layer and pcs" if rnd == 1 else f"the better half of round {rnd - 1}") if halving else \
            f"refine {', '.join(key(c) for c in st['beam'])}"
        count = f"the best {self.a.screen_keep} screened" if screening else len(cands)
        print(f"== round {rnd}: {what} ({count} candidates, {n} prompts each)", flush=True)
        # the screen, builds, a selection per sweep width, the report's build, the report, its short thinking
        stages = 4 + screening + len({c.get("wide", False) for c in cands})
        with outer(self.progress, f"round {rnd}", stages, sys.stdout) as self.round_bar:
            if screening:
                self.round_bar.stage("screen")
                cands = self.screen(d, n_layer)
            self.round_bar.stage("builds")
            cands = self.build(d, cands)
            ranked = self.settle_residuals(d, rank(cands, self.select(d, cands, n), self.a.max_kld), n)
            print_ranked(ranked)
            best, outcome = self.report(d, ranked[0], st["best"], min(self.a.report_prompts, SELECT_OFFSET), rnd)
        seen = {*st["seen"], *(key(c) for c in cands)}
        nxt, beam = plan(ranked, halving, n, pool, seen, self.a)
        new = {"round": rnd, "n": n, "step": step, "candidates": nxt, "beam": beam, "halving": bool(nxt),
               "seen": sorted(seen), "best": best,
               "unchanged": unchanged_after(st["unchanged"], outcome, n, pool)}
        (d / "round.json").write_text(json.dumps({"n": n, "candidates": ranked, "outcome": outcome, "best": best},
                                                 indent=2) + "\n")
        self.finish(new, outcome)
        return new

    def screen(self, d, n_layer, known=None):
        """every layer of the grid's range with each first pcs count, built a pass per count
        and scored on the first token alone: how unlikely the base's own first reply token
        becomes where the kl reaches the budget. returns the --screen-keep least likely, round
        1's candidates. known {name: refusals} is printed beside them, to check the ranking."""
        indices, _ = grid(n_layer)
        cands = crossed(range(indices[0], indices[-1] + 1), self.a.first_pcs)
        self.build(d, cands)
        sweeps = self.score(f"{d.name}-screen", [d / f"{key(c)}.gguf" for c in cands], self.a.screen_prompts,
                            SELECT_OFFSET, SCREEN_SCALES, SELECT_CHUNKS, d / "screen.json", extra=SCREEN_ONLY)
        ranked = sorted(((at_budget(sweeps[f"{key(c)}.gguf"], self.a.max_kld, "reply_logp"), c) for c in cands),
                        key=lambda x: (x[0]["reply_logp"], x[0]["kl"]))
        print(f"{'screened':24} {'reply_logp@kl':>13} {'scale':>7}" + (f" {'round 1':>7}" if known else ""))
        for at, c in ranked:
            was = f" {known[key(c)]:7.3f}" if key(c) in (known or {}) else ""
            print(f"{key(c):24} {at['reply_logp']:13.3f} {at['scale']:7.3g}{was}")
        return [c for _, c in ranked[:self.a.screen_keep]]

    def screen_only(self, n_layer):
        """the screen alone in OUT/screen, beside round 1's refusals where it has them;
        the run itself is left as it was."""
        r1 = self.out / "round-1" / "round.json"
        known = {key(x["cand"]): x["refusals"] for x in json.loads(r1.read_text())["candidates"]} if r1.exists() else {}
        d = self.out / "screen"
        d.mkdir(exist_ok=True)
        self.screen(d, n_layer, known)

    def settle_residuals(self, d, ranked, n):
        """ranked, after one more try for each residual that landed far from the weight it was
        captured under: a capture under a much stronger adapter is not the residual that weight
        would serve, so its first score says little. building it again recaptures it."""
        moved = [x["cand"] for x in ranked if x["cand"]["residual"] and self.capture_moved(x["cand"])]
        if not moved:
            return ranked
        again = d / f"{d.name}-residuals"
        again.mkdir(parents=True, exist_ok=True)
        kept = self.build(again, moved)
        retried = {key(x["cand"]): x for x in rank(kept, self.select(again, kept, n), self.a.max_kld)}
        return sorted((retried.get(key(x["cand"]), x) for x in ranked), key=lambda x: (x["refusals"], x["kl"]))

    def capture_moved(self, c):
        """whether c's weight is too far from the one its residual was captured under to reuse it."""
        done = self.root / "residual" / key({**c, "residual": False}) / "weight.json"
        return done.exists() and abs(json.loads(done.read_text())["weight"] / c["weight"] - 1) > RESIDUAL_REUSE

    def select(self, d, cands, n):
        """{adapter file name: sweep} on n selection prompts, swept wide where a candidate's
        weight is not known yet and narrow around it where it is."""
        sweeps = {}
        for wide, suffix, scales in ((True, "-wide", WIDE_SCALES), (False, "", NARROW_SCALES)):
            group = [d / f"{key(c)}.gguf" for c in cands if c.get("wide", False) == wide]
            if group:
                self.round_bar.stage(f"select{suffix}")
                sweeps |= self.score(f"{d.name}{suffix}", group, n, SELECT_OFFSET, scales, SELECT_CHUNKS,
                                     d / f"select{suffix}.json", extra=SELECT_ONLY)
        return sweeps

    def report(self, d, leader, best, n, rnd):
        """the leader, rebuilt at the weight that reaches the budget, beside the current best
        on n held-out prompts, thinking off and with short thinking. returns the best from now
        on and the outcome: "new" when a different recipe takes over, "refined" when the best's
        own recipe is rebuilt at its new weight, "kept" when the best stays as it was."""
        self.round_bar.stage("report builds")
        rd = d / "report"
        rd.mkdir(exist_ok=True)
        mine = self.build(rd, [leader["cand"]])
        self.round_bar.stage("report")
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
        scales = {x: at_budget(sweeps[x.name], self.a.max_kld)["scale"] or 1 for x in loras}
        think = self.think_score(d.name, rd, scales, min(n, self.a.think_prompts))
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

    def think_score(self, label, rd, scales, n):
        """{adapter file name: {think, think_base, think_n}}: refusals with thinking on, each
        adapter of {path: scale} at the scale its report score was measured at, on the first
        n report prompts. the reasoning is kept short, which is not how the model is served;
        the final check covers that."""
        if not n:
            return {}
        self.round_bar.stage("think")
        a = self.a
        capped = a.think_budget >= 0
        budget = ["--reasoning-budget", a.think_budget, "--reasoning-budget-message", BUDGET_MESSAGE] if capped else []
        by_scale, sweeps = {}, {}
        for lora, scale in scales.items():
            by_scale.setdefault(scale, []).append(lora)
        for i, (scale, loras) in enumerate(by_scale.items()):
            part = rd / f"think-{i}.json"
            self.eval(f"{label}-think",
                        *self.serve(a.think_budget + a.max_tokens if capped else a.think_max_tokens), *loras_of(loras),
                        "--harmful", confirm_sets(), "--limit", n, "--harmless-offset", 0, "--scales", f"{scale:g}",
                        "--think", "--reasoning-effort", a.think_effort, *budget,
                        "--think-kwargs", json.dumps(a.think_kwargs), "--judge", a.judge, "--no-sidecar",
                        "--results", part, "--server-log", self.logs / f"server-{label}-think.log")
            sweeps |= json.loads(part.read_text())
            part.unlink()
        (rd / "think.json").write_text(json.dumps(sweeps, indent=2) + "\n")
        at = {name: {p["scale"]: p for p in s} for name, s in sweeps.items()}
        return {Path(lora).name: {"think": at[Path(lora).name][s]["refusals"], "think_n": n,
                                  "think_base": at[Path(lora).name][0]["refusals"],
                                  "think_reasoned": reasoned(at[Path(lora).name][s])}
                for lora, s in scales.items()}

    def promote(self, lora):
        """the adapter and its sidecar become best.*, each replaced in one step."""
        for ext in (".gguf", ".json"):
            tmp = self.out / f"best{ext}.tmp"
            shutil.copy(lora.with_suffix(ext), tmp)
            os.replace(tmp, self.out / f"best{ext}")

    def finish(self, st, outcome):
        b = st["best"]
        verb = {"new": "new best", "refined": "best refined", "kept": "best unchanged"}[outcome]
        thinking = f" ({b['think']:.3f} with short thinking{no_reasoning(b)})" if "think" in b else ""
        line = f"{b['key']} (round {b['round']})  refusals {b['refusals']:.3f}{thinking}, kl {b['kl']:.4f}"
        print(f">> {verb}: {line}\n   {self.out / 'best.gguf'}", flush=True)
        with open(self.out / "history.txt", "a") as f:
            f.write(f"round {st['round']} ({st['n']} prompts): {verb}: {line}\n")
        self.write_best(st)
        tmp = self.out / "state.json.tmp"
        tmp.write_text(json.dumps(st, indent=2) + "\n")
        os.replace(tmp, self.out / "state.json")
        if not self.a.keep_rounds:
            remove(dead_files(self.root))

    def compare(self):
        """--lora adapters beside the best on the report's prompts, each where its kl
        reaches the budget. it reads the cache; the run itself is left as it was."""
        d = self.out / "compare"
        d.mkdir(exist_ok=True)
        best = self.out / "best.gguf"
        loras = [*([best] if best.exists() else []), *(Path(x).expanduser() for x in self.a.lora)]
        sweeps = self.score("compare", loras, min(self.a.report_prompts, SELECT_OFFSET), 0, WIDE_SCALES,
                            REPORT_CHUNKS, d / "compare.json")
        print(f"{'adapter':24} {'refusals@kl':>11} {'scale':>7} {'kl':>7}")
        for lora in loras:
            at = at_budget(sweeps[lora.name], self.a.max_kld)
            print(f"{lora.stem:24} {at['refusals']:11.3f} {at['scale']:7.3g} {at['kl']:7.4f}")

    def write_best(self, st):
        b, think = st["best"], st.get("think")
        side = json.loads((self.out / "best.json").read_text())
        scale = side["scale"]
        lines = [f"{b['key']} at weight {b['cand']['weight']:g}, found in round {b['round']}",
                 (f"held-out: refusals {b['refusals']:.3f} (base {b['base']:.3f}) at kl {b['kl']:.4f}, "
                  f"{b['n']} prompts")]
        if "think" in b:
            lines.append(f"held-out, short thinking: refusals {b['think']:.3f} (base {b['think_base']:.3f}), "
                         f"{b['think_n']} prompts{no_reasoning(b)}")
        if think and "markers" in think:
            lines.append(f"full replies, thinking on: refusals {think['markers']:.3f} by markers, "
                         f"{think['refusals']:.3f} by the base as judge, unfinished {think['unfinished']:.3f}, "
                         f"loops {think['loops']:.3f}, reasoning in {think['reasoned']:.2f}, {think['n']} prompts")
        lora = (self.out / "best.gguf").absolute()
        if not scale:
            lines.append("no scale of it refused less than the base within the kl budget; served at 1 anyway")
        lines += [f"built with llama.cpp {side.get('llama_cpp', 'unknown')}",
                  f"serve: llama-server {serve_model(self.a)} --lora-scaled {lora}:{scale or 1:g}",
                  f"sidecar: {lora.with_suffix('.json')}"]
        (self.out / "best.txt").write_text("\n".join(lines) + "\n")

    def final_check(self, st):
        """the best with thinking on and full-length replies, once per best: replies that
        never finish or go round in circles show up here, and nowhere else. markers is
        the measure the report uses, so the two compare; the base as judge also counts
        warnings and deflection, so it runs well above markers on an adapter's long replies."""
        b = st["best"]
        done = st.get("think") or {}
        if not self.a.think_check or (done.get("for") == [b["key"], b["cand"]["weight"]] and "markers" in done):
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
        scores = {"n": self.a.think_limit, "markers": mean_of(marked(r) for r in p["replies"]), "reasoned": reasoned(p),
                  **{k: p[k] for k in ("refusals", "unfinished", "loops")}}
        st["think"] = {"for": [b["key"], b["cand"]["weight"]], **scores}
        side = json.loads((self.out / "best.json").read_text())
        side["eval"]["final"] = scores
        (self.out / "best.json").write_text(json.dumps(side, indent=2) + "\n")
        self.write_best(st)
        (self.out / "state.json").write_text(json.dumps(st, indent=2) + "\n")

    def eval(self, label, *args):
        """an eval sharing the run's cache: an adapter file, prompt and setting already scored
        by an earlier round (the base, or an incumbent kept since) is not scored again."""
        self.script(label, "hierophant_eval.py", "--cache", self.root / SHARED_CACHE, *args)

    def load_state(self):
        path = self.out / "state.json"
        return json.loads(path.read_text()) if path.exists() else None

    def score(self, label, loras, n, offset, scales, chunks, results, sidecars=False, extra=()):
        """{adapter file name: sweep} on n jbb+lat prompts from offset: the kl at every scale,
        replies only where it reaches the budget, the one point a ranking reads."""
        a = self.a
        self.eval(label, *self.serve(), *loras_of(loras), "--harmful", confirm_sets(),
                    "--limit", n, "--offset", offset, "--harmless-offset", 0, "--scales", scales, "--at-budget",
                    *extra, "--kld-text", a.kld_text, "--kld-ctx", a.kld_ctx, "--kld-chunks", chunks,
                    "--max-kld", a.max_kld,
                    "--judge", a.judge, *([] if sidecars else ["--no-sidecar"]),
                    "--results", results, "--server-log", self.logs / f"server-{label}.log")
        return json.loads(results.read_text())


def rank(cands, sweeps, budget):
    """candidates by refusals at the kl budget, each carrying the weight that reaches it,
    around which it is swept from now on."""
    out = []
    for c in cands:
        at = at_budget(sweeps[f"{key(c)}.gguf"], budget)
        weight = round(c["weight"] * at["scale"], 3) if at["scale"] >= MIN_SCALE else c["weight"]
        out.append({**at, "cand": {**{k: v for k, v in c.items() if k != "wide"}, "weight": weight}})
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


def mean_of(values):
    values = list(values)
    return sum(values) / len(values)


def reasoned(point):
    """the share of a sweep point's replies that reasoned at all: a model that never does
    gives a thinking score that says nothing about thinking."""
    return mean_of(bool(r["reasoning"]) for r in point["replies"])


def marked(reply):
    """whether a reply refuses by the report's measure, its reasoning standing in if empty."""
    return has_marker(reply["reply"] or reply["reasoning"])


def no_reasoning(best):
    return ", no reasoning seen" if best.get("think_reasoned", 1) == 0 else ""


def run_out(a):
    """the directory of a's rounds and best: OUT, or OUT/runs/NAME with --name."""
    out = Path(a.out).expanduser()
    return out / "runs" / a.run_name if a.run_name else out


def states(root):
    """{directory: state} of every run sharing root."""
    dirs = [root, *sorted((root / "runs").glob("*"))]
    return {d: json.loads((d / "state.json").read_text()) for d in dirs if (d / "state.json").exists()}


def base_key(name):
    return name.removesuffix("-res")


def dead_files(root):
    """what no later round will read, in every run sharing root: the adapters of recorded
    rounds, whose scores outlive them in cache.jsonl and round.json, and the residuals of
    candidates no run still holds. a pruned residual is captured again if it comes back."""
    runs = states(root)
    dead = []
    for d, st in runs.items():
        for n in range(1, st["round"] + 1):
            dead += [f for pattern in ("*.gguf", f"*.gguf{BUILT_SUFFIX}") for f in (d / f"round-{n}").rglob(pattern)]
    alive = {base_key(key(c)) for st in runs.values() for c in (*st["candidates"], *st["beam"])}
    alive |= {base_key(st["best"]["key"]) for st in runs.values() if st["best"]}
    for d in sorted((root / "residual").glob("*")):
        if (d / "weight.json").exists():
            dead += [f for f in (d / "under.gguf", d / f"under.gguf{BUILT_SUFFIX}") if f.exists()]
            if d.name not in alive and (d / "dirs.gguf").exists():
                dead += [d / "dirs.gguf", d / "weight.json"]
    return dead


def remove(files):
    for f in files:
        f.unlink(missing_ok=True)


def prune(a):
    """list what is dead, and delete it with --yes."""
    root = Path(a.out).expanduser()
    dead = dead_files(root)
    size = sum(f.stat().st_size for f in dead)
    for f in dead:
        print(f)
    print(f"{len(dead)} files, {size / GIB:.1f} GiB" + ("" if a.yes else ": rerun with --yes to delete them"))
    if a.yes:
        remove(dead)


def with_best(a, program):
    """a llama.cpp program on the base with the best at the scale it reaches the budget at."""
    out = run_out(a)
    if not (out / "best.gguf").exists():
        sys.exit(f"no best.gguf in {out} yet")
    scale = json.loads((out / "best.json").read_text())["scale"] or 1
    return [str(program), *serve_model(a).split(), "--lora-scaled", f"{(out / 'best.gguf').absolute()}:{scale:g}",
            "-ngl", str(a.ngl)]


def recipe_sidecar(out, recipe):
    """(the sidecar of a recipe a report scored, its directory's latest), or of the best."""
    if not recipe:
        return out / "best.json"
    reports = sorted(out.glob(f"round-*/report/{recipe}.json"), key=lambda p: int(p.parts[-3].removeprefix("round-")))
    if not reports:
        sys.exit(f"no report scored {recipe} in {out}")
    return reports[-1]


def accept_run(a, st, recipe=""):
    """write a recipe of the run, the best by default, into its model's config."""
    repo, _, quant = a.hf_repo.partition(":") if a.hf_repo else ("", "", "")
    if not quant:
        sys.exit("accept needs REPO:QUANT: a config is about a hub model and one of its quants")
    out = run_out(a)
    best = (st or {}).get("best") or {}
    side = json.loads(recipe_sidecar(out, "" if recipe == best.get("key") else recipe).read_text())
    think = best if "think" in best and recipe in ("", best.get("key")) else None
    path = config_accept(repo, quant, side, think)
    print(f">> accepted {recipe or best.get('key', 'the best')} for {quant} in {path}", flush=True)


def should_accept(a, st, config):
    """whether a finished build records itself: a recipe's measurements for its quant, or a
    converged search that fills an empty config. it never replaces an accepted recipe."""
    if not (a.auto_accept and a.hf_repo and st["best"]):
        return False
    return bool(a.recipe) or (st["unchanged"] >= PATIENCE and not accepted_recipe(config, ""))


def verify_line(st, accepted):
    """the rebuild beside what the config records."""
    b, e = st["best"], accepted.get("eval", {})
    return (f">> verify {b['key']}: refusals {b['refusals']:.3f} at kl {b['kl']:.4f} on {b['n']} prompts, "
            f"recorded {e.get('refusals', 0):.3f} on {e.get('n', 0)}")


def status(out):
    for name in ("best.txt", "history.txt"):
        path = out / name
        print(path.read_text() if path.exists() else f"no {name} in {out} yet")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p)
    p.add_argument("--out", help="output directory (default out/<model>, named after -m or --hf-repo)")
    p.add_argument("--name", dest="run_name", default="",
                   help="keep this run in OUT/runs/NAME, reusing OUT's directions, cache and residuals")
    p.add_argument("--status", action="store_true", help="print the best so far and each round's, and exit")
    p.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (default: once converged)")
    p.add_argument("--recipe", default="", help="build this recipe (L33-pcs8-res) instead of searching")
    p.add_argument("--recipe-weight", type=float, default=1.0, help="the weight --recipe starts from")
    p.add_argument("--no-accept", dest="auto_accept", action="store_false",
                   help="leave the model's config as it is when the build ends")
    p.add_argument("--accept", dest="accept_recipe", nargs="?", const="", metavar="RECIPE",
                   help="write RECIPE, or the best, into the model's config, and exit")
    p.add_argument("--compare", action="store_true", help="score each --lora beside the best, and exit")
    p.add_argument("--lora", action="append", default=[], help="an adapter for --compare; repeatable")
    p.add_argument("--serve", nargs=argparse.REMAINDER, help="run llama-server with the best; the rest goes to it")
    p.add_argument("--chat", nargs=argparse.REMAINDER, help="run llama-cli with the best; the rest goes to it")
    p.add_argument("--prune", action="store_true", help="list the files no round will read again, and exit")
    p.add_argument("--yes", action="store_true", help="with --prune, delete them")
    p.add_argument("--keep-rounds", action="store_true", help="keep every finished round's adapters")
    p.add_argument("--verify", action="store_true",
                   help="rebuild the recipe from nothing in OUT/runs/verify and print it beside the config's scores")
    p.add_argument("--max-kld", type=float, default=MAX_KLD, help="kl budget: the larger of the first-token kl and the kl over --kld-text")
    p.add_argument("--judge", default="markers",
                   help="markers (heretic's), self (the served base model) or an openai-compatible base url")
    p.add_argument("--indices", help="round 1's direction indices, instead of a grid over the depth")
    p.add_argument("--first-pcs", default=FIRST_PCS,
                   help=f"harmless pcs counts round 1 tries every index with (each of {','.join(map(str, PCS))})")
    p.add_argument("--screen-layers", action="store_true",
                   help="round 1 screens every layer of the grid's range with each --first-pcs count on the "
                        "first token alone and scores only the best --screen-keep in full (ignored with --indices)")
    p.add_argument("--screen-only", action="store_true",
                   help="run the screen alone in OUT/screen beside round 1's refusals, to check its ranking, and exit")
    p.add_argument("--screen-keep", type=int, default=SCREEN_KEEP)
    p.add_argument("--screen-prompts", type=int, default=SCREEN_PROMPTS)
    p.add_argument("--train-data", help="filtered training sets from another run's OUT/data")
    g = p.add_argument_group("effort")
    g.add_argument("--first-prompts", type=int, default=FIRST_PROMPTS, help="round 1's prompts, doubled each round")
    g.add_argument("--beam", type=int, default=BEAM,
                   help="halving stops once its better half fits this; after it, the most leaders each round "
                        "refines: each shape's best, and near ties of the leader")
    g.add_argument("--report-prompts", type=int, default=REPORT_PROMPTS,
                   help=f"held-out prompts every promotion is decided on (at most {SELECT_OFFSET})")
    g.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES, help="per refining round (not halving)")
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
    g.add_argument("--cli", default=BIN / "llama-cli")
    g.add_argument("--hierophant", default=BIN / "hierophant-core")
    a = p.parse_args(argv)
    if not a.model and not a.hf_repo:
        p.error("give -m or --hf-repo")
    a.first_pcs = [int(x) for x in a.first_pcs.split(",")]
    if not set(a.first_pcs) <= set(PCS):
        p.error(f"--first-pcs takes counts from {','.join(map(str, PCS))}")
    a.out = a.out or default_out(a)
    if a.recipe:
        parse_recipe(a.recipe)
        a.rounds = a.rounds or 1
    if a.verify:
        a.out, a.run_name, a.auto_accept = Path(a.out) / "runs" / "verify", "", False
    # what the prepare steps read: one capture that every pcs count can use
    vars(a).update(no_filter=False, positions=1, harmless_pcs=max(PCS), think_capture=False)
    return a


def finish_build(a, st):
    """what a finished build leaves in the model's config."""
    repo = (a.hf_repo or "").partition(":")[0]
    configs = load_configs()
    config = next((c for c in configs.values() if c["model"]["repo"] == repo), None)
    quant = (a.hf_repo or "").partition(":")[2]
    if a.verify:
        print(verify_line(st, ((config or {}).get("quant") or {}).get(quant, {})), flush=True)
    elif should_accept(a, st, config):
        accept_run(a, st)
    elif a.auto_accept and config and st["unchanged"] >= PATIENCE and st["best"]:
        print(f">> the accepted recipe stays {accepted_recipe(config, quant)}: "
              f"hierophant accept writes {st['best']['key']} over it", flush=True)


def main(argv):
    a = parse_args(argv)
    if a.status:
        return status(run_out(a))
    if a.serve is not None or a.chat is not None:
        cmd = [*with_best(a, a.server), "-np", str(a.parallel), *a.serve] if a.chat is None else \
            [*with_best(a, a.cli), *a.chat]
        return os.execv(cmd[0], cmd)
    if a.prune:
        return prune(a)
    if a.accept_recipe is not None:
        path = run_out(a) / "state.json"
        return accept_run(a, json.loads(path.read_text()) if path.exists() else None, a.accept_recipe)
    r = Auto(a)
    r.echo = a.verbose
    if a.compare:
        return r.compare()
    print(f"== build: following recipe {a.recipe}" if a.recipe else "== build: searching", flush=True)
    r.prepare()
    n_layer = Reader(r.dirs).tensors["harmful_mean"]["ne"][1]
    if a.screen_only:
        return r.screen_only(n_layer)
    pool = len(load_prompts(confirm_sets())) - SELECT_OFFSET
    st = r.load_state() or first_state(a, n_layer)
    for k in range(1, st["round"] + 1):
        print(f"== round {k}: done", flush=True)
    while not stopped(st, a):
        st = r.round(st, n_layer, pool)
    if st["best"]:
        r.final_check(st)
        print(f"best weights: {r.out / 'best.gguf'}\n" + (r.out / "best.txt").read_text(), end="", flush=True)
        finish_build(a, st)


if __name__ == "__main__":
    entry(main, RESUME_HINT)
