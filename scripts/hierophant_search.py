#!/usr/bin/env python3
"""search heretic's weight curves, one per site group, without writing a file per trial.

`hierophant adapter --split` writes one weight-1 adapter per group (or tensor
kind) and layer. llama-server loads them all unapplied, and a trial is one set
of per-request lora scales: each curve's weight at each layer. several split
directories, one per direction index, make the index a searched choice too.
the leaders can be re-scored on other prompts, and the best trial's curves
build the merged adapter with `hierophant adapter --curve NAME=...`.
"""

import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_lite import Reader
from hierophant_eval import OFF, Judge, Server, add_server_args, load_prompts, resolve_model, saved, score

# groups weighted flat, as one layer-free weight: the embedding.
FLAT = ("embd", "token_embd")
UNIFORM = "uniform"  # the one parameter of --uniform
# heretic's ranges: max weight, its position as a fraction of the last layer,
# min weight as a fraction of max weight, and min distance as a fraction of
# the last layer (at least one layer).
MAX_WEIGHT = (0.8, 1.5)
POSITION = (0.6, 1.0)
MIN_FRACTION = (0.0, 1.0)
DISTANCE = (0.0, 0.6)
FLAT_WEIGHT = (0.0, 1.5)
STEP = 0.1  # refinement noise, as a fraction of each range
KEEP_INDEX = 0.8  # chance a refinement keeps the best trial's direction index
SCORES = ("refusals", "kld", "top1_same", "reply_logp", "unfinished", "loops")


def number(s):
    """a direction index as recorded ("33", "33.5", "33+20") in its natural json form."""
    try:
        f = float(s)
    except ValueError:
        return s
    return int(f) if f.is_integer() else f


def parts_of(dirs):
    """[{index, group, layer, path}] for the split adapters in each directory."""
    out = []
    for d in dirs:
        paths = sorted(Path(d).glob("*.gguf"))
        if not paths:
            sys.exit(f"error: no adapters in {d}")
        for p in paths:
            kv = Reader(p).kv
            if "hierophant.split_group" not in kv:
                sys.exit(f"error: {p} is not a split adapter (hierophant adapter --split)")
            out.append({"index": number(kv["hierophant.direction_indices"]), "group": kv["hierophant.split_group"],
                        "layer": kv["hierophant.split_layer"], "path": p})
    return out


def groups_of(parts, index):
    return sorted({p["group"] for p in parts if p["index"] == index}, key=lambda g: (g in FLAT, g))


def weight(curve, layer):
    """the same weight hierophant's curve gives a layer, 0 where it does not reach."""
    max_weight, max_pos, min_weight, min_dist = curve
    d = abs(max(layer, 0) - max_pos)
    if d > min_dist:
        return 0.0
    return max_weight if math.isinf(min_dist) else max_weight + (d / min_dist) * (min_weight - max_weight)


def pair(s):
    lo, hi = map(float, s.split(","))
    return lo, hi


def ranges(groups, last, args):
    """{(group, i): (lo, hi)} for every searched parameter of one index; with
    --uniform, one weight shared by every group."""
    if args.uniform:
        return {(UNIFORM, 0): pair(args.max_weight)}
    position, distance = pair(args.position), pair(args.distance)
    out = {}
    for g in groups:
        if g in FLAT:
            out[(g, 0)] = FLAT_WEIGHT
            continue
        out[(g, 0)] = pair(args.max_weight)
        out[(g, 1)] = (position[0] * last, position[1] * last)
        out[(g, 2)] = MIN_FRACTION
        out[(g, 3)] = (max(1.0, distance[0] * last), max(1.0, distance[1] * last))
    return out


def curves_of(params, groups):
    """{group: (max_weight, max_pos, min_weight, min_dist)} from searched parameters."""
    out = {}
    for g in groups:
        if (UNIFORM, 0) in params:
            w = params[(UNIFORM, 0)]
            out[g] = (w, 0.0, w, math.inf)
        elif g in FLAT:
            w = params[(g, 0)]
            out[g] = (w, 0.0, w, math.inf)
        else:
            mw = params[(g, 0)]
            out[g] = (mw, params[(g, 1)], params[(g, 2)] * mw, params[(g, 3)])
    return out


def sample(rng, spaces, around=None):
    """(index, params): uniform over an index's space, or usually gaussian steps
    around a point at its own index, clipped to the space."""
    if around is not None and rng.random() < KEEP_INDEX:
        index, centre = around
    else:
        index, centre = rng.choice(sorted(spaces, key=str)), None
    params = {}
    for k, (lo, hi) in spaces[index].items():
        v = rng.uniform(lo, hi) if centre is None else rng.gauss(centre[k], STEP * (hi - lo))
        params[k] = min(hi, max(lo, v))
    return index, params


def rank(trial, args):
    """the best objective within the kl budget; over budget, least kl. a re-scored
    trial is held to the budget by budget_kld, its kl on the search prompts the
    budget was set on: kl on another prompt slice can sit several times higher."""
    kld = trial.get("budget_kld", trial["kld"])
    if kld <= args.max_kld:
        return (0, trial[args.objective_key], kld)
    return (1, kld, trial[args.objective_key])


def adapter_args(index, curves, targets):
    """the hierophant adapter flags for a trial; one flat weight for all groups is --weight."""
    flat = {(c[0], c[2], c[3]) for c in map(tuple, curves.values())}
    w, w_min, dist = next(iter(flat))
    if len(flat) == 1 and w == w_min and math.isinf(dist):
        specs = [f"--weight {float(w)!r}"]
    else:
        specs = [f"--curve {g}={','.join(map(repr, map(float, c)))}" for g, c in curves.items()]
    return " ".join([f"--direction-index {index}", *specs, f"--targets {','.join(targets)}"])


def run_trial(server, pool, judge, parts, index, curves, prompts, base):
    loras = [(i, s) for i, p in enumerate(parts)
             if p["index"] == index and (s := weight(curves[p["group"]], p["layer"])) != 0]
    point = saved(score(server, pool, judge, loras or OFF, *prompts, base))
    return {"direction_index": index, "curves": {g: list(c) for g, c in curves.items()},
            **{k: point[k] for k in SCORES}}


def describe(t):
    return (f"refusals {t['refusals']:.3f} kld {t['kld']:.4f} reply_logp {t['reply_logp']:.3f} "
            f"index {t['direction_index']} "
            + " ".join(f"{g}={','.join(f'{v:.3g}' for v in c)}" for g, c in t["curves"].items()))


class Search:
    """the trials so far, the parameters behind each, and the rescored leaders."""

    def __init__(self, args, server, parts):
        self.args, self.server, self.parts = args, server, parts
        self.spaces = {i: ranges(groups_of(parts, i), max(p["layer"] for p in parts if p["index"] == i), args)
                       for i in {p["index"] for p in parts}}
        self.trials, self.params, self.rescored = [], [], []

    def best(self):
        return min(range(len(self.trials)), key=lambda i: rank(self.trials[i], self.args))

    def run(self, pool, judge, prompts):
        rng = random.Random(self.args.seed)
        base = score(self.server, pool, judge, OFF, *prompts)
        print(f"base: refusals {base['refusals']:.3f} reply_logp {base['reply_logp']:.3f}")
        for n in range(self.args.trials):
            around = None
            if n >= self.args.startup and self.trials:
                b = self.best()
                around = (self.trials[b]["direction_index"], self.params[b])
            index, p = sample(rng, self.spaces, around)
            curves = curves_of(p, groups_of(self.parts, index))
            self.trials.append(run_trial(self.server, pool, judge, self.parts, index, curves, prompts, base))
            self.params.append(p)
            print(f"trial {n}: {describe(self.trials[-1])}")
            self.write()
        return base

    def rescore(self, pool, judge):
        """score the --rescore best trials again on --rescore-offset/--rescore-limit."""
        a = self.args
        prompts = tuple(load_prompts(f, a.rescore_limit, a.rescore_offset) for f in (a.harmful, a.harmless))
        base = score(self.server, pool, judge, OFF, *prompts)
        leaders = sorted(range(len(self.trials)), key=lambda i: rank(self.trials[i], a))[:a.rescore]
        for i in leaders:
            t = self.trials[i]
            curves = {g: tuple(c) for g, c in t["curves"].items()}
            point = run_trial(self.server, pool, judge, self.parts, t["direction_index"], curves, prompts, base)
            self.rescored.append({"trial": i, "offset": a.rescore_offset, "limit": a.rescore_limit,
                                  **{k: point[k] for k in SCORES}})
            print(f"rescored trial {i}: {describe({**t, **point})}")
            self.write()

    def summary(self):
        """every trial, the rescored leaders and the best, with the flags that build it."""
        scored = [{**r, "budget_kld": self.trials[r["trial"]]["kld"]} for r in self.rescored] or \
                 [{"trial": i, **{k: t[k] for k in SCORES}} for i, t in enumerate(self.trials)]
        best = dict(min(scored, key=lambda r: rank(r, self.args)))
        t = self.trials[best["trial"]]
        mine = [p for p in self.parts if p["index"] == t["direction_index"]]
        targets = sorted({k for p in mine for k in Reader(p["path"]).kv["hierophant.targets"].split(",")})
        best.update(direction_index=t["direction_index"], curves=t["curves"],
                    scales={f"{p['group']}-{p['layer']}": weight(t["curves"][p["group"]], p["layer"]) for p in mine},
                    adapter_args=adapter_args(t["direction_index"], t["curves"], targets))
        return {"trials": self.trials, "rescored": self.rescored, "best": best}

    def write(self):
        """rewritten after each trial, so a stopped search keeps its results."""
        if self.args.results:
            Path(self.args.results).write_text(json.dumps(self.summary(), indent=2) + "\n")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_server_args(p)
    p.add_argument("--split-dir", required=True, action="append",
                   help="directory of adapters from `hierophant adapter --split`; repeat for more direction indices")
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--startup", type=int, default=20, help="random trials before refining around the best")
    p.add_argument("--objective", choices=("refusals", "logp"), default="refusals",
                   help="rank by the refusal rate, or by reply_logp, which moves smoothly and so is less noisy")
    p.add_argument("--max-kld", type=float, default=0.05, help="kl budget; the best trial stays within it")
    p.add_argument("--position", default=",".join(map(str, POSITION)),
                   help="range of the peak position, as fractions of the last layer")
    p.add_argument("--max-weight", default=",".join(map(str, MAX_WEIGHT)), help="range of the peak weight")
    p.add_argument("--uniform", action="store_true",
                   help="search one flat weight for every site (from --max-weight) instead of curves")
    p.add_argument("--distance", default=",".join(map(str, DISTANCE)),
                   help="range of the curve's reach either side of the peak, as fractions of the last layer; "
                        "1 reaches every layer from any peak")
    p.add_argument("--rescore", type=int, default=0, help="score this many leaders again on other prompts")
    p.add_argument("--rescore-offset", type=int, default=0)
    p.add_argument("--rescore-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", help="write every trial and the best one here")
    p.add_argument("--summarize-only", action="store_true",
                   help="pick the best again from an existing --results file, without running anything")
    args = resolve_model(p, p.parse_args(argv))
    args.objective_key = "reply_logp" if args.objective == "logp" else "refusals"
    return args


def resummarize(args, parts):
    """the best trial of an existing --results file under the current ranking."""
    if not args.results or not Path(args.results).exists():
        sys.exit("error: --summarize-only needs an existing --results file")
    old = json.loads(Path(args.results).read_text())
    s = Search(args, None, parts)
    s.trials, s.rescored = old["trials"], old["rescored"]
    s.write()
    best = s.summary()["best"]
    print(f"best: trial {best['trial']}, {best['adapter_args']}")


def main(argv):
    args = parse_args(argv)
    parts = parts_of(args.split_dir)
    args.lora = [p["path"] for p in parts]
    if args.summarize_only:
        resummarize(args, parts)
        return
    prompts = tuple(load_prompts(f, args.limit, args.offset) for f in (args.harmful, args.harmless))
    server = Server(args)
    s = Search(args, server, parts)
    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            judge = Judge(args, server)
            base = s.run(pool, judge, prompts)
            if args.rescore:
                s.rescore(pool, judge)
    finally:
        server.close()
    best = s.summary()["best"]
    print(f"best: trial {best['trial']}, refusals {best['refusals']:.3f} kld {best['kld']:.4f} "
          f"reply_logp {best['reply_logp']:.3f} (base refusals {base['refusals']:.3f})")
    print(f"build it with: hierophant adapter MODEL -d DIRECTIONS {best['adapter_args']} -o ADAPTER")


if __name__ == "__main__":
    main(sys.argv[1:])
