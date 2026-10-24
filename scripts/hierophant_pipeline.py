#!/usr/bin/env python3
"""run the whole adapter recipe for one model, resumably, into one output directory.

filter -> capture -> split -> search -> build -> residual -> confirm -> think -> summary:
keep the training prompts the base model treats as labelled, capture
directions from them, split adapters at each direction index, search the
weight curves (re-scoring the leaders on the other half of the test split),
build the winner, optionally add the direction of what it still refuses as
a rank-2 variant, score them and any --ref adapters on harmful sets the
search never saw (jailbreakbench, llm-lat) with full-sequence kl, which picks
each one's scale and writes the sidecars, and check them with thinking on at
that scale.

--screen compares recipes quickly instead: no split or search, every
--indices entry built at weight 1 (a uniform weight is the runtime scale),
all scored on a small held-out slice over a sweep of scales.

each finished step leaves a marker in OUT/.done, so a rerun picks up where the
last one stopped and --from STEP reruns a step and everything after it. the
configuration is kept in OUT/config.json; a rerun with a different one starts
over at the first step the change affects. every command's output also goes
to OUT/logs/<step>.log.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
BIN = ROOT / "build" / "bin"
DATA = ROOT / "data"
sys.path.insert(0, str(SCRIPTS))
from hierophant_eval import add_model_args, model_flags, pick
from progress import CLEAR, COUNT

STEPS = ["filter", "capture", "split", "search", "build", "residual", "confirm", "think", "summary"]
# options that say how to run, not what to make: changing them needs no --from.
RUNTIME = {"from_step", "ngl", "parallel", "server", "perplexity", "hierophant", "offline", "progress"}
# the first step that reads each option; a change to any other reruns everything.
# split and search do nothing with --screen, so its builds start over at split too.
AFFECTS = {
    "capture": ("positions", "harmless_pcs", "think_capture"),
    "split": ("screen", "indices", "split_by", "targets", "type"),
    "search": ("trials", "startup", "objective", "search_limit", "rescore", "max_kld", "distance", "position",
               "max_weight", "uniform", "seed"),
    "build": ("name",),
    "residual": ("residual",),
    "confirm": ("ref", "confirm_limit", "confirm_offset", "confirm_scales", "confirm_max_kld", "kld_text", "kld_ctx",
                "kld_chunks"),
    "think": ("think_check", "think_limit", "think_max_tokens", "think_scales"),
}
FIRST_STEP = {option: step for step, options in AFFECTS.items() for option in options}
CONFIRM_SETS = ("harmful-jbb.txt", "harmful-lat.txt")
SUMMARY_KEYS = ("scale", "refusals", "kld", "kld_seq", "reply_logp", "unfinished", "loops")
OUT = Path("lora")  # under the current directory, one subdirectory per model
MIN_RESIDUAL = 4  # still-refused prompts below which a residual direction is noise
READ_CHUNK = 1 << 16
PROGRESS_HELP = "draw each command's progress bars in place: a request, a kl chunk, a captured prompt, a site"
LINE_END = re.compile(rb"(?<=[\r\n])")
# --screen: a slice of jbb + lat past the 300 a full confirmation uses, a short kl text, a wide sweep.
SCREEN = {"confirm_limit": 60, "confirm_offset": 300, "confirm_scales": "0.8,1,1.2,1.4,1.6", "kld_chunks": 4,
          "think_limit": 16, "think_max_tokens": 2048}


def segments(stream):
    """a child's output as it arrives, cut after each newline and each carriage return
    (a progress bar's redraw), so a redraw shows before the line it is on ends."""
    buf = b""
    while data := stream.read1(READ_CHUNK):
        *done, buf = LINE_END.split(buf + data)
        yield from (d.decode(errors="replace") for d in done)
    if buf:
        yield buf.decode(errors="replace")


class Run:
    """paths, the shared command-line pieces, and logged command execution."""

    echo = True  # stream each command's output to the terminal, not only to its log
    bar = None  # a bar over several commands, drawn instead of a line per command
    rollup = ()  # labels of the children's bars that advance it

    def __init__(self, args):
        self.a = args
        self.out = Path(args.out).expanduser()
        self.logs, self.done = self.out / "logs", self.out / ".done"
        self.data = Path(args.train_data).expanduser() if args.train_data else self.out / "data"
        for d in (self.out, self.logs, self.done):
            d.mkdir(parents=True, exist_ok=True)
        self.dirs, self.resid_dirs = self.out / "dirs.gguf", self.out / "resid.gguf"
        self.adapter = self.out / f"{args.name}.gguf"
        self.resid_adapter = self.out / f"{args.name}-res.gguf"
        self.model = model_flags(args)

    def serve(self, max_tokens=None):
        a = self.a
        return ["-ngl", a.ngl, "-np", a.parallel, "--server", a.server, "--hierophant", a.hierophant,
                "--max-tokens", max_tokens or a.max_tokens]

    def sh(self, step, *cmd):
        """run a command, streaming its output to the step's log and, with echo, the terminal;
        without echo the terminal gets one line naming the command and its log."""
        cmd = list(map(str, cmd))
        log = self.logs / f"{step}.log"
        if self.echo:
            print("$ " + shlex.join(cmd), flush=True)
        elif not self.bar:
            what = Path(cmd[3]).stem if cmd[1:3] == ["-I", "-u"] else f"{Path(cmd[0]).name} {cmd[1]}"
            print(f"   {what} (log: {log})", flush=True)
        with open(log, "a") as f:
            f.write("$ " + shlex.join(cmd) + "\n")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            drawn = False
            for seg in segments(p.stdout):
                # a progress bar's redraw is drawn over the last; only finished lines are kept
                if seg.endswith("\r"):
                    drawn = self.redraw(seg) or drawn
                    continue
                line = seg.replace(CLEAR, "")
                if self.echo:
                    print((CLEAR if drawn else "") + line, end="", flush=True)
                    drawn = False
                f.write(line)
                f.flush()
            if self.bar:
                self.bar.draw()
            elif drawn:
                print(CLEAR, end="", flush=True)
            if p.wait() != 0:
                sys.exit(f"error: step {step} failed with {p.returncode}; see {log}")

    def redraw(self, seg):
        """a child's bar: folded into the enclosing bar when it is one of the units that bar
        rolls up, else drawn as it is. returns whether anything was drawn."""
        if self.bar and (m := COUNT.match(seg)) and m["label"] in self.rollup:
            self.bar.part(seg)
        elif self.progress:
            print(seg, end="", flush=True)
        return self.progress

    @property
    def progress(self):
        return getattr(self.a, "progress", False)

    def hierophant(self, step, command, *args):
        self.sh(step, self.a.hierophant, command, *self.model, *args, *(["--progress"] if self.progress else []))

    def script(self, step, name, *args):
        # -u: a child writing to our pipe would otherwise hold its output in a
        # block buffer until it exits (-I ignores PYTHONUNBUFFERED)
        self.sh(step, sys.executable, "-I", "-u", SCRIPTS / name, *self.model, *args,
                *(["--progress"] if self.progress else []))

    def train_sets(self):
        if self.a.no_filter:
            return DATA / "harmful-train.txt", DATA / "harmless-train.txt"
        return self.data / "harmful-train.filtered.txt", self.data / "harmless-train.filtered.txt"

    def pcs(self):
        """adapter options every build shares."""
        a = self.a
        return [*(["--harmless-pcs", a.harmless_pcs] if a.harmless_pcs else []),
                *(["--targets", a.targets] if a.targets else []), "--type", a.type]

    def capture_flags(self):
        a = self.a
        return ["-ngl", a.ngl, "--positions", a.positions, "--limit", a.train_limit,
                *(["--keep-rows"] if a.harmless_pcs else []), *(["--think"] if a.think_capture else [])]

    def built(self):
        """the adapters this run builds: the winner, or one per index with --screen."""
        if not self.a.screen:
            return [self.adapter]
        return [self.out / f"{self.a.name}-{i}.gguf" for i in self.a.indices.split(",")]

    def base_args(self):
        """how the first built adapter was made, which a residual variant repeats."""
        if self.a.screen:
            return ["--direction-index", self.a.indices.split(",")[0]]
        return shlex.split(json.loads((self.out / "search.json").read_text())["best"]["adapter_args"])

    def candidates(self):
        """the built adapters, the residual variant and the --ref adapters, the
        refs linked under OUT/refs so the eval writes their sidecars there
        instead of beside the originals."""
        out = [*self.built(), *([self.resid_adapter] if self.resid_adapter.exists() else [])]
        for ref in self.a.ref:
            link = self.out / "refs" / Path(ref).name
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                link.symlink_to(Path(ref).expanduser().resolve())
            out.append(link)
        return out


def seed_manifest(d):
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "manifest.json").exists():
        shutil.copy(DATA / "manifest.json", d / "manifest.json")


def step_filter(r):
    if r.a.no_filter:
        print("filtering off (--no-filter)")
        return
    if r.a.train_data and all(p.exists() for p in r.train_sets()):
        print(f"using the filtered sets in {r.data}")
        return
    seed_manifest(r.data)
    r.script("filter", "filter_prompts.py", *r.serve(), "--harmful", DATA / "harmful-train.txt",
             "--harmless", DATA / "harmless-train.txt", "--limit", r.a.train_limit, "--out-dir", r.data,
             "--server-log", r.logs / "server-filter.log")


def step_capture(r):
    harmful, harmless = r.train_sets()
    r.hierophant("capture", "capture", "--harmful", harmful, "--harmless", harmless, *r.capture_flags(), "-o", r.dirs)


def step_split(r):
    if r.a.screen:
        print("no split with --screen")
        return
    for index in r.a.indices.split(","):
        parts = r.out / f"parts-{index}"
        shutil.rmtree(parts, ignore_errors=True)
        parts.mkdir()
        r.hierophant("split", "adapter", "-d", r.dirs, "--direction-index", index, "--split",
                     "--split-by", r.a.split_by, *r.pcs(), "-o", parts / "s-{}.gguf")


def step_search(r):
    a = r.a
    if a.screen:
        print("no search with --screen: every index is built at weight 1 and swept by scale")
        return
    dirs = [arg for i in a.indices.split(",") for arg in ("--split-dir", r.out / f"parts-{i}")]
    r.script("search", "hierophant_search.py", *r.serve(), *dirs, "--limit", a.search_limit,
             "--rescore", a.rescore, "--rescore-offset", a.search_limit, "--rescore-limit", a.search_limit,
             "--trials", a.trials, "--startup", a.startup, "--objective", a.objective, "--max-kld", a.max_kld,
             "--distance", a.distance, "--position", a.position, "--max-weight", a.max_weight, "--seed", a.seed,
             *(["--uniform"] if a.uniform else []),
             "--results", r.out / "search.json", "--server-log", r.logs / "server-search.log")


def step_build(r):
    if r.a.screen:
        r.hierophant("build", "adapter", "-d", r.dirs, "--direction-index", r.a.indices, *r.pcs(),
                     "-o", r.out / f"{r.a.name}-{{}}.gguf")
        return
    r.hierophant("build", "adapter", "-d", r.dirs, *r.base_args(), *r.pcs(), "-o", r.adapter)


def step_residual(r):
    """the direction of what the first built adapter still refuses, captured with it
    applied at --residual, removed beside its own as OUT/NAME-res.gguf."""
    r.resid_adapter.unlink(missing_ok=True)
    if not r.a.residual:
        print("no residual variant (--residual SCALE adds one)")
        return
    first, (harmful, harmless) = r.built()[0], r.train_sets()
    under = ["--lora", first, "--lora-scale", r.a.residual]
    d = r.out / "residual"
    seed_manifest(d)
    r.script("residual", "filter_prompts.py", *r.serve(), "--harmful", harmful, *under, "--out-dir", d,
             "--server-log", r.logs / "server-residual.log")
    still = d / f"{harmful.stem}.filtered.txt"
    n = len(still.read_text().splitlines())
    if n < MIN_RESIDUAL:
        print(f"only {n} prompts still refused under {first.name}; no residual variant")
        return
    r.hierophant("residual", "capture", "--harmful", still, "--harmless", harmless, *r.capture_flags(), *under,
                 "-o", r.resid_dirs)
    r.hierophant("residual", "adapter", "-d", r.dirs, "--residual", r.resid_dirs, *r.base_args(), *r.pcs(),
                 "-o", r.resid_adapter)


def confirm_sets():
    return ",".join(str(DATA / name) for name in CONFIRM_SETS)


def loras_of(paths):
    return [arg for lora in paths for arg in ("--lora", lora)]


def picked_scales(r):
    """{adapter name: the scale the confirmation picked under its kl budget}."""
    results = json.loads((r.out / "confirm.json").read_text())
    return {name: pick(points, r.a.confirm_max_kld)["scale"] for name, points in results.items()}


def step_confirm(r):
    a = r.a
    r.script("confirm", "hierophant_eval.py", *r.serve(), *loras_of(r.candidates()), "--harmful", confirm_sets(),
             "--limit", a.confirm_limit, "--offset", a.confirm_offset, "--harmless-offset", 0,
             "--scales", a.confirm_scales,
             "--kld-text", a.kld_text, "--kld-ctx", a.kld_ctx, "--kld-chunks", a.kld_chunks,
             "--max-kld", a.confirm_max_kld, "--perplexity", a.perplexity,
             "--results", r.out / "confirm.json", "--server-log", r.logs / "server-confirm.log")


def think_groups(loras, picked, override=""):
    """{scales: adapters checked at them}: each at its picked scale unless overridden,
    leaving out those picked at 0 (the base is already the best within the budget)."""
    groups = {}
    for lora in loras:
        scales = override or f"{picked[Path(lora).name]:g}"
        if scales != "0":
            groups.setdefault(scales, []).append(lora)
    return groups


def step_think(r):
    """thinking on, each adapter at the scale the confirmation picked (or --think-scales),
    one eval per distinct scale; replies that loop or never finish show up here."""
    a = r.a
    if not a.think_check:
        print("thinking check off (--no-think-check)")
        return
    groups = think_groups(r.candidates(), picked_scales(r), a.think_scales)
    merged = {}
    for n, (scales, loras) in enumerate(groups.items()):
        out = r.out / f"think-{n}.json"
        r.script("think", "hierophant_eval.py", *r.serve(a.think_max_tokens), *loras_of(loras),
                 "--harmful", confirm_sets(), "--limit", a.think_limit, "--offset", a.confirm_offset,
                 "--harmless-offset", 0, "--scales", scales, "--think", "--judge", "self", "--no-sidecar", "--results", out,
                 "--server-log", r.logs / f"server-think-{n}.log")
        merged.update(json.loads(out.read_text()))
        out.unlink()
    (r.out / "think.json").write_text(json.dumps(merged, indent=2) + "\n")


def summary_rows(r):
    """{adapter: {"confirm": rows, "think": rows}} without the scale-0 rows."""
    adapters = {}
    for kind in ("confirm", "think"):
        path = r.out / f"{kind}.json"
        for name, points in (json.loads(path.read_text()) if path.exists() else {}).items():
            rows = [{k: p[k] for k in SUMMARY_KEYS if k in p} for p in points if p["scale"] != 0]
            adapters.setdefault(name, {})[kind] = rows
    return adapters


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def step_summary(r):
    """each adapter's confirmation and thinking scores side by side, printed and in
    summary.json; * marks the scale picked under --confirm-max-kld."""
    adapters, picked = summary_rows(r), picked_scales(r)
    search = r.out / "search.json"
    best = json.loads(search.read_text())["best"] if search.exists() and not r.a.screen else None
    (r.out / "summary.json").write_text(json.dumps({"best": best, "picked": picked, "adapters": adapters},
                                                   indent=2) + "\n")
    print(f"{'adapter':28} {'scale':>6} {'refusals':>8} {'kld':>7} {'kld_seq':>7} "
          f"{'think':>6} {'unfin':>6} {'loops':>6}")
    for name, kinds in adapters.items():
        think = {row["scale"]: row for row in kinds.get("think", [])}
        for row in kinds.get("confirm", []):
            th = think.get(row["scale"], {})
            mark = "*" if picked.get(name) == row["scale"] else " "
            print(f"{name:28} {row['scale']:5g}{mark} {row['refusals']:8.3f} {row['kld']:7.4f} "
                  f"{fmt(row.get('kld_seq'), '.4f'):>7} {fmt(th.get('refusals'), '.3f'):>6} "
                  f"{fmt(th.get('unfinished'), '.3f'):>6} {fmt(th.get('loops'), '.3f'):>6}")
    for lora in r.built():
        print(f"adapter: {lora}\nsidecar: {lora.with_suffix('.json')}")


FUNCS = {name: globals()[f"step_{name}"] for name in STEPS}


def check_config(r, args):
    """never mix outputs of two configurations: a changed one reruns from the first
    step it affects, unless --from says where to start."""
    path = r.out / "config.json"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in RUNTIME}
    start = args.from_step
    if path.exists() and not start:
        old = json.loads(path.read_text())
        # an option only one side knows is one added or removed since; it changed no output
        changed = sorted(k for k in set(old) & set(config) if old[k] != config[k])
        if changed:
            start = min((FIRST_STEP.get(k, STEPS[0]) for k in changed), key=STEPS.index)
            print(f"== config: {', '.join(changed)} changed; rerunning from {start}", flush=True)
    if start:
        for step in STEPS[STEPS.index(start):]:
            (r.done / step).unlink(missing_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def default_out(args):
    """lora/<org>--<repo>[--<quant or file stem>], or lora/<file stem> for -m."""
    if not args.hf_repo:
        return OUT / Path(args.model).name.removesuffix(".gguf")
    repo, _, tag = args.hf_repo.partition(":")
    parts = [repo.replace("/", "--"), Path(args.hf_file).name.removesuffix(".gguf") if args.hf_file else tag]
    return OUT / "--".join(p for p in parts if p)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(p)
    p.add_argument("--out", help=f"output directory (default {OUT}/<model>, named after -m or --hf-repo)")
    p.add_argument("--name", default="hierophant", help="the final adapter is OUT/NAME.gguf")
    p.add_argument("--indices", required=True, help="direction indices to search, e.g. 32,33,34 or 32,32+20")
    p.add_argument("--ref", action="append", default=[], help="an adapter to compare against; repeatable")
    p.add_argument("--from", dest="from_step", choices=STEPS, help="rerun this step and the ones after it")
    p.add_argument("--screen", action="store_true",
                   help="compare recipes fast: no search, each index at weight 1, a small held-out slice "
                        f"({', '.join(f'{k}={v}' for k, v in SCREEN.items())})")
    g = p.add_argument_group("directions")
    g.add_argument("--train-data", help="filtered training sets from another run's OUT/data, to skip filtering")
    g.add_argument("--residual", type=float, default=0, metavar="SCALE",
                   help="also build NAME-res.gguf, adding the direction of what the first adapter still "
                        "refuses with it applied at SCALE")
    g.add_argument("--targets", default="", help="site kinds to edit, e.g. to leave out token_embd")
    g.add_argument("--no-filter", action="store_true", help="capture from every training prompt")
    g.add_argument("--train-limit", type=int, default=0)
    g.add_argument("--positions", type=int, default=1, help="final prompt tokens averaged per prompt")
    g.add_argument("--harmless-pcs", type=int, default=0, help="harmless principal components to avoid too")
    g.add_argument("--think-capture", action="store_true", help="capture with the template's thinking on")
    g.add_argument("--split-by", choices=("group", "kind"), default="group")
    g = p.add_argument_group("search (defaults suit qwen3.8-flash-next: edits must reach nearly every layer)")
    g.add_argument("--trials", type=int, default=60)
    g.add_argument("--startup", type=int, default=20)
    g.add_argument("--objective", choices=("refusals", "logp"), default="logp")
    g.add_argument("--search-limit", type=int, default=50,
                   help="harmful/harmless test prompts searched on; the next as many re-score the leaders")
    g.add_argument("--rescore", type=int, default=5)
    g.add_argument("--max-kld", type=float, default=0.05)
    g.add_argument("--distance", default="0.5,1")
    g.add_argument("--position", default="0.5,0.9")
    g.add_argument("--max-weight", default="1,1.8")
    g.add_argument("--uniform", action="store_true",
                   help="search one flat weight (from --max-weight) per index instead of curves; on qwen3.8 "
                        "curves only tied the uniform adapter on held-out prompts")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("confirmation")
    g.add_argument("--confirm-limit", type=int, default=300, help="prompts of jailbreakbench + llm-lat")
    g.add_argument("--confirm-offset", type=int, default=0)
    g.add_argument("--confirm-scales", default="1,1.25")
    g.add_argument("--confirm-max-kld", type=float, default=0.1, help="wikitext kl budget of the picked scale")
    g.add_argument("--no-think-check", dest="think_check", action="store_false")
    g.add_argument("--think-limit", type=int, default=50)
    g.add_argument("--think-max-tokens", type=int, default=1536)
    g.add_argument("--think-scales", default="", help="instead of each adapter's picked scale")
    g.add_argument("--kld-text", default=DATA / "wikitext-test.txt")
    g.add_argument("--kld-ctx", type=int, default=512)
    g.add_argument("--kld-chunks", type=int, default=16)
    g = p.add_argument_group("running")
    g.add_argument("--progress", action="store_true", help=PROGRESS_HELP)
    g.add_argument("--type", choices=("f16", "f32"), default="f16")
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("-ngl", "--ngl", type=int, default=-1)
    g.add_argument("-np", "--parallel", type=int, default=4)
    g.add_argument("--server", default=BIN / "llama-server")
    g.add_argument("--perplexity", default=BIN / "llama-perplexity")
    g.add_argument("--hierophant", default=BIN / "hierophant")
    if "--screen" in argv:
        p.set_defaults(**SCREEN)
    args = p.parse_args(argv)
    if not args.model and not args.hf_repo:
        p.error("give -m or --hf-repo")
    args.out = args.out or default_out(args)
    return args


def main(argv):
    args = parse_args(argv)
    missing = [n for n in (*CONFIRM_SETS, "wikitext-test.txt") if not (DATA / n).exists()]
    if missing:
        sys.exit(f"error: {', '.join(missing)} missing from {DATA}; run make datasets")
    r = Run(args)
    check_config(r, args)
    for step in STEPS:
        if (r.done / step).exists():
            print(f"== {step}: done, skipped", flush=True)
            continue
        print(f"== {step}", flush=True)
        FUNCS[step](r)
        (r.done / step).touch()


if __name__ == "__main__":
    main(sys.argv[1:])
