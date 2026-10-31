"""what a long run shares: its directories, logged commands and the steps that prepare it."""

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
from hierophant_eval import model_flags, stop
from progress import CLEAR, COUNT, OPEN, fit

CONFIRM_SETS = ("harmful-jbb.txt", "harmful-lat.txt")
OUT = Path("out")  # under the current directory, one subdirectory per model
MIN_RESIDUAL = 4  # still-refused prompts below which a residual direction is noise
READ_CHUNK = 1 << 16
PROGRESS_HELP = "draw each command's progress bars in place: a request, a kl chunk, a captured prompt, a site"
LINE_END = re.compile(rb"(?<=[\r\n])")
RESUME_HINT = ": rerun the same command to continue"


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
    doing = ""  # what the bar's unit under way is, named in its detail

    def __init__(self, args):
        self.a = args
        self.out = Path(args.out).expanduser()
        self.logs, self.done = self.out / "logs", self.out / ".done"
        self.data = Path(args.train_data).expanduser() if args.train_data else self.out / "data"
        for d in (self.out, self.logs, self.done):
            d.mkdir(parents=True, exist_ok=True)
        self.dirs = self.out / "dirs.gguf"
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
            try:
                drawn = self.stream(p, f)
            except BaseException:
                # stopped with the run, it would outlive it holding the model's memory
                stop(p)
                raise
            if self.bar:
                self.bar.draw()
            elif drawn:
                print(CLEAR, end="", flush=True)
            if p.wait() != 0:
                sys.exit(f"error: step {step} failed with {p.returncode}; see {log}")

    def stream(self, p, f):
        """a child's output to its log f and, with echo, the terminal. returns whether a
        redraw is left on the terminal."""
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
        return drawn

    def redraw(self, seg):
        """a child's bar: under a bar over several commands, it moves that bar when it is
        one of the units the bar rolls up; else under an open outer bar (a round) it is the
        share of its stage; else it is drawn as it is. returns whether it was drawn."""
        text = seg.replace(CLEAR, "").rstrip("\r")
        m = COUNT.match(text)
        if self.bar:
            if m:
                rolled = m["label"] in self.rollup and int(m["total"])
                self.bar.follow(int(m["done"]) / int(m["total"]) if rolled else 0, f"{self.doing} {m['label']}")
        elif OPEN:
            if m:
                OPEN[-1].follow(float(m["pct"]) / 100, m["detail"] or m["label"])
        elif self.progress:
            print(fit(text) + CLEAR + "\r", end="", flush=True)
            return True
        return False

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

    def capture_flags(self):
        a = self.a
        return ["-ngl", a.ngl, "--positions", a.positions, "--limit", a.train_limit,
                *(["--keep-rows"] if a.harmless_pcs else []), *(["--think"] if a.think_capture else [])]


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


def confirm_sets():
    return ",".join(str(DATA / name) for name in CONFIRM_SETS)


def loras_of(paths):
    return [arg for lora in paths for arg in ("--lora", lora)]


def default_out(args):
    """out/<org>--<repo>[--<quant or file stem>], or out/<file stem> for -m."""
    if not args.hf_repo:
        return OUT / Path(args.model).name.removesuffix(".gguf")
    repo, _, tag = args.hf_repo.partition(":")
    parts = [repo.replace("/", "--"), Path(args.hf_file).name.removesuffix(".gguf") if args.hf_file else tag]
    return OUT / "--".join(p for p in parts if p)
