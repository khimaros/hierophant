#!/usr/bin/env python3
"""fetch the prompt sets heretic uses into one-prompt-per-line text files.

directions come from the train splits, evaluation from the test splits, so the
prompts that pick the direction are never the ones that score it. a slice of
wikitext is fetched as plain text for llama-perplexity's kl divergence.
"""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"
INFO_API = "https://huggingface.co/api/datasets/"
PAGE = 100
HARMFUL = "mlabonne/harmful_behaviors"
HARMLESS = "mlabonne/harmless_alpaca"
WIKITEXT = "Salesforce/wikitext"


@dataclass
class Set:
    name: str
    dataset: str
    split: str
    limit: int
    column: str = "text"
    config: str = "default"
    prompts: bool = True  # one escaped prompt per line, else the rows as plain text
    exclude: tuple = ()  # sets already written whose prompts this one must not repeat


SETS = [
    Set("harmful-train", HARMFUL, "train", 416),
    Set("harmless-train", HARMLESS, "train", 416),
    Set("harmful-test", HARMFUL, "test", 100),
    Set("harmless-test", HARMLESS, "test", 100),
    Set("wikitext-test", WIKITEXT, "test", 600, config="wikitext-2-raw-v1", prompts=False),
]

# more harmful prompts to score with, never to pick directions. both draw
# partly on advbench, which harmful_behaviors is, so anything its splits hold
# is left out.
HARMFUL_SPLITS = ("harmful-train", "harmful-test")
MORE = [
    Set("harmful-jbb", "JailbreakBench/JBB-Behaviors", "harmful", 100, column="Goal", config="behaviors",
        exclude=HARMFUL_SPLITS),
    Set("harmful-lat", "LLM-LAT/harmful-dataset", "train", 1000, column="prompt", exclude=HARMFUL_SPLITS),
]


def normalized(prompt):
    return prompt.strip().lower().rstrip(".")


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def rows(s):
    """one column of the first `limit` rows of a split."""
    out = []
    while len(out) < s.limit:
        q = urllib.parse.urlencode({"dataset": s.dataset, "config": s.config, "split": s.split,
                                    "offset": len(out), "length": min(PAGE, s.limit - len(out))})
        page = get_json(f"{ROWS_API}?{q}")["rows"]
        if not page:
            break
        out += [r["row"][s.column] for r in page]
    return out


def escape(prompt):
    return prompt.replace("\\", "\\\\").replace("\n", "\\n")


def parse_extra(spec):
    """NAME=DATASET:SPLIT:N[:COLUMN[:CONFIG]], a prompt set beside heretic's."""
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    if not name or not 3 <= len(parts) <= 5:
        raise argparse.ArgumentTypeError(f"--extra wants NAME=DATASET:SPLIT:N[:COLUMN[:CONFIG]], got '{spec}'")
    return Set(name, parts[0], parts[1], int(parts[2]), *parts[3:])


def fetch(s, out, pinned):
    """write one set and return its manifest entry. the data files are not in the
    repo, so with a pinned entry the rows must hash as it says: the hub serves only
    the latest rows, and a changed upstream would otherwise swap the prompts."""
    got = [p.strip() for p in rows(s)]
    seen = {normalized(line) for name in s.exclude for line in (out / f"{name}.txt").read_text().splitlines()}
    kept = []
    for p in got:
        key = normalized(escape(p))
        if not s.exclude or key not in seen:
            kept.append(p)
            seen.add(key)
    text = "".join(escape(p) + "\n" for p in kept if p) if s.prompts else "\n".join(kept) + "\n"
    sha = hashlib.sha256(text.encode()).hexdigest()
    if pinned and pinned["sha256"] != sha:
        raise SystemExit(f"error: {s.name} from {s.dataset} has changed upstream (sha256 {sha[:12]}, manifest.json "
                         f"pins {pinned['sha256'][:12]}); rerun with --update to take the new rows")
    path = out / f"{s.name}.txt"
    path.write_text(text)
    already = f", {len(got) - len(kept)} already in {' or '.join(s.exclude)}" if s.exclude else ""
    print(f"wrote {path} ({len(kept)} rows{already})")
    entry = {
        "dataset": s.dataset,
        "revision": get_json(INFO_API + s.dataset)["sha"],
        "config": s.config,
        "split": s.split,
        "column": s.column,
        "n": text.count("\n") if s.prompts else len(kept),
        "sha256": sha,
    }
    return {**entry, "excluded": len(got) - len(kept)} if s.exclude else entry


def add_output_args(p):
    """where scripts that derive sets from others write them."""
    p.add_argument("--out-dir", default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--manifest", help="defaults to manifest.json in --out-dir")


class Manifest:
    """data/manifest.json, extended with sets derived from others (translated or
    filtered). a derived set carries its source's dataset, revision and split,
    so it reads as the same split, and is written as soon as it is added."""

    def __init__(self, args):
        self.path = Path(args.manifest or Path(args.out_dir) / "manifest.json")
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else {}

    def add(self, source, path, **extra):
        text = Path(path).read_bytes()
        src = self.entries.get(Path(source).stem, {})
        self.entries[Path(path).stem] = {
            **{k: src[k] for k in ("dataset", "revision", "config", "split", "column") if k in src},
            "source": Path(source).stem,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "n": text.count(b"\n"),
            "sha256": hashlib.sha256(text).hexdigest(),
            **extra,
        }
        self.path.write_text(json.dumps(self.entries, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default="data")
    p.add_argument("--extra", type=parse_extra, action="append", default=[],
                   help="also fetch NAME=DATASET:SPLIT:N[:COLUMN[:CONFIG]] as a prompt set")
    p.add_argument("--only-extra", action="store_true", help="fetch only the --extra and --more sets")
    p.add_argument("--more", action="store_true",
                   help="also fetch more harmful test prompts (jailbreakbench, llm-lat), minus the train split's")
    p.add_argument("--update", action="store_true", help="take the hub's current rows instead of the pinned ones")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for s in ([] if args.only_extra else SETS) + (MORE if args.more else []) + args.extra:
        manifest[s.name] = fetch(s, out, None if args.update else manifest.get(s.name))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
