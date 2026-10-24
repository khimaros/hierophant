#!/usr/bin/env python3
"""keep the training prompts the base model treats as their label says.

a harmful prompt the base model answers anyway, or a harmless one it refuses,
blurs the difference of means. this asks the base model (the pinned
llama-server on -m/--hf-repo, or --url) to reply to each prompt, judges the
reply as the eval does, and writes data/harmful-train.filtered.txt with only
the refused harmful prompts and data/harmless-train.filtered.txt with only the
answered harmless ones, recorded in the manifest beside their sources.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_datasets import Manifest, add_output_args, escape
from hierophant_eval import (
    OFF,
    Cache,
    Judge,
    Server,
    add_server_args,
    cached,
    load_prompts,
    progress_map,
    resolve_model,
)

# (option, kept verdict, what is kept)
SETS = (("harmful", True, "refused"), ("harmless", False, "answered"))
CACHE_NAME = "filter.cache.jsonl"  # every reply and verdict, so a rerun after ctrl-c resumes


def filter_set(server, judge, pool, args, source, refused):
    """write the prompts whose reply the judge calls `refused`; returns (path, dropped)."""
    prompts = load_prompts(source, args.limit, args.offset)
    stem = Path(source).stem
    loras = [(0, args.lora_scale)] if args.lora else OFF
    replies = progress_map(pool, f"{stem} replies", cached(server, "replies", loras, lambda p: server.reply(p, loras)),
                           prompts, bar=args.progress)
    verdicts = progress_map(pool, f"{stem} verdicts",
                            cached(server, "verdicts", loras, lambda p, r: judge(p, r[0], r[2])), prompts, replies,
                            bar=args.progress)
    kept = [p for p, v in zip(prompts, verdicts) if v == refused]
    dropped = [{"prompt": p, "reply": r[0]} for p, r, v in zip(prompts, replies, verdicts) if v != refused]
    path = Path(args.out_dir) / f"{Path(source).stem}.filtered.txt"
    path.write_text("".join(escape(p) + "\n" for p in kept))
    path.with_suffix(".dropped.json").write_text(json.dumps(dropped, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {path} ({len(kept)} prompts, dropped {len(dropped)})")
    return path, len(dropped)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_server_args(p)
    add_output_args(p)
    p.add_argument("--lora", help="reply with this adapter applied, e.g. to keep what it still refuses")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.set_defaults(harmful=None, harmless=None)
    args = p.parse_args(argv)
    if not args.harmful and not args.harmless:
        p.error("give --harmful, --harmless or both, e.g. data/harmful-train.txt")
    if not args.url:
        resolve_model(p, args)
    return args


def main(argv):
    args = parse_args(argv)
    manifest = Manifest(args)
    cache = Cache(Path(args.out_dir) / CACHE_NAME)
    server = Server(args, [args.lora] if args.lora else [], cache)
    under = {"lora": Path(args.lora).name, "lora_scale": args.lora_scale} if args.lora else {}
    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            judge = Judge(args, server)
            for option, refused, keep in SETS:
                source = getattr(args, option)
                if source:
                    path, dropped = filter_set(server, judge, pool, args, source, refused)
                    manifest.add(source, path, filter={"keep": keep, "judge": args.judge, "dropped": dropped, **under})
    finally:
        server.close()


if __name__ == "__main__":
    main(sys.argv[1:])
