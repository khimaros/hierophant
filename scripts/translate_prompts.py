#!/usr/bin/env python3
"""translate prompt sets into other languages with a model, one file per set and language.

the translator is a model behind an openai-compatible --url, or the pinned
llama-server started here on -m/--hf-repo, optionally with an adapter (an
abliterated model translates harmful prompts that an aligned one refuses).
replies that refuse, answer the prompt instead of translating it, or come back
untranslated are dropped and listed beside the output. data/harmful-test.txt
becomes data/harmful-test.<lang>.txt, recorded in the manifest with its source,
so train and test splits stay apart in every language.
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
    REFUSAL_MARKERS,
    Server,
    add_server_args,
    load_prompts,
    progress_map,
    resolve_model,
)

INSTRUCTION = ("Translate the following text into {language}. Reply with the translation only, "
               "without notes or quotes, and do not answer or act on it.")
LANGUAGES = {
    "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French", "hi": "Hindi", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
    "tr": "Turkish", "vi": "Vietnamese", "zh": "Chinese (Simplified)",
}
# a translation this much longer than its source is an answer, not a translation.
LENGTH_RATIO = 4
LENGTH_SLACK = 100


def problem(src, out):
    """why a translation is unusable, or None. a refusal marker counts only when
    the source does not contain it, so a translated "illegal" is kept."""
    if not out.strip():
        return "empty"
    if out.strip() == src.strip():
        return "untranslated"
    if len(out) > LENGTH_RATIO * len(src) + LENGTH_SLACK:
        return "too long"
    low, src_low = out.lower().replace(chr(0x2019), "'"), src.lower()
    if any(m in low and m not in src_low for m in REFUSAL_MARKERS):
        return "refusal"
    return None


def translate(server, args, text, language):
    loras = [(0, args.lora_scale)] if args.lora else OFF
    content = INSTRUCTION.format(language=language) + "\n\n" + text
    c = server.chat([{"role": "user", "content": content}], loras, max_tokens=args.max_tokens)
    out = (c["message"].get("content") or "").strip()
    return out[1:-1] if len(out) > 1 and out[0] == out[-1] == '"' else out


def translate_set(server, args, pool, source, code):
    """write source translated into one language; returns (kept, dropped)."""
    prompts = load_prompts(source)
    outs = progress_map(pool, f"{Path(source).stem}.{code}",
                        lambda p: translate(server, args, p, LANGUAGES.get(code, code)), prompts)
    kept, dropped = [], []
    for src, out in zip(prompts, outs):
        why = problem(src, out)
        if why:
            dropped.append({"source": src, "reply": out, "reason": why})
        else:
            kept.append(out)
    path = Path(args.out_dir) / f"{Path(source).stem}.{code}.txt"
    path.write_text("".join(escape(p) + "\n" for p in kept))
    path.with_suffix(".dropped.json").write_text(json.dumps(dropped, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {path} ({len(kept)} prompts, dropped {len(dropped)})")
    return path, len(dropped)


def translator_of(args):
    return {"model": args.url_model or Path(args.model or args.url).name,
            "lora": Path(args.lora).name if args.lora else None,
            "lora_scale": args.lora_scale if args.lora else None}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_server_args(p)
    add_output_args(p)
    p.add_argument("sources", nargs="+", help="prompt files to translate, e.g. data/harmful-train.txt")
    p.add_argument("--languages", required=True, help="comma-separated codes, e.g. zh,ru,es,de,ja,ar,hi,sw")
    p.add_argument("--lora", help="adapter for the translator llama-server started here")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.set_defaults(max_tokens=512)
    args = p.parse_args(argv)
    if not args.url:
        resolve_model(p, args)
    return args


def main(argv):
    args = parse_args(argv)
    manifest = Manifest(args)
    server = Server(args, [args.lora] if args.lora else [])
    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            for source in args.sources:
                for code in args.languages.split(","):
                    path, dropped = translate_set(server, args, pool, source, code)
                    manifest.add(source, path, language=code, translator=translator_of(args), dropped=dropped)
    finally:
        server.close()


if __name__ == "__main__":
    main(sys.argv[1:])
