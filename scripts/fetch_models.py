#!/usr/bin/env python3
"""fetch the small pinned ggufs the end-to-end tests run against."""

import hashlib
import sys
import urllib.request
from pathlib import Path

RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{file}"
CHUNK = 1 << 20

STORIES = ("ggml-org/stories15M_MOE", "b6dd737497465570b5f5e962dbc9d9454ed1e0eb")

# (repo, revision, file, sha256): a dense instruct model with tied embeddings,
# a mixture-of-experts model with untied embeddings at f16 and q8_0, and a
# lora for it that hierophant did not build.
MODELS = [
    ("bartowski/SmolLM2-135M-Instruct-GGUF", "09816acd5d99df7be770d85ea30822623dab342c",
     "SmolLM2-135M-Instruct-Q8_0.gguf", "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"),
    (*STORIES, "stories15M_MOE-F16.gguf", "1240dfc1957df9f3550dd6c1d9e64b466fc2f452d8bc34bd4e45e1a1e2ca6055"),
    (*STORIES, "stories15M_MOE-Q8_0.gguf", "c7aa6863f9a4b3cdf19716e2c95622dcbd3bd06989324bf1ac8e60486ef8e881"),
    (*STORIES, "moe_shakespeare15M.gguf", "d1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo, rev, file, digest, out):
    path = out / file
    if path.exists() and sha256(path) == digest:
        return
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(RESOLVE.format(repo=repo, rev=rev, file=file), timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
    if sha256(tmp) != digest:
        tmp.unlink()
        sys.exit(f"error: checksum mismatch for {repo}/{file}")
    tmp.rename(path)
    print(f"wrote {path}")


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        fetch(*m, out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models")
