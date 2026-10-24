#!/usr/bin/env python3
"""how much of an adapter's direction is left in the residual stream.

compares a base capture with one taken through the adapter (`hierophant capture
--lora`). each layer is measured along the direction the adapter removes at
that layer, read from its own B = -w r, so the report can never disagree with
the adapter. a global direction at scale 1 leaves ~0 at every layer after the
first site, apart from what no LoRA can reach (tied embeddings, ple conv).
"""

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_lite import Reader

SETS = ("harmful_mean", "harmless_mean")
SITE = re.compile(r"blk\.(\d+)\..*\.lora_b$")


def layer_means(path, name):
    r = Reader(path)
    n_embd, n_layer = r.tensors[name]["ne"]
    flat = r.tensor(name)
    return [flat[i * n_embd:(i + 1) * n_embd] for i in range(n_layer)]


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def unit(v):
    n = math.sqrt(dot(v, v))
    return [x / n for x in v]


def adapter_directions(path, n_layer):
    """per layer, the unit direction of the first site there (the first column of
    a rank-k B); layers without a site take the nearest earlier one (or the
    first, before any)."""
    r = Reader(path)
    found = {}
    for name, info in r.tensors.items():
        m = SITE.match(name)
        if m and int(m.group(1)) not in found:
            k, n_embd = info["ne"][:2]
            found[int(m.group(1))] = unit([-x for x in r.tensor(name)[:k * n_embd:k]])
    if not found:
        sys.exit(f"error: {path} has no block lora_b tensors")
    first = found[min(found)]
    out, last = [], first
    for layer in range(n_layer):
        last = found.get(layer, last)
        out.append(last)
    return out


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", help="directions from `hierophant capture`")
    p.add_argument("adapted", help="directions from `hierophant capture --lora ADAPTER`")
    p.add_argument("--adapter", required=True, help="the adapter the second capture ran through")
    args = p.parse_args(argv)
    means = [layer_means(f, s) for s in SETS for f in (args.base, args.adapted)]
    dirs = adapter_directions(args.adapter, len(means[0]))
    print("layer  " + "  ".join(f"{s.split('_')[0] + ' base':>13} {'adapted':>9} {'left':>6}" for s in SETS))
    for layer, rows in enumerate(zip(*means)):
        cells = []
        for k in range(len(SETS)):
            before, after = dot(dirs[layer], rows[2 * k]), dot(dirs[layer], rows[2 * k + 1])
            cells.append(f"{before:+13.4f} {after:+9.4f} {after / before if before else 0:6.1%}")
        print(f"{layer:5d}  " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
