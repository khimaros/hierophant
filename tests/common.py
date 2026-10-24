"""shared paths and helpers for the end-to-end tests."""

import json
import math
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gguf_lite import Reader

BIN = ROOT / "build" / "bin"
HIEROPHANT = BIN / "hierophant"
LLAMA_SERVER = BIN / "llama-server"
LLAMA_PERPLEXITY = BIN / "llama-perplexity"
DATA = ROOT / "data"
MOE = ROOT / "models" / "stories15M_MOE-F16.gguf"
MOE_Q8 = ROOT / "models" / "stories15M_MOE-Q8_0.gguf"
FOREIGN_LORA = ROOT / "models" / "moe_shakespeare15M.gguf"
TIED = ROOT / "models" / "SmolLM2-135M-Instruct-Q8_0.gguf"
HARMFUL = DATA / "harmful-train.txt"
HARMLESS = DATA / "harmless-train.txt"
N_PROMPTS = 8


def raw(cmd):
    """subprocess.run with its output decoded but untranslated: text mode would turn a
    progress bar's carriage returns into newlines."""
    p = subprocess.run(list(map(str, cmd)), capture_output=True, check=False)
    return subprocess.CompletedProcess(p.args, p.returncode, p.stdout.decode(), p.stderr.decode())


def run(*args):
    """run hierophant, returning stdout+stderr; fails the test on a non-zero exit."""
    p = raw([HIEROPHANT, *args])
    if p.returncode != 0:
        raise AssertionError(f"hierophant {' '.join(map(str, args))} exited {p.returncode}:\n{p.stdout}{p.stderr}")
    return p.stdout + p.stderr


def tmpdir(add_cleanup):
    """a scratch directory removed by a test's addCleanup or addClassCleanup."""
    d = tempfile.TemporaryDirectory(prefix="hierophant-")
    add_cleanup(d.cleanup)
    return Path(d.name)


def fake_openai(add_cleanup, reply):
    """an openai-compatible chat endpoint answering reply(request body), which is
    the content, a whole choice ({"message": ..., "finish_reason": ...}) or an http
    error status. returns its base url and the list of request bodies it received."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            r = reply(body)
            status = r if isinstance(r, int) else 200
            choice = {"finish_reason": "stop", **r} if isinstance(r, dict) else \
                {"message": {"content": r}, "finish_reason": "stop"}
            out = json.dumps({"error": {"code": status}} if isinstance(r, int) else {"choices": [choice]}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    add_cleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_port}", requests


def capture(model, out, *extra):
    run("capture", "-m", model, "--harmful", HARMFUL, "--harmless", HARMLESS,
        "--limit", N_PROMPTS, "-t", 4, "-o", out, *extra)
    return out


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


def adapter_direction(path, site="blk.0.attn_output.weight"):
    """the unit direction an adapter removes at a site, read from its B = -w r."""
    r = Reader(path)
    ne = r.tensors[site + ".lora_b"]["ne"]
    b = r.tensor(site + ".lora_b")[:ne[1]]
    return unit([-x for x in b]), math.sqrt(dot(b, b))


def lora_columns(path, site):
    """the k columns of a site's B = -w Q (first expert), as lists of n_embd floats."""
    r = Reader(path)
    k, n_embd = (r.tensors[site + ".lora_b"]["ne"] + [1])[:2]
    b = r.tensor(site + ".lora_b")
    return [[b[i * k + d] for i in range(n_embd)] for d in range(k)]


def refusal_direction(path, layer):
    h, b = layer_means(path, "harmful_mean"), layer_means(path, "harmless_mean")
    return unit([x - y for x, y in zip(h[layer], b[layer])])
