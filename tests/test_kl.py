"""hierophant kl: llama-perplexity's kl over every position of a text, for several adapters
and scales from one model load."""

import json
import re
import tempfile
import unittest

from common import HARMLESS, LLAMA_PERPLEXITY, MOE, capture, raw, run, tmpdir

CTX, CHUNKS = 128, 2
SCALES = (0, 0.5, 1)
KLD_MEAN = re.compile(r"Mean\s+KLD:\s+(\S+)")
SAME_TOP = re.compile(r"Same top p:\s+(\S+)")


def perplexity(*args):
    p = raw([LLAMA_PERPLEXITY, "-m", MOE, "-c", CTX, "--chunks", CHUNKS, "-ngl", 0, *args])
    if p.returncode != 0:
        raise AssertionError(p.stdout + p.stderr)
    return p.stdout + p.stderr


class Kl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        dirs = capture(MOE, cls.dir / "dirs.gguf")
        cls.adapter = cls.dir / "a.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", cls.adapter, "--direction-index", 3)
        runs = [arg for s in SCALES for arg in ("--lora", cls.adapter, "--scale", s)]
        out = run("kl", "-m", MOE, "--text", HARMLESS, "-c", CTX, "--chunks", CHUNKS, "-ngl", 0, *runs)
        cls.runs = [json.loads(line) for line in out.splitlines() if line.startswith("{")]

    def reference(self, scale):
        with tempfile.NamedTemporaryFile(dir=self.dir, suffix=".kld") as base:
            perplexity("-f", HARMLESS, "--kl-divergence-base", base.name)
            out = perplexity("--lora-scaled", f"{self.adapter}:{scale}", "--kl-divergence-base", base.name,
                             "--kl-divergence")
        return float(KLD_MEAN.search(out).group(1)), float(SAME_TOP.search(out).group(1)) / 100

    def test_one_line_per_run_in_order(self):
        self.assertEqual([(r["lora"], r["scale"]) for r in self.runs], [(str(self.adapter), s) for s in SCALES])

    def test_scale_zero_is_the_base(self):
        self.assertEqual((self.runs[0]["kld"], self.runs[0]["same_top"]), (0, 1))

    def test_it_matches_llama_perplexity(self):
        # llama-perplexity keeps the base's log-probabilities in 16 bits; this keeps them in 32
        for r in self.runs[1:]:
            kld, same = self.reference(r["scale"])
            self.assertAlmostEqual(r["kld"], kld, delta=0.01 * kld + 1e-5)
            self.assertAlmostEqual(r["same_top"], same, delta=0.01)

    def test_a_stronger_edit_diverges_more(self):
        self.assertGreater(self.runs[2]["kld"], self.runs[1]["kld"])
        self.assertGreater(self.runs[1]["kld"], 0)


if __name__ == "__main__":
    unittest.main()
