"""staging adapters for a hugging face repo: one per base quant, renamed after it, with a
README written from their sidecars, and an upload only when asked."""

import json
import subprocess
import sys
import unittest

from common import LLAMA_SERVER, MOE, MOE_Q8, ROOT, capture, run, tmpdir

PUBLISH = ROOT / "scripts" / "hierophant_publish.py"
EVAL = ROOT / "scripts" / "hierophant_eval.py"
BASE_REPO = "tester/stories15M_MOE-GGUF"
REPO = "tester/stories15M_MOE-hierophant-LoRA-GGUF"
# the quant last, as GGUF files name it: a variant goes before it
NAMES = {"F16": "stories15M_MOE-hierophant-F16", "Q8_0": "stories15M_MOE-hierophant-Q8_0"}


def adapter_with_sidecar(model, d):
    """an adapter for model, scored so that its sidecar names BASE_REPO and model's file."""
    dirs = capture(model, d / "dirs.gguf")
    a = d / "best.gguf"
    run("adapter", "-m", model, "-d", dirs, "-o", a, "--direction-index", 3)
    p = subprocess.run([sys.executable, "-I", str(EVAL), "-m", str(model), "--lora", str(a), "--limit", "3",
                        "--scales", "1", "--max-tokens", "8", "-np", "2", "--server", str(LLAMA_SERVER),
                        "--base-repo", BASE_REPO, "--max-kld", "100"], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    return a


def publish(*args):
    return subprocess.run([sys.executable, "-I", str(PUBLISH), *map(str, args)], capture_output=True, text=True,
                          check=False)


class Publish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = tmpdir(cls.addClassCleanup)
        cls.adapters = {}
        for quant, model in (("F16", MOE), ("Q8_0", MOE_Q8)):
            (root / quant).mkdir()
            cls.adapters[quant] = adapter_with_sidecar(model, root / quant)
        cls.stage = root / "stage"
        cls.first = publish(cls.adapters["F16"], "--user", "tester", "--license", "mit", "--out", cls.stage)
        cls.only_first = (cls.stage / "README.md").read_text() if cls.first.returncode == 0 else ""
        cls.fake_hf = root / "hf"
        cls.fake_hf.write_text(f"#!/bin/sh\necho \"$@\" > {root / 'hf-args.txt'}\n")
        cls.fake_hf.chmod(0o755)
        cls.second = publish(cls.adapters["Q8_0"], "--user", "tester", "--license", "mit", "--out", cls.stage,
                             "--upload", "--hf", cls.fake_hf)
        cls.hf_args = root / "hf-args.txt"
        cls.card = (cls.stage / "README.md").read_text()

    def test_both_succeed(self):
        for p in (self.first, self.second):
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_each_adapter_is_staged_under_its_quant_with_its_sidecar(self):
        for quant, name in NAMES.items():
            self.assertEqual((self.stage / f"{name}.gguf").read_bytes(), self.adapters[quant].read_bytes())
            staged = json.loads((self.stage / f"{name}.json").read_text())
            self.assertEqual(staged, json.loads(self.adapters[quant].with_suffix(".json").read_text()))

    def test_the_card_carries_the_hub_metadata(self):
        front = self.card.split("---\n")[1]
        for line in (f"base_model: {BASE_REPO}", "base_model_relation: adapter", "license: mit", "- gguf", "- lora",
                     "- llama.cpp", "- hierophant", "- abliteration"):
            self.assertIn(line, front)

    def test_the_card_says_how_to_get_and_serve_each_quant(self):
        self.assertIn("https://github.com/khimaros/hierophant", self.card)
        for quant, name in NAMES.items():
            # a pick of 0 (nothing refused less within the budget) is served at 1 anyway, as best.txt says
            scale = json.loads((self.stage / f"{name}.json").read_text())["scale"] or 1
            self.assertIn(f"hf download {REPO} {name}.gguf --local-dir .", self.card)
            self.assertIn(f"llama-server -hf {BASE_REPO}:{quant} --lora-scaled {name}.gguf:{scale:g}", self.card)

    def test_the_table_has_a_row_per_quant_with_its_base_sha256(self):
        for quant, name in NAMES.items():
            sha = json.loads((self.stage / f"{name}.json").read_text())["base"]["sha256"]
            row = next(line for line in self.card.splitlines() if line.startswith(f"| {quant} |"))
            self.assertIn(f"{name}.gguf", row)
            self.assertIn(sha, row)

    def test_a_row_scores_the_scale_its_serve_line_gives(self):
        for quant, name in NAMES.items():
            s = json.loads((self.stage / f"{name}.json").read_text())
            at = next(p for p in s["eval"]["sweep"] if p["scale"] == (s["scale"] or 1))
            row = next(line for line in self.card.splitlines() if line.startswith(f"| {quant} |")).split(" | ")
            self.assertEqual(row[-2], f"{at['kld']:.3f}")
            self.assertTrue(row[-3].endswith(f"-> {at['refusals']:.2f} (n=3)"), row)

    def test_a_later_quant_rewrites_the_card_from_every_sidecar(self):
        self.assertIn("| F16 |", self.only_first)
        self.assertNotIn("| Q8_0 |", self.only_first)
        self.assertIn("| F16 |", self.card)

    def test_nothing_is_uploaded_unless_asked(self):
        self.assertIn(f"hf upload {REPO} {self.stage}", self.first.stdout)
        self.assertEqual(self.hf_args.read_text().split(), ["upload", REPO, str(self.stage), ".", "--repo-type",
                                                            "model"])


class Names(unittest.TestCase):
    """the model and quant come from the base's repo and file, as unsloth and bartowski name them."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import hierophant_publish

        cls.p = hierophant_publish

    def test_a_split_file_in_a_quant_directory(self):
        base = {"repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
                "file": "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"}
        self.assertEqual(self.p.names_of(base), ("Qwen3.8-Flash-Next", "UD-IQ4_XS"))

    def test_a_single_file(self):
        base = {"repo": "bartowski/SmolLM2-135M-Instruct-GGUF", "file": "SmolLM2-135M-Instruct-Q8_0.gguf"}
        self.assertEqual(self.p.names_of(base), ("SmolLM2-135M-Instruct", "Q8_0"))

    def test_a_file_not_named_after_its_repo_needs_the_quant_given(self):
        base = {"repo": "someone/Model-GGUF", "file": "weights.gguf"}
        self.assertEqual(self.p.names_of(base, "Q4_K_M"), ("Model", "Q4_K_M"))
        with self.assertRaises(SystemExit):
            self.p.names_of(base)


if __name__ == "__main__":
    unittest.main()
