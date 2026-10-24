"""the whole pipeline on the tiny dense model, and resuming it."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import LLAMA_PERPLEXITY, LLAMA_SERVER, ROOT, TIED, Reader, tmpdir

PIPELINE = ROOT / "scripts" / "hierophant_pipeline.py"
SLOW_SECONDS = 10
STEPS = ["filter", "capture", "split", "search", "build", "residual", "confirm", "think", "summary"]
# the tiny model's kl is far above any real budget; a loose one still picks a non-zero scale
LOOSE_KLD = 100
SCREEN = ["--screen", "--train-limit", "40", "--confirm-limit", "4", "--kld-chunks", "2", "--think-scales", "1"]


def run_pipeline(out, *args):
    cmd = [sys.executable, "-I", str(PIPELINE), "-m", str(TIED), "--out", str(out), "--name", "tiny",
           "--max-tokens", "8", "--think-limit", "2", "--think-max-tokens", "16", "--kld-ctx", "128",
           "--kld-chunks", "2", "--confirm-max-kld", LOOSE_KLD, "-np", "2", "--server", str(LLAMA_SERVER),
           "--perplexity", str(LLAMA_PERPLEXITY), *args]
    return subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=False)


def pipeline(out, *extra):
    return run_pipeline(out, "--indices", "15,20", "--positions", "2", "--harmless-pcs", "2", "--train-limit", "8",
                        "--trials", "2", "--startup", "2", "--uniform", "--search-limit", "3", "--rescore", "1",
                        "--confirm-limit", "4", "--think-scales", "1", *extra)


class ThinkScales(unittest.TestCase):
    """the thinking check runs each adapter at the scale its confirmation picked."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from hierophant_pipeline import think_groups

        cls.groups = staticmethod(think_groups)

    def test_adapters_sharing_a_scale_share_a_run(self):
        picked = {"a.gguf": 1.2, "b.gguf": 1.4, "c.gguf": 1.2}
        self.assertEqual(self.groups(["x/a.gguf", "x/b.gguf", "x/c.gguf"], picked),
                         {"1.2": ["x/a.gguf", "x/c.gguf"], "1.4": ["x/b.gguf"]})

    def test_an_adapter_picked_at_zero_is_left_out(self):
        self.assertEqual(self.groups(["a.gguf", "b.gguf"], {"a.gguf": 0.0, "b.gguf": 1.0}), {"1": ["b.gguf"]})

    def test_think_scales_override_the_pick(self):
        self.assertEqual(self.groups(["a.gguf"], {"a.gguf": 0.0}, "1,1.25"), {"1,1.25": ["a.gguf"]})


class DefaultOut(unittest.TestCase):
    """without --out, each model gets its own directory under ./lora."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from hierophant_pipeline import default_out

        cls.out = staticmethod(lambda *argv: default_out(SimpleNamespace(**{"model": None, "hf_repo": None,
                                                                            "hf_file": None, **dict(argv)})))

    def test_a_repo_and_quant(self):
        self.assertEqual(self.out(("hf_repo", "unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS")),
                         Path("lora") / "unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS")

    def test_a_repo_and_file(self):
        self.assertEqual(self.out(("hf_repo", "org/m-GGUF"), ("hf_file", "Q8_0/m-Q8_0-00001-of-00002.gguf")),
                         Path("lora") / "org--m-GGUF--m-Q8_0-00001-of-00002")

    def test_a_bare_repo(self):
        self.assertEqual(self.out(("hf_repo", "org/m-GGUF")), Path("lora") / "org--m-GGUF")

    def test_a_local_file(self):
        self.assertEqual(self.out(("model", "/models/tiny-Q8_0.gguf")), Path("lora") / "tiny-Q8_0")


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir(cls.addClassCleanup) / "run"
        cls.first = pipeline(cls.out)
        cls.again = pipeline(cls.out)

    def test_every_step_runs_once(self):
        self.assertEqual(self.first.returncode, 0, self.first.stdout + self.first.stderr)
        for step in STEPS:
            self.assertIn(f"== {step}", self.first.stdout)

    def test_the_winner_is_built_from_the_filtered_capture_with_its_sidecar(self):
        kv = Reader(self.out / "tiny.gguf").kv
        self.assertEqual(kv["hierophant.positions"], 2)
        self.assertEqual(kv["hierophant.harmless_pcs"], 2)
        self.assertIn("harmful-train.filtered.txt", kv["hierophant.harmful_set"])
        best = json.loads((self.out / "search.json").read_text())["best"]
        self.assertEqual(kv["hierophant.direction_indices"], str(best["direction_index"]))
        w = best["curves"]["attn"][0]
        self.assertEqual(kv["hierophant.curve"], f"{w!r},0,{w!r},inf")
        s = json.loads((self.out / "tiny.json").read_text())
        self.assertIn("JailbreakBench/JBB-Behaviors@", s["eval"]["refusals"]["set"])
        self.assertIn("LLM-LAT/harmful-dataset@", s["eval"]["refusals"]["set"])
        self.assertIn("kld_seq", s["eval"])
        self.assertEqual(s["eval"]["max_kld"], LOOSE_KLD)

    def test_summary_has_thinking_scores_and_the_sidecar_pick(self):
        summary = json.loads((self.out / "summary.json").read_text())
        row = summary["adapters"]["tiny.gguf"]
        self.assertEqual(sorted(row), ["confirm", "think"])
        self.assertIn("kld_seq", row["confirm"][0])
        self.assertIn("loops", row["think"][0])
        self.assertEqual([p["scale"] for p in row["think"]], [1])
        self.assertEqual(json.loads((self.out / "tiny.json").read_text())["scale"], summary["picked"]["tiny.gguf"])

    def test_no_residual_unless_asked(self):
        self.assertIn("no residual variant", self.first.stdout)
        self.assertFalse((self.out / "tiny-res.gguf").exists())

    def test_a_rerun_skips_finished_steps(self):
        self.assertEqual(self.again.returncode, 0, self.again.stdout + self.again.stderr)
        for step in STEPS:
            self.assertIn(f"== {step}: done, skipped", self.again.stdout)

    def test_a_changed_configuration_reruns_from_the_first_step_it_affects(self):
        p = pipeline(self.out, "--trials", "3")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("trials changed; rerunning from search", p.stdout)
        self.assertIn("== split: done, skipped", p.stdout)
        self.assertNotIn("== search: done, skipped", p.stdout)

    def test_an_option_removed_since_the_run_reruns_nothing(self):
        path = self.out / "config.json"
        config = json.loads(path.read_text())
        path.write_text(json.dumps({**config, "hf_revision": "main"}))
        p = pipeline(self.out, "--trials", config["trials"])
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertNotIn("rerunning", p.stdout)

    def test_from_reruns_that_step_and_the_ones_after(self):
        p = pipeline(self.out, "--from", "confirm")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("== build: done, skipped", p.stdout)
        self.assertNotIn("== confirm: done, skipped", p.stdout)
        self.assertNotIn("== summary: done, skipped", p.stdout)


class Screen(unittest.TestCase):
    """weight-1 adapters per index plus a residual variant, scored on a held-out slice; a
    second recipe reuses the first one's filtered prompts."""

    @classmethod
    def setUpClass(cls):
        d = tmpdir(cls.addClassCleanup)
        cls.out, cls.reuse = d / "screen", d / "reuse"
        cls.first = run_pipeline(cls.out, *SCREEN, "--indices", "15,15+20", "--residual", "0.05")
        cls.again = cls.reuse_run("15")

    @classmethod
    def reuse_run(cls, indices):
        return run_pipeline(cls.reuse, *SCREEN, "--indices", indices, "--train-data", cls.out / "data",
                            "--targets", "attn_output")

    def test_both_runs_succeed(self):
        self.assertEqual(self.first.returncode, 0, self.first.stdout + self.first.stderr)
        self.assertEqual(self.again.returncode, 0, self.again.stdout + self.again.stderr)

    def test_no_search_and_every_index_is_built_at_weight_one(self):
        self.assertIn("no search with --screen", self.first.stdout)
        self.assertFalse(list(self.out.glob("parts-*")))
        for name in ("tiny-15", "tiny-15+20"):
            kv = Reader(self.out / f"{name}.gguf").kv
            self.assertEqual(kv["hierophant.direction_indices"], name.removeprefix("tiny-"))
            self.assertEqual(kv["hierophant.curve"], "1,0,1,inf")

    def test_the_residual_variant_adds_what_the_first_still_refuses(self):
        kv = Reader(self.out / "tiny-res.gguf").kv
        self.assertEqual(kv["adapter.lora.alpha"], 2.0)
        self.assertEqual(kv["hierophant.residual_lora"], "tiny-15.gguf")
        self.assertAlmostEqual(kv["hierophant.residual_lora_scale"], 0.05, places=6)
        self.assertIn("harmful-train.filtered.filtered.txt", kv["hierophant.residual_set"])

    def test_every_candidate_is_scored_past_the_full_confirmation_slice(self):
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertIsNone(summary["best"])
        self.assertEqual(sorted(summary["adapters"]), ["tiny-15+20.gguf", "tiny-15.gguf", "tiny-res.gguf"])
        self.assertEqual(len(summary["adapters"]["tiny-15.gguf"]["confirm"]), 5)
        s = json.loads((self.out / "tiny-15.json").read_text())
        self.assertEqual(s["eval"]["refusals"]["offset"], 300)

    def test_the_thinking_check_runs_past_the_end_of_the_short_harmless_set(self):
        think = json.loads((self.out / "think.json").read_text())
        self.assertEqual([p["scale"] for p in think["tiny-15.gguf"]], [0, 1])

    def test_changed_indices_rebuild_without_capturing_again(self):
        p = self.reuse_run("15+20")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("indices changed; rerunning from split", p.stdout)
        self.assertIn("== capture: done, skipped", p.stdout)
        self.assertTrue((self.reuse / "tiny-15+20.gguf").exists())

    def test_a_second_recipe_reuses_the_filtered_prompts(self):
        self.assertIn("using the filtered sets in", self.again.stdout)
        kv = Reader(self.reuse / "tiny-15.gguf").kv
        self.assertIn("harmful-train.filtered.txt", kv["hierophant.harmful_set"])
        self.assertEqual(kv["hierophant.targets"], "attn_output")


class Streaming(unittest.TestCase):
    """a step's output reaches its log while the step still runs, not when it exits."""

    def test_script_output_is_logged_before_the_script_exits(self):
        import threading
        import time

        sys.path.insert(0, str(ROOT / "scripts"))
        from hierophant_pipeline import Run, parse_args

        d = tmpdir(self.addCleanup)
        slow = d / "slow.py"
        slow.write_text(f"import time\nprint('started')\ntime.sleep({SLOW_SECONDS})\n")
        r = Run(parse_args(["-m", str(TIED), "--out", str(d / "out"), "--indices", "1"]))
        worker = threading.Thread(target=r.script, args=("slow", slow))
        worker.start()
        self.addCleanup(worker.join)
        log = d / "out" / "logs" / "slow.log"
        deadline = time.time() + SLOW_SECONDS / 2
        while time.time() < deadline and "started" not in (log.read_text() if log.exists() else ""):
            time.sleep(0.2)
        self.assertIn("started", log.read_text())


if __name__ == "__main__":
    unittest.main()
