"""the curve search over split adapters, and the merged adapter it recommends."""

import json
import shlex
import subprocess
import sys
import unittest

from common import LLAMA_SERVER, ROOT, TIED, Reader, adapter_direction, capture, run, tmpdir

SEARCH = ROOT / "scripts" / "hierophant_search.py"
INDEX = 20
TRIALS = 4


class Search(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        cls.dirs = capture(TIED, cls.dir / "dirs.gguf")
        split = cls.dir / "split"
        split.mkdir()
        run("adapter", "-m", TIED, "-d", cls.dirs, "-o", split / "s-{}.gguf", "--direction-index", INDEX, "--split")
        cls.parts = {(r.kv["hierophant.split_group"], r.kv["hierophant.split_layer"]): p
                     for p in split.glob("*.gguf") for r in [Reader(p)]}
        cmd = [sys.executable, "-I", str(SEARCH), "-m", str(TIED), "--split-dir", str(split), "--trials", str(TRIALS),
               "--startup", "2", "--limit", "3", "--max-tokens", "8", "-np", "2", "--server", str(LLAMA_SERVER),
               "--results", str(cls.dir / "search.json")]
        cls.proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    @staticmethod
    def parts_of(split):
        return [(r.kv["hierophant.split_group"], r.kv["hierophant.split_layer"])
                for p in split.glob("*.gguf") for r in [Reader(p)]]

    def results(self):
        return json.loads((self.dir / "search.json").read_text())

    def test_search_succeeds(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_every_trial_is_scored(self):
        trials = self.results()["trials"]
        self.assertEqual(len(trials), TRIALS)
        for t in trials:
            self.assertEqual(sorted(t["curves"]), ["attn", "mlp"])
            self.assertGreaterEqual(t["refusals"], 0)
            self.assertGreaterEqual(t["kld"], 0)

    def test_recommended_curves_build_the_adapter_the_trial_scored(self):
        best = self.results()["best"]
        merged = self.dir / "merged.gguf"
        run("adapter", "-m", TIED, "-d", self.dirs, "-o", merged, "--direction-index", INDEX,
            *shlex.split(best["adapter_args"]))
        tensors = Reader(merged).tensors
        for (group, layer), part in self.parts.items():
            site = next(n for n in Reader(part).tensors if n.endswith(".lora_b")).removesuffix(".lora_b")
            scale = best["scales"][f"{group}-{layer}"]
            if scale == 0:
                self.assertNotIn(site + ".lora_b", tensors)
                continue
            _, w = adapter_direction(merged, site)
            self.assertAlmostEqual(w, scale, places=5)


def split(d, index, *extra):
    """split adapters of TIED at one index, in their own directory."""
    out = d / f"split-{index}"
    out.mkdir()
    run("adapter", "-m", TIED, "-d", d / "dirs.gguf", "-o", out / "s-{}.gguf", "--direction-index", index, "--split",
        *extra)
    return out


def search(d, *extra):
    cmd = [sys.executable, "-I", str(SEARCH), "-m", str(TIED), "--limit", "3", "--max-tokens", "8", "-np", "2",
           "--server", str(LLAMA_SERVER), "--results", str(d / "search.json"), *extra]
    p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise AssertionError(p.stdout + p.stderr)
    return json.loads((d / "search.json").read_text()), p.stdout


class Objective(unittest.TestCase):
    """ranking by the base replies' log-probability, then re-scoring the leaders on other prompts."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        capture(TIED, cls.dir / "dirs.gguf")
        cls.out, cls.stdout = search(cls.dir, "--split-dir", split(cls.dir, INDEX), "--trials", 3, "--startup", 3,
                         "--objective", "logp", "--max-kld", 10, "--rescore", 2, "--rescore-offset", 3,
                         "--rescore-limit", 3)

    def test_trials_print_one_line_each_without_progress(self):
        self.assertNotIn("replies:", self.stdout)

    def test_trials_carry_the_reply_logp(self):
        for t in self.out["trials"]:
            self.assertLessEqual(t["reply_logp"], 0)

    def test_leaders_are_rescored_on_other_prompts(self):
        rescored = self.out["rescored"]
        self.assertEqual(len(rescored), 2)
        by_logp = sorted(range(3), key=lambda i: self.out["trials"][i]["reply_logp"])
        self.assertEqual([r["trial"] for r in rescored], by_logp[:2])
        self.assertEqual(rescored[0]["offset"], 3)

    def test_best_is_the_best_rescored(self):
        best = min(self.out["rescored"], key=lambda r: r["reply_logp"])
        self.assertEqual(self.out["best"]["trial"], best["trial"])
        self.assertEqual(self.out["best"]["reply_logp"], best["reply_logp"])


class Indices(unittest.TestCase):
    """one search over split sets at two direction indices, one split by group, one by kind."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        capture(TIED, cls.dir / "dirs.gguf")
        cls.dirs = {15: split(cls.dir, 15), 20: split(cls.dir, 20, "--split-by", "kind")}
        cls.out, _ = search(cls.dir, "--split-dir", cls.dirs[15], "--split-dir", cls.dirs[20], "--trials", 6,
                         "--startup", 6, "--seed", 1)

    def test_trials_use_both_indices_with_their_own_groups(self):
        groups = {15: ["attn", "mlp"], 20: ["attn_output", "ffn_down"]}
        seen = set()
        for t in self.out["trials"]:
            seen.add(t["direction_index"])
            self.assertEqual(sorted(t["curves"]), groups[t["direction_index"]])
        self.assertEqual(seen, {15, 20})

    def test_best_builds_from_its_own_index(self):
        best = self.out["best"]
        merged = self.dir / "merged.gguf"
        run("adapter", "-m", TIED, "-d", self.dir / "dirs.gguf", "-o", merged, *shlex.split(best["adapter_args"]))
        self.assertEqual(Reader(merged).kv["hierophant.direction_indices"], str(best["direction_index"]))
        for key, scale in best["scales"].items():
            group, layer = key.rsplit("-", 1)
            kind = {"attn": "attn_output", "mlp": "ffn_down"}.get(group, group)
            site = f"blk.{layer}.{kind}.weight"
            if scale:
                self.assertAlmostEqual(adapter_direction(merged, site)[1], scale, places=5)


class Uniform(unittest.TestCase):
    """one flat weight for every site, as the uniform adapters that work on qwen3.8."""

    def test_every_group_shares_one_flat_weight(self):
        d = tmpdir(self.addCleanup)
        capture(TIED, d / "dirs.gguf")
        out, _ = search(d, "--split-dir", split(d, INDEX), "--trials", 3, "--startup", 2, "--uniform",
                     "--max-weight", "1,1.5")
        for t in out["trials"]:
            curves = list(t["curves"].values())
            w = curves[0][0]
            self.assertTrue(1 <= w <= 1.5)
            for c in curves:
                self.assertEqual(c[:3], [w, 0.0, w])
                self.assertEqual(c[3], float("inf"))
        merged = d / "merged.gguf"
        run("adapter", "-m", TIED, "-d", d / "dirs.gguf", "-o", merged, *shlex.split(out["best"]["adapter_args"]))
        w = out["best"]["curves"]["attn"][0]
        for site in ("blk.0.attn_output.weight", "blk.29.ffn_down.weight"):
            self.assertAlmostEqual(adapter_direction(merged, site)[1], w, places=5)


class Resummarize(unittest.TestCase):
    """re-scored leaders are judged against the kl budget by the kl measured where the
    budget was set (the search prompts), since another prompt slice can sit far higher."""

    def test_rescored_leaders_keep_their_search_kl_for_the_budget(self):
        d = tmpdir(self.addCleanup)
        capture(TIED, d / "dirs.gguf")
        parts = split(d, INDEX)

        def trial(w, kld, logp):
            flat = [w, 0.0, w, float("inf")]
            return {"direction_index": INDEX, "curves": {"attn": flat, "mlp": flat}, "refusals": 0.1, "kld": kld,
                    "top1_same": 1.0, "reply_logp": logp, "unfinished": 0.0}

        trials = [trial(1.4, 0.049, -3.7), trial(1.3, 0.041, -3.1)]
        # on the rescore slice every kl is ~3x the search's, so all would be over a 0.05 budget
        rescored = [{**trial(1.4, 0.15, -4.3), "trial": 0, "offset": 3, "limit": 3},
                    {**trial(1.3, 0.12, -3.6), "trial": 1, "offset": 3, "limit": 3}]
        for r in rescored:
            del r["curves"], r["direction_index"]
        (d / "search.json").write_text(json.dumps({"trials": trials, "rescored": rescored, "best": {}}))
        cmd = [sys.executable, "-I", str(SEARCH), "-m", str(TIED), "--split-dir", str(parts), "--uniform",
               "--objective", "logp", "--max-kld", "0.05", "--results", str(d / "search.json"), "--summarize-only"]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        best = json.loads((d / "search.json").read_text())["best"]
        self.assertEqual(best["trial"], 0)
        self.assertIn("--weight 1.4", best["adapter_args"])


class Ranges(unittest.TestCase):
    def test_weight_and_distance_ranges_are_respected(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(TIED, d / "dirs.gguf")
        (d / "split").mkdir()
        run("adapter", "-m", TIED, "-d", dirs, "-o", d / "split" / "s-{}.gguf", "--direction-index", INDEX, "--split")
        cmd = [sys.executable, "-I", str(SEARCH), "-m", str(TIED), "--split-dir", str(d / "split"), "--trials", "2",
               "--startup", "1", "--limit", "2", "--max-tokens", "4", "-np", "2", "--server", str(LLAMA_SERVER),
               "--max-weight", "1.25,1.25", "--distance", "1,1", "--results", str(d / "search.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        last = max(layer for _, layer in Search.parts_of(d / "split"))
        for t in json.loads((d / "search.json").read_text())["trials"]:
            for curve in t["curves"].values():
                self.assertEqual(curve[0], 1.25)
                self.assertEqual(curve[3], last)


if __name__ == "__main__":
    unittest.main()
