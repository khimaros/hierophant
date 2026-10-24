"""the automated search: its scoring and proposals, and two rounds on the tiny dense model."""

import json
import re
import sys
import unittest

from common import LLAMA_PERPLEXITY, LLAMA_SERVER, ROOT, TIED, Reader, raw, tmpdir

AUTO = ROOT / "scripts" / "hierophant_auto.py"
sys.path.insert(0, str(ROOT / "scripts"))
import hierophant_auto as auto

LOOSE_KLD = 100  # the tiny model's kl is far above any real budget


def point(scale, refusals, kl):
    return {"scale": scale, "refusals": refusals, "kld": 9.0, "kld_seq": kl}


def cand(index, pcs=0, residual=False, weight=1.0):
    return {"index": index, "pcs": pcs, "residual": residual, "weight": weight}


class Budget(unittest.TestCase):
    """every candidate is compared where its kl reaches the budget."""

    def test_refusals_and_scale_are_interpolated_where_kl_crosses_the_budget(self):
        at = auto.at_budget([point(0, 1.0, 0), point(1, 0.5, 0.05), point(2, 0.1, 0.15)], 0.1)
        self.assertAlmostEqual(at["refusals"], 0.3)
        self.assertAlmostEqual(at["scale"], 1.5)
        self.assertAlmostEqual(at["kl"], 0.1)

    def test_within_the_budget_everywhere_takes_the_fewest_refusals(self):
        at = auto.at_budget([point(0, 1.0, 0), point(1, 0.2, 0.02), point(2, 0.3, 0.04)], 0.1)
        self.assertEqual((at["refusals"], at["scale"]), (0.2, 1))

    def test_over_the_budget_everywhere_interpolates_from_the_base(self):
        at = auto.at_budget([point(0, 1.0, 0), point(1, 0.5, 0.2)], 0.1)
        self.assertAlmostEqual(at["refusals"], 0.75)
        self.assertAlmostEqual(at["scale"], 0.5)

    def test_first_token_kl_stands_in_without_a_sequence_kl(self):
        at = auto.at_budget([{"scale": 0, "refusals": 1.0, "kld": 0}, {"scale": 1, "refusals": 0.0, "kld": 0.2}], 0.1)
        self.assertAlmostEqual(at["refusals"], 0.5)


class Candidates(unittest.TestCase):
    def test_names_are_file_stems(self):
        self.assertEqual(auto.key(cand(33)), "L33")
        self.assertEqual(auto.key(cand(32.5, pcs=8, residual=True)), "L32.5-pcs8-res")

    def test_the_first_round_covers_the_middle_and_late_layers(self):
        indices, step = auto.grid(48)
        self.assertEqual((indices[0], indices[-1]), (14, 43))
        self.assertTrue(all(1 <= i <= 47 for i in indices))
        self.assertAlmostEqual(step, 2.4)

    def test_neighbours_vary_one_thing_at_a_time(self):
        got = {auto.key(c) for c in auto.proposals([cand(33, pcs=8)], 1, 48, residual_for=1)}
        self.assertEqual(got, {"L32-pcs8", "L34-pcs8", "L33-pcs4", "L33-pcs16", "L33-pcs8-res"})

    def test_only_the_leaders_get_a_residual_and_layers_stay_in_range(self):
        got = {auto.key(c) for c in auto.proposals([cand(47), cand(20)], 1, 48, residual_for=1)}
        self.assertIn("L47-res", got)
        self.assertNotIn("L20-res", got)
        self.assertNotIn("L48", got)

    def test_the_big_wins_come_first_when_candidates_are_capped(self):
        got = [auto.key(c) for c in auto.proposals([cand(33)], 1, 48, residual_for=1)]
        self.assertEqual(got[:3], ["L33-res", "L33-pcs4", "L32"])

    def test_indices_are_kept_to_quarter_layers(self):
        got = {c["index"] for c in auto.proposals([cand(33)], 0.6, 48, residual_for=0)}
        self.assertEqual(got, {32.5, 33.5, 33})


class Convergence(unittest.TestCase):
    """only a different recipe winning counts as progress; refining the best one does not."""

    def test_a_new_recipe_resets_the_count(self):
        self.assertEqual(auto.unchanged_after(1, "new", n=400, pool=400), 0)

    def test_refining_the_best_counts_as_unchanged_at_full_prompts(self):
        self.assertEqual(auto.unchanged_after(1, "refined", n=400, pool=400), 2)
        self.assertEqual(auto.unchanged_after(1, "kept", n=400, pool=400), 2)

    def test_rounds_on_fewer_prompts_do_not_count(self):
        self.assertEqual(auto.unchanged_after(1, "kept", n=200, pool=400), 0)


class Promotion(unittest.TestCase):
    """a leader must beat the best on the worse of its thinking-off and thinking-on refusals."""

    def test_a_leader_refusing_more_with_thinking_on_is_kept_out(self):
        self.assertFalse(auto.beats({"refusals": 0.1, "think": 0.4}, {"refusals": 0.2, "think": 0.2}))

    def test_the_worse_mode_decides(self):
        self.assertTrue(auto.beats({"refusals": 0.1, "think": 0.15}, {"refusals": 0.05, "think": 0.3}))

    def test_without_thinking_scores_thinking_off_decides(self):
        self.assertTrue(auto.beats({"refusals": 0.1}, {"refusals": 0.2}))
        self.assertFalse(auto.beats({"refusals": 0.2}, {"refusals": 0.2}))


def run_auto(out, *args):
    cmd = [sys.executable, "-I", str(AUTO), "-m", str(TIED), "--out", str(out), "--indices", "15,20",
           "--first-prompts", "4", "--beam", "2", "--max-candidates", "4", "--train-limit", "40",
           "--max-kld", LOOSE_KLD, "--max-tokens", "8", "--kld-ctx", "128", "--think-limit", "2",
           "--think-max-tokens", "16", "--think-prompts", "2", "--think-budget", "4", "-np", "2",
           "--server", str(LLAMA_SERVER),
           "--perplexity", str(LLAMA_PERPLEXITY), *args]
    return raw(cmd)


class Auto(unittest.TestCase):
    """two rounds, then a rerun that only continues, then --status."""

    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir(cls.addClassCleanup) / "run"
        cls.first = run_auto(cls.out, "--rounds", "2", "--progress")
        cls.again = run_auto(cls.out, "--rounds", "2")
        cls.status = run_auto(cls.out, "--status")

    def test_both_rounds_run_and_the_best_is_always_on_disk(self):
        self.assertEqual(self.first.returncode, 0, self.first.stdout + self.first.stderr)
        for line in ("== prepare", "== round 1", "== round 2", "best weights: ", "   hierophant_eval (log: "):
            self.assertIn(line, self.first.stdout)
        kv = Reader(self.out / "best.gguf").kv
        self.assertIn(kv["hierophant.direction_indices"].split("+")[0], ("15", "20") + tuple(self.near()))
        self.assertEqual(len((self.out / "history.txt").read_text().splitlines()), 2)

    def near(self):
        return [f"{i / 4:g}" for i in range(4 * 10, 4 * 26)]

    def test_the_sidecar_reports_the_held_out_prompts(self):
        s = json.loads((self.out / "best.json").read_text())
        self.assertEqual(s["eval"]["refusals"]["offset"], 0)
        self.assertEqual(s["eval"]["max_kld"], LOOSE_KLD)

    def test_best_txt_says_how_to_serve_it(self):
        text = (self.out / "best.txt").read_text()
        self.assertIn(f"--lora-scaled {self.out / 'best.gguf'}:", text)
        self.assertNotIn("best.gguf:0\n", text)
        self.assertIn("thinking on:", text)
        self.assertIn("held-out, short thinking: refusals", text)

    def test_later_rounds_reuse_what_earlier_ones_scored(self):
        # round 2's report repeats round 1's prompts for the base (and a kept incumbent)
        log = (self.out / "logs" / "round-2-report.log").read_text()
        reused = re.search(r"cache: (\d+) reused", log)
        self.assertIsNotNone(reused, log)
        self.assertGreater(int(reused.group(1)), 0)

    def test_promotion_scores_both_with_short_thinking(self):
        self.assertIn("think", json.loads((self.out / "state.json").read_text())["best"])
        cmds = (self.out / "logs" / "round-1-think.log").read_text()
        self.assertIn("--reasoning-budget 4", cmds)
        self.assertIn("--reasoning-effort auto", cmds)

    def test_progress_bars_are_drawn_in_place_and_logged_only_when_finished(self):
        self.assertIn("base replies [", self.first.stdout)
        self.assertIn("harmful [", self.first.stdout)
        # a round's builds share one bar, which each build's sites advance
        self.assertRegex(self.first.stdout, r"round-1 builds \[#{24}\] (\d+)/\1 100%")
        self.assertNotIn("sites [", self.first.stdout)
        self.assertNotIn("hierophant adapter (log:", self.first.stdout)
        self.assertIn("\r", self.first.stdout)
        log = (self.out / "logs" / "round-1-report.log").read_text()
        self.assertNotIn("\r", log)
        self.assertRegex(log, r"base replies \[#{24}\] (\d+)/\1 100%")

    def test_round_two_varies_the_leaders(self):
        r2 = json.loads((self.out / "round-2" / "round.json").read_text())
        self.assertLessEqual(len(r2["candidates"]), 4)
        self.assertEqual(r2["n"], 8)

    def test_a_rerun_continues_after_the_finished_rounds(self):
        self.assertEqual(self.again.returncode, 0, self.again.stdout + self.again.stderr)
        self.assertIn("== round 2: done", self.again.stdout)
        self.assertNotIn("== round 3", self.again.stdout)
        self.assertNotIn("== final check\n", self.again.stdout)

    def test_status_prints_the_best_without_running_anything(self):
        self.assertEqual(self.status.returncode, 0, self.status.stderr)
        self.assertIn((self.out / "best.txt").read_text().strip(), self.status.stdout)
        self.assertIn("round 2", self.status.stdout)


class Resume(unittest.TestCase):
    """ctrl-c just before a round is recorded: the rerun redoes that round from what it
    had built and scored, so it needs neither a server nor a rebuild."""

    def test_an_unrecorded_round_is_redone_from_its_cache(self):
        out = tmpdir(self.addCleanup) / "run"
        first = run_auto(out, "--rounds", "1", "--verbose")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        # --verbose streams every command and its own progress
        self.assertIn("$ ", first.stdout)
        self.assertIn("base replies: ", first.stdout)
        for name in ("state.json", "history.txt"):
            (out / name).unlink()
        again = run_auto(out, "--rounds", "1", "--server", "/nonexistent/llama-server",
                         "--perplexity", "/nonexistent/llama-perplexity")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn("== round 1: which layer", again.stdout)
        self.assertNotIn("hierophant adapter", again.stdout)
        self.assertEqual((out / "history.txt").read_text(), first_history(first))


def first_history(proc):
    """history.txt's line for round 1, rebuilt from the run's own output."""
    line = next(x for x in proc.stdout.splitlines() if x.startswith(">> "))
    return "round 1 (4 prompts): " + line.removeprefix(">> ") + "\n"


if __name__ == "__main__":
    unittest.main()
