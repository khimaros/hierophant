"""the automated search: its scoring and proposals, and two rounds on the tiny dense model."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import CLI, LLAMA_SERVER, ROOT, TIED, Reader, raw, tmpdir

sys.path.insert(0, str(ROOT / "scripts"))
import hierophant_auto as auto
import hierophant_config
from hierophant_config import ConfigError

LOOSE_KLD = 100  # the tiny model's kl is far above any real budget
REPORT_PROMPTS = 6
SLOW_SECONDS = 10


def point(scale, refusals, kl):
    return {"scale": scale, "refusals": refusals, "kld": 0.0, "kld_seq": kl}


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

    def test_points_without_replies_only_place_the_budget(self):
        # --at-budget: the kl at every scale, replies only where it reaches the budget
        kl_only = [{"scale": 1, "kld": 0.0, "kld_seq": 0.05}, {"scale": 2, "kld": 0.0, "kld_seq": 0.15}]
        at = auto.at_budget([point(0, 1.0, 0), *kl_only, point(1.5, 0.3, 0.1)], 0.1)
        self.assertEqual((at["refusals"], at["scale"]), (0.3, 1.5))

    def test_another_measure_can_rank(self):
        # the layer screen ranks by the base's own first reply tokens, which fall as refusals do
        pts = [{**point(0, 0, 0), "reply_logp": -0.1}, {**point(1, 0, 0.2), "reply_logp": -2.1}]
        at = auto.at_budget([{k: v for k, v in p.items() if k != "refusals"} for p in pts], 0.1, "reply_logp")
        self.assertAlmostEqual(at["reply_logp"], -1.1)

    def test_the_larger_of_the_two_kls_is_the_measure(self):
        # harmless pcs keep the first token near the base's, which the wikitext kl alone misses
        at = auto.at_budget([{"scale": 0, "refusals": 1.0, "kld": 0, "kld_seq": 0},
                             {"scale": 1, "refusals": 0.0, "kld": 0.2, "kld_seq": 0.05}], 0.1)
        self.assertAlmostEqual(at["refusals"], 0.5)
        self.assertAlmostEqual(at["scale"], 0.5)

    def test_the_base_never_stands_in_for_an_adapter_with_a_point_of_its_own(self):
        at = auto.at_budget([point(0, 0.95, 0), point(0.6, 0.97, 0.04)], 0.05)
        self.assertEqual((at["refusals"], at["scale"]), (0.97, 0.6))

    def test_an_adapter_with_no_point_of_its_own_reads_as_the_base(self):
        at = auto.at_budget([point(0, 0.95, 0), {"scale": 1, "kld": 0.0, "kld_seq": 0.2}], 0.05)
        self.assertEqual((at["refusals"], at["scale"]), (0.95, 0))


class Refining(unittest.TestCase):
    """a rebuild of the best's own recipe has to earn the place, as a different recipe does."""

    def report(self, reported):
        run = auto.Auto.__new__(auto.Auto)
        c = cand(18, weight=0.744)
        best = {"key": "L18", "cand": cand(18, weight=0.875), "round": 1, "refusals": 0.180, "n": 300}
        sweep = [point(0, 0.97, 0), point(1, reported, 0.044)]
        run.a = SimpleNamespace(max_kld=0.05, think_prompts=0)
        run.build = lambda d, cands: cands
        run.score = lambda *args, **kw: {"L18.gguf": sweep}
        run.think_score = lambda *args: {}
        run.promoted = []
        run.promote = run.promoted.append
        with tempfile.TemporaryDirectory() as d:
            got = run.report(Path(d), {"cand": c}, best, 300, 2)
        return got, best, run.promoted

    def test_a_worse_rebuild_leaves_the_best_in_place(self):
        (kept, outcome), best, promoted = self.report(0.267)
        self.assertEqual((kept, outcome, promoted), (best, "kept", []))

    def test_a_better_rebuild_replaces_it_and_keeps_the_round_it_was_found_in(self):
        (kept, outcome), _, promoted = self.report(0.150)
        self.assertEqual((kept["refusals"], kept["round"], outcome, len(promoted)), (0.150, 1, "refined", 1))


class Candidates(unittest.TestCase):
    def test_names_are_file_stems(self):
        self.assertEqual(auto.key(cand(33)), "L33")
        self.assertEqual(auto.key(cand(32.5, pcs=8, residual=True)), "L32.5-pcs8-res")

    def test_the_first_round_covers_every_middle_and_late_layer(self):
        # on qwen3.8 a grid 5 layers apart missed L33, and a first-token screen of the rest
        # ranked the best layers 36th and 38th of 60
        indices, step = auto.grid(48)
        self.assertEqual(indices, list(range(14, 44)))
        self.assertEqual(step, 1.0)

    def test_neighbours_vary_one_thing_and_pcs_steps_to_its_nearest_counts(self):
        got = {auto.key(c) for c in auto.proposals([cand(33, pcs=8)], 1, 48, residual=True)}
        self.assertEqual(got, {"L32-pcs8", "L34-pcs8", "L33-pcs4", "L33-pcs16", "L33-pcs8-res"})

    def test_a_residual_member_only_moves_its_layer(self):
        # any other change to it needs a capture of its own
        got = {auto.key(c) for c in auto.proposals([cand(33, residual=True)], 1, 48, residual=True)}
        self.assertEqual(got, {"L32-res", "L34-res"})

    def test_every_member_gets_a_residual_and_layers_stay_in_range(self):
        got = {auto.key(c) for c in auto.proposals([cand(47), cand(20)], 1, 48, residual=True)}
        self.assertTrue({"L47-res", "L20-res"} <= got)
        self.assertNotIn("L48", got)
        self.assertFalse(any(k.endswith("-res") for k in
                             (auto.key(c) for c in auto.proposals([cand(33)], 1, 48, residual=False))))

    def test_each_member_s_moves_come_together_in_rank_order_cheapest_first(self):
        # layers share one pass over the weights; a residual needs a filter and a capture
        got = [auto.key(c) for c in auto.proposals([cand(33), cand(20, residual=True), cand(20)], 1, 48,
                                                   residual=True)]
        self.assertEqual(got, ["L32", "L34", "L33-pcs4", "L33-res", "L19-res", "L21-res", "L19", "L21",
                               "L20-pcs4", "L20-res"])

    def test_a_full_beam_of_one_shape_still_tries_the_leader_s_shapes(self):
        # qwen3.8 round 2: four plain members within noise took 4 of 9 slots, and their 8 layer
        # moves left no room for a pcs count or a residual
        st = {"step": 4.8, "beam": [cand(34), cand(19), cand(24), cand(14)], "seen": []}
        a = auto.parse_args(["-m", "model.gguf"])
        got = [auto.key(c) for c in auto.next_candidates(st, 48, a)[0]]
        self.assertEqual(len(got), a.max_candidates)
        self.assertIn("L34-pcs4", got)
        self.assertIn("L34-res", got)

    def test_a_changed_shape_is_swept_wide(self):
        props = {auto.key(c): c for c in auto.proposals([cand(33)], 1, 48, residual=True)}
        self.assertTrue(props["L33-res"]["wide"])
        self.assertTrue(props["L33-pcs4"]["wide"])
        self.assertFalse(props["L34"].get("wide"))

    def test_indices_are_kept_to_quarter_layers(self):
        got = {c["index"] for c in auto.proposals([cand(33)], 0.6, 48, residual=False)}
        self.assertEqual(got, {32.5, 33.5, 33})


def ranked(*rows):
    return [{"refusals": r, "kl": 0.1, "cand": c} for c, r in rows]


class Beam(unittest.TestCase):
    """the beam keeps one leader per shape (pcs, residual), and more of a shape only
    while they are within a standard error of the leader: closer is noise."""

    def test_near_ties_of_one_shape_stay_and_a_distant_one_goes(self):
        got = auto.select_beam(ranked((cand(34), 0.10), (cand(33), 0.11), (cand(30), 0.30)), n=128, size=3)
        self.assertEqual([auto.key(c) for c in got], ["L34", "L33"])

    def test_every_shape_keeps_its_best(self):
        got = auto.select_beam(ranked((cand(34), 0.05), (cand(33), 0.30), (cand(33, pcs=8), 0.30),
                                      (cand(34, residual=True), 0.32)), n=400, size=3)
        self.assertEqual([auto.key(c) for c in got], ["L34", "L33-pcs8", "L34-res"])

    def test_a_shape_s_best_comes_before_near_ties_of_another(self):
        # round 2 on qwen3.8: L34 and L38-res took the last slots, so pcs4 + residual never came up
        got = auto.select_beam(ranked((cand(38), 0.081), (cand(34), 0.094), (cand(34, residual=True), 0.095),
                                      (cand(38, residual=True), 0.104), (cand(38, pcs=4), 0.111)), n=128, size=4)
        self.assertEqual([auto.key(c) for c in got], ["L38", "L34", "L34-res", "L38-pcs4"])

    def test_the_size_caps_it(self):
        rows = [(cand(30 + i), 0.10) for i in range(6)]
        self.assertEqual(len(auto.select_beam(ranked(*rows), n=32, size=4)), 4)

    def test_rank_forgets_the_wide_sweep_once_a_weight_is_found(self):
        sweeps = {"L33-res.gguf": [point(0, 1.0, 0), point(1, 0.2, 0.2)]}
        got = auto.rank([{**cand(33, residual=True), "wide": True}], sweeps, 0.1)
        self.assertNotIn("wide", got[0]["cand"])


class Halving(unittest.TestCase):
    """the first rounds try every index with every first pcs count, then keep the better
    half by rank each round and whatever is within noise of it, on twice the prompts, until
    no more than a beam is left."""

    def test_round_one_crosses_every_index_with_the_first_pcs_counts(self):
        st = auto.first_state(auto.parse_args(["-m", "model.gguf", "--indices", "16.5,33"]), 48)
        self.assertEqual({auto.key(c) for c in st["candidates"]}, {"L16.5", "L16.5-pcs8", "L33", "L33-pcs8"})
        self.assertTrue(all(c["wide"] for c in st["candidates"]))
        self.assertTrue(st["halving"])

    def test_qwen_round_2_keeps_the_layers_near_33_the_beam_dropped(self):
        # a standard error on 256 prompts cut L31.5 and L34 by 0.005, so pcs and a residual
        # were only ever tried on L16.5
        rows = ranked((cand(16.5), 0.098), (cand(31.5), 0.129), (cand(19), 0.129), (cand(34), 0.133),
                      (cand(14), 0.147), (cand(24), 0.184), (cand(21.5), 0.195), (cand(36.5), 0.222),
                      (cand(26.5), 0.680))
        self.assertNotIn("L31.5", [auto.key(c) for c in auto.select_beam(rows, n=256, size=4)])
        cands, beam = auto.halve(rows, n=256, size=4)
        self.assertEqual([auto.key(c) for c in cands[:5]], ["L16.5", "L31.5", "L19", "L34", "L14"])
        self.assertEqual(beam, [])

    def test_residuals_join_once_few_survive_each_at_its_survivor_s_weight(self):
        # each needs a filter and a capture of its own, about 10 minutes on qwen3.8
        rows = ranked(*((cand(14 + i), i / 100) for i in range(2 * auto.RESIDUAL_SURVIVORS + 1)))
        cands, _ = auto.halve(rows, n=32, size=1)
        self.assertFalse(any(c["residual"] for c in cands))
        rows = ranked((cand(33, pcs=8, weight=1.3), 0.1), (cand(16.5), 0.12), (cand(40), 0.5))
        cands, _ = auto.halve(rows, n=128, size=1)
        self.assertEqual([auto.key(c) for c in cands], ["L33-pcs8", "L16.5", "L33-pcs8-res", "L16.5-res"])
        self.assertEqual(cands[2]["weight"], 1.3)
        self.assertTrue(cands[2]["wide"])
        self.assertFalse(cands[0].get("wide"))

    def test_a_residual_tried_before_is_not_tried_again(self):
        rows = ranked((cand(33, pcs=8), 0.1), (cand(16.5), 0.12), (cand(40), 0.5))
        cands, _ = auto.halve(rows, n=128, size=1, seen={"L33-pcs8-res"})
        self.assertEqual([auto.key(c) for c in cands], ["L33-pcs8", "L16.5", "L16.5-res"])

    def test_a_tie_with_the_last_survivor_survives(self):
        # qwen3.8 round 1: L34-pcs8 and L38 both refused 14 of 64, and the kl split them
        rows = ranked((cand(14, pcs=8, residual=True), 0.109), (cand(34, residual=True), 0.116),
                      (cand(38, residual=True), 0.219), (cand(34, pcs=8, residual=True), 0.219),
                      (cand(43), 0.328), (cand(29), 0.750))
        cands, _ = auto.halve(rows, n=64, size=1)
        self.assertEqual([auto.key(c) for c in cands], ["L14-pcs8-res", "L34-res", "L38-res", "L34-pcs8-res"])

    def test_qwen_round_3_keeps_l33_pcs8_within_noise_of_the_cut(self):
        # it missed the better half by 2 of 128 prompts, so its residual (the manual recipe)
        # was never built
        rows = ranked((cand(14, pcs=8), 0.102), (cand(18, pcs=8), 0.109), (cand(31, pcs=8), 0.109),
                      (cand(15, pcs=8), 0.117), (cand(20, pcs=8), 0.133), (cand(16, pcs=8), 0.141),
                      (cand(16), 0.141), (cand(20), 0.148), (cand(33, pcs=8), 0.163), (cand(34), 0.164),
                      (cand(19), 0.164), (cand(31), 0.180), (cand(36), 0.195), (cand(17, pcs=8), 0.197),
                      (cand(38), 0.320), (cand(39), 0.320))
        got = [auto.key(c) for c in auto.halve(rows, n=128, size=4)[0]]
        self.assertIn("L33-pcs8", got)
        self.assertNotIn("L36", got)
        self.assertNotIn("L38", got)

    def test_lfm3_s_last_halving_round_gets_the_top_eight_and_their_residuals(self):
        # within noise of the cut, 16 of 17 survived 256 prompts, so at most 8 never came about
        # and the search left halving without a residual tried
        scores = (("L9-pcs8", 0.070), ("L15-pcs8", 0.074), ("L16", 0.074), ("L14", 0.074), ("L9", 0.082),
                  ("L16-pcs8", 0.082), ("L14-pcs8", 0.086), ("L13", 0.086), ("L10-pcs8", 0.086), ("L12-pcs8", 0.086),
                  ("L17-pcs8", 0.086), ("L8-pcs8", 0.090), ("L15", 0.090), ("L13-pcs8", 0.102), ("L12", 0.102),
                  ("L11-pcs8", 0.102), ("L10", 0.105))
        rows = ranked(*((hierophant_config.parse_recipe(k) | {"weight": 1.0}, r) for k, r in scores))
        a = auto.parse_args(["-m", "model.gguf"])
        nxt, beam = auto.plan(rows, True, n=256, pool=402, seen=set(), a=a)
        self.assertEqual(beam, [])
        top = [k for k, _ in scores[:auto.RESIDUAL_SURVIVORS]]
        self.assertEqual([auto.key(c) for c in nxt], top + [f"{k}-res" for k in top])

    def test_before_the_last_round_many_survivors_get_no_residuals(self):
        rows = ranked(*((cand(14 + i), 0.08 + i / 1000) for i in range(17)))
        a = auto.parse_args(["-m", "model.gguf"])
        nxt, _ = auto.plan(rows, True, n=128, pool=402, seen=set(), a=a)
        self.assertGreater(len(nxt), auto.RESIDUAL_SURVIVORS)
        self.assertFalse(any(c["residual"] for c in nxt))

    def test_a_last_round_that_fits_the_beam_ends_halving(self):
        rows = ranked((cand(14), 0.05), (cand(15), 0.06), (cand(16), 0.5), (cand(17), 0.6))
        a = auto.parse_args(["-m", "model.gguf"])
        self.assertEqual(auto.plan(rows, True, n=256, pool=402, seen=set(), a=a)[0], [])

    def test_it_ends_at_full_prompts_however_many_tie(self):
        # more prompts would split a tie, and there are no more: a model refusing nothing
        # anywhere would halve forever
        rows = ranked(*((cand(30 + i, residual=True), 0.0) for i in range(6)))
        a = auto.parse_args(["-m", "model.gguf", "--beam", "2"])
        nxt, beam = auto.plan(rows, True, n=128, pool=402, seen=set(), a=a)
        self.assertEqual(len(nxt), 6)
        self.assertEqual(beam, [])
        nxt, beam = auto.plan(rows, True, n=402, pool=402, seen=set(), a=a)
        self.assertEqual(nxt, [])
        self.assertEqual(beam, auto.select_beam(rows, 402, 2))

    def test_it_ends_once_the_better_half_fits_the_beam(self):
        rows = ranked(*((cand(30 + i), 0.1 + i / 100) for i in range(7)))
        cands, beam = auto.halve(rows, n=4000, size=4)
        self.assertEqual(cands, [])
        self.assertEqual([auto.key(c) for c in beam], ["L30", "L31", "L32", "L33"])


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
    """a leader must refuse less than the best with thinking off, on the report's prompts,
    and not refuse more with short thinking by more than its fewer prompts can tell."""

    def test_a_leader_refusing_far_more_with_thinking_on_is_kept_out(self):
        self.assertFalse(auto.beats({"refusals": 0.1, "think": 0.4, "think_n": 64},
                                    {"refusals": 0.2, "think": 0.2, "think_n": 64}))

    def test_thinking_off_decides(self):
        self.assertFalse(auto.beats({"refusals": 0.1, "think": 0.15, "think_n": 64},
                                    {"refusals": 0.05, "think": 0.3, "think_n": 64}))

    def test_a_short_thinking_gap_within_its_noise_does_not_keep_a_leader_out(self):
        # qwen3.8: L31-pcs8-res's short thinking went from 0.047 to 0.094 between two
        # rounds on the same 64 prompts, and 0.094 against 0.078 decided a promotion
        self.assertTrue(auto.beats({"refusals": 0.09, "think": 0.125, "think_n": 64},
                                   {"refusals": 0.107, "think": 0.078, "think_n": 64}))

    def test_without_thinking_scores_thinking_off_decides(self):
        self.assertTrue(auto.beats({"refusals": 0.1}, {"refusals": 0.2}))
        self.assertFalse(auto.beats({"refusals": 0.2}, {"refusals": 0.2}))


def run_auto(out, *args, indices=("--indices", "15,20"), command="build"):
    cmd = [CLI, command, TIED, "--out", out, *indices,
           "--first-prompts", "4", "--beam", "2", "--max-candidates", "4", "--train-limit", "40",
           "--max-kld", LOOSE_KLD, "--max-tokens", "8", "--kld-ctx", "128", "--think-limit", "2",
           "--think-max-tokens", "16", "--think-prompts", "2", "--think-budget", "4", "-np", "2",
           "--report-prompts", REPORT_PROMPTS,
           "--server", str(LLAMA_SERVER), *args]
    return raw(cmd)


class Auto(unittest.TestCase):
    """two rounds, then a rerun that only continues, then --status."""

    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir(cls.addClassCleanup) / "run"
        # the tiny model's four round 1 candidates tie, so a beam of four refines from round 2
        refine = ("--beam", "4", "--max-candidates", "6")
        cls.first = run_auto(cls.out, "--rounds", "2", "--progress", *refine)
        cls.again = run_auto(cls.out, "--rounds", "2", *refine)
        cls.status = run_auto(cls.out, command="status")

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
        self.assertIn("full replies, thinking on: refusals", text)
        self.assertIn("by markers", text)
        self.assertIn("by the base as judge", text)
        self.assertIn("no reasoning seen", text)
        self.assertIn("held-out, short thinking: refusals", text)
        side = json.loads((self.out / "best.json").read_text())
        self.assertIn(f"built with llama.cpp {side['llama_cpp']}\n", text)

    def test_the_sidecar_keeps_the_final_check_s_scores(self):
        final = json.loads((self.out / "best.json").read_text())["eval"]["final"]
        self.assertEqual(sorted(final), ["loops", "markers", "n", "reasoned", "refusals", "unfinished"])

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
        self.assertIn("harmful [", self.first.stdout)
        # a round's builds share one bar, which each build's sites advance
        self.assertRegex(self.first.stdout, r"round-1 builds \[#{24}\] (\d+)/\1 100%")
        self.assertNotIn("sites [", self.first.stdout)
        self.assertNotIn("hierophant adapter (log:", self.first.stdout)
        self.assertIn("\r", self.first.stdout)
        # a round draws one bar, its stages advanced by what each one reports, and it ends on
        # a line of its own; the bars of what runs inside it are not drawn over it
        self.assertRegex(self.first.stdout, r"round 1 select-wide \[[#-]{24}\] 1/5 [\d.]+% .*\| ")
        # its detail names the candidates being built, not the stage again
        self.assertRegex(self.first.stdout, r"round 1 builds \[[#-]{24}\] 0/5 [\d.]+% [^|\r\n]*\| L[\d.]+(,L[\d.]+)* sites")
        self.assertNotRegex(self.first.stdout, r"\| round-\d+ builds")
        self.assertRegex(self.first.stdout, r"round 1 \[#{24}\] 5/5 100\.0%")
        self.assertNotIn("base replies [", self.first.stdout)
        for rnd in (1, 2):
            shown = [float(m) for m in re.findall(rf"round {rnd} [^\[\r\n]*\[[#-]+\] \d+/\d+ ([\d.]+)%",
                                                  self.first.stdout)]
            self.assertGreater(len(shown), 4)
            self.assertEqual(shown, sorted(shown), f"round {rnd} went backwards")
        log = (self.out / "logs" / "round-1-report.log").read_text()
        self.assertNotIn("\r", log)
        self.assertRegex(log, r"base replies \[#{24}\] (\d+)/\1 100%")

    def test_selection_replies_only_at_the_budget_and_skips_the_reply_logp(self):
        cmds = (self.out / "logs" / "round-1-wide.log").read_text()
        self.assertIn("--at-budget", cmds)
        self.assertIn("--logp-tokens 0", cmds)
        for points in json.loads((self.out / "round-1" / "select-wide.json").read_text()).values():
            self.assertEqual(len([p for p in points if "refusals" in p and p["scale"]]), 1, points)
            self.assertFalse(any("reply_logp" in p for p in points))

    def test_the_report_keeps_the_reply_logp_of_the_adapter_it_promotes(self):
        cmds = (self.out / "logs" / "round-1-report.log").read_text()
        self.assertIn("--at-budget", cmds)
        self.assertNotIn("--logp-tokens", cmds)
        self.assertTrue(any("reply_logp" in p for p in json.loads((self.out / "best.json").read_text())["eval"]["sweep"]))

    def test_every_round_reports_on_the_same_held_out_prompts(self):
        for rnd in (1, 2):
            r = json.loads((self.out / f"round-{rnd}" / "round.json").read_text())
            self.assertEqual(r["best"]["n"], REPORT_PROMPTS)

    def test_round_two_varies_the_leaders(self):
        r1, r2 = (json.loads((self.out / f"round-{k}" / "round.json").read_text()) for k in (1, 2))
        self.assertTrue({x["cand"]["index"] for x in r2["candidates"]} - {x["cand"]["index"] for x in r1["candidates"]})
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
        # an unrecorded round has not been pruned; recording it is what this test undoes
        first = run_auto(out, "--rounds", "1", "--verbose", "--keep-rounds")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        # --verbose streams every command and its own progress
        self.assertIn("$ ", first.stdout)
        self.assertIn("base replies: ", first.stdout)
        for name in ("state.json", "history.txt"):
            (out / name).unlink()
        again = run_auto(out, "--rounds", "1", "--server", "/nonexistent/llama-server")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn("== round 1: which layer", again.stdout)
        self.assertNotIn("hierophant adapter", again.stdout)
        # nothing measured again, the kl over wikitext included
        evals = [log for log in (out / "logs").glob("round-1*.log") if "hierophant_eval.py" in log.read_text()]
        self.assertTrue(evals)
        for log in evals:
            self.assertRegex(log.read_text().rsplit("cache: ", 1)[-1], r"^\d+ reused, 0 new", log.name)
        self.assertEqual((out / "history.txt").read_text(), first_history(first))


class ScreenLayers(unittest.TestCase):
    """--screen-layers: every layer of the grid's range scored on the first token alone (the
    base's own first reply token, where the kl reaches the budget), and only the best few
    scored in full in round 1."""

    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir(cls.addClassCleanup) / "run"
        cls.proc = run_auto(cls.out, "--rounds", "1", "--screen-layers", "--screen-keep", "2", "--screen-prompts", "4",
                            "--no-think-check", indices=())

    def test_it_runs(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_every_layer_of_the_range_is_screened_with_each_first_pcs_count_without_replies(self):
        # on qwen3.8 8 pcs helped at L33 and hurt at L34, so a plain screen would miss L33-pcs8
        n_layer = Reader(self.out / "dirs.gguf").tensors["harmful_mean"]["ne"][1]
        lo, hi = auto.grid(n_layer)[0][0], auto.grid(n_layer)[0][-1]
        screen = json.loads((self.out / "round-1" / "screen.json").read_text())
        self.assertEqual(sorted(screen), sorted(f"L{i}{s}.gguf" for i in range(lo, hi + 1) for s in ("", "-pcs8")))
        for points in screen.values():
            self.assertFalse(any("refusals" in p for p in points))
            self.assertEqual(len([p for p in points if "reply_logp" in p]), 2, points)

    def test_round_one_scores_the_best_screened_in_full(self):
        screen = json.loads((self.out / "round-1" / "screen.json").read_text())
        best = sorted(screen, key=lambda k: auto.at_budget(screen[k], LOOSE_KLD, "reply_logp")["reply_logp"])[:2]
        r1 = json.loads((self.out / "round-1" / "round.json").read_text())
        self.assertEqual(sorted(auto.key(x["cand"]) + ".gguf" for x in r1["candidates"]), sorted(best))
        self.assertIn("screen", self.proc.stdout)

    def test_screen_only_ranks_beside_round_one_and_leaves_the_run_alone(self):
        state = (self.out / "state.json").read_text()
        proc = run_auto(self.out, "--screen-prompts", "4", indices=(), command="screen")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual((self.out / "state.json").read_text(), state)
        self.assertNotIn("== round", proc.stdout)
        r1 = json.loads((self.out / "round-1" / "round.json").read_text())
        for x in r1["candidates"]:
            self.assertRegex(proc.stdout, rf"\n{re.escape(auto.key(x['cand']))} +-?[\d.]+ +[\d.]+ +{x['refusals']:.3f}\n")
        self.assertTrue((self.out / "screen" / "screen.json").exists())


class HalvingRun(unittest.TestCase):
    """with a beam of one, round 2 still halves: round 1's better half (and its ties) and,
    as few survive, their residuals, on twice the prompts."""

    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir(cls.addClassCleanup) / "run"
        cls.proc = run_auto(cls.out, "--rounds", "2", "--beam", "1", "--no-think-check")

    def test_round_two_is_what_halving_round_one_gives(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)
        self.assertIn("== round 1: which layer and pcs (4 candidates, 4 prompts each)", self.proc.stdout)
        self.assertIn("== round 2: the better half of round 1", self.proc.stdout)
        r1 = json.loads((self.out / "round-1" / "round.json").read_text())
        r2 = json.loads((self.out / "round-2" / "round.json").read_text())
        want = {auto.key(c) for c in auto.halve(r1["candidates"], n=r1["n"], size=1)[0]}
        got = {auto.key(x["cand"]) for x in r2["candidates"]}
        # a residual is dropped where too few prompts are still refused, as the tiny model's are
        self.assertEqual({k for k in want if not k.endswith("-res")}, {k for k in got if not k.endswith("-res")})
        self.assertTrue(got <= want)
        self.assertEqual(r2["n"], 8)
        self.assertTrue(json.loads((self.out / "state.json").read_text())["halving"])


class DefaultOut(unittest.TestCase):
    """without --out, each model gets its own directory under ./out."""

    @staticmethod
    def out(**fields):
        return auto.default_out(SimpleNamespace(**{"model": None, "hf_repo": None, "hf_file": None, **fields}))

    def test_a_repo_and_quant(self):
        self.assertEqual(self.out(hf_repo="unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS"),
                         Path("out") / "unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS")

    def test_a_repo_and_file(self):
        self.assertEqual(self.out(hf_repo="org/m-GGUF", hf_file="Q8_0/m-Q8_0-00001-of-00002.gguf"),
                         Path("out") / "org--m-GGUF--m-Q8_0-00001-of-00002")

    def test_a_bare_repo(self):
        self.assertEqual(self.out(hf_repo="org/m-GGUF"), Path("out") / "org--m-GGUF")

    def test_a_local_file(self):
        self.assertEqual(self.out(model="/models/tiny-Q8_0.gguf"), Path("out") / "tiny-Q8_0")


class Streaming(unittest.TestCase):
    """a step's output reaches its log while the step still runs, not when it exits."""

    def test_script_output_is_logged_before_the_script_exits(self):
        d = tmpdir(self.addCleanup)
        slow = d / "slow.py"
        slow.write_text(f"import time\nprint('started')\ntime.sleep({SLOW_SECONDS})\n")
        r = auto.Auto(auto.parse_args(["-m", str(TIED), "--out", str(d / "out"), "--indices", "1"]))
        worker = threading.Thread(target=r.script, args=("slow", slow))
        worker.start()
        self.addCleanup(worker.join)
        log = d / "out" / "logs" / "slow.log"
        deadline = time.time() + SLOW_SECONDS / 2
        while time.time() < deadline and "started" not in (log.read_text() if log.exists() else ""):
            time.sleep(0.2)
        self.assertIn("started", log.read_text())


class ShortThinking(unittest.TestCase):
    """short thinking is scored where the report scored each adapter: scale 1 against a
    report at 0.97 compared two different strengths. the tiny model's report lands on scale
    0, so this drives think_score with an eval that answers at whatever scale it is asked."""

    def run_think(self, reasoning=""):
        d = tmpdir(self.addCleanup)
        r = auto.Auto(auto.parse_args(["-m", "model.gguf", "--out", str(d / "run")]))
        self.asked = []

        def fake_eval(label, *args):
            flags = list(map(str, args))
            scale = float(flags[flags.index("--scales") + 1])
            names = [Path(flags[i + 1]).name for i, x in enumerate(flags) if x == "--lora"]
            self.asked.append((scale, names))
            replies = [{"reply": "yes", "reasoning": reasoning}]
            Path(flags[flags.index("--results") + 1]).write_text(json.dumps({n: [
                {"scale": 0, "refusals": 0.9, "replies": replies}, {"scale": scale, "refusals": scale / 10,
                                                                    "replies": replies}] for n in names}))

        r.eval = fake_eval
        scales = {d / "a.gguf": 0.97, d / "b.gguf": 1.03, d / "c.gguf": 0.97}
        return r.think_score("x", d, scales, 4), d

    def test_each_adapter_is_scored_at_its_own_scale_sharing_a_run_with_its_equals(self):
        got, d = self.run_think()
        self.assertEqual(self.asked, [(0.97, ["a.gguf", "c.gguf"]), (1.03, ["b.gguf"])])
        self.assertAlmostEqual(got["a.gguf"]["think"], 0.097)
        self.assertAlmostEqual(got["b.gguf"]["think"], 0.103)
        self.assertEqual(sorted(json.loads((d / "think.json").read_text())), ["a.gguf", "b.gguf", "c.gguf"])

    def test_replies_without_any_reasoning_say_so(self):
        # lfm3 never reasoned, so its short thinking scores were plain replies
        self.assertEqual(self.run_think()[0]["a.gguf"]["think_reasoned"], 0)
        self.assertEqual(self.run_think("because")[0]["a.gguf"]["think_reasoned"], 1)


class ResidualRetry(unittest.TestCase):
    """a residual is captured under the adapter at its weight before that, and on lfm3 L16-res
    scored 0.211 at a capture from weight 1.03 and 0.080 once recaptured near where it lands."""

    def setUp(self):
        self.d = tmpdir(self.addCleanup)
        self.r = auto.Auto(auto.parse_args(["-m", "model.gguf", "--out", str(self.d / "run")]))
        self.built = []
        self.r.build = lambda d, cands: self.built.extend(auto.key(c) for c in cands) or cands
        self.r.select = lambda d, cands, n: {f"{auto.key(c)}.gguf": [point(1, 0.08, 0.05)] for c in cands}

    def captured(self, name, weight):
        d = self.r.root / "residual" / name
        d.mkdir(parents=True)
        (d / "weight.json").write_text(json.dumps({"weight": weight}))

    def settle(self, *rows):
        return self.r.settle_residuals(self.r.out / "round-5", ranked(*rows), 402)

    def test_a_residual_far_from_its_capture_is_scored_again_and_reranked(self):
        self.captured("L16", 1.03)
        got = self.settle((cand(14), 0.09), (cand(16, residual=True, weight=0.741), 0.211))
        self.assertEqual(self.built, ["L16-res"])
        self.assertEqual([(auto.key(x["cand"]), x["refusals"]) for x in got], [("L16-res", 0.08), ("L14", 0.09)])

    def test_a_retry_names_its_round_once_in_its_builds_and_its_log(self):
        self.captured("L16", 1.03)
        steps, labels = [], []
        self.r.a.progress = False
        self.r.build = lambda d, cands: auto.Auto.build(self.r, d, cands)
        self.r.select = lambda d, cands, n: auto.Auto.select(self.r, d, cands, n)
        self.r.residual_dirs = lambda c: "dirs.gguf"
        self.r.make_adapter = lambda step, *args: steps.append(step)
        self.r.score = lambda label, *args, **kw: labels.append(label) or {"L16-res.gguf": [point(1, 0.08, 0.05)]}
        self.settle((cand(16, residual=True, weight=0.741), 0.211))
        self.assertEqual((set(steps), labels), ({"round-5-residuals"}, ["round-5-residuals"]))

    def test_a_residual_near_its_capture_stands(self):
        self.captured("L16", 0.78)
        got = self.settle((cand(16, residual=True, weight=0.741), 0.211))
        self.assertEqual(self.built, [])
        self.assertEqual(got[0]["refusals"], 0.211)

    def test_without_any_residual_nothing_runs(self):
        self.settle((cand(14), 0.09))
        self.assertEqual(self.built, [])

    def test_a_residual_with_too_few_prompts_for_a_capture_is_left_alone(self):
        self.settle((cand(16, residual=True, weight=0.741), 0.211))
        self.assertEqual(self.built, [])


class Defaults(unittest.TestCase):
    def test_the_kl_budget(self):
        # lfm3: 0.1 bought 0.04 fewer refusals than 0.05, at twice the kl
        self.assertEqual(auto.parse_args(["-m", "model.gguf"]).max_kld, 0.05)


class Accepting(unittest.TestCase):
    """a build fills an empty config and records a recipe's measurements for its quant, and
    never replaces an accepted recipe."""

    def args(self, *extra):
        return auto.parse_args(["--hf-repo", "org/m-GGUF:Q8_0", *extra])

    def test_a_converged_search_fills_an_empty_config(self):
        st = {"best": {"key": "L3"}, "unchanged": auto.PATIENCE}
        self.assertTrue(auto.should_accept(self.args(), st, None))
        self.assertTrue(auto.should_accept(self.args(), st, {"quant": {"Q8_0": {}}}))

    def test_a_search_stopped_early_is_not_accepted(self):
        self.assertFalse(auto.should_accept(self.args(), {"best": {"key": "L3"}, "unchanged": 0}, None))

    def test_a_search_never_replaces_an_accepted_recipe(self):
        st = {"best": {"key": "L3"}, "unchanged": auto.PATIENCE}
        self.assertFalse(auto.should_accept(self.args(), st, {"quant": {"Q8_0": {"recipe": {"index": 9}}}}))

    def test_a_recipe_build_records_its_quant(self):
        args = self.args("--recipe", "L9")
        self.assertTrue(auto.should_accept(args, {"best": {"key": "L9"}, "unchanged": 0}, {"quant": {}}))

    def test_no_accept_turns_both_off(self):
        st = {"best": {"key": "L3"}, "unchanged": auto.PATIENCE}
        self.assertFalse(auto.should_accept(self.args("--no-accept"), st, None))
        self.assertFalse(auto.should_accept(self.args("--no-accept", "--recipe", "L9"), st, {}))

    def test_a_local_file_has_no_config_to_fill(self):
        args = auto.parse_args(["-m", "model.gguf"])
        self.assertFalse(auto.should_accept(args, {"best": {"key": "L3"}, "unchanged": auto.PATIENCE}, None))

    def test_a_bad_recipe_fails_before_anything_runs(self):
        with self.assertRaises(ConfigError):
            self.args("--recipe", "best")


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


class Prune(unittest.TestCase):
    """only what no later round will read: finished rounds' adapters and lost residuals."""

    def setUp(self):
        self.root = tmpdir(self.addCleanup)
        self.state(self.root, round=2, candidates=[cand(33, pcs=8, residual=True)], best="L15-pcs8")

    @staticmethod
    def state(d, round, candidates=(), best=None):
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps({"round": round, "candidates": list(candidates), "beam": [],
                                                  "best": {"key": best} if best else None}))

    def residual(self, name):
        d = self.root / "residual" / name
        return [touch(d / "dirs.gguf"), touch(d / "under.gguf"), touch(d / "weight.json")]

    def test_a_recorded_round_s_adapters_are_dead_and_its_scores_are_not(self):
        dead = touch(self.root / "round-1" / "L3.gguf")
        built = touch(self.root / "round-1" / "L3.gguf.built")
        report = touch(self.root / "round-2" / "report" / "incumbent.gguf")
        touch(self.root / "round-1" / "round.json")
        touch(self.root / "round-2" / "report" / "L3.json")
        self.assertEqual(sorted(auto.dead_files(self.root)), sorted([dead, built, report]))

    def test_the_round_under_way_is_left_alone(self):
        touch(self.root / "round-3" / "L3.gguf")
        self.assertEqual(auto.dead_files(self.root), [])

    def test_the_best_and_the_directions_are_left_alone(self):
        for name in ("best.gguf", "dirs.gguf", "cache.jsonl"):
            touch(self.root / name)
        self.assertEqual(auto.dead_files(self.root), [])

    def test_a_lost_residual_goes_with_its_marker_so_it_is_captured_again(self):
        dirs, under, weight = self.residual("L20")
        self.assertEqual(sorted(auto.dead_files(self.root)), sorted([dirs, under, weight]))

    def test_a_residual_still_in_the_running_keeps_its_directions(self):
        _, under, _ = self.residual("L33-pcs8")
        self.assertEqual(auto.dead_files(self.root), [under])

    def test_the_best_s_residual_keeps_its_directions(self):
        _, under, _ = self.residual("L15-pcs8")
        self.assertEqual(auto.dead_files(self.root), [under])

    def test_an_unfinished_capture_is_left_alone(self):
        touch(self.root / "residual" / "L20" / "under.gguf")
        self.assertEqual(auto.dead_files(self.root), [])

    def test_a_named_run_keeps_its_own_residuals_alive(self):
        self.state(self.root / "runs" / "other", round=1, best="L20")
        dead = touch(self.root / "runs" / "other" / "round-1" / "L20.gguf")
        _, under, _ = self.residual("L20")
        self.assertEqual(sorted(auto.dead_files(self.root)), sorted([dead, under]))


class Finished(unittest.TestCase):
    """the commands that read a finished run, on one run of the tiny model."""

    @classmethod
    def setUpClass(cls):
        root = tmpdir(cls.addClassCleanup)
        cls.out, cls.configs = root / "run", root / "configs"
        cls.configs.mkdir()
        cls.env = {**os.environ, "HIEROPHANT_CONFIGS": str(cls.configs)}
        cls.first = run_auto(cls.out, "--rounds", "1", "--beam", "4", "--max-candidates", "6")
        cls.best = json.loads((cls.out / "state.json").read_text())["best"]

    def cli(self, *args):
        return subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True, check=False, env=self.env)

    def test_it_runs_and_says_it_searched(self):
        self.assertEqual(self.first.returncode, 0, self.first.stdout + self.first.stderr)
        self.assertIn("== build: searching", self.first.stdout)

    def test_a_finished_round_leaves_no_adapters_behind(self):
        self.assertEqual(list((self.out / "round-1").rglob("*.gguf")), [])
        self.assertTrue((self.out / "round-1" / "round.json").exists())
        self.assertTrue((self.out / "best.gguf").exists())

    def test_accept_writes_the_best_into_a_new_config(self):
        p = self.cli("accept", "org/Tiny-GGUF:Q8_0", "--out", self.out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        config = tomllib.loads((self.configs / "org--Tiny-GGUF.toml").read_text())
        self.assertEqual(config["model"]["name"], "Tiny")
        q = config["quant"]["Q8_0"]
        self.assertEqual(hierophant_config.recipe_name(q["recipe"]), self.best["key"])
        self.assertEqual(q["eval"]["think_n"], self.best["think_n"])

    def test_accept_needs_a_quant(self):
        self.assertNotEqual(self.cli("accept", "org/Tiny-GGUF", "--out", self.out).returncode, 0)

    def test_compare_scores_an_adapter_beside_the_best(self):
        other = self.out / "other.gguf"
        shutil.copy(self.out / "best.gguf", other)
        p = run_auto(self.out, "--lora", other, indices=(), command="compare")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"\nbest +[\d.]+ +[\d.]+ +[\d.]+\n")
        self.assertRegex(p.stdout, r"\nother +[\d.]+ +[\d.]+ +[\d.]+\n")

    def test_serve_runs_the_server_on_the_best_at_its_scale(self):
        fake = tmpdir(self.addCleanup) / "server"
        fake.write_text('#!/bin/sh\necho "$@"\n')
        fake.chmod(0o755)
        p = self.cli("serve", TIED, "--out", self.out, "--server", fake, "--", "--port", "9")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        scale = json.loads((self.out / "best.json").read_text())["scale"] or 1
        self.assertIn(f"--lora-scaled {(self.out / 'best.gguf').absolute()}:{scale:g}", p.stdout)
        self.assertTrue(p.stdout.strip().endswith("--port 9"), p.stdout)

    def test_chat_runs_llama_cli_on_the_best_at_its_scale(self):
        fake = tmpdir(self.addCleanup) / "cli"
        fake.write_text('#!/bin/sh\necho "$@"\n')
        fake.chmod(0o755)
        p = self.cli("chat", TIED, "--out", self.out, "--cli", fake, "--", "--temp", "0")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        scale = json.loads((self.out / "best.json").read_text())["scale"] or 1
        self.assertIn(f"-m {TIED.absolute()} --lora-scaled {(self.out / 'best.gguf').absolute()}:{scale:g}", p.stdout)
        self.assertTrue(p.stdout.strip().endswith("--temp 0"), p.stdout)

    def test_chat_finds_the_real_llama_cli(self):
        self.assertTrue((ROOT / "build" / "bin" / "llama-cli").exists())

    def test_serve_without_a_best_says_so(self):
        p = self.cli("serve", TIED, "--out", tmpdir(self.addCleanup))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("no best.gguf", p.stderr)

    def test_prune_only_lists_unless_told(self):
        keep = self.out / "round-1" / "planted.gguf"
        keep.write_text("x")
        self.addCleanup(keep.unlink, missing_ok=True)
        p = self.cli("prune", TIED, "--out", self.out)
        self.assertIn(str(keep), p.stdout)
        self.assertTrue(keep.exists())
        p = self.cli("prune", TIED, "--out", self.out, "--yes")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertFalse(keep.exists())


class Recipe(unittest.TestCase):
    """a known recipe needs one round, and a named run reuses what its directory has."""

    @classmethod
    def setUpClass(cls):
        cls.out = tmpdir(cls.addClassCleanup) / "run"
        cls.proc = run_auto(cls.out, "--recipe", "L15-pcs8", "--recipe-weight", "1.2", "--no-think-check", indices=())
        cls.named = run_auto(cls.out, "--recipe", "L20", "--name", "other", "--no-think-check", "--keep-rounds",
                             indices=())

    def test_one_candidate_in_one_round(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)
        self.assertIn("== build: following recipe L15-pcs8", self.proc.stdout)
        self.assertIn("(1 candidates", self.proc.stdout)
        self.assertNotIn("== round 2", self.proc.stdout)
        st = json.loads((self.out / "state.json").read_text())
        self.assertEqual(st["best"]["key"], "L15-pcs8")

    def test_a_named_run_reuses_the_capture_and_the_cache(self):
        self.assertEqual(self.named.returncode, 0, self.named.stdout + self.named.stderr)
        self.assertIn("capture: done", self.named.stdout)
        run = self.out / "runs" / "other"
        self.assertTrue((run / "best.gguf").exists())
        self.assertFalse((run / "dirs.gguf").exists())
        self.assertFalse((run / "cache.jsonl").exists())

    def test_keep_rounds_keeps_the_adapters(self):
        self.assertTrue(list((self.out / "runs" / "other" / "round-1").rglob("*.gguf")))


def first_history(proc):
    """history.txt's line for round 1, rebuilt from the run's own output."""
    line = next(x for x in proc.stdout.splitlines() if x.startswith(">> "))
    return "round 1 (4 prompts): " + line.removeprefix(">> ") + "\n"


if __name__ == "__main__":
    unittest.main()
