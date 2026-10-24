"""capture, adapter and info, checked against llama.cpp's own forward pass."""

import math
import subprocess
import sys
import unittest

from common import (
    HARMFUL,
    HARMLESS,
    HIEROPHANT,
    MOE,
    MOE_Q8,
    N_PROMPTS,
    ROOT,
    TIED,
    Reader,
    adapter_direction,
    capture,
    dot,
    layer_means,
    lora_columns,
    refusal_direction,
    run,
    tmpdir,
    unit,
)

PROJECTION = ROOT / "scripts" / "projection.py"


def run_capture(model, out, prompts, *extra):
    run("capture", "-m", model, *prompts, "-o", out, *extra)
    return out

DIRECTION_LAYER = 3


def projections(dirs, r):
    return [dot(r, m) for name in ("harmful_mean", "harmless_mean") for m in layer_means(dirs, name)]


class Info(unittest.TestCase):
    def test_moe_sites_include_experts_and_embeddings(self):
        out = run("info", "-m", MOE)
        self.assertIn("site token_embd.weight", out)
        self.assertIn("site blk.0.ffn_down_exps.weight", out)
        self.assertIn("site blk.5.attn_output.weight", out)
        self.assertIn("13 sites", out)

    def test_tied_embeddings_are_skipped(self):
        out = run("info", "-m", TIED)
        self.assertNotIn("site token_embd.weight", out)
        self.assertIn("token_embd is tied to the output head", out)

    def test_targets_restrict_kinds(self):
        out = run("info", "-m", MOE, "--targets", "attn_output")
        self.assertIn("6 sites", out)
        self.assertNotIn("ffn_down_exps", out)


class Ablation(unittest.TestCase):
    def ablate(self, model, *adapter_args):
        d = tmpdir(self.addCleanup)
        dirs = capture(model, d / "dirs.gguf")
        adapter = d / "adapter.gguf"
        run("adapter", "-m", model, "-d", dirs, "-o", adapter, "-t", 4, *adapter_args)
        return d, dirs, adapter

    def test_global_direction_is_removed_from_every_layer(self):
        d, dirs, adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        r, _ = adapter_direction(adapter)
        before = projections(dirs, r)
        after = projections(capture(MOE, d / "after.gguf", "--lora", adapter), r)
        self.assertGreater(max(map(abs, before)), 0.5)
        self.assertLess(max(map(abs, after)), 1e-3 * max(map(abs, before)))

    def projection_report(self, *adapter_args):
        """rows of scripts/projection.py, for a base capture on fewer prompts than the adapter's."""
        d, _, adapter = self.ablate(MOE, *adapter_args)
        few = ["--harmful", HARMFUL, "--harmless", HARMLESS, "--limit", 3, "-t", 4]
        base = run_capture(MOE, d / "base.gguf", few)
        run_capture(MOE, d / "after.gguf", few, "--lora", adapter)
        cmd = [sys.executable, "-I", str(PROJECTION), base, d / "after.gguf", "--adapter", adapter]
        out = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=True).stdout
        return [line.split() for line in out.splitlines()[1:]], out, adapter, base

    def test_projection_report_reads_the_direction_from_the_adapter(self):
        rows, out, _, _ = self.projection_report("--direction-index", DIRECTION_LAYER)
        left = [float(c.rstrip("%")) for row in rows for c in row[3::3]]
        self.assertLess(max(map(abs, left)), 1.0, out)

    def test_projection_report_follows_per_layer_directions(self):
        rows, out, adapter, base = self.projection_report("--direction-index", "layer")
        for layer in (1, 4):
            r, _ = adapter_direction(adapter, f"blk.{layer}.attn_output.weight")
            expected = dot(r, layer_means(base, "harmful_mean")[layer])
            self.assertAlmostEqual(float(rows[layer][1]), expected, places=3, msg=out)

    def test_quantized_experts_are_ablated_too(self):
        d, dirs, adapter = self.ablate(MOE_Q8, "--direction-index", DIRECTION_LAYER)
        r, _ = adapter_direction(adapter)
        before = projections(dirs, r)
        after = projections(capture(MOE_Q8, d / "after.gguf", "--lora", adapter), r)
        # q8_0 activations in the base matmul are the remaining error
        self.assertLess(max(map(abs, after)), 1e-2 * max(map(abs, before)))

    def test_tied_model_keeps_only_the_embedding_component(self):
        layer = 15
        d, dirs, adapter = self.ablate(TIED, "--direction-index", layer)
        r, _ = adapter_direction(adapter)
        before = projections(dirs, r)
        after = projections(capture(TIED, d / "after.gguf", "--lora", adapter), r)
        self.assertLess(max(map(abs, after)), 0.02 * max(map(abs, before)))

    def test_scale_zero_is_the_base_model(self):
        d, dirs, adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        zero = capture(MOE, d / "zero.gguf", "--lora", adapter, "--lora-scale", 0)
        for name in ("harmful_mean", "harmless_mean"):
            for a, b in zip(layer_means(dirs, name), layer_means(zero, name)):
                self.assertEqual(a, b)

    def test_adapter_layout_matches_llama_cpp(self):
        _, _, adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        r = Reader(adapter)

        def ne(name):  # gguf drops trailing unit dimensions
            return (r.tensors[name]["ne"] + [1, 1])[:3]

        self.assertEqual(r.kv["general.type"], "adapter")
        self.assertEqual(r.kv["adapter.type"], "lora")
        self.assertEqual(r.kv["adapter.lora.alpha"], 1.0)
        self.assertEqual(r.kv["hierophant.sites"], 13)
        self.assertEqual(ne("token_embd.weight.lora_a"), [1, 32000, 1])
        self.assertEqual(ne("token_embd.weight.lora_b"), [1, 288, 1])
        self.assertEqual(ne("blk.0.ffn_down_exps.weight.lora_a"), [768, 1, 4])
        self.assertEqual(ne("blk.0.ffn_down_exps.weight.lora_b"), [1, 288, 4])
        self.assertEqual(ne("blk.0.attn_output.weight.lora_a"), [288, 1, 1])

    def test_curve_limits_layers_and_sets_weights(self):
        _, _, adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER, "--curve", "1.5,2,0.5,1")
        r = Reader(adapter)
        layers = sorted({int(n.split(".")[1]) for n in r.tensors if n.startswith("blk.")})
        self.assertEqual(layers, [1, 2, 3])
        self.assertNotIn("token_embd.weight.lora_a", r.tensors)
        for layer, weight in ((1, 0.5), (2, 1.5), (3, 0.5)):
            _, w = adapter_direction(adapter, f"blk.{layer}.attn_output.weight")
            self.assertAlmostEqual(w, weight, places=4)

    def test_several_indices_in_one_pass_match_separate_runs(self):
        d, dirs, single = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "multi-{}.gguf", "-t", 4,
            "--direction-index", f"2.5,{DIRECTION_LAYER}")
        a, b = Reader(single), Reader(d / f"multi-{DIRECTION_LAYER}.gguf")
        self.assertEqual(sorted(a.tensors), sorted(b.tensors))
        for name in a.tensors:
            self.assertEqual(a.tensor(name), b.tensor(name))
        self.assertEqual(Reader(d / "multi-2.5.gguf").kv["hierophant.direction_index"], 2.5)

    def test_per_layer_joins_a_list_of_indices(self):
        d, dirs, _ = self.ablate(MOE)
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "multi-{}.gguf", "-t", 4,
            "--direction-index", f"layer,{DIRECTION_LAYER}")
        a, b = Reader(d / "adapter.gguf"), Reader(d / "multi-layer.gguf")
        for name in a.tensors:
            self.assertEqual(a.tensor(name), b.tensor(name))
        self.assertEqual(b.kv["hierophant.direction_index"], -1)
        self.assertEqual(Reader(d / f"multi-{DIRECTION_LAYER}.gguf").kv["hierophant.direction_index"], DIRECTION_LAYER)

    def test_several_indices_need_a_placeholder(self):
        d, dirs, _ = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        p = subprocess.run([str(HIEROPHANT), "adapter", "-m", str(MOE), "-d", str(dirs), "-o", str(d / "x.gguf"),
                            "--direction-index", "2,3"], capture_output=True, text=True, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("{}", p.stderr)

    def test_rebase_onto_another_quant_matches_a_direct_build(self):
        d, dirs, f16_adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER, "--curve", "1.5,2,0.5,3")
        direct, rebased = d / "direct.gguf", d / "rebased.gguf"
        run("adapter", "-m", MOE_Q8, "-d", dirs, "-o", direct, "-t", 4, "--direction-index", DIRECTION_LAYER,
            "--curve", "1.5,2,0.5,3")
        run("adapter", "-m", MOE_Q8, "--from-adapter", f16_adapter, "-o", rebased, "-t", 4)
        a, b = Reader(direct), Reader(rebased)
        self.assertEqual(sorted(a.tensors), sorted(b.tensors))
        for name in a.tensors:
            x, y = a.tensor(name), b.tensor(name)
            scale = max(map(abs, x))
            self.assertLess(max(abs(p - q) for p, q in zip(x, y)), 1e-5 * scale, name)
        self.assertEqual(b.kv["hierophant.base_file"], MOE_Q8.name)

    def test_directions_leave_the_harmless_mean_by_default(self):
        _, dirs, adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        harmless = layer_means(dirs, "harmless_mean")[DIRECTION_LAYER]
        r, _ = adapter_direction(adapter)
        self.assertLess(abs(dot(r, harmless)), 1e-5 * math.sqrt(dot(harmless, harmless)))
        self.assertGreater(dot(r, refusal_direction(dirs, DIRECTION_LAYER)), 0.5)
        self.assertTrue(Reader(adapter).kv["hierophant.project_harmless"])

    def test_no_project_harmless_keeps_the_plain_difference_of_means(self):
        _, dirs, adapter = self.ablate(MOE, "--direction-index", DIRECTION_LAYER, "--no-project-harmless")
        r, _ = adapter_direction(adapter)
        self.assertAlmostEqual(dot(r, refusal_direction(dirs, DIRECTION_LAYER)), 1.0, places=5)
        self.assertFalse(Reader(adapter).kv["hierophant.project_harmless"])

    def test_f16_adapter_halves_the_size(self):
        d, dirs, f32 = self.ablate(MOE, "--direction-index", DIRECTION_LAYER)
        f16 = d / "f16.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", f16, "--direction-index", DIRECTION_LAYER, "--type", "f16")
        self.assertLess(f16.stat().st_size, 0.6 * f32.stat().st_size)


class Curves(unittest.TestCase):
    def build(self, *curves):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "a.gguf", "--direction-index", DIRECTION_LAYER, *curves)
        return d / "a.gguf"

    def weight(self, adapter, site):
        return adapter_direction(adapter, site)[1] if site + ".lora_b" in Reader(adapter).tensors else None

    def test_group_curve_overrides_the_default_for_its_group(self):
        adapter = self.build("--curve", "1.5,2,0.5,1", "--curve", "attn=1.25,0,1.25,inf")
        for layer in range(6):
            self.assertAlmostEqual(self.weight(adapter, f"blk.{layer}.attn_output.weight"), 1.25, places=4)
        mlp = [self.weight(adapter, f"blk.{layer}.ffn_down_exps.weight") for layer in range(6)]
        self.assertEqual([w is not None for w in mlp], [False, True, True, True, False, False])
        self.assertAlmostEqual(mlp[2], 1.5, places=4)
        self.assertIsNone(self.weight(adapter, "token_embd.weight"))
        self.assertIn("attn=1.25,0,1.25,inf", Reader(adapter).kv["hierophant.curve"])

    def test_embedding_has_its_own_group(self):
        adapter = self.build("--curve", "embd=0.75,0,0.75,inf")
        self.assertAlmostEqual(self.weight(adapter, "token_embd.weight"), 0.75, places=4)
        self.assertAlmostEqual(self.weight(adapter, "blk.0.attn_output.weight"), 1.0, places=4)

    def test_unknown_group_fails(self):
        p = subprocess.run([str(HIEROPHANT), "adapter", "-m", str(MOE), "-d", "x", "-o", "y",
                            "--curve", "ffn=1,0,1,inf"], capture_output=True, text=True, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("attn, mlp or embd", p.stderr)

    def test_kind_curve_overrides_its_group(self):
        adapter = self.build("--curve", "attn_output=0.5,0,0.5,inf", "--curve", "attn=1.25,0,1.25,inf")
        self.assertAlmostEqual(self.weight(adapter, "blk.2.attn_output.weight"), 0.5, places=4)
        self.assertAlmostEqual(self.weight(adapter, "blk.2.ffn_down_exps.weight"), 1.0, places=4)
        self.assertIn("attn_output=0.5,0,0.5,inf", Reader(adapter).kv["hierophant.curve"])


class Split(unittest.TestCase):
    def test_split_adapters_add_up_to_the_whole(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "whole.gguf", "--direction-index", DIRECTION_LAYER)
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "s-{}.gguf", "--direction-index", DIRECTION_LAYER, "--split")
        parts = sorted(d.glob("s-*.gguf"))
        self.assertEqual(len(parts), 13)
        whole, seen = Reader(d / "whole.gguf"), {}
        for part in parts:
            r = Reader(part)
            group, layer = r.kv["hierophant.split_group"], r.kv["hierophant.split_layer"]
            self.assertEqual(part.name, f"s-{group}.gguf" if layer < 0 else f"s-{group}-{layer}.gguf")
            for name in r.tensors:
                seen[name] = r.tensor(name)
        self.assertEqual(sorted(seen), sorted(whole.tensors))
        for name, values in seen.items():
            self.assertEqual(values, whole.tensor(name), name)

    def test_split_by_kind_names_parts_after_tensor_kinds(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "s-{}.gguf", "--direction-index", DIRECTION_LAYER,
            "--split", "--split-by", "kind")
        names = sorted(p.name for p in d.glob("s-*.gguf"))
        self.assertIn("s-token_embd.gguf", names)
        self.assertIn("s-ffn_down_exps-5.gguf", names)
        self.assertEqual(Reader(d / "s-attn_output-0.gguf").kv["hierophant.split_group"], "attn_output")

    def test_split_takes_one_index(self):
        p = subprocess.run([str(HIEROPHANT), "adapter", "-m", str(MOE), "-d", "x", "-o", "s-{}.gguf",
                            "--direction-index", "2,3", "--split"], capture_output=True, text=True, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--split", p.stderr)


class RankK(unittest.TestCase):
    """several directions removed at once: B = -w Q with Q orthonormal, A = Q^T W."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        cls.dirs = capture(MOE, cls.dir / "dirs.gguf")
        cls.adapter = cls.dir / "a-3+1.gguf"
        run("adapter", "-m", MOE, "-d", cls.dirs, "-o", cls.dir / "a-{}.gguf", "--direction-index", "3+1,3")

    def test_layout_has_rank_two(self):
        r = Reader(self.adapter)
        self.assertEqual(r.kv["adapter.lora.alpha"], 2.0)
        self.assertEqual(r.kv["hierophant.direction_indices"], "3+1")
        self.assertEqual(r.tensors["blk.0.attn_output.weight.lora_a"]["ne"], [288, 2])
        self.assertEqual(r.tensors["blk.0.attn_output.weight.lora_b"]["ne"], [2, 288])
        self.assertEqual(r.tensors["blk.0.ffn_down_exps.weight.lora_a"]["ne"], [768, 2, 4])
        self.assertEqual(r.tensors["blk.0.ffn_down_exps.weight.lora_b"]["ne"], [2, 288, 4])
        self.assertEqual(r.tensors["token_embd.weight.lora_a"]["ne"], [2, 32000])
        self.assertEqual(r.tensors["token_embd.weight.lora_b"]["ne"], [2, 288])

    def test_columns_are_orthonormal_and_start_with_the_first_index(self):
        q1, q2 = lora_columns(self.adapter, "blk.2.attn_output.weight")
        r, _ = adapter_direction(self.dir / "a-3.gguf", "blk.2.attn_output.weight")
        self.assertAlmostEqual(dot(q1, q1), 1.0, places=5)
        self.assertAlmostEqual(dot(q2, q2), 1.0, places=5)
        self.assertAlmostEqual(dot(q1, q2), 0.0, places=5)
        self.assertAlmostEqual(dot(q1, [-x for x in r]), 1.0, places=5)

    def test_both_directions_are_removed(self):
        q1, q2 = (unit([-x for x in q]) for q in lora_columns(self.adapter, "blk.0.attn_output.weight"))
        after = capture(MOE, self.dir / "after.gguf", "--lora", self.adapter)
        for q in (q1, q2):
            before = projections(self.dirs, q)
            left = projections(after, q)
            self.assertGreater(max(map(abs, before)), 0.1)
            self.assertLess(max(map(abs, left)), 1e-3 * max(map(abs, before)))

    def test_rank_two_rebases_like_a_direct_build(self):
        direct, rebased = self.dir / "direct.gguf", self.dir / "rebased.gguf"
        run("adapter", "-m", MOE_Q8, "-d", self.dirs, "-o", direct, "--direction-index", "3+1")
        run("adapter", "-m", MOE_Q8, "--from-adapter", self.adapter, "-o", rebased)
        a, b = Reader(direct), Reader(rebased)
        self.assertEqual(b.kv["adapter.lora.alpha"], 2.0)
        for name in a.tensors:
            x, y = a.tensor(name), b.tensor(name)
            self.assertLess(max(abs(p - q) for p, q in zip(x, y)), 1e-5 * max(map(abs, x)), name)

    def test_a_repeated_direction_leaves_a_zero_column(self):
        run("adapter", "-m", MOE, "-d", self.dirs, "-o", self.dir / "same.gguf", "--direction-index", "3+3")
        _, q2 = lora_columns(self.dir / "same.gguf", "blk.0.attn_output.weight")
        self.assertEqual(max(map(abs, q2)), 0)


class Residual(unittest.TestCase):
    """a second direction captured with the first adapter applied, removed beside the first."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        cls.dirs = capture(MOE, cls.dir / "dirs.gguf")
        cls.first = cls.dir / "first.gguf"
        plain = ["--direction-index", "3", "--no-project-harmless"]
        run("adapter", "-m", MOE, "-d", cls.dirs, "-o", cls.first, *plain)
        cls.resid = capture(MOE, cls.dir / "resid.gguf", "--lora", cls.first, "--lora-scale", "0.5")
        cls.adapter = cls.dir / "res.gguf"
        run("adapter", "-m", MOE, "-d", cls.dirs, "--residual", cls.resid, "-o", cls.adapter, *plain)

    def test_capture_records_the_adapter_it_ran_under(self):
        kv = Reader(self.resid).kv
        self.assertEqual(kv["hierophant.capture_lora"], "first.gguf")
        self.assertEqual(kv["hierophant.capture_lora_scale"], 0.5)
        self.assertEqual(Reader(self.dirs).kv["hierophant.capture_lora"], "")

    def test_first_column_is_the_first_direction_and_the_second_the_residual(self):
        site = "blk.2.attn_output.weight"
        q1, q2 = (unit([-x for x in q]) for q in lora_columns(self.adapter, site))
        r1, _ = adapter_direction(self.first, site)
        self.assertAlmostEqual(dot(q1, r1), 1.0, places=5)
        r = refusal_direction(self.resid, 3)
        rest = unit([x - dot(r, r1) * y for x, y in zip(r, r1)])
        self.assertAlmostEqual(abs(dot(q2, rest)), 1.0, places=4)

    def test_residual_provenance_is_recorded(self):
        kv = Reader(self.adapter).kv
        self.assertEqual(kv["adapter.lora.alpha"], 2.0)
        self.assertEqual(kv["hierophant.direction_indices"], "3")
        self.assertEqual(kv["hierophant.residual_lora"], "first.gguf")
        self.assertEqual(kv["hierophant.residual_lora_scale"], 0.5)
        self.assertEqual(kv["hierophant.residual_set"], HARMFUL.name)
        self.assertEqual(kv["hierophant.n_residual"], N_PROMPTS)

    def test_sidecar_describes_the_residual(self):
        from hierophant_eval import directions_of

        res = directions_of(Reader(self.adapter).kv, {})["residual"]
        self.assertEqual(res["lora"], "first.gguf")
        self.assertEqual(res["lora_scale"], 0.5)
        self.assertIn(f"{HARMFUL.name} n={N_PROMPTS}", res["harmful"])
        self.assertNotIn("residual", directions_of(Reader(self.first).kv, {}))

    def test_residual_needs_a_direction_index(self):
        p = subprocess.run([str(HIEROPHANT), "adapter", "-m", str(MOE), "-d", str(self.dirs), "--residual",
                            str(self.resid), "-o", str(self.dir / "x.gguf")], capture_output=True, text=True, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--residual needs --direction-index", p.stderr)


class CaptureOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        cls.last = capture(MOE, cls.dir / "last.gguf")
        cls.rows = capture(MOE, cls.dir / "rows.gguf", "--keep-rows")
        cls.span = capture(MOE, cls.dir / "span.gguf", "--positions", 3)

    def test_positions_average_the_last_tokens(self):
        self.assertEqual(Reader(self.span).kv["hierophant.positions"], 3)
        self.assertEqual(Reader(self.last).kv["hierophant.positions"], 1)
        a, b = layer_means(self.last, "harmful_mean")[2], layer_means(self.span, "harmful_mean")[2]
        self.assertGreater(max(abs(x - y) for x, y in zip(a, b)), 1e-3)

    def test_think_is_recorded(self):
        self.assertFalse(Reader(self.last).kv["hierophant.think"])

    def test_rows_average_to_the_harmless_mean(self):
        r = Reader(self.rows)
        n_embd, n, _ = r.tensors["harmless_rows"]["ne"]
        self.assertEqual(n, N_PROMPTS)
        flat = r.tensor("harmless_rows")
        mean = layer_means(self.rows, "harmless_mean")[2]
        rows = [flat[(2 * n + i) * n_embd:(2 * n + i + 1) * n_embd] for i in range(n)]
        for j in range(n_embd):
            self.assertAlmostEqual(sum(row[j] for row in rows) / n, mean[j], delta=1e-2 * (abs(mean[j]) + 1))

    def test_all_harmless_components_leave_every_harmless_row_alone(self):
        adapter = self.dir / "pcs.gguf"
        run("adapter", "-m", MOE, "-d", self.rows, "-o", adapter, "--direction-index", 2,
            "--harmless-pcs", N_PROMPTS - 1)
        r, _ = adapter_direction(adapter)
        reader = Reader(self.rows)
        n_embd, n, _ = reader.tensors["harmless_rows"]["ne"]
        flat = reader.tensor("harmless_rows")
        rows = [flat[(2 * n + i) * n_embd:(2 * n + i + 1) * n_embd] for i in range(n)]
        centre = [sum(col) / n for col in zip(*rows)]
        for row in rows:
            x = [a - b for a, b in zip(row, centre)]
            self.assertLess(abs(dot(r, x)), 1e-3 * math.sqrt(dot(x, x)))
        self.assertGreater(dot(r, refusal_direction(self.rows, 2)), 0.1)
        self.assertEqual(Reader(adapter).kv["hierophant.harmless_pcs"], N_PROMPTS - 1)

    def test_components_need_kept_rows(self):
        p = subprocess.run([str(HIEROPHANT), "adapter", "-m", str(MOE), "-d", str(self.last), "-o",
                            str(self.dir / "x.gguf"), "--direction-index", "2", "--harmless-pcs", "2"],
                           capture_output=True, text=True, check=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--keep-rows", p.stderr)


class CaptureSets(unittest.TestCase):
    def test_several_prompt_files_are_pooled(self):
        d = tmpdir(self.addCleanup)
        lines = HARMFUL.read_text().splitlines()
        (d / "a.txt").write_text("\n".join(lines[:3]) + "\n")
        (d / "b.txt").write_text("\n".join(lines[3:6]) + "\n")
        common = ["--harmless", HARMLESS, "--limit", 6, "-t", 4]
        whole = run_capture(MOE, d / "whole.gguf", ["--harmful", HARMFUL, *common])
        pooled = run_capture(MOE, d / "pooled.gguf", ["--harmful", f"{d / 'a.txt'},{d / 'b.txt'}", *common])
        self.assertEqual(Reader(pooled).kv["hierophant.harmful_set"], "a.txt,b.txt")
        self.assertEqual(Reader(pooled).kv["hierophant.n_harmful"], 6)
        for x, y in zip(layer_means(whole, "harmful_mean"), layer_means(pooled, "harmful_mean")):
            self.assertLess(max(abs(p - q) for p, q in zip(x, y)), 1e-4)


class Progress(unittest.TestCase):
    """--progress redraws a bar in place per prompt and per site, ending each on a line of its own."""

    def test_capture_and_adapter_draw_bars(self):
        d = tmpdir(self.addCleanup)
        out = run("capture", "-m", MOE, "--harmful", HARMFUL, "--harmless", HARMLESS, "--limit", N_PROMPTS,
                  "-t", 4, "-o", d / "dirs.gguf", "--progress")
        self.assertIn(f"harmful [{'#' * 24}] {N_PROMPTS}/{N_PROMPTS} 100%", out)
        self.assertIn(f"harmless [{'#' * 24}] {N_PROMPTS}/{N_PROMPTS} 100%", out)
        self.assertIn("\r", out)
        out = run("adapter", "-m", MOE, "-d", d / "dirs.gguf", "-o", d / "a.gguf", "--progress")
        self.assertIn(f"sites [{'#' * 24}] 13/13 100%", out)
        self.assertIn("\r", out)

    def test_without_it_there_are_no_redraws(self):
        d = tmpdir(self.addCleanup)
        out = run("capture", "-m", MOE, "--harmful", HARMFUL, "--harmless", HARMLESS, "--limit", N_PROMPTS,
                  "-t", 4, "-o", d / "dirs.gguf")
        self.assertIn(f"harmful: {N_PROMPTS}/{N_PROMPTS} prompts", out)
        self.assertNotIn("\r", out)


if __name__ == "__main__":
    unittest.main()
