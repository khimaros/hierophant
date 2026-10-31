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
REPO = "tester/tester-stories15M_MOE-hierophant-LoRA-GGUF"  # the base's uploader in the name: "--" is forbidden
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


def rung(index, refusals, pcs=0, residual=False, weight=1.5):
    return {"refusals": refusals, "kl": 0.1, "scale": 1, "cand": {"index": index, "pcs": pcs, "residual": residual,
                                                                  "weight": weight}}


def write_rounds(d):
    """two rounds of a search: L3 and L4 go on, and L4 improves its score."""
    rounds = (("round-1", 32, [rung(3, 0.1), rung(4, 0.2, pcs=8), rung(5, 0.5)]),
              ("round-2", 64, [rung(3, 0.09), rung(4, 0.15, pcs=8, residual=True, weight=1.2)]),
              ("round-10", 402, [rung(3, 0.08)]))
    for name, n, candidates in rounds:
        (d / name).mkdir()
        (d / name / "round.json").write_text(json.dumps({"n": n, "candidates": candidates}))


def publish(*args):
    return subprocess.run([sys.executable, "-I", str(PUBLISH), *map(str, args)], capture_output=True, text=True,
                          check=False)


class Publish(unittest.TestCase):
    @staticmethod
    def p_about():
        sys.path.insert(0, str(ROOT / "scripts"))
        import hierophant_publish

        return hierophant_publish.COLLECTION_ABOUT

    @classmethod
    def setUpClass(cls):
        root = tmpdir(cls.addClassCleanup)
        cls.adapters = {}
        for quant, model in (("F16", MOE), ("Q8_0", MOE_Q8)):
            (root / quant).mkdir()
            cls.adapters[quant] = adapter_with_sidecar(model, root / quant)
        write_rounds(root / "F16")
        cls.stage = root / "stage"
        cls.first = publish(cls.adapters["F16"], "--user", "tester", "--license", "mit", "--out", cls.stage)
        cls.only_first = (cls.stage / "README.md").read_text() if cls.first.returncode == 0 else ""
        cls.fake_hf = root / "hf"
        cls.fake_hf.write_text(f"#!/bin/sh\necho \"$@\" >> {root / 'hf-args.txt'}\n"
                               'if [ "$2" = create ]; then echo "$5/$3-64f9a55b"; fi\n')
        cls.about = "--description " + cls.p_about()
        cls.fake_hf.chmod(0o755)
        cls.second = publish(cls.adapters["Q8_0"], "--user", "tester", "--license", "mit", "--out", cls.stage,
                             "--upload", "--hf", cls.fake_hf)
        cls.hf_args = root / "hf-args.txt"
        cls.hf_calls = cls.hf_args.read_text().splitlines()
        cls.card = (cls.stage / "README.md").read_text()
        cls.hf_args.unlink()
        cls.named = publish(cls.adapters["F16"], "--user", "tester", "--license", "mit", "--out", cls.stage, "--upload",
                            "--hf", cls.fake_hf, "--collection", "tester/mine-123")
        cls.named_calls = cls.hf_args.read_text().splitlines()
        cls.hf_args.unlink()
        cls.none = publish(cls.adapters["F16"], "--user", "tester", "--license", "mit", "--out", cls.stage, "--upload",
                           "--hf", cls.fake_hf, "--collection", "")
        cls.none_calls = cls.hf_args.read_text().splitlines()

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
            # short enough to leave the KL columns on screen; the sidecar has all of it
            self.assertIn(f"| {sha[:12]} |", row)

    def test_a_row_scores_the_scale_its_serve_line_gives(self):
        for quant, name in NAMES.items():
            s = json.loads((self.stage / f"{name}.json").read_text())
            at = next(p for p in s["eval"]["sweep"] if p["scale"] == (s["scale"] or 1))
            row = next(line for line in self.card.splitlines() if line.startswith(f"| {quant} |")).split(" | ")
            self.assertEqual(row[-3], f"{at['kld']:.3f}")
            self.assertTrue(row[-4].endswith(f"-> {at['refusals']:.2f} (n=3)"), row)

    def test_the_ladder_is_near_the_end_with_each_recipe_at_its_last_round_best_first(self):
        self.assertLess(self.card.index("## limitations"), self.card.index("## every recipe tried"))
        self.assertTrue(self.card.rstrip().endswith("</details>"))
        self.assertIn("<details><summary>F16: 4 recipes</summary>", self.card)
        rows = [line for line in self.card.splitlines() if line.startswith("| L")]
        self.assertEqual([r.split(" | ")[0].removeprefix("| ") for r in rows],
                         ["L3 (published)", "L4-pcs8-res", "L4-pcs8", "L5"])
        self.assertEqual(rows[0], "| L3 (published) | 10 | 402 | 0.080 | 1.5 | 0.100 |")
        self.assertEqual(rows[1], "| L4-pcs8-res | 2 | 64 | 0.150 | 1.2 | 0.100 |")

    def test_a_quant_built_from_a_recipe_alone_has_no_ladder(self):
        self.assertNotIn("Q8_0: ", self.card.split("## every recipe tried")[1])

    def test_a_card_says_the_ladder_is_not_held_out(self):
        self.assertIn("not held out", self.card.split("## every recipe tried")[1])

    def test_a_ladder_is_staged_outside_the_sidecars(self):
        self.assertEqual(len(json.loads((self.stage / "ladder-F16.json").read_text())), 4)
        self.assertEqual(len(list(self.stage.glob("stories15M_MOE-hierophant-*.json"))), 2)

    def test_the_preview_opens_the_ladder_as_the_hub_does(self):
        page = (self.stage.parent / "stage.preview.html").read_text()
        self.assertIn("<details><summary>F16: 4 recipes</summary>", page)
        self.assertIn("<td>L4-pcs8-res</td>", page)

    def test_a_later_quant_rewrites_the_card_from_every_sidecar(self):
        self.assertIn("| F16 |", self.only_first)
        self.assertNotIn("| Q8_0 |", self.only_first)
        self.assertIn("| F16 |", self.card)

    def test_nothing_is_uploaded_unless_asked(self):
        self.assertIn(f"hf upload {REPO} {self.stage}", self.first.stdout)
        self.assertEqual(self.hf_calls[0].split(), ["upload", REPO, str(self.stage), ".", "--repo-type", "model"])

    def test_an_upload_joins_the_hierophant_collection_by_default(self):
        self.assertEqual(self.hf_calls[1:],
                         [f"collections create hierophant --namespace tester {self.about} --exists-ok --format quiet",
                          f"collections add-item tester/hierophant-64f9a55b {REPO} model --exists-ok"])

    def test_the_collection_s_description_fits_the_hub_s_limit(self):
        self.assertLessEqual(len(self.p_about()), 150)

    def test_the_preview_shows_the_card_outside_what_is_uploaded(self):
        page = (self.stage.parent / "stage.preview.html").read_text()
        self.assertIn("preview: ", self.first.stdout)
        self.assertFalse(list(self.stage.glob("*.html")))
        self.assertIn(f"<h1 class=\"repo\">{REPO}</h1>", page)
        self.assertIn('<span class="tag">mit</span>', page)
        self.assertIn('<img alt="hierophant" src="stage/hierophant-banner.jpeg">', page)
        self.assertIn("<h2>use</h2>", page)
        self.assertIn("<th>quant</th>", page)
        self.assertIn(f"<td>{NAMES['F16']}.gguf</td>", page)
        self.assertIn("<pre><code># F16", page)
        self.assertIn("<ul><li>only the listed base files", page)

    def test_the_banner_is_staged_beside_the_card_that_shows_it(self):
        self.assertIn("![hierophant](hierophant-banner.jpeg)", self.card)
        self.assertEqual((self.stage / "hierophant-banner.jpeg").read_bytes(),
                         (ROOT / "templates" / "hierophant-banner.jpeg").read_bytes())

    def test_a_slug_names_a_collection_that_already_exists(self):
        self.assertEqual(self.named.returncode, 0, self.named.stderr)
        self.assertEqual(self.named_calls[1:], [f"collections add-item tester/mine-123 {REPO} model --exists-ok"])

    def test_an_empty_collection_joins_none(self):
        self.assertEqual(self.none.returncode, 0, self.none.stderr)
        self.assertEqual(len(self.none_calls), 1)

    def test_staging_alone_touches_no_collection(self):
        self.assertNotIn("collection", self.first.stdout)


class Loops(unittest.TestCase):
    """the final check's loops are the visible cost of an edit, so the card shows them."""

    def row(self, **eval_extra):
        sys.path.insert(0, str(ROOT / "scripts"))
        import hierophant_publish

        evals = {"sweep": [], "refusals": {"base": 1, "adapted": 0.1, "n": 3}, "kld": {"mean": 0.1}, **eval_extra}
        side = {"base": {"sha256": "abc"}, "scale": 1, "directions": {"index": 3}, "eval": evals}
        return hierophant_publish.row_of("Q8_0", "m", side)

    def test_the_card_shows_the_final_check_s_loops(self):
        self.assertTrue(self.row(final={"loops": 0.3, "n": 50}).endswith("| 0.30 (n=50) |"))

    def test_an_adapter_with_no_final_check_shows_none(self):
        self.assertTrue(self.row().endswith("| - |"))

    def test_a_row_names_the_llama_cpp_that_built_it(self):
        self.assertIn(" | aa588a346 | ", self.row_of_commit("aa588a3463357ee990b314f41dda242869130704"))

    def row_of_commit(self, commit):
        sys.path.insert(0, str(ROOT / "scripts"))
        import hierophant_publish

        side = {"base": {"sha256": "abc"}, "scale": 1, "directions": {"index": 3}, "llama_cpp": commit,
                "eval": {"sweep": [], "refusals": {"base": 1, "adapted": 0.1, "n": 3}, "kld": {"mean": 0.1}}}
        return hierophant_publish.row_of("Q8_0", "m", side)


class Licenses(unittest.TestCase):
    """the hub's license field only takes its own identifiers."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import hierophant_preview
        import hierophant_publish

        cls.p, cls.preview = hierophant_publish, hierophant_preview

    def front(self, license):
        return self.p.card("me/M-hierophant-LoRA-GGUF", "org/M-GGUF", license, []).split("---\n")[1]

    def test_a_hub_license_is_named_directly(self):
        self.assertIn("license: apache-2.0\n", self.front("apache-2.0"))
        self.assertNotIn("license_name", self.front("apache-2.0"))

    def test_a_model_s_own_license_is_other_with_its_name_and_a_link_to_the_base(self):
        front = self.front("qwen-community-1.0")
        self.assertIn("license: other\nlicense_name: qwen-community-1.0\n", front)
        self.assertIn("license_link: https://huggingface.co/org/M-GGUF\n", front)

    def test_the_preview_shows_a_model_s_own_license_by_name(self):
        card = self.p.card("me/M-hierophant-LoRA-GGUF", "org/M-GGUF", "lfm1.0", [])
        meta, _ = self.preview.split_card(card)
        self.assertEqual(self.preview.license_of(meta), "lfm1.0")


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

    def test_a_repo_names_the_base_s_uploader_as_a_config_does(self):
        got = self.p.repo_of("khimaros", "LiquidAI/LFM3-Omni-2B-GGUF", "LFM3-Omni-2B")
        self.assertEqual(got, "khimaros/LiquidAI-LFM3-Omni-2B-hierophant-LoRA-GGUF")
        self.assertNotIn("--", got)

    def test_a_card_is_titled_after_the_model_alone(self):
        card = self.p.card("me/org-M-hierophant-LoRA-GGUF", "org/M-GGUF", "mit", [])
        self.assertIn("# M hierophant LoRA\n", card)

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
