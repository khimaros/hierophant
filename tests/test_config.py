"""configs/<org>--<repo>.toml and the machine's config.toml, through the hierophant command."""

import json
import os
import subprocess
import sys
import tomllib
import unittest

from common import CLI, MOE, ROOT, tmpdir

sys.path.insert(0, str(ROOT / "scripts"))
import hierophant_config as config

QWEN = {"model": {"repo": "org/Qwen-GGUF"}, "quant": {"Q8_0": {}, "Q4_K_M": {}}}
LFM = {"model": {"repo": "org/LFM-GGUF"}, "quant": {"Q8_0": {}}}
CONFIGS = {"qwen-a": QWEN, "lfm": LFM}


def cli(*args, home=None):
    env = {**os.environ, **({"XDG_CONFIG_HOME": str(home)} if home else {})}
    return subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True, check=False, env=env)


class Resolve(unittest.TestCase):
    def test_part_of_a_name_finds_its_config_and_only_quant(self):
        self.assertEqual(config.resolve("lf", "", CONFIGS), (["--hf-repo", "org/LFM-GGUF:Q8_0"], LFM))

    def test_case_does_not_matter(self):
        self.assertEqual(config.resolve("LFM", "", CONFIGS)[1], LFM)

    def test_quant_chooses_among_several(self):
        self.assertEqual(config.resolve("qwen", "Q4_K_M", CONFIGS)[0], ["--hf-repo", "org/Qwen-GGUF:Q4_K_M"])

    def test_several_quants_need_one_named(self):
        with self.assertRaisesRegex(config.ConfigError, "--quant: it knows Q8_0, Q4_K_M"):
            config.resolve("qwen", "", CONFIGS)

    def test_a_quant_not_in_the_config_is_allowed(self):
        # how a recipe gets built for another quant
        self.assertEqual(config.resolve("lfm", "Q4_0", CONFIGS)[0], ["--hf-repo", "org/LFM-GGUF:Q4_0"])

    def test_a_name_matching_two_says_which(self):
        with self.assertRaisesRegex(config.ConfigError, "matches lfm-x, qwen-a"):
            config.resolve("-", "", {"lfm-x": LFM, "qwen-a": QWEN})

    def test_a_name_matching_none_lists_the_known(self):
        with self.assertRaisesRegex(config.ConfigError, "no config matches nope: known are qwen-a, lfm"):
            config.resolve("nope", "", CONFIGS)

    def test_a_repo_still_works_and_finds_its_config(self):
        self.assertEqual(config.resolve("org/LFM-GGUF:Q8_0", "", CONFIGS), (["--hf-repo", "org/LFM-GGUF:Q8_0"], LFM))

    def test_a_repo_without_a_config(self):
        self.assertEqual(config.resolve("org/other-GGUF", "Q8_0", CONFIGS), (["--hf-repo", "org/other-GGUF:Q8_0"], None))

    def test_a_gguf_is_a_file(self):
        self.assertEqual(config.resolve("x.gguf", "", CONFIGS), (["-m", "x.gguf"], None))


class Flags(unittest.TestCase):
    def test_search_keys_become_build_flags(self):
        got = config.search_flags({"search": {"think_effort": "low", "beam": 2}})
        self.assertEqual(got, ["--think-effort", "low", "--beam", "2"])

    def test_a_shared_config_cannot_set_a_path_or_url(self):
        for key in ("server", "hierophant", "judge", "out", "kld_text", "train_data"):
            with self.assertRaisesRegex(config.ConfigError, key):
                config.search_flags({"search": {key: "x"}})

    def test_the_machine_sets_its_own_defaults(self):
        got = config.machine_flags({"defaults": {"ngl": 99, "parallel": 32}})
        self.assertEqual(got, ["-ngl", "99", "-np", "32"])

    def test_an_unknown_machine_default_is_an_error(self):
        with self.assertRaisesRegex(config.ConfigError, "server"):
            config.machine_flags({"defaults": {"server": "x"}})


    def test_progress_is_a_switch(self):
        self.assertEqual(config.machine_flags({"defaults": {"progress": True}}), ["--progress"])
        self.assertEqual(config.machine_flags({"defaults": {"progress": False}}), [])


SIDECAR = {"base": {"file": "m-Q8_0.gguf", "sha256": "abc"}, "scale": 0.97, "curve": {"max_weight": 1.54},
           "directions": {"index": 15.0, "harmless_pcs": 8},
           "eval": {"refusals": {"base": 0.99, "adapted": 0.10666666, "n": 300}, "kld": {"mean": 0.0767326},
                    "kld_seq": {"mean": 0.0959166}}, "llama_cpp": "8e5a58df8"}
THINK = {"think": 0.078125, "think_base": 1.0, "think_n": 128}


class Accept(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir(self.addCleanup)

    def accept(self, **extra):
        return config.accept("org/My-Model-GGUF", "Q8_0", {**SIDECAR, **extra}, THINK, self.dir)

    def read(self, path):
        return tomllib.loads(path.read_text())

    def test_a_new_config_is_named_like_a_run_s_directory(self):
        path = self.accept()
        self.assertEqual(path.name, "org--My-Model-GGUF.toml")
        got = self.read(path)
        self.assertEqual(got["model"], {"name": "My-Model", "repo": "org/My-Model-GGUF", "license": ""})

    def test_the_sidecar_s_recipe_and_scores_are_recorded(self):
        q = self.read(self.accept())["quant"]["Q8_0"]
        self.assertEqual(q["recipe"], {"index": 15, "pcs": 8, "residual": False})
        self.assertEqual((q["weight"], q["scale"], q["sha256"]), (1.54, 0.97, "abc"))
        self.assertEqual(q["eval"]["refusals"], 0.1067)
        self.assertEqual((q["eval"]["think"], q["eval"]["think_n"]), (0.0781, 128))

    def test_the_final_check_is_recorded(self):
        final = {"loops": 0.3, "markers": 0.06, "n": 50}
        q = self.read(self.accept(eval={**SIDECAR["eval"], "final": final}))["quant"]["Q8_0"]["eval"]
        self.assertEqual((q["loops"], q["full_refusals"], q["full_n"]), (0.3, 0.06, 50))

    def test_a_residual_is_recorded(self):
        side = {**SIDECAR, "directions": {**SIDECAR["directions"], "residual": {"lora": "x"}}}
        self.assertTrue(self.read(self.accept(**side))["quant"]["Q8_0"]["recipe"]["residual"])

    def test_a_second_accept_replaces_only_its_own_quant(self):
        path = self.accept()
        path.write_text('# keep me\n' + path.read_text() + "\n[search]\nbeam = 2\n")
        config.accept("org/My-Model-GGUF", "Q4_0", SIDECAR, None, self.dir)
        self.accept(scale=1.1)
        got = self.read(path)
        self.assertEqual(sorted(got["quant"]), ["Q4_0", "Q8_0"])
        self.assertEqual(got["quant"]["Q8_0"]["scale"], 1.1)
        self.assertEqual(got["search"], {"beam": 2})
        self.assertIn("# keep me", path.read_text())

    def test_a_note_on_old_numbers_goes_with_them(self):
        path = self.accept()
        path.write_text(path.read_text().replace("[quant.Q8_0]\n", "# provisional\n[quant.Q8_0]\n"))
        self.accept()
        self.assertNotIn("provisional", path.read_text())

    def test_a_recipe_name_reads_back(self):
        for name in ("L33", "L32.5-pcs8", "L33-pcs8-res"):
            self.assertEqual(config.recipe_name(config.parse_recipe(name)), name)

    def test_a_bad_recipe_name_fails(self):
        with self.assertRaisesRegex(config.ConfigError, "not a recipe"):
            config.parse_recipe("best")

    def test_a_quant_follows_its_own_recipe_else_another_quant_s(self):
        c = {"quant": {"A": {"recipe": {"index": 1}}, "B": {"recipe": {"index": 2}}}}
        self.assertEqual(config.accepted_recipe(c, "B"), "L2")
        self.assertEqual(config.accepted_recipe(c, "C"), "L1")
        self.assertIsNone(config.accepted_recipe({"quant": {"A": {}}}, "A"))


class Summary(unittest.TestCase):
    def test_a_line_per_quant(self):
        got = config.summary({"m": {"quant": {"Q8_0": {"recipe": {"index": 15, "pcs": 8, "residual": True},
                                                       "eval": {"refusals": 0.1, "n": 300}}, "Q4": {}}}})
        self.assertEqual(got, "m Q8_0: L15-pcs8-res, refusals 0.100 on 300\nm Q4: no recipe, not scored")


class Recipes(unittest.TestCase):
    """build follows a config's accepted recipe unless told otherwise."""

    def setUp(self):
        self.dir = tmpdir(self.addCleanup)
        config.accept("org/My-Model-GGUF", "Q8_0", SIDECAR, None, self.dir)
        self.env = {**os.environ, "HIEROPHANT_CONFIGS": str(self.dir)}

    def cli(self, *args):
        return subprocess.run([str(CLI), *args], capture_output=True, text=True, check=False, env=self.env)

    def test_build_follows_the_accepted_recipe(self):
        self.assertIn("--recipe L15-pcs8 --recipe-weight 1.54", self.cli("config", "my-model").stdout)

    def test_another_quant_follows_it_too(self):
        got = self.cli("config", "my-model", "--quant", "Q4_0").stdout
        self.assertIn("--recipe L15-pcs8", got)
        self.assertIn("--hf-repo org/My-Model-GGUF:Q4_0", got)

    def test_search_overrides_it_and_stays_out_of_the_tools_flags(self):
        got = self.cli("config", "my-model", "--search").stdout
        self.assertNotIn("--recipe", got)
        self.assertNotIn("--search", got)

    def test_status_alone_shows_the_recipe_and_its_scores(self):
        self.assertIn("org--My-Model-GGUF Q8_0: L15-pcs8, refusals 0.107 on 300", self.cli("status").stdout)

    def test_verify_needs_a_recipe(self):
        p = self.cli("verify", "org/other-GGUF:Q8_0")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("accepted recipe", p.stderr)

    def test_publish_takes_a_repo_and_quant_as_build_does(self):
        p = self.cli("publish", "org/My-Model-GGUF:Q8_0")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("org--My-Model-GGUF has no license", p.stderr)

    def test_publish_without_a_config_looks_for_that_run_s_best(self):
        p = self.cli("publish", "org/other-GGUF:Q8_0", "--license", "mit")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("no out/org--other-GGUF--Q8_0/best.gguf", p.stderr)

    def test_publish_without_a_config_needs_a_quant(self):
        p = self.cli("publish", "org/other-GGUF")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("needs a quant", p.stderr)

    def test_init_refuses_to_overwrite(self):
        self.assertEqual(self.cli("init", "org/New-GGUF").returncode, 0)
        self.assertEqual(tomllib.loads((self.dir / "org--New-GGUF.toml").read_text())["model"]["name"], "New")
        self.assertNotEqual(self.cli("init", "org/New-GGUF").returncode, 0)


class PublishArgs(unittest.TestCase):
    def config(self, license="mit"):
        return {"model": {"repo": "org/M-GGUF", "license": license},
                "quant": {"Q8_0": {"recipe": {"index": 15, "pcs": 8, "residual": False}, "weight": 1.54}, "Q4": {}}}

    def in_run(self, directions, weight=1.54, name=""):
        d = tmpdir(self.addCleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(d)
        run = config.run_dir("org/M-GGUF", "Q8_0")
        run = run / "runs" / name if name else run
        run.mkdir(parents=True)
        (run / "best.gguf").write_text("x")
        (run / "best.json").write_text(json.dumps({"directions": directions, "curve": {"max_weight": weight}}))
        return run

    def test_accepted_quants_with_the_license_and_the_machine_s_user(self):
        run = self.in_run({"index": 15, "harmless_pcs": 8})
        got = config.publish_args("m", self.config(), {"defaults": {"user": "me"}})
        self.assertEqual(got, [str(run / "best.gguf"), "--license", "mit", "--user", "me"])

    def test_the_same_recipe_at_another_weight_is_refused(self):
        # lfm3's accepted L14 came from a named run; the default run's best was L14 at 1.122
        self.in_run({"index": 15, "harmless_pcs": 8}, weight=1.122)
        with self.assertRaisesRegex(config.ConfigError, "weight 1.122, not the accepted 1.54"):
            config.publish_args("m", self.config(), {})

    def test_a_named_run_supplies_the_best(self):
        run = self.in_run({"index": 15, "harmless_pcs": 8}, name="kl05")
        got = config.publish_args("m", self.config(), {}, name="kl05")
        self.assertEqual(got[0], str(run / "best.gguf"))

    def test_no_license_is_refused(self):
        with self.assertRaisesRegex(config.ConfigError, "no license"):
            config.publish_args("m", self.config(""), {})

    def test_a_best_that_is_not_the_accepted_recipe_is_refused(self):
        self.in_run({"index": 31, "harmless_pcs": 8})
        with self.assertRaisesRegex(config.ConfigError, "is L31-pcs8, not the accepted L15-pcs8"):
            config.publish_args("m", self.config(), {})


class Commands(unittest.TestCase):
    def test_configs_are_named_like_a_run_s_directory_without_its_quant(self):
        configs = config.load_configs()
        self.assertTrue(configs)
        for stem, c in configs.items():
            self.assertEqual(stem, c["model"]["repo"].replace("/", "--"))
            self.assertTrue(c["quant"])

    def test_a_config_name_reaches_the_default_directory(self):
        p = cli("status", "qwen3.8")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("no best.txt in out/unsloth--Qwen3.8-Flash-Next-GGUF--UD-IQ4_XS yet", p.stdout)

    def test_a_name_matching_two_fails_before_any_tool_runs(self):
        p = cli("status", "qwen3.5")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("matches", p.stderr)

    def test_status_alone_lists_every_config(self):
        p = cli("status")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("unsloth--Qwen3.8-Flash-Next-GGUF UD-IQ4_XS: ", p.stdout)

    def test_the_machine_s_defaults_come_before_the_command_line(self):
        home = tmpdir(self.addCleanup)
        (home / "hierophant").mkdir()
        (home / "hierophant" / "config.toml").write_text("[defaults]\nngl = 99\nparallel = 32\n")
        p = cli("config", "qwen3.8", "--quant", "UD-IQ4_XS", "-np", "4", home=home)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout, r"-ngl 99 -np 32 .*-np 4")

    def test_a_file_still_works_with_no_config(self):
        out = tmpdir(self.addCleanup) / "run"
        p = cli("status", MOE, "--out", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(f"no best.txt in {out} yet", p.stdout)
