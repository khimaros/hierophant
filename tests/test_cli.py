"""the hierophant entrypoint: one command in front of the scripts and the C++ tools."""

import unittest

from common import CLI, HIEROPHANT, MOE, raw, tmpdir


def cli(*args):
    return raw([CLI, *args])


class Help(unittest.TestCase):
    def test_every_command_is_listed(self):
        p = cli("--help")
        self.assertEqual(p.returncode, 0, p.stderr)
        for word in ("init", "build", "status", "screen", "eval", "compare", "accept", "verify", "serve", "chat", "publish",
                     "prune", "curves", "config", "data fetch", "data models", "data filter", "data translate", "core"):
            self.assertIn(word, p.stdout)

    def test_no_command_prints_the_usage_and_fails(self):
        p = cli()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("usage: hierophant", p.stderr)

    def test_an_unknown_command_fails_and_names_itself(self):
        p = cli("frobnicate")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("frobnicate", p.stderr)

    def test_an_unknown_data_command_fails(self):
        self.assertNotEqual(cli("data", "frobnicate").returncode, 0)


class Dispatch(unittest.TestCase):
    def test_each_command_reaches_its_tool(self):
        for args, flag in ((("build", "--help"), "--first-pcs"), (("eval", "--help"), "--lora"),
                           (("publish", "--help"), "--license"), (("data", "fetch", "--help"), "--more"),
                           (("data", "filter", "--help"), "--harmful"), (("data", "translate", "--help"), "sources")):
            p = cli(*args)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn(flag, p.stdout)

    def test_core_passes_through_to_the_cpp_tools(self):
        p = cli("core", "version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, raw([HIEROPHANT, "version"]).stdout)

    def test_a_symlink_on_the_path_finds_the_checkout(self):
        link = tmpdir(self.addCleanup) / "hierophant"
        link.symlink_to(CLI)
        p = raw([link, "core", "version"])
        self.assertEqual(p.returncode, 0, p.stderr)


class Model(unittest.TestCase):
    """the first argument names the model, as a file or as REPO[:QUANT]."""

    def test_a_file_is_a_model_path(self):
        out = tmpdir(self.addCleanup) / "run"
        p = cli("status", MOE, "--out", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(f"no best.txt in {out} yet", p.stdout)

    def test_anything_else_is_a_repo(self):
        p = cli("status", "unsloth/Some-Model-GGUF:Q8_0")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("no best.txt in out/unsloth--Some-Model-GGUF--Q8_0 yet", p.stdout)

    def test_the_model_flags_still_work(self):
        out = tmpdir(self.addCleanup) / "run"
        p = cli("status", "-m", MOE, "--out", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(f"no best.txt in {out} yet", p.stdout)
