"""naming the base by huggingface repo and quant, resolved as llama-server's -hf does."""

import os
import subprocess
import sys
import unittest
from unittest import mock

from common import HIEROPHANT, MOE, ROOT, Reader, capture, run, tmpdir

REPO = "hierophant-test/stories15M_MOE"  # not on the hub; every call here is --offline
COMMIT = "0123456789abcdef0123456789abcdef01234567"
FILES = ["stories-Q4_K_M.gguf", "stories-Q8_0.gguf", "big-Q6_K-00001-of-00002.gguf", "big-Q6_K-00002-of-00002.gguf"]


def fake_hub(case):
    """a hub cache holding MOE under several quant names, laid out as huggingface_hub does."""
    hub = tmpdir(case.addCleanup)
    repo = hub / ("models--" + REPO.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(COMMIT)
    snap = repo / "snapshots" / COMMIT
    snap.mkdir(parents=True)
    for name in FILES:
        (snap / name).symlink_to(MOE)
    return snap


def resolve(*model):
    return subprocess.run([str(HIEROPHANT), "resolve", *model, "--offline"], capture_output=True, text=True,
                          check=False)


class HfRepo(unittest.TestCase):
    def setUp(self):
        self.snap = fake_hub(self)
        env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.snap.parent.parent.parent)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LLAMA_CACHE", None)  # llama.cpp reads it before HF_HUB_CACHE

    def assertResolves(self, model, name):
        p = resolve(*model)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), str(self.snap / name))

    def test_a_quant_tag_picks_that_file(self):
        self.assertResolves(["--hf-repo", f"{REPO}:Q8_0"], "stories-Q8_0.gguf")

    def test_the_tag_is_case_insensitive_and_hf_is_short_for_hf_repo(self):
        self.assertResolves(["-hf", f"{REPO}:q8_0"], "stories-Q8_0.gguf")

    def test_no_tag_takes_q4_k_m(self):
        self.assertResolves(["--hf-repo", REPO], "stories-Q4_K_M.gguf")

    def test_a_split_model_resolves_to_its_first_shard(self):
        self.assertResolves(["--hf-repo", f"{REPO}:Q6_K"], "big-Q6_K-00001-of-00002.gguf")

    def test_hf_file_names_one_file_exactly(self):
        self.assertResolves(["--hf-repo", REPO, "--hf-file", "stories-Q8_0.gguf"], "stories-Q8_0.gguf")

    def test_an_uncached_quant_fails_offline(self):
        p = resolve("--hf-repo", f"{REPO}:Q2_K")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn(f"{REPO}:Q2_K", p.stderr)

    def test_adapter_records_the_repo_and_the_file_it_resolved_to(self):
        d = tmpdir(self.addCleanup)
        src = ["--hf-repo", f"{REPO}:Q8_0", "--offline"]
        self.assertIn("13 sites", run("info", *src))
        dirs = capture(MOE, d / "dirs.gguf")
        run("adapter", *src, "-d", dirs, "-o", d / "a.gguf", "--direction-index", 3)
        kv = Reader(d / "a.gguf").kv
        self.assertEqual(kv["hierophant.base_repo"], REPO)
        self.assertEqual(kv["hierophant.base_file"], "stories-Q8_0.gguf")

    def test_the_scripts_take_a_quant_tag(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import hierophant_eval
        import hierophant_pipeline

        args = hierophant_eval.parse_args(["--hf-repo", f"{REPO}:Q8_0", "--offline", "--lora", "x.gguf"])
        self.assertEqual(args.model, str(self.snap / "stories-Q8_0.gguf"))
        r = hierophant_pipeline.Run(hierophant_pipeline.parse_args(
            ["--hf-repo", f"{REPO}:Q8_0", "--offline", "--out", str(tmpdir(self.addCleanup)), "--indices", "1"]))
        self.assertEqual(r.model, ["--hf-repo", f"{REPO}:Q8_0", "--offline"])


if __name__ == "__main__":
    unittest.main()
