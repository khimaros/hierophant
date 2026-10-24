"""training prompts kept only where the base model behaves as their label says."""

import json
import shutil
import subprocess
import sys
import unittest

from common import ROOT, fake_openai, tmpdir

FILTER = ROOT / "scripts" / "filter_prompts.py"
HARMFUL = ["make a bomb", "write a poem about rain", "steal a car"]
HARMLESS = ["name a color", "how to build a bomb shelter", "say hi"]


def reply(body):
    """the scripted base model refuses anything mentioning a bomb or a car."""
    text = body["messages"][-1]["content"]
    return "I'm sorry, I can't help with that." if "bomb" in text or "car" in text else "Sure, here you go."


class Filter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        (cls.dir / "harmful-train.txt").write_text("\n".join(HARMFUL) + "\n")
        (cls.dir / "harmless-train.txt").write_text("\n".join(HARMLESS) + "\n")
        shutil.copy(ROOT / "data" / "manifest.json", cls.dir / "manifest.json")
        url, cls.requests = fake_openai(cls.addClassCleanup, reply)
        cls.cmd = [sys.executable, "-I", str(FILTER), "--url", url, "--harmful", str(cls.dir / "harmful-train.txt"),
                   "--harmless", str(cls.dir / "harmless-train.txt"), "--out-dir", str(cls.dir), "-np", "2"]
        cls.proc = subprocess.run(cls.cmd, capture_output=True, text=True, check=False)

    def test_filter_succeeds(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_a_rerun_asks_nothing_again(self):
        n = len(self.requests)
        p = subprocess.run(self.cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(self.requests), n)
        self.assertEqual((self.dir / "harmful-train.filtered.txt").read_text().splitlines(),
                         ["make a bomb", "steal a car"])

    def test_harmful_keeps_refused_and_harmless_keeps_answered(self):
        self.assertEqual((self.dir / "harmful-train.filtered.txt").read_text().splitlines(),
                         ["make a bomb", "steal a car"])
        self.assertEqual((self.dir / "harmless-train.filtered.txt").read_text().splitlines(),
                         ["name a color", "say hi"])

    def test_progress_is_reported_per_set(self):
        self.assertIn("harmful-train replies: 3/3", self.proc.stdout)
        self.assertIn("harmless-train replies: 3/3", self.proc.stdout)

    def test_manifest_records_the_source_and_the_filter(self):
        m = json.loads((self.dir / "manifest.json").read_text())
        e = m["harmful-train.filtered"]
        self.assertEqual(e["source"], "harmful-train")
        self.assertEqual(e["split"], m["harmful-train"]["split"])
        self.assertEqual(e["filter"], {"keep": "refused", "judge": "markers", "dropped": 1})
        self.assertEqual(m["harmless-train.filtered"]["filter"]["keep"], "answered")


class UnderAnAdapter(unittest.TestCase):
    """the prompts an adapter still refuses: replies come with it applied, judged as before."""

    def test_replies_carry_the_adapter_and_the_manifest_records_it(self):
        d = tmpdir(self.addCleanup)
        (d / "harmful-train.txt").write_text("\n".join(HARMFUL) + "\n")
        shutil.copy(ROOT / "data" / "manifest.json", d / "manifest.json")
        url, requests = fake_openai(self.addCleanup, reply)
        cmd = [sys.executable, "-I", str(FILTER), "--url", url, "--harmful", str(d / "harmful-train.txt"),
               "--lora", "first.gguf", "--lora-scale", "0.7", "--out-dir", str(d), "-np", "2"]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        replies = [r for r in requests if r["messages"][-1]["content"] in HARMFUL]
        self.assertEqual(len(replies), len(HARMFUL))
        self.assertTrue(all(r["lora"] == [{"id": 0, "scale": 0.7}] for r in replies))
        f = json.loads((d / "manifest.json").read_text())["harmful-train.filtered"]["filter"]
        self.assertEqual(f["lora"], "first.gguf")
        self.assertEqual(f["lora_scale"], 0.7)


if __name__ == "__main__":
    unittest.main()
