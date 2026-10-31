"""ctrl-c, kill or a closed terminal: every child (llama-server above all, which holds the
model's memory) is stopped and waited for before the command returns."""

import signal
import sys
import unittest

from common import HARMFUL, HARMLESS, LLAMA_SERVER, MOE, ROOT, TIED, capture, interrupt, run, tmpdir

EVAL = ROOT / "scripts" / "hierophant_eval.py"
AUTO = ROOT / "scripts" / "hierophant_auto.py"
INTERRUPTED = 130
AUTO_MARKER = "| base replies"  # the round's first eval is asking its server
STOP_PROMPTLY = 10  # seconds; well under the 30 a parent gives its child before killing it


class Interrupt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        dirs = capture(MOE, cls.dir / "dirs.gguf")
        cls.adapter = cls.dir / "a.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", cls.adapter)

    def eval_cmd(self):
        # enough requests that the signal lands while they are in flight
        return [sys.executable, "-I", EVAL, "-m", MOE, "--lora", self.adapter, "--harmful", HARMFUL,
                "--harmless", HARMLESS, "--limit", 40, "--scales", "0.5,1,1.5,2,2.5,3", "--max-tokens", 64,
                "-np", 2, "--server", LLAMA_SERVER, "--no-sidecar", "--progress"]

    def assert_stopped(self, p, left):
        self.assertEqual(p.returncode, INTERRUPTED, p.stdout)
        self.assertNotIn("Traceback", p.stdout)
        self.assertIn("interrupted", p.stdout)
        self.assertEqual(left, [])

    def test_ctrl_c_stops_the_eval_and_its_server(self):
        self.assert_stopped(*interrupt(self.eval_cmd(), "replies [", signal.SIGINT, group=True))

    def test_kill_stops_the_eval_and_its_server(self):
        # only the eval gets the signal; its server must be stopped by it
        self.assert_stopped(*interrupt(self.eval_cmd(), "replies [", signal.SIGTERM, group=False))

    def test_kill_with_long_replies_in_flight_stops_at_once(self):
        # llama-server finishes what it is generating before it honours a sigterm; nested
        # under the automated run, waiting it out let the run kill the eval first
        cmd = [*self.eval_cmd(), "--max-tokens", 4000]
        p, left = interrupt(cmd, "replies [", signal.SIGTERM, group=False)
        self.assert_stopped(p, left)
        self.assertLess(p.stop_seconds, STOP_PROMPTLY, p.stdout)

    def auto_cmd(self, out):
        return [sys.executable, "-I", AUTO, "-m", TIED, "--out", out, "--indices", "15,20", "--first-prompts", 4,
                "--train-limit", 40, "--max-tokens", 8, "--kld-ctx", 128, "-np", 2, "--server", LLAMA_SERVER,
                "--progress"]

    def test_ctrl_c_stops_the_whole_automated_run(self):
        p, left = interrupt(self.auto_cmd(tmpdir(self.addCleanup)), AUTO_MARKER, signal.SIGINT, group=True)
        self.assert_stopped(p, left)
        self.assertIn("rerun the same command to continue", p.stdout)

    def test_a_closed_terminal_stops_the_whole_automated_run(self):
        # sighup reaches only the run; it stops the eval, which stops its server
        p, left = interrupt(self.auto_cmd(tmpdir(self.addCleanup)), AUTO_MARKER, signal.SIGHUP, group=False)
        self.assert_stopped(p, left)
