"""the scale sweep against a real llama-server, and the sidecars it writes."""

import json
import subprocess
import sys
import unittest

from common import (
    FOREIGN_LORA,
    HARMLESS,
    LLAMA_PERPLEXITY,
    LLAMA_SERVER,
    MOE,
    MOE_Q8,
    ROOT,
    TIED,
    capture,
    fake_openai,
    raw,
    run,
    tmpdir,
)

EVAL = ROOT / "scripts" / "hierophant_eval.py"
HARMFUL_TEST = ROOT / "data" / "harmful-test.txt"
N_EVAL = 6
INDICES = (15, 20)


class Eval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        dirs = capture(TIED, cls.dir / "dirs.gguf")
        run("adapter", "-m", TIED, "-d", dirs, "-o", cls.dir / "a-{}.gguf",
            "--direction-index", ",".join(map(str, INDICES)))
        cls.adapters = [cls.dir / f"a-{i}.gguf" for i in INDICES]
        loras = [arg for a in cls.adapters for arg in ("--lora", str(a))]
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(TIED), *loras, "--limit", str(N_EVAL),
               "--scales", "1", "--max-tokens", "16", "-np", "2", "--server", str(LLAMA_SERVER),
               "--base-repo", "bartowski/SmolLM2-135M-Instruct-GGUF", "--results", str(cls.dir / "results.json")]
        cls.proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    def results(self, adapter):
        return json.loads((self.dir / "results.json").read_text())[adapter.name]

    def sidecar(self, adapter):
        return json.loads(adapter.with_suffix(".json").read_text())

    def test_eval_succeeds(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_scale_zero_has_no_divergence(self):
        for a in self.adapters:
            zero = next(r for r in self.results(a) if r["scale"] == 0)
            self.assertEqual(zero["kld"], 0)
            self.assertEqual(zero["top1_same"], 1)

    def test_each_adapter_changes_the_distribution_differently(self):
        klds = [next(r for r in self.results(a) if r["scale"] == 1)["kld"] for a in self.adapters]
        self.assertGreater(min(klds), 0)
        self.assertNotEqual(klds[0], klds[1])

    def test_each_adapter_gets_its_own_sidecar(self):
        self.assertEqual([self.sidecar(a)["directions"]["index"] for a in self.adapters], list(INDICES))

    def test_progress_is_reported_for_the_base_and_each_scale(self):
        self.assertIn(f"base replies: {N_EVAL}/{N_EVAL}", self.proc.stdout)
        self.assertIn(f"a-15.gguf scale 1 replies: {N_EVAL}/{N_EVAL}", self.proc.stdout)
        self.assertIn(f"a-15.gguf scale 1 first tokens: {N_EVAL}/{N_EVAL}", self.proc.stdout)

    def test_results_keep_every_reply_with_its_verdict(self):
        point = next(r for r in self.results(self.adapters[0]) if r["scale"] == 1)
        self.assertEqual(len(point["replies"]), N_EVAL)
        self.assertEqual(sorted(point["replies"][0]),
                         ["looped", "prompt", "reasoning", "refusal", "reply", "unfinished"])
        self.assertEqual(point["refusals"], sum(r["refusal"] for r in point["replies"]) / N_EVAL)
        self.assertEqual(point["loops"], sum(r["looped"] for r in point["replies"]) / N_EVAL)

    def test_base_replies_grow_less_likely_under_the_adapter(self):
        for a in self.adapters:
            points = self.results(a)
            zero, one = (next(r for r in points if r["scale"] == s) for s in (0, 1))
            self.assertLessEqual(zero["reply_logp"], 0)
            self.assertLess(one["reply_logp"], zero["reply_logp"])

    def test_sidecar_follows_the_contract(self):
        s = self.sidecar(self.adapters[0])
        self.assertEqual(s["kind"], "abliteration")
        self.assertEqual(s["rank"], 1)
        self.assertEqual(s["base"]["repo"], "bartowski/SmolLM2-135M-Instruct-GGUF")
        self.assertEqual(s["base"]["file"], TIED.name)
        self.assertEqual(len(s["base"]["sha256"]), 64)
        self.assertEqual(s["sites"], 60)
        self.assertEqual(s["arch"], "llama")
        self.assertEqual(sorted(s["targets"]), ["attn_output", "ffn_down"])
        self.assertEqual(s["directions"]["position"], "first-token")
        self.assertFalse(s["directions"]["think"])
        self.assertIn("reply_logp", s["eval"]["sweep"][0])
        self.assertIn("mlabonne/harmful_behaviors@", s["directions"]["harmful"])
        self.assertIn("train n=8", s["directions"]["harmful"])
        self.assertEqual(s["eval"]["refusals"]["n"], N_EVAL)
        self.assertIn("test n=6", s["eval"]["kld"]["set"])
        self.assertIn(s["scale"], [0.0, 1.0])
        self.assertEqual(len(s["llama_cpp"]), 40)


class ForeignAdapter(unittest.TestCase):
    """an adapter hierophant did not build is scored beside one it did, without a sidecar."""

    def test_foreign_adapter_is_scored_without_a_sidecar(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        ours = d / "ours.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", ours, "--direction-index", 3)
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(MOE), "--lora", str(ours), "--lora", str(FOREIGN_LORA),
               "--limit", "3", "--scales", "1", "--max-tokens", "8", "-np", "2", "--server", str(LLAMA_SERVER),
               "--results", str(d / "results.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("not built by hierophant, no sidecar", p.stdout)
        self.assertTrue(ours.with_suffix(".json").exists())
        self.assertFalse(FOREIGN_LORA.with_suffix(".json").exists())
        results = json.loads((d / "results.json").read_text())
        self.assertEqual(sorted(results), sorted([ours.name, FOREIGN_LORA.name]))


class Prefill(unittest.TestCase):
    def test_prefilled_tokens_score_what_generation_reported(self):
        from types import SimpleNamespace

        from hierophant_eval import OFF, Server, load_prompts

        args = SimpleNamespace(url=None, lora=[], model=TIED, server=LLAMA_SERVER, ctx=4096, parallel=1, ngl=-1,
                               server_log=None, system="", think=False, top_k=50, logp_tokens=6)
        server = Server(args)
        self.addCleanup(server.close)
        for p in load_prompts(HARMFUL_TEST, 3):
            c = server.chat(server.messages(p), OFF, max_tokens=6, logprobs=True, top_logprobs=1)
            tokens = c["logprobs"]["content"]
            generated = sum(t["logprob"] for t in tokens) / len(tokens)
            prefilled = server.prefix_logp(p, OFF, [(t["id"], t["token"]) for t in tokens])
            self.assertAlmostEqual(prefilled, generated, places=3)


class Reasoning(unittest.TestCase):
    """with thinking on, each reply keeps its reasoning, so a reply cut off mid-thought can be read."""

    def test_replies_keep_their_reasoning(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from hierophant_eval import OFF, Judge, Server, score

        token = {"id": 7, "token": "ok", "logprob": -0.5, "top_logprobs": [{"id": 7, "logprob": -0.5}]}

        def reply(body):
            if body.get("logprobs"):
                return {"message": {"content": "ok"}, "logprobs": {"content": [token]}}
            prompt = body["messages"][-1]["content"]
            if "loop" in prompt:
                return {"message": {"content": "", "reasoning_content": "step one, then " * 50},
                        "finish_reason": "length"}
            if "decline" in prompt:
                return {"message": {"content": "", "reasoning_content": "hmm " * 50 + "i cannot help with this"},
                        "finish_reason": "length"}
            return {"message": {"content": "sure, here", "reasoning_content": "the user asks for steps"}}

        url, _ = fake_openai(self.addCleanup, reply)
        args = SimpleNamespace(url=url, lora=[], system="", think=True, top_k=5, max_tokens=8, logp_tokens=1,
                               judge="markers", judge_model="")
        server = Server(args)
        with ThreadPoolExecutor(2) as pool:
            point = score(server, pool, Judge(args, server), OFF, ["explain", "loop forever", "decline slowly"],
                          ["hi"])
        first, cut, declined = point["replies"]
        self.assertEqual(first["reasoning"], "the user asks for steps")
        self.assertFalse(first["unfinished"])
        self.assertEqual(cut["reasoning"], "step one, then " * 50)
        self.assertTrue(cut["unfinished"])
        self.assertFalse(cut["refusal"], "a reply cut off while working out the answer complies")
        self.assertTrue(declined["unfinished"])
        self.assertTrue(declined["refusal"], "a reply cut off after deciding to refuse refuses")


class ShortThinking(unittest.TestCase):
    """with --think, reasoning can be kept short: a low effort, unless the model's chat
    template rejects that level, and a token budget after which llama-server forces the answer."""

    def serve(self, reply, **opts):
        from types import SimpleNamespace

        from hierophant_eval import Server

        url, requests = fake_openai(self.addCleanup, reply)
        args = SimpleNamespace(url=url, lora=[], system="", think=True, top_k=5, max_tokens=8, logp_tokens=1, **opts)
        return Server(args), requests

    def test_replies_carry_a_low_effort_and_the_budget(self):
        from hierophant_eval import OFF

        server, requests = self.serve(lambda body: "ok", reasoning_effort="auto", reasoning_budget=32,
                                      reasoning_budget_message="time to answer.", think_kwargs={"think_fast": True})
        server.reply("explain", OFF)
        body = requests[-1]
        self.assertEqual((body["reasoning_effort"], body["reasoning_budget_tokens"], body["reasoning_budget_message"]),
                         ("low", 32, "time to answer."))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "think_fast": True})
        server.first_token("hi", OFF)
        self.assertNotIn("reasoning_effort", requests[-1])
        self.assertNotIn("reasoning_budget_tokens", requests[-1])
        self.assertEqual(requests[-1]["chat_template_kwargs"], {"enable_thinking": False})

    def test_a_template_without_low_keeps_its_own_default(self):
        from hierophant_eval import OFF

        server, requests = self.serve(lambda body: 400 if body.get("reasoning_effort") == "low" else "ok",
                                      reasoning_effort="auto", reasoning_budget=-1, reasoning_budget_message="")
        server.reply("explain", OFF)
        self.assertNotIn("reasoning_effort", requests[-1])
        self.assertNotIn("reasoning_budget_tokens", requests[-1])

    def test_llama_server_takes_them_and_the_sidecar_records_them(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(TIED, d / "dirs.gguf")
        run("adapter", "-m", TIED, "-d", dirs, "-o", d / "a.gguf", "--direction-index", 15)
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(TIED), "--lora", str(d / "a.gguf"), "--limit", "2",
               "--scales", "1", "--max-tokens", "8", "-np", "2", "--server", str(LLAMA_SERVER), "--think",
               "--reasoning-effort", "auto", "--reasoning-budget", "4"]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("thinking: effort low, budget 4 tokens", p.stdout)
        refusals = json.loads((d / "a.json").read_text())["eval"]["refusals"]
        self.assertEqual((refusals["reasoning_effort"], refusals["reasoning_budget"]), ("low", 4))


class Resume(unittest.TestCase):
    """every request's outcome is kept as it arrives, so a rerun after ctrl-c repeats only
    what was in flight, and a rerun of a finished eval needs no server at all."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        dirs = capture(MOE, cls.dir / "dirs.gguf")
        cls.adapter = cls.dir / "a.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", cls.adapter, "--direction-index", 3)
        cls.first = cls.eval()
        cls.results = json.loads((cls.dir / "results.json").read_text())
        cls.first_cache = (cls.dir / "results.json.cache.jsonl").read_text().splitlines()

    @classmethod
    def eval(cls, *extra, server=LLAMA_SERVER, perplexity=LLAMA_PERPLEXITY):
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(MOE), "--lora", str(cls.adapter), "--limit", "3",
               "--scales", "0.5,1", "--max-tokens", "8", "-np", "2", "--server", str(server), "--perplexity",
               str(perplexity), "--kld-text", str(HARMLESS), "--kld-ctx", "128", "--kld-chunks", "2",
               "--results", str(cls.dir / "results.json"), *extra]
        return subprocess.run(list(map(str, cmd)), capture_output=True, text=True, check=False)

    def cache(self):
        return self.dir / "results.json.cache.jsonl"

    def test_a_finished_eval_reruns_without_a_server(self):
        self.assertEqual(self.first.returncode, 0, self.first.stdout + self.first.stderr)
        p = self.eval(server="/nonexistent/llama-server", perplexity="/nonexistent/llama-perplexity")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(json.loads((self.dir / "results.json").read_text()), self.results)

    def test_an_interrupted_eval_repeats_only_what_is_missing(self):
        lines = self.first_cache
        # the last few outcomes never arrived, and the line being written was cut off
        self.cache().write_text("\n".join(lines[:-4]) + "\n" + lines[-4][:10])
        p = self.eval()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(self.cache().read_text().splitlines()), len(lines) + 1)
        # regenerated outcomes can differ in the last digits (llama-server batches)
        got = json.loads((self.dir / "results.json").read_text())[self.adapter.name]
        self.assertEqual([(r["scale"], len(r["replies"])) for r in got],
                         [(r["scale"], len(r["replies"])) for r in self.results[self.adapter.name]])

    def test_another_eval_sharing_the_cache_reuses_its_outcomes(self):
        offline = {"server": "/nonexistent/llama-server", "perplexity": "/nonexistent/llama-perplexity"}
        own = self.eval(**offline)
        self.assertEqual(own.returncode, 0, own.stdout + own.stderr)
        other = self.dir / "other.json"
        p = self.eval("--results", other, "--cache", self.cache(), **offline)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r"cache: \d+ reused, 0 new")
        self.assertEqual(json.loads(other.read_text()), json.loads((self.dir / "results.json").read_text()))

    def test_a_changed_judge_rejudges_the_same_replies(self):
        n = len(self.cache().read_text().splitlines())
        p = self.eval("--judge", "self", perplexity="/nonexistent/llama-perplexity")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        # one new verdict per prompt at the base and each scale, and nothing else
        self.assertEqual(len(self.cache().read_text().splitlines()), n + 3 * 3)


class MissingLogprobs(unittest.TestCase):
    """llama-server leaves out logprobs when its one token is half a utf-8 character
    or a stop word; that prompt goes unmeasured instead of ending the run."""

    def test_a_reply_without_logprobs_is_left_out(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from hierophant_eval import OFF, Judge, Server, score

        token = {"id": 7, "token": "ok", "logprob": -0.5, "top_logprobs": [{"id": 7, "logprob": -0.5}]}

        def reply(body):
            if not body.get("logprobs"):
                return {"message": {"content": "sure"}}
            split = "split" in body["messages"][0]["content"]
            if split and (body.get("cache_prompt") or body["top_logprobs"] > 1):
                return {"message": {"content": ""}, "finish_reason": "length"}
            return {"message": {"content": "ok"}, "logprobs": {"content": [token]}}

        url, _ = fake_openai(self.addCleanup, reply)
        args = SimpleNamespace(url=url, lora=[], system="", think=False, top_k=5, max_tokens=8, logp_tokens=1,
                               judge="markers", judge_model="")
        server = Server(args)
        with ThreadPoolExecutor(2) as pool:
            base = score(server, pool, Judge(args, server), OFF, ["explain", "split"], ["hi", "split"])
            point = score(server, pool, Judge(args, server), OFF, ["explain", "split"], ["hi", "split"], base)
        self.assertEqual(point["kld"], 0)
        self.assertEqual(point["reply_logp"], -0.5)


class ServerDeath(unittest.TestCase):
    def test_a_dead_server_is_reported_with_its_log(self):
        from types import SimpleNamespace

        from hierophant_eval import OFF, Server

        log = tmpdir(self.addCleanup) / "server.log"
        args = SimpleNamespace(url=None, lora=[], model=TIED, server=LLAMA_SERVER, ctx=4096, parallel=1, ngl=-1,
                               server_log=log, system="", think=False, top_k=50, max_tokens=4)
        server = Server(args)
        self.addCleanup(server.close)
        server.start()
        server.proc.kill()
        server.proc.wait()
        with self.assertRaises(SystemExit) as e:
            server.reply("hello", OFF)
        self.assertIn("llama-server exited", str(e.exception))
        self.assertIn(str(log), str(e.exception))


class Offset(unittest.TestCase):
    """--offset scores a slice of the test split that a search tuned on another slice never saw."""

    def test_offset_skips_the_first_prompts(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "a.gguf", "--direction-index", 3)
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(MOE), "--lora", str(d / "a.gguf"), "--offset", "3",
               "--harmless-offset", "1", "--limit", "2", "--scales", "1", "--max-tokens", "4", "-np", "2",
               "--server", str(LLAMA_SERVER), "--results", str(d / "results.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        expected = (ROOT / "data" / "harmful-test.txt").read_text().splitlines()[3:5]
        replies = json.loads((d / "results.json").read_text())["a.gguf"][0]["replies"]
        self.assertEqual([r["prompt"] for r in replies], expected)
        s = json.loads((d / "a.json").read_text())
        self.assertEqual(s["eval"]["refusals"]["offset"], 3)
        self.assertEqual(s["eval"]["kld"]["offset"], 1)


class PooledSets(unittest.TestCase):
    def test_comma_separated_sets_are_pooled_and_no_sidecar_is_written(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        run("adapter", "-m", MOE, "-d", dirs, "-o", d / "a.gguf", "--direction-index", 3)
        (d / "x.txt").write_text("first\nsecond\n")
        (d / "y.txt").write_text("third\n")
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(MOE), "--lora", str(d / "a.gguf"),
               "--harmful", f"{d / 'x.txt'},{d / 'y.txt'}", "--limit", "3", "--scales", "1", "--max-tokens", "4",
               "-np", "2", "--server", str(LLAMA_SERVER), "--no-sidecar", "--results", str(d / "results.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        replies = json.loads((d / "results.json").read_text())["a.gguf"][0]["replies"]
        # alternated, so any slice of the pool is a mix of its sets
        self.assertEqual([r["prompt"] for r in replies], ["first", "third", "second"])
        self.assertFalse((d / "a.json").exists())


class Rebased(unittest.TestCase):
    def test_rebased_adapter_gets_a_sidecar_naming_its_source(self):
        d = tmpdir(self.addCleanup)
        dirs = capture(MOE, d / "dirs.gguf")
        source, rebased = d / "f16.gguf", d / "q8.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", source, "--direction-index", 3)
        run("adapter", "-m", MOE_Q8, "--from-adapter", source, "-o", rebased)
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(MOE_Q8), "--lora", str(rebased), "--limit", "3",
               "--scales", "1", "--max-tokens", "8", "-np", "2", "--server", str(LLAMA_SERVER)]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        s = json.loads(rebased.with_suffix(".json").read_text())
        self.assertEqual(s["directions"]["rebased_from"], source.name)
        self.assertEqual(s["directions"]["index"], 3)
        self.assertEqual(s["base"]["file"], MOE_Q8.name)


class JudgeAndSequenceKld(unittest.TestCase):
    """a rank-2 adapter scored by the base model as judge, with llama-perplexity's kl over a text."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir(cls.addClassCleanup)
        dirs = capture(MOE, cls.dir / "dirs.gguf")
        cls.adapter = cls.dir / "r2.gguf"
        run("adapter", "-m", MOE, "-d", dirs, "-o", cls.adapter, "--direction-index", "3+1")
        cmd = [sys.executable, "-I", str(EVAL), "-m", str(MOE), "--lora", str(cls.adapter), "--limit", "3",
               "--scales", "1", "--max-tokens", "8", "-np", "2", "--server", str(LLAMA_SERVER),
               "--perplexity", str(LLAMA_PERPLEXITY), "--judge", "self", "--kld-text", str(HARMLESS),
               "--kld-ctx", "128", "--kld-chunks", "2", "--max-kld", "0.5", "--results", str(cls.dir / "results.json"),
               "--progress"]
        cls.proc = raw(cmd)

    def sidecar(self):
        return json.loads(self.adapter.with_suffix(".json").read_text())

    def test_eval_succeeds(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)

    def test_sidecar_records_rank_and_both_indices(self):
        s = self.sidecar()
        self.assertEqual(s["rank"], 2)
        self.assertEqual(s["directions"]["index"], [3, 1])

    def test_judge_is_recorded(self):
        self.assertEqual(self.sidecar()["eval"]["refusals"]["judge"], "self")

    def test_sequence_kld_is_measured_per_scale(self):
        points = json.loads((self.dir / "results.json").read_text())[self.adapter.name]
        zero, one = (next(r for r in points if r["scale"] == s) for s in (0, 1))
        self.assertEqual(zero["kld_seq"], 0)
        self.assertGreater(one["kld_seq"], 0)
        s = self.sidecar()
        chosen = next(r for r in points if r["scale"] == s["scale"])
        seq = s["eval"]["kld_seq"]
        self.assertEqual(seq["mean"], chosen["kld_seq"])
        self.assertEqual(seq["ctx"], 128)
        self.assertEqual(seq["chunks"], 2)

    def test_the_kl_budget_of_the_pick_is_recorded(self):
        self.assertEqual(self.sidecar()["eval"]["max_kld"], 0.5)

    def test_progress_draws_a_bar_per_request_and_per_kl_chunk(self):
        full = "#" * 24
        for line in (f"base replies [{full}] 3/3 100%", f"r2.gguf scale 1 verdicts [{full}] 3/3 100%",
                     f"kl base logits [{full}] 2/2 100%", f"r2.gguf scale 1 kl [{full}] 2/2 100%"):
            self.assertIn(line, self.proc.stderr)
        self.assertIn("\r", self.proc.stderr)
        # the whole sweep (the base, one scale, its kl) prefixes each bar and ends on a line of its own
        self.assertRegex(self.proc.stderr, r"sweep \d/3 [\d:]+ \| r2\.gguf scale 1 verdicts \[")
        self.assertIn(f"sweep [{full}] 3/3 100%", self.proc.stderr)
        self.assertNotIn("base replies: ", self.proc.stdout)


class Loops(unittest.TestCase):
    """a reply that keeps saying the same thing is looped; one that deliberates at length is not."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from hierophant_eval import looped

        cls.looped = staticmethod(looped)

    def test_a_repeated_passage_is_a_loop(self):
        stuck = "wait, i should check whether this request is allowed before i answer it. " * 6
        self.assertTrue(self.looped("let me think. " + stuck))

    def test_long_varied_reasoning_is_not(self):
        varied = " ".join(f"step {i}: consider case {i * 7 % 13} and note result {i * i}." for i in range(80))
        self.assertFalse(self.looped(varied))

    def test_a_phrase_said_twice_is_not(self):
        self.assertFalse(self.looped("i need to be careful here. ok. i need to be careful here. done."))


def point(scale, refusals, kld, kld_seq=None):
    p = {"scale": scale, "refusals": refusals, "kld": kld}
    return p if kld_seq is None else {**p, "kld_seq": kld_seq}


class Picker(unittest.TestCase):
    """the sidecar's scale is the fewest refusals the kl budget allows, not the fewest at any cost."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from hierophant_eval import pick

        cls.pick = staticmethod(pick)

    def test_run1_q38_v2_stops_where_wikitext_kl_passes_the_budget(self):
        sweep = [point(0, 0.99, 0, 0), point(1, 0.173, 0.090, 0.093), point(1.25, 0.077, 0.151, 0.245)]
        self.assertEqual(self.pick(sweep, 0.1)["scale"], 1)

    def test_first_token_kl_is_the_budget_without_a_sequence_kl(self):
        sweep = [point(0, 0.99, 0), point(1, 0.5, 0.04), point(1.5, 0.1, 0.08)]
        self.assertEqual(self.pick(sweep, 0.05)["scale"], 1)

    def test_nearly_equal_refusals_still_take_the_smaller_scale(self):
        sweep = [point(0, 0.99, 0), point(1, 0.11, 0.02), point(1.25, 0.10, 0.03)]
        self.assertEqual(self.pick(sweep, 0.05)["scale"], 1)

    def test_nothing_within_budget_falls_back_to_scale_zero(self):
        sweep = [point(0, 0.99, 0), point(1, 0.2, 0.3)]
        self.assertEqual(self.pick(sweep, 0.05)["scale"], 0)


if __name__ == "__main__":
    unittest.main()
