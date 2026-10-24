"""fetching the prompt sets against the hashes manifest.json pins, from a fake hub."""

import hashlib
import sys
import unittest
from unittest import mock

from common import ROOT, tmpdir

sys.path.insert(0, str(ROOT / "scripts"))
import fetch_datasets

ROWS = ["first prompt", "second prompt"]


def fake_hub(url):
    if url.startswith(fetch_datasets.INFO_API):
        return {"sha": "abc123"}
    return {"rows": [{"row": {"text": r}} for r in ROWS] if "offset=0" in url else []}


def pin(text):
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()}


class Pinned(unittest.TestCase):
    """the data files are not in the repo, so a fetch must reproduce the ones manifest.json
    describes, or refuse instead of quietly swapping the prompts."""

    def setUp(self):
        self.out = tmpdir(self.addCleanup)
        self.set = fetch_datasets.Set("harmful-train", "x/y", "train", 2)
        patch = mock.patch.object(fetch_datasets, "get_json", fake_hub)
        patch.start()
        self.addCleanup(patch.stop)

    def test_rows_matching_the_pin_are_written(self):
        entry = fetch_datasets.fetch(self.set, self.out, pin("first prompt\nsecond prompt\n"))
        self.assertEqual((self.out / "harmful-train.txt").read_text(), "first prompt\nsecond prompt\n")
        self.assertEqual(entry["revision"], "abc123")

    def test_changed_upstream_rows_are_refused_and_the_old_file_kept(self):
        (self.out / "harmful-train.txt").write_text("old prompt\n")
        with self.assertRaises(SystemExit) as e:
            fetch_datasets.fetch(self.set, self.out, pin("old prompt\n"))
        self.assertIn("--update", str(e.exception.code))
        self.assertEqual((self.out / "harmful-train.txt").read_text(), "old prompt\n")

    def test_without_a_pin_the_rows_are_taken(self):
        entry = fetch_datasets.fetch(self.set, self.out, None)
        self.assertEqual(entry, {**entry, **pin("first prompt\nsecond prompt\n")})


if __name__ == "__main__":
    unittest.main()
