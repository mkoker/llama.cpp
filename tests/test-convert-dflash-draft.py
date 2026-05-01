#!/usr/bin/env python3
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "convert-dflash-draft.py"

class ConvertDFlashDraftCliTest(unittest.TestCase):
    def test_help_mentions_dflash_and_conversion_options(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("dflash", proc.stdout.lower())
        self.assertIn("--outfile", proc.stdout)
        self.assertIn("--outtype", proc.stdout)

if __name__ == "__main__":
    unittest.main(verbosity=2)
