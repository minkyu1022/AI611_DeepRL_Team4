"""Lightweight checks for the final-assignment deliverable artifacts.

These tests intentionally avoid importing torch, OMatG, MACE, or ASE. They lock
the report/package structure that should remain valid even on a machine without
the full GPU environment.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


class AssignmentPackageTests(unittest.TestCase):
    def test_full_eval_results_have_expected_models(self) -> None:
        data = json.loads((ROOT / "experiments" / "full_test_eval.json").read_text())
        self.assertEqual(data["n_structures"], 9046)

        results = data["results"]
        expected = {
            "baseline (pretrained, Nt=50)",
            "velocity-based OMatG-IRL",
            "score-based OMatG-IRL",
        }
        self.assertEqual(set(results), expected)

        baseline = results["baseline (pretrained, Nt=50)"]
        velocity = results["velocity-based OMatG-IRL"]
        score = results["score-based OMatG-IRL"]
        self.assertLess(velocity["mean_rmsd"], baseline["mean_rmsd"])
        self.assertLess(score["mean_rmsd"], baseline["mean_rmsd"])
        self.assertLess(velocity["relative_energy_per_atom"], baseline["relative_energy_per_atom"])
        self.assertLess(score["relative_energy_per_atom"], baseline["relative_energy_per_atom"])

    def test_reproduction_notebook_is_present_and_on_topic(self) -> None:
        nb = json.loads((ROOT / "omatg_irl" / "omatg-irl-reproduce.ipynb").read_text())
        joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        self.assertIn("Section 4.2", joined)
        self.assertIn("OMatG/MP-20-CSP/Trig-SDE-Gamma", joined)
        self.assertIn("velocity-based", joined)
        self.assertIn("score-based", joined)

    def test_dng_extension_notebook_is_present_and_on_topic(self) -> None:
        nb = json.loads((WORKSPACE / "omatg-irl-extension" / "omatg-irl-extension.ipynb").read_text())
        joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        self.assertIn("de novo", joined.lower())
        self.assertIn("S.U.N", joined)
        self.assertIn("research question", joined.lower())


if __name__ == "__main__":
    unittest.main()
