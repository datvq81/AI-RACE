"""Unit tests for radical experiment gate evaluation."""

from __future__ import annotations

import unittest

from scripts.run_radical_suite import _evaluate_gate


def _artifact(score: float, count: int = 4) -> dict:
    return {
        "metrics": {"score": score, "num_images": count},
        "render_image_count": count,
        "split_validation_count": count,
    }


class RadicalGateTest(unittest.TestCase):
    def test_score_and_render_gate_pass(self) -> None:
        suite = {
            "radical_gate": {
                "requirements": [
                    {"kind": "render_count_matches", "runs": ["P0", "P1"]},
                    {
                        "kind": "score_delta_min",
                        "baseline": "P0",
                        "candidate": "P1",
                        "value": 0.2,
                    },
                ]
            }
        }
        result = _evaluate_gate(
            suite,
            {"P0": _artifact(72.0), "P1": _artifact(72.25)},
        )
        self.assertEqual(result["status"], "passed")

    def test_missing_metrics_make_gate_unavailable(self) -> None:
        suite = {
            "radical_gate": {
                "requirements": [
                    {
                        "kind": "score_abs_delta_max",
                        "baseline": "R0C0",
                        "candidate": "R0C1",
                        "value": 0.1,
                    }
                ]
            }
        }
        result = _evaluate_gate(suite, {})
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
