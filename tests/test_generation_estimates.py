import tempfile
import unittest
from pathlib import Path

from studio.generation_estimates import GenerationEstimates, estimate_message


class EstimateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.estimates = GenerationEstimates(Path(self.temp.name) / "timings.json")

    def test_no_history_has_no_made_up_eta(self):
        self.assertIsNone(self.estimates.estimate("model-a", 100))
        self.assertIn("Learning speed", estimate_message(None))

    def test_estimate_scales_only_comparable_successful_runs(self):
        self.estimates.record("model-a", 100, 20)
        self.assertEqual(self.estimates.estimate("model-a", 200), (20, 80))
        self.assertIsNone(self.estimates.estimate("model-b", 200))
        self.assertIsNone(self.estimates.estimate("model-a", 1000))

    def test_corrupt_history_and_invalid_samples_are_ignored(self):
        self.estimates.path.write_text("not json")
        self.assertIsNone(self.estimates.estimate("model-a", 100))
        self.estimates.record("model-a", 0, 20)
        self.estimates.record("model-a", 100, float("nan"))
        self.assertIsNone(self.estimates.estimate("model-a", 100))

    def test_exceeded_estimate_does_not_display_zero_remaining(self):
        self.assertIn("Taking longer", estimate_message((10, 40), 45))
        self.assertIn("5s–35s", estimate_message((10, 40), 5))

    def test_history_persists_across_restarts(self):
        self.estimates.record("model-a", 100, 20)
        restored = GenerationEstimates(self.estimates.path)
        self.assertEqual(restored.estimate("model-a", 100), (10, 40))


if __name__ == "__main__":
    unittest.main()
