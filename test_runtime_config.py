import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime_config import detect_hardware, recommend_steps


class HardwareDetectionTests(unittest.TestCase):
    def detect(self, cuda=False, mps=False, requested="auto", count=1):
        with patch("runtime_config.torch.cuda.is_available", return_value=cuda), \
             patch("runtime_config.torch.backends.mps.is_available", return_value=mps), \
             patch("runtime_config.torch.cuda.device_count", return_value=count), \
             patch("runtime_config.torch.cuda.current_device", return_value=0), \
             patch("runtime_config.torch.cuda.get_device_properties", return_value=SimpleNamespace(
                 name="Test GPU", total_memory=12 * 1024**3)):
            return detect_hardware(requested)

    def test_cuda_preferred_when_both_available(self):
        info = self.detect(cuda=True, mps=True)
        self.assertEqual(info["device"], "cuda")
        self.assertEqual(info["gpu_memory_gib"], 12)

    def test_mps_when_cuda_unavailable(self):
        self.assertEqual(self.detect(mps=True)["device"], "mps")

    def test_cpu_when_no_accelerator_available(self):
        self.assertEqual(self.detect()["device"], "cpu")

    def test_explicit_cpu_overrides_gpu(self):
        self.assertEqual(self.detect(cuda=True, mps=True, requested="CPU")["device"], "cpu")

    def test_indexed_cuda(self):
        self.assertEqual(self.detect(cuda=True, requested="cuda:1", count=2)["device"], "cuda:1")

    def test_unavailable_or_invalid_override_fails(self):
        for requested in ("mps", "cuda", "cuda:0", "other"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                self.detect(requested=requested)
        for requested in ("cuda:2", "cuda:-1", "cuda:", "cuda:no"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                self.detect(cuda=True, requested=requested)

    def test_recommended_steps_follow_selected_backend(self):
        for device, default, low, high in (
            ("cpu", 6, 4, 10), ("mps", 10, 8, 15),
            ("cuda", 10, 10, 20), ("cuda:1", 10, 10, 20),
        ):
            with self.subTest(device=device):
                self.assertEqual(recommend_steps(device), {
                    "default": default, "range_min": low, "range_max": high,
                })

    def test_cuda_capability_tiers(self):
        for memory, processors, major, default, maximum in (
            (6, 20, 8, 8, 10),
            (12, 46, 8, 10, 20),
            (24, 82, 8, 15, 25),
            (24, 128, 8, 20, 30),
            (48, 128, 7, 10, 20),
            (8, 128, 8, 10, 20),
        ):
            with self.subTest(memory=memory, processors=processors, major=major):
                result = recommend_steps({
                    "device": "cuda", "gpu_memory_gib": memory,
                    "gpu_multiprocessors": processors, "gpu_compute_major": major,
                })
                self.assertEqual(result["default"], default)
                self.assertEqual(result["range_max"], maximum)

    def test_missing_cuda_specs_use_baseline(self):
        self.assertEqual(recommend_steps({"device": "cuda", "gpu_memory_gib": 24})["default"], 10)


if __name__ == "__main__":
    unittest.main()
