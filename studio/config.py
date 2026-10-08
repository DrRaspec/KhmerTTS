"""Runtime settings and data locations shared by the studio."""

import os
from pathlib import Path

from .runtime_config import detect_hardware, recommend_steps
from .generation_estimates import GenerationEstimates


HARDWARE = detect_hardware(os.environ.get("VOXCPM_DEVICE", "auto"))
RUNTIME_DEVICE = HARDWARE["device"]
STEP_RECOMMENDATION = recommend_steps(HARDWARE)
ESTIMATES = GenerationEstimates(".generation_timings.json")

OUTPUT_DIR = Path("outputs")
VOICE_DIR = Path("voices")
VOICE_DB = Path("voices.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)
