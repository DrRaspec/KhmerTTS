"""Approximate synthesis time from successful, comparable local runs."""

import json
import math
from pathlib import Path
from statistics import median
from threading import Lock


def format_duration(seconds):
    seconds = max(1, math.ceil(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class GenerationEstimates:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = Lock()

    def _read(self):
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def record(self, key, characters, seconds):
        if characters <= 0 or not math.isfinite(seconds) or seconds <= 0:
            return
        with self.lock:
            data = self._read()
            history = data.get(key, [])
            if not isinstance(history, list):
                history = []
            data[key] = (history + [{"characters": characters, "seconds": seconds}])[-8:]
            temporary = self.path.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(data), encoding="utf-8")
                temporary.replace(self.path)
            except OSError as error:
                # Timing history must never turn a successful generation into a failure.
                print(f"Could not save timing history: {error}")

    def estimate(self, key, characters):
        with self.lock:
            history = self._read().get(key, [])
        if not isinstance(history, list):
            return None
        rates = []
        for sample in history:
            if not isinstance(sample, dict):
                continue
            count, seconds = sample.get("characters"), sample.get("seconds")
            if (isinstance(count, (int, float)) and isinstance(seconds, (int, float))
                    and count > 0 and math.isfinite(seconds) and seconds > 0
                    and 0.25 <= characters / count <= 4):
                rates.append(seconds / count)
        if not rates:
            return None
        center = median(rates) * characters
        # Deliberately broad: script delivery and competing apps can change speed.
        return max(1, min(rates) * characters * 0.5), max(2, max(rates) * characters * 2, center * 2)


def estimate_value(estimate, elapsed=0):
    if estimate is None:
        return "Learning speed"
    low, high = estimate
    if elapsed >= high:
        return "Taking longer"
    return (
        f"{format_duration(max(0, low - elapsed))}–{format_duration(high - elapsed)}"
    )


def estimate_message(estimate, elapsed=0):
    return f"**Remaining (approx.):** {estimate_value(estimate, elapsed)}"
