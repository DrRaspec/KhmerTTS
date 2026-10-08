"""Select a supported inference backend from the machine running the app."""

import os
import platform

import torch


def recommend_steps(hardware):
    """Conservative starting points, not benchmarked speed or safety limits."""
    info = hardware if isinstance(hardware, dict) else {"device": hardware}
    device = info["device"]
    if device.startswith("cuda"):
        memory = info.get("gpu_memory_gib")
        processors = info.get("gpu_multiprocessors")
        architecture = info.get("gpu_compute_major")
        # SM counts are only a rough capability indicator across generations.
        # Require both memory headroom and compute capacity for higher defaults.
        if memory is not None and processors is not None and architecture is not None:
            if architecture >= 8 and memory >= 24 and processors >= 120:
                return {"default": 20, "range_min": 10, "range_max": 30}
            if architecture >= 8 and memory >= 16 and processors >= 80:
                return {"default": 15, "range_min": 10, "range_max": 25}
            if memory < 8 or processors < 30:
                return {"default": 8, "range_min": 4, "range_max": 10}
        return {"default": 10, "range_min": 10, "range_max": 20}
    if device == "mps":
        return {"default": 10, "range_min": 8, "range_max": 15}
    return {"default": 6, "range_min": 4, "range_max": 10}


def detect_hardware(requested_device="auto"):
    requested = (requested_device or "auto").strip().lower()
    mps = getattr(torch.backends, "mps", None)
    cuda_available = torch.cuda.is_available()
    mps_available = mps is not None and mps.is_available()

    if requested == "auto":
        device = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif requested == "cpu":
        device = "cpu"
    elif requested == "mps":
        if not mps_available:
            raise ValueError("Requested MPS GPU is unavailable. Use VOXCPM_DEVICE=auto or cpu.")
        device = "mps"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available:
            raise ValueError("Requested CUDA GPU is unavailable. Use VOXCPM_DEVICE=auto or cpu.")
        index_text = requested.partition(":")[2]
        if requested != "cuda" and not index_text.isdecimal():
            raise ValueError("CUDA device must be cuda or cuda:<non-negative index>.")
        index = int(index_text) if index_text else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable.")
        device = requested
    else:
        raise ValueError("VOXCPM_DEVICE must be auto, cpu, mps, cuda, or cuda:<index>.")

    info = {
        "device": device,
        "system": platform.system(),
        "architecture": platform.machine(),
        "cpu_threads": os.cpu_count(),
        "accelerator": "CPU",
        "gpu_memory_gib": None,
        "gpu_multiprocessors": None,
        "gpu_compute_major": None,
    }
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(device))
        info["accelerator"] = props.name
        info["gpu_memory_gib"] = props.total_memory / 1024**3
        info["gpu_multiprocessors"] = getattr(props, "multi_processor_count", None)
        info["gpu_compute_major"] = getattr(props, "major", None)
    elif device == "mps":
        info["accelerator"] = "Apple GPU (shared system memory)"
    return info
