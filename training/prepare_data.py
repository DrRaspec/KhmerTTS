"""Download a bounded Khmer pilot set; keep speaker identities and provenance.

Run from the project root. This prepares data; it does not start training.
"""

import argparse
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import soundfile as sf
import yaml
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
MIN_DURATION = 2
MAX_DURATION = 20
MIN_RMS = 0.0001
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class Source:
    dataset: str
    text_field: str
    license: str
    attribution: str
    config: str = "default"
    split: str = "train"
    evaluation_only: bool = False


SOURCES = {
    "openslr42": Source(
        "deepdml/openslr42-khmer-tts", "transcription", "CC-BY-SA-4.0",
        "Google LLC, OpenSLR resource 42; Hugging Face mirror by deepdml",
    ),
    "ddd": Source(
        "Digital-Divide-Data/khmer-speech-dataset", "transcript", "CC-BY-SA-4.0",
        "Digital Divide Data, Khmer ASR Cultural Dataset",
    ),
    "fleurs": Source(
        "google/fleurs", "raw_transcription", "CC-BY-4.0",
        "Google, FLEURS (Conneau et al., 2022)", config="km_kh",
        split="validation", evaluation_only=True,
    ),
}


def get_response(session, url, **kwargs):
    response = session.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    return response


def write_jsonl(path, records):
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def valid_audio(audio, rate):
    duration = len(audio) / rate
    return (
        rate == SAMPLE_RATE
        and audio.ndim == 1
        and np.isfinite(audio).all()
        and MIN_DURATION <= duration <= MAX_DURATION
        and np.sqrt(np.mean(audio**2)) >= MIN_RMS
    )


def convert_audio(raw):
    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all() or not MIN_DURATION <= len(audio) / rate <= MAX_DURATION:
        return None
    if np.sqrt(np.mean(audio**2)) < MIN_RMS:
        return None
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    return audio


def audio_asset(row):
    assets = row.get("audio", [])
    return assets[0].get("src") if isinstance(assets, list) and assets else None


def split_records(records):
    """Hold out whole transcript groups, including repeated recordings."""
    groups = {}
    for record in records:
        key = " ".join(record["text"].split())
        groups.setdefault(key, []).append(record)
    keys = sorted(groups, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    if len(keys) < 3:
        raise ValueError("Need at least three distinct transcripts for a train/validation split.")
    held_out = set(keys[:max(1, math.ceil(len(keys) * 0.1))])
    train, validation = [], []
    for key, group in groups.items():
        (validation if key in held_out else train).extend(group)
    return train, validation


def write_pilot_config(folder):
    # Start with a short pipeline test, not a production-quality training run.
    config = {
        "pretrained_path": str(Path("models/VoxCPM2").resolve()),
        "train_manifest": str((folder / "train.jsonl").resolve()),
        "val_manifest": str((folder / "validation.jsonl").resolve()),
        "sample_rate": 16000, "out_sample_rate": 48000,
        "batch_size": 1, "grad_accum_steps": 8,
        "num_workers": 0, "preprocessing_num_workers": 1,
        "num_iters": 100, "max_steps": 100,
        "log_interval": 10, "valid_interval": 50, "save_interval": 50,
        "learning_rate": 0.0001, "weight_decay": 0.01, "warmup_steps": 10,
        "max_batch_tokens": 2048, "max_grad_norm": 1.0,
        "save_path": str(Path("checkpoints/khmer_pilot", folder.name).resolve()),
        "tensorboard": str(Path("logs/khmer_pilot", folder.name).resolve()),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {"enable_lm": True, "enable_dit": True, "enable_proj": False,
                 "r": 32, "alpha": 32, "dropout": 0.0},
    }
    (folder / "pilot_lora.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def prepare_source(session, label, root, count, speaker=None):
    spec = SOURCES[label]
    folder = root / label
    audio_folder = folder / "audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    # Fetch only one small preview page: never load the complete parquet dataset.
    preview = get_response(session, DATASET_ROWS_URL, params={
        "dataset": spec.dataset, "config": spec.config, "split": spec.split,
        "offset": 0, "length": min(100, count * 3),
    }).json()
    accepted, provenance = [], []
    chosen_speaker = speaker
    for item in preview["rows"]:
        row = item["row"]
        if item.get("truncated_cells"):
            continue
        text = row.get(spec.text_field, "").strip()
        if not text or not re.search(r"[\u1780-\u17ff]", text):
            continue
        identity = str(row.get("speaker_id", "unknown"))
        if not spec.evaluation_only:
            if chosen_speaker is None:
                chosen_speaker = identity
            if identity == "unknown" or identity != chosen_speaker:
                continue
        duration = row.get("duration")
        if duration is not None and not MIN_DURATION <= float(duration) <= MAX_DURATION:
            continue
        url = audio_asset(row)
        if not url:
            continue
        path = audio_folder / f'{item["row_idx"]:06d}.wav'
        if not path.exists():
            audio = convert_audio(get_response(session, url).content)
            if audio is None:
                continue
            sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        audio, rate = sf.read(path)
        if not valid_audio(audio, rate):
            raise ValueError(f"Invalid cached audio: {path}")
        record = {"audio": str(path.resolve()), "text": text, "duration": len(audio) / rate}
        accepted.append(record)
        revision = re.search(r"/--/([a-f0-9]{40})/--/", url)
        provenance.append({
            **record, "dataset": spec.dataset, "config": spec.config,
            "source_split": spec.split, "row_index": item["row_idx"],
            "source_revision": revision.group(1) if revision else None,
            "speaker_id": identity, "license": spec.license,
            "attribution": spec.attribution,
            "source_id": row.get("filename", row.get("sentence_id", row.get("id"))),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "listening_review": "pending",
        })
        print(f"{label}: {len(accepted)}/{count} clips", flush=True)
        if len(accepted) >= count:
            break
    if not accepted:
        raise ValueError(f"No usable clips found for {label} in the bounded preview page.")
    write_jsonl(folder / "provenance.jsonl", provenance)
    if spec.evaluation_only:
        write_jsonl(folder / "evaluation.jsonl", accepted)
        partitions = {"evaluation": len(accepted)}
    else:
        train, validation = split_records(accepted)
        write_jsonl(folder / "train.jsonl", train)
        write_jsonl(folder / "validation.jsonl", validation)
        write_pilot_config(folder)
        partitions = {"train": len(train), "validation": len(validation)}
    report = {
        "dataset": spec.dataset, "speaker_id": chosen_speaker,
        "clips": len(accepted), "seconds": round(sum(r["duration"] for r in accepted), 2),
        "partitions": partitions, "license": spec.license,
        "listening_review": "pending", "purpose": "pilot, not a production training corpus",
    }
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("data/khmer_pilot"))
    args = parser.parse_args()
    if not 3 <= args.count <= 33:
        parser.error("Pilot count must be between 3 and 33; use a full dataset pipeline for larger runs.")
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"],
    )))
    failures = []
    for label in args.sources:
        try:
            print(json.dumps(prepare_source(session, label, args.output, args.count)), flush=True)
        except (requests.RequestException, ValueError, OSError) as error:
            failures.append(label)
            detail = str(error)
            if isinstance(error, requests.RequestException):
                status = error.response.status_code if error.response is not None else "network error"
                detail = f"Download failed ({status}); rerun to resume cached clips."
            print(f"{label}: FAILED: {detail}", flush=True)
    if failures:
        raise SystemExit("Incomplete sources: " + ", ".join(failures))


if __name__ == "__main__":
    main()
