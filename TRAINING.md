# Khmer voice training starter

The app runs VoxCPM2. `training/prepare_data.py` downloads a bounded pilot,
not a full corpus, and does not start training or change the studio's voices.
The earlier Dara and Rithy references are generated samples, not dataset speakers.

## Prepare clips

Run from this project directory:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m training.prepare_data --count 24
```

The downloader accepts 2–20 second Khmer clips, checks for empty/nonfinite audio,
converts to mono PCM16 WAV at 16 kHz, and preserves the original transcript.
It uses a small Hugging Face preview page rather than downloading full parquet
shards. Re-running reuses existing WAVs and rebuilds absolute manifest paths.
Run it again after moving this project to a training machine.

| Source | Use in this pilot | License / attribution |
| --- | --- | --- |
| [OpenSLR42 mirror](https://huggingface.co/datasets/deepdml/openslr42-khmer-tts) | One speaker, `0308`; separate train/validation | CC-BY-SA-4.0; Google LLC; mirror by deepdml |
| [Digital Divide Data](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset) | One speaker, `f-adt1-0001`; separate train/validation | CC-BY-SA-4.0; Digital Divide Data |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | Khmer `km_kh`, validation split; evaluation only | CC-BY-4.0; Google, Conneau et al. (2022) |
| [Common Voice / Mozilla Data Collective](https://mozilladatacollective.com/api-reference) | Pending authenticated download and acceptance of dataset terms | Preserve the exact downloaded release's license and terms |

OpenSLR and DDD remain separate so each pilot adapts one identified speaker.
FLEURS' preview has no speaker identity field; gender and sentence IDs are not
speaker IDs, so those clips are not used to define a named person. Common Voice
needs account access and dataset terms accepted through Mozilla; do not bypass
that access step or substitute an unofficial mirror.

Each source under `data/khmer_pilot/` has `audio/`, `provenance.jsonl`, and
`summary.json`. Provenance includes dataset revision, source row, speaker ID,
license, attribution, transcript, audio hash, and listening-review status.
Single-speaker sources also have `train.jsonl`, `validation.jsonl`, and
`pilot_lora.yaml`. Identical normalized transcripts stay in the same partition.
FLEURS has only `evaluation.jsonl`. Downloaded data is excluded from Git.

Listen to every pilot clip and check its transcript before a quality run. These
automated checks cannot establish pronunciation, recording cleanliness, or text
accuracy. The small, sequential sample is for testing the pipeline, not a
representative language training corpus. Preserve the listed attribution and
license terms when using or redistributing dataset material.

## Actual training

Use the [official VoxCPM trainer](https://github.com/OpenBMB/VoxCPM/blob/main/scripts/train_voxcpm_finetune.py)
and [fine-tuning guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html).
The current studio environment has VoxCPM 2.0.3; set up training in a separate
environment with the trainer's matching dependencies.

The guide estimates about 20 GB VRAM for VoxCPM2 LoRA under its reference
configuration; that is not a guarantee for this pilot. CUDA training is the
documented route. This Mac's training compatibility and memory capacity have
not been validated, and no training process has been launched.

For a chosen training machine:

1. Prepare the data there, or re-run the importer to refresh local paths.
2. Obtain the official trainer and install its training dependencies in an
   isolated environment. Download the `openbmb/VoxCPM2` base model there.
3. Set `pretrained_path` in the chosen source's `pilot_lora.yaml` to that model
   directory. The generated `models/VoxCPM2` path is a placeholder.
4. Validate both manifests using `voxcpm validate --manifest PATH --sample-rate 16000`.
5. From the official trainer checkout, run:

```bash
python scripts/train_voxcpm_finetune.py --config_path /absolute/path/to/data/khmer_pilot/openslr42/pilot_lora.yaml
```

The generated configuration is a 100-step pipeline test. Batch size 1 and
gradient accumulation 8 are starting settings, not measured memory guarantees.
Do not treat a completed test as a quality improvement: compare held-out speech,
speaker similarity, pronunciation, stopping behavior, and FLEURS evaluation.
Then expand the selected speaker's recordings and tune training duration based
on those comparisons. Training separate speaker adapters requires separate runs.

No adapter is integrated into the studio until one has actually been trained
and evaluated. A saved reference voice can be used without model training.
