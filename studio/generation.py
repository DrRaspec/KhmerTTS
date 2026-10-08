"""Speech generation, model switching, cancellation, and timing estimates."""

import hashlib
import json
import re
import time
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from uuid import uuid4

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from .config import HARDWARE, RUNTIME_DEVICE, STEP_RECOMMENDATION, ESTIMATES, OUTPUT_DIR
from .generation_estimates import estimate_value
from .speech_models import (
    FULL_MODEL, VOX_MODELS, VOX_CHUNK_CHARS, GenerationCancelled,
    SpeechModelManager, generation_executor, model_language, split_vox_text,
)
from .voices import (
    VOICES, SCRIPT_SAMPLES, build_voice_prompt, get_voice_reference,
    safe_filename, styles_for_model, voices_for_model,
)


MODEL_MANAGER = SpeechModelManager(RUNTIME_DEVICE)


def prepare_waveform(wav):
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()

    wav = np.asarray(wav)
    wav = np.squeeze(wav)

    return wav


def generation_worker(task, text, options):
    engine = MODEL_MANAGER.load(task)
    task.check()
    task.phase = "Creating your audio"
    task.inference_started = time.perf_counter()
    print(f"Model ready | {task.model_key} | Device: {engine.device}")
    if isinstance(text, list):
        print(f"Creating {len(text)} short sections")
        sections = []
        for index, section in enumerate(text):
            task.check()
            task.phase = f"Creating section {index + 1} of {len(text)}"
            print(f"Section {index + 1}/{len(text)}")
            audio = prepare_waveform(engine.generate(section, options, task))
            task.check()
            if audio.size == 0:
                raise ValueError(f"Section {index + 1} produced no audio.")
            if sections:
                sections.append(np.zeros(round(engine.sample_rate * 0.25), dtype=np.float32))
            sections.append(audio)
        wav = np.concatenate(sections)
    else:
        wav = engine.generate(text, options, task)
    task.check()
    return wav, engine.sample_rate, time.perf_counter() - task.inference_started


def cancel_generation(request: gr.Request):
    owner = request.session_hash if request else "local"
    if MODEL_MANAGER.cancel(owner):
        return "**Stopping…** Waiting for the current computation to finish. No audio will be saved."
    return "**No active generation to cancel in this session.**"


def model_description(key):
    language = model_language(key)
    if key not in VOX_MODELS:
        return (
            f"One {language} voice · Low memory · **Noncommercial only**"
        )
    return (
        f"{language} voice design & cloning · High memory"
    )


def switch_model(key, script=None):
    try:
        MODEL_MANAGER.switch(key)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    full = key in VOX_MODELS
    voices = voices_for_model(key)
    language = model_language(key)
    script_update = gr.update(
        placeholder="Write your English script here…" if language == "English" else "សរសេរអត្ថបទខ្មែររបស់អ្នកនៅទីនេះ…",
    )
    if not script or script in SCRIPT_SAMPLES.values():
        script_update["value"] = SCRIPT_SAMPLES[language]
    return (
        key, model_description(key),
        gr.update(
            choices=voices, value=voices[0], interactive=full,
            info=None,
        ),
        gr.update(choices=styles_for_model(key), value="Natural", interactive=full),
        gr.update(interactive=full), gr.update(interactive=full),
        gr.update(interactive=full), gr.update(interactive=full),
        gr.update(interactive=full), gr.update(interactive=full),
        "**Ready.** Model selected.",
        script_update,
    )


def estimate_signature(model_key, steps, voice, style, custom_style, reference):
    full = model_key in VOX_MODELS
    signature = {
        "hardware": HARDWARE, "torch": str(torch.__version__), "model": model_key,
        "steps": int(steps) if full else None,
        "voice": voice if full else None, "style": style if full else None,
        "custom_style": custom_style if full else None,
        "reference": str(reference) if full and reference else None,
        "chunk_chars": VOX_CHUNK_CHARS if full else None,
    }
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def preview_estimate(text, voice, style, custom_style, reference, steps, model_key):
    reference = reference or VOICES.get(voice, {}).get("reference")
    key = estimate_signature(model_key, steps, voice, style, custom_style, reference)
    count = len(re.sub(r"\s", "", text or ""))
    return f"Estimate: **{estimate_value(ESTIMATES.estimate(key, count))}**"


def generate_audio(
    text, selected_voice, selected_style, custom_style, temporary_reference,
    cfg_value, inference_steps, model_key=FULL_MODEL,
    request: gr.Request = None, progress=gr.Progress(),
):
    if not text or not text.strip():
        raise gr.Error("Please enter a script.")
    text = text.strip()
    if len(text) > 4000:
        raise gr.Error("Please keep the script below 4000 characters.")
    owner = request.session_hash if request else "local"
    try:
        task = MODEL_MANAGER.begin(owner, model_key)
    except (RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    try:
        yield None, None, "### Preparing voice"
        progress(None, desc="Preparing text and voice")
        full = model_key in VOX_MODELS
        reference_path = None
        generation_text = text
        options = {}
        if full:
            if not selected_voice:
                raise gr.Error("Please choose a speaker.")
            reference_path = get_voice_reference(selected_voice, temporary_reference)
            voice_prompt = build_voice_prompt(
                selected_voice, selected_style, custom_style, has_reference=bool(reference_path),
            )
            chunks = split_vox_text(text)
            prompted_chunks = [f"({voice_prompt}) {chunk}" for chunk in chunks]
            generation_text = prompted_chunks[0] if len(prompted_chunks) == 1 else prompted_chunks
            options = {
                "cfg_value": float(cfg_value), "inference_timesteps": int(inference_steps),
                "normalize": False, "retry_badcase": False,
            }
            if reference_path:
                options["reference_wav_path"] = reference_path
            if int(inference_steps) > STEP_RECOMMENDATION["range_max"]:
                gr.Info("This step setting exceeds the suggested range and may take longer.")

        started = time.perf_counter()
        estimate_key = estimate_signature(
            model_key, inference_steps, selected_voice, selected_style, custom_style, reference_path,
        )
        character_count = len(re.sub(r"\s", "", text))
        estimate = ESTIMATES.estimate(estimate_key, character_count)
        print(f"\nStarting speech generation | {model_key}")
        if full:
            print(
                f"Voice: {selected_voice} | Style: {selected_style} | "
                f"CFG: {float(cfg_value):g} | Steps: {int(inference_steps)} | "
                f"Reference: {'yes' if reference_path else 'none'}"
            )
        try:
            progress(None, desc="Loading model / creating audio")
            with generation_executor(task) as executor:
                context = copy_context()
                future = executor.submit(context.run, generation_worker, task, generation_text, options)
                while True:
                    elapsed = time.perf_counter() - started
                    minutes, seconds = divmod(int(elapsed), 60)
                    stopping = task.cancelled.is_set()
                    phase = "Stopping" if stopping else task.phase.replace("your audio", "audio")
                    inference_elapsed = (
                        time.perf_counter() - task.inference_started
                        if task.inference_started is not None else 0
                    )
                    timing = (
                        "—" if stopping or task.phase == "Loading model"
                        else estimate_value(estimate, inference_elapsed)
                    )
                    yield None, None, (
                        f"**{phase}**\n\n"
                        f"Elapsed **{minutes:02d}:{seconds:02d}** · Remaining **{timing}**"
                    )
                    try:
                        wav, sample_rate, inference_seconds = future.result(timeout=1)
                        break
                    except FutureTimeout:
                        if future.done():
                            wav, sample_rate, inference_seconds = future.result()
                            break
            task.check()
        except GenerationCancelled:
            progress(None, desc="Cancelled")
            yield None, None, "**Generation cancelled.** No audio was saved. You can switch models or try again."
            return
        except Exception as error:
            print(f"Generation error: {error}")
            if task.cancelled.is_set():
                yield None, None, "**Generation cancelled.** No audio was saved."
                return
            yield None, None, (
                "**We couldn’t finish this audio.** Check the terminal for details. "
                "Try a shorter script or switch models."
            )
            return

        wav = prepare_waveform(wav)
        generation_seconds = time.perf_counter() - started
        audio_seconds = len(wav) / sample_rate
        progress(None, desc="Saving your audio")
        yield None, None, "**Saving your audio…** Preparing the player and WAV download."
        # Cancellation after inference still discards the result before saving.
        if task.cancelled.is_set():
            yield None, None, "**Generation cancelled.** No audio was saved."
            return
        filename = safe_filename(selected_voice if full else f"mms_{model_language(model_key).lower()}")
        output_path = OUTPUT_DIR / f"{filename}_{uuid4().hex[:8]}.wav"
        try:
            sf.write(output_path, wav, sample_rate)
        except Exception as error:
            print(f"Audio save error: {error}")
            yield None, None, "**Audio could not be saved.** Check disk space and the terminal error."
            return
        try:
            MODEL_MANAGER.commit(task)
        except GenerationCancelled:
            output_path.unlink(missing_ok=True)
            yield None, None, "**Generation cancelled.** No audio was saved."
            return
        print(
            f"Audio ready | Duration: {audio_seconds:.1f}s | "
            f"Total time: {generation_seconds:.1f}s "
            f"(inference: {inference_seconds:.1f}s)\n"
            f"Saved to: {output_path}"
        )
        ESTIMATES.record(estimate_key, character_count, inference_seconds)
        progress(1, desc="Audio ready")
        yield str(output_path), str(output_path), (
            f"### Your audio is ready\n\n"
            f"**Audio:** {audio_seconds:.1f}s · **Generation:** {generation_seconds:.1f}s"
        )
    finally:
        # The executor has joined before this point. Switching cannot race a worker.
        MODEL_MANAGER.finish(task)


def generate_named_take(
    text, selected_voice, selected_style, custom_style, temporary_reference,
    cfg_value, inference_steps, model_key=FULL_MODEL,
    request: gr.Request = None, progress=gr.Progress(),
):
    for audio, download, status in generate_audio(
        text, selected_voice, selected_style, custom_style, temporary_reference,
        cfg_value, inference_steps, model_key, request, progress,
    ):
        take = {"path": audio, "language": model_language(model_key)} if audio else None
        yield audio, download, status, take
