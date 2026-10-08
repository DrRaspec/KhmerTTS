from pathlib import Path
from uuid import uuid4
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context

import json
import hashlib
import os
import re
import shutil
import tempfile
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from runtime_config import detect_hardware, recommend_steps
from generation_estimates import GenerationEstimates, estimate_value
from speech_models import (
    FULL_MODEL, LITE_MODEL, ENGLISH_MODEL, ENGLISH_LITE_MODEL, VOX_MODELS,
    MODEL_CHOICES, GenerationCancelled, SpeechModelManager, generation_executor, model_language,
    split_vox_text, VOX_CHUNK_CHARS,
)


# ============================================================
# Configuration
# ============================================================

# Detect supported accelerators on the host; explicit choices are validated.
HARDWARE = detect_hardware(os.environ.get("VOXCPM_DEVICE", "auto"))
RUNTIME_DEVICE = HARDWARE["device"]
STEP_RECOMMENDATION = recommend_steps(HARDWARE)
ESTIMATES = GenerationEstimates(".generation_timings.json")

OUTPUT_DIR = Path("outputs")
VOICE_DIR = Path("voices")
VOICE_DB = Path("voices.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Built-in Voice Design Presets
# ============================================================

RETIRED_KHMER_VOICES = {
    "Khmer - Plain text",
    "Khmer Male - Playful", "Khmer Female - Playful", "Khmer Funny - Cartoon",
    "Khmer Funny - Cheeky Uncle", "Khmer Funny - Dramatic Auntie",
}

DEFAULT_VOICES = {
    "Khmer Male - Young": {
        "reference": None,
        "description": (
            "Native Khmer speaker, "
            "young Cambodian male around 20 to 25 years old, "
            "warm natural voice, "
            "medium-low pitch, "
            "clear Khmer pronunciation"
        ),
        "permanent": False,
    },

    "Khmer Male - Deep": {
        "reference": None,
        "description": (
            "Native Khmer speaker, "
            "Cambodian male, "
            "deep mature voice, "
            "low pitch, "
            "clear Khmer pronunciation, "
            "strong and confident voice"
        ),
        "permanent": False,
    },

    "Khmer Male - Soft": {
        "reference": None,
        "description": (
            "Native Khmer speaker, "
            "young Cambodian male, "
            "soft warm voice, "
            "relaxed natural tone, "
            "clear Khmer pronunciation"
        ),
        "permanent": False,
    },

    "Khmer Female - Young": {
        "reference": None,
        "description": (
            "Native Khmer speaker, "
            "young Cambodian female around 20 to 25 years old, "
            "soft bright voice, "
            "warm and friendly tone, "
            "clear Khmer pronunciation"
        ),
        "permanent": False,
    },

    "Khmer Female - Mature": {
        "reference": None,
        "description": (
            "Native Khmer speaker, "
            "mature Cambodian female, "
            "professional voice, "
            "clear Khmer pronunciation, "
            "calm and confident tone"
        ),
        "permanent": False,
    },
}

for name, description in {
    "English Male - Warm": "Native English male speaker, warm conversational voice, clear pronunciation",
    "English Male - Deep": "Native English male speaker, deep mature voice, confident narration",
    "English Female - Bright": "Native English female speaker, bright friendly voice, clear pronunciation",
    "English Female - Calm": "Native English female speaker, soft calm voice, relaxed delivery",
}.items():
    DEFAULT_VOICES[name] = {
        "reference": None, "description": description, "permanent": False, "language": "English",
    }


# ============================================================
# Speaking Styles
# ============================================================

STYLES = {
    "Natural": (
        "warm conversational tone, "
        "comfortable speaking pace, "
        "gentle expression and natural pauses"
    ),

    "Cybersecurity": (
        "professional cybersecurity narrator, "
        "calm and confident, "
        "slightly serious, "
        "clear educational delivery, "
        "medium speaking speed"
    ),

    "News": (
        "professional Khmer news presenter, "
        "authoritative and clear, "
        "controlled delivery, "
        "medium speaking speed"
    ),

    "Storytelling": (
        "warm expressive storytelling, "
        "natural emotional delivery, "
        "slightly slower speaking speed, "
        "natural pauses"
    ),

    "Documentary": (
        "serious documentary narrator, "
        "calm and controlled, "
        "slightly deep delivery, "
        "medium-slow speaking speed"
    ),

    "Energetic": (
        "energetic and friendly, "
        "enthusiastic delivery, "
        "slightly faster speaking speed"
    ),

    "Calm": (
        "very calm and relaxed, "
        "soft delivery, "
        "slower speaking speed"
    ),

    "Serious": (
        "serious and confident, "
        "controlled emotion, "
        "clear articulation, "
        "medium speaking speed"
    ),

    "Playful": (
        "slightly cheerful tone, natural pace, clear pronunciation"
    ),

    "Comedy": (
        "cheeky comic storytelling delivery, expressive emphasis, "
        "short pauses before punchlines, amused conversational tone, clear pronunciation"
    ),

    "Cartoon": (
        "animated cartoon delivery, bouncy rhythm, exaggerated pitch changes, "
        "playful surprised expression, clear pronunciation"
    ),
}


def styles_for_model(key):
    # Keep the exaggerated styles out of Khmer mode after unintelligible output.
    return [name for name in STYLES if model_language(key) != "Khmer" or name not in ("Playful", "Comedy", "Cartoon")]


# ============================================================
# Voice Database
# ============================================================

def create_voice_database_if_needed():
    if not VOICE_DB.exists():
        VOICE_DB.write_text("{}\n", encoding="utf-8")


def load_saved_voices():
    create_voice_database_if_needed()

    try:
        data = json.loads(VOICE_DB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("Warning: voices.json is invalid.")
        return {}
    return data if isinstance(data, dict) else {}


def save_voice_database(voices):
    payload = json.dumps(voices, indent=4, ensure_ascii=False) + "\n"
    temporary = VOICE_DB.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(VOICE_DB)


# ============================================================
# Load Voice Library
# ============================================================

SAVED_VOICES = load_saved_voices()

VOICES = {
    **DEFAULT_VOICES,
    **SAVED_VOICES,
}

SCRIPT_SAMPLES = {
    "Khmer": "សួស្តីអ្នកទាំងអស់គ្នា។",
    "English": "Hello everyone. Welcome to Voice Studio.",
}


def voices_for_model(key):
    language = model_language(key)
    if key not in VOX_MODELS:
        return [f"MMS {language} (built-in)"]
    names = [name for name, voice in VOICES.items() if voice.get("language", "Khmer") == language]
    return sorted(names, key=lambda name: not bool(VOICES[name].get("reference")))


# ============================================================
# Lazy speech model manager
# ============================================================

MODEL_MANAGER = SpeechModelManager(RUNTIME_DEVICE)
print(f"Hardware: {HARDWARE['accelerator']} ({RUNTIME_DEVICE})")
print("Studio ready. Speech models load only when you generate audio.")


# ============================================================
# Helpers
# ============================================================

def prepare_waveform(wav):
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()

    wav = np.asarray(wav)
    wav = np.squeeze(wav)

    return wav


def safe_filename(name):
    """Return a filesystem-safe name, e.g. Khmer Male 04 -> khmer_male_04."""
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return filename or f"voice_{uuid4().hex[:8]}"


def get_voice_reference(
    selected_voice,
    temporary_reference,
):
    """
    Priority:

    1. Temporary uploaded reference
    2. Permanent saved reference
    3. Voice Design
    """

    if temporary_reference:
        return temporary_reference

    voice = VOICES.get(selected_voice)
    if not voice:
        raise gr.Error("Selected voice could not be found.")
    reference = voice.get("reference")
    if not reference:
        return None
    path = Path(reference)
    if not path.exists():
        raise gr.Error(f"Reference audio does not exist:\n\n{reference}")
    return str(path)


def build_voice_prompt(
    selected_voice,
    selected_style,
    custom_style,
    has_reference=False,
):
    if selected_voice in RETIRED_KHMER_VOICES:
        raise gr.Error("This experimental preset was removed. Choose Khmer Male - Young and Natural.")
    voice = VOICES.get(selected_voice)

    if not voice:
        raise gr.Error(
            "Selected voice could not be found."
        )

    voice_description = voice.get(
        "description",
        "Native Khmer speaker, clear pronunciation",
    )

    style_description = STYLES.get(
        selected_style,
        STYLES["Natural"],
    )
    if voice.get("language", "Khmer") == "Khmer" and selected_style in ("Playful", "Comedy", "Cartoon"):
        raise gr.Error("This experimental Khmer style was removed. Choose Natural.")
    if voice.get("language") == "English" and selected_style == "News":
        style_description = style_description.replace("Khmer", "English")

    # A reference supplies speaker identity. Only describe delivery in that
    # case so a built-in age/gender/pitch preset cannot compete with the clip.
    parts = [style_description] if has_reference else [
        voice_description, style_description,
    ]

    if custom_style:
        custom_style = custom_style.strip()

        if custom_style:
            parts.append(custom_style)

    return ", ".join(parts)


# ============================================================
# UI Lock / Unlock
# ============================================================

def lock_for_generation():
    return (
        gr.update(value="Generating…", interactive=False),
        gr.update(interactive=False),
        "**Preparing audio…**",
        gr.update(interactive=True),
    )


def unlock_after_generation():
    return (
        gr.update(value="Generate audio", interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def lock_for_voice_save():
    """
    Disable important actions while saving a voice.
    """

    return (
        gr.Button(
            value="Generate audio",
            interactive=False,
        ),

        gr.Button(
            value="Saving…",
            interactive=False,
        ),

        "⏳ **Saving permanent voice...**",
    )


def unlock_after_voice_save():
    return (
        gr.Button(
            value="Generate audio",
            interactive=True,
        ),

        gr.Button(
            value="Save voice",
            interactive=True,
        ),
    )


# ============================================================
# Add Permanent Voice
# ============================================================

def required_text(value, field_name):
    cleaned = value.strip() if value else ""
    if not cleaned:
        raise gr.Error(f"{field_name} is required.")
    return cleaned


def voice_dropdown(model_key, selected_name=None, language=None):
    choices = voices_for_model(model_key)
    can_select_saved = (
        selected_name in choices
        and model_key in VOX_MODELS
        and language == model_language(model_key)
    )
    return gr.Dropdown(
        choices=choices,
        value=selected_name if can_select_saved else choices[0],
        interactive=model_key in VOX_MODELS,
    )

def add_permanent_voice(
    voice_name,
    voice_description,
    audio_file,
    language="Khmer",
    model_key=FULL_MODEL,
):
    if language not in SCRIPT_SAMPLES:
        raise gr.Error("Choose Khmer or English for this voice.")
    voice_name = required_text(voice_name, "Voice Name")
    voice_description = required_text(voice_description, "Voice Description")
    if not audio_file:
        raise gr.Error("Reference Audio is required.")
    if voice_name in VOICES:
        raise gr.Error(
            f'A voice named "{voice_name}" already exists.\n\n'
            "Please choose another name."
        )
    source_path = Path(audio_file)
    if not source_path.exists():
        raise gr.Error("Uploaded audio file could not be found.")

    destination = VOICE_DIR / f"{safe_filename(voice_name)}_{uuid4().hex[:6]}.wav"
    voice_data = {
        "reference": str(destination),
        "description": voice_description,
        "permanent": True,
        "language": language,
    }
    saved_voices = load_saved_voices()
    saved_voices[voice_name] = voice_data

    shutil.copy2(source_path, destination)
    try:
        save_voice_database(saved_voices)
    except OSError:
        destination.unlink(missing_ok=True)
        raise
    VOICES[voice_name] = voice_data

    print()
    print("=" * 60)
    print("Permanent voice added")
    print(f"Name: {voice_name}")
    print(f"Audio: {destination}")
    print("=" * 60)

    return (
        voice_dropdown(model_key, voice_name, language),
        "",
        "",
        None,
        f"Saved **{voice_name}**. Select it in Create with a {language} VoxCPM2 model.",
    )


# ============================================================
# Generate Audio
# ============================================================

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


# ============================================================
# UI
# ============================================================

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


def save_generated_voice(name, take, model_key):
    if not take or not take.get("path"):
        raise gr.Error("Generate and listen to an audio sample first.")
    # Keep the reference short even when the completed script is long.
    with sf.SoundFile(take["path"]) as recording:
        rate = recording.samplerate
        audio = recording.read(frames=rate * 20, dtype="float32")
    if not len(audio) or not np.isfinite(audio).all():
        raise gr.Error("This sample has no usable audio. Generate another sample.")
    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "reference.wav"
        sf.write(reference, audio, rate)
        saved = add_permanent_voice(
            name, "Reusable speaker from a selected audio sample", str(reference),
            take["language"], model_key,
        )
    # Clear a temporary override so the next generation uses the named voice.
    return saved[0], "", saved[4], None

STUDIO_CSS = """
.gradio-container { width: 100% !important; max-width: 1100px !important; box-sizing: border-box;
    margin: auto; padding: 28px 24px !important;
    --block-label-background-fill: transparent; --block-label-text-color: var(--body-text-color);
    --block-title-background-fill: transparent; --block-title-text-color: var(--body-text-color); }
#studio-header { padding: 8px 0 26px; }
.studio-heading { display: flex; align-items: center; gap: 14px; }
.studio-mark { background: #0f766e; color: white; border-radius: 16px; width: 52px; height: 52px;
    display: flex; align-items: center; justify-content: center; font-size: 26px; }
.studio-heading h1 { margin: 0 !important; font-size: 27px !important; letter-spacing: -0.8px; }
.studio-heading p { margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 14px; }
#studio-tabs > .tab-nav { border-bottom: 1px solid var(--border-color-primary); margin-bottom: 22px; gap: 8px; }
#studio-tabs > .tab-nav button { padding: 12px 24px; font-size: 14px; }
#studio-tabs > .tabitem { padding: 0; border: none; background: transparent; }
.studio-card { border: 1px solid var(--border-color-primary) !important; border-radius: 20px !important;
    padding: 24px !important; background: var(--block-background-fill) !important; }
.studio-card h2 { font-size: 19px !important; margin: 0 0 4px !important; letter-spacing: -0.3px; }
.studio-card .section-intro p { color: var(--body-text-color-subdued); font-size: 13px; margin: 0 0 8px; }
#script-box textarea { font-size: 18px; line-height: 1.9; min-height: 240px; }
#script-box { border-radius: 14px; }
#generate-action { min-height: 50px; border-radius: 12px; font-size: 15px; }
#progress-card { padding: 12px 14px; border-radius: 12px; background: var(--background-fill-secondary); }
#progress-card p { margin: 0; font-size: 13px; }
#progress-card h3 { margin: 0 0 6px; font-size: 15px; }
#timing-preview { color: var(--body-text-color-subdued); font-size: 12px; }
#timing-preview p { margin: 0; }
#runtime-card { color: var(--body-text-color-subdued); font-size: 13px; }
#preview-card { gap: 16px; }
#audio-preview { min-height: 140px; border-radius: 14px; }
@media (max-width: 640px) {
    .gradio-container { padding: 16px 12px !important; }
    .studio-card { padding: 18px !important; }
    .studio-heading h1 { font-size: 23px !important; }
    #studio-tabs > .tab-nav button { padding: 10px 16px; }
}
"""

with gr.Blocks(title="Voice Studio") as app:
    gr.HTML(
        '<div class="studio-heading"><div class="studio-mark" aria-hidden="true">ក</div>'
        '<div><h1>Voice Studio</h1><p>Khmer & English. Your words, your voice.</p></div></div>',
        elem_id="studio-header",
    )
    active_model = gr.State(FULL_MODEL)
    generated_take = gr.State(None)

    with gr.Tabs(elem_id="studio-tabs"):
        with gr.Tab("Create", id="create"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=320, elem_classes="studio-card"):
                    gr.Markdown("## Your script")
                    with gr.Row():
                        selected_voice = gr.Dropdown(
                            choices=voices_for_model(FULL_MODEL), value=voices_for_model(FULL_MODEL)[0],
                            label="Voice", filterable=True, scale=2,
                            info="Saved voices reuse a speaker. Design presets can change between takes.",
                        )
                        selected_style = gr.Dropdown(
                            choices=styles_for_model(FULL_MODEL), value="Natural", label="Style", scale=1,
                        )
                    text_input = gr.Textbox(
                        label="Script", show_label=False,
                        placeholder="សរសេរអត្ថបទខ្មែររបស់អ្នកនៅទីនេះ…",
                        lines=8,
                        value=SCRIPT_SAMPLES["Khmer"],
                        elem_id="script-box",
                    )
                    with gr.Accordion("Voice options", open=False):
                        custom_style = gr.Textbox(
                            label="Extra direction", placeholder="Warm and conversational…",
                            lines=2,
                        )
                        temporary_reference = gr.Audio(
                            label="Reference voice", sources=["upload", "microphone"],
                            type="filepath", format="wav",
                        )
                        gr.Markdown("A clean 10–20s clip overrides the selected voice.", elem_classes="section-intro")
                with gr.Column(scale=2, min_width=300, elem_id="preview-card", elem_classes="studio-card"):
                    gr.Markdown("## Audio preview")
                    gr.Markdown("Create a take, then listen or download.", elem_classes="section-intro")
                    audio_output = gr.Audio(
                        label="Audio preview", show_label=False, type="filepath", format="wav",
                        interactive=False, elem_id="audio-preview",
                    )
                    generation_status = gr.Markdown(
                        "Ready when you are.", elem_id="progress-card", sanitize_html=True,
                    )
                    timing_preview = gr.Markdown(
                        "Estimate: **Run a sample first**", elem_id="timing-preview",
                    )
                    with gr.Row():
                        generate_button = gr.Button(
                            "Generate audio", variant="primary", size="lg", scale=3,
                            elem_id="generate-action",
                        )
                        cancel_button = gr.Button("Stop", interactive=False, scale=1, min_width=70)
                    download_button = gr.DownloadButton(label="Download WAV", variant="secondary")
                    with gr.Accordion("Keep this speaker", open=False):
                        gr.Markdown("Like this voice? Save it once, then choose its name for future scripts. A short, clear sample works best.")
                        generated_voice_name = gr.Textbox(label="Speaker name", placeholder="Dara, Sophea…")
                        save_generated_button = gr.Button("Save this speaker")
                        generated_voice_status = gr.Markdown()

        with gr.Tab("Voices", id="voices"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                    gr.Markdown("## Save a voice")
                    gr.Markdown("Record once. Reuse in any script.", elem_classes="section-intro")
                    new_voice_audio = gr.Audio(
                        label="Reference recording", sources=["upload", "microphone"],
                        type="filepath", format="wav",
                    )
                    gr.Markdown("Use a clean 10–20s clip of one speaker.", elem_classes="section-intro")
                with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                    gr.Markdown("## Voice details")
                    new_voice_language = gr.Dropdown(choices=list(SCRIPT_SAMPLES), value="Khmer", label="Language")
                    new_voice_name = gr.Textbox(label="Name", placeholder="Dara, Sophea…")
                    new_voice_description = gr.Textbox(
                        label="Description", placeholder="Warm, clear speaking voice", lines=3,
                    )
                    save_voice_button = gr.Button("Save voice", variant="primary")
                    permanent_voice_status = gr.Markdown("Saved voices appear in the Create tab.", elem_classes="section-intro")

        with gr.Tab("Settings", id="settings"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                    gr.Markdown("## Speech model")
                    model_choice = gr.Dropdown(
                        choices=MODEL_CHOICES, value=FULL_MODEL, label="Model", show_label=False,
                    )
                    model_info = gr.Markdown(model_description(FULL_MODEL), elem_classes="section-intro")
                    switch_model_button = gr.Button("Apply model", variant="secondary")
                    gr.Markdown(f"Device: **{HARDWARE['accelerator']}**", elem_id="runtime-card")
                with gr.Column(scale=1, min_width=300, elem_classes="studio-card"):
                    gr.Markdown("## Generation")
                    inference_steps = gr.Slider(
                        minimum=4, maximum=30, value=STEP_RECOMMENDATION["default"], step=1,
                        label="Detail steps",
                        info=f"Lower is faster · Recommended: {STEP_RECOMMENDATION['range_min']}–{STEP_RECOMMENDATION['range_max']}",
                    )
                    with gr.Row():
                        quick_test_button = gr.Button("Fast · 4 steps", size="sm")
                        recommended_button = gr.Button("Recommended", size="sm")
                    with gr.Accordion("Advanced", open=False):
                        cfg_value = gr.Slider(
                            minimum=1.0, maximum=3.0, value=2.0, step=0.1,
                            label="Voice guidance", info="Default: 2.0",
                        )
            with gr.Accordion("About timing & models", open=False):
                gr.Markdown(
                    "Estimates learn from completed takes and exclude model loading. "
                    "Stop cancels at the next computation checkpoint.\n\n"
                    "MMS Khmer and English each have one voice, without cloning or style controls. "
                    "[CC-BY-NC 4.0](https://huggingface.co/facebook/mms-tts-khm) · Noncommercial only."
                )

    quick_test_button.click(fn=lambda: 4, outputs=inference_steps, queue=False)
    recommended_button.click(
        fn=lambda: STEP_RECOMMENDATION["default"], outputs=inference_steps, queue=False,
    )

    switch_model_button.click(
        fn=switch_model, inputs=[model_choice, text_input],
        outputs=[
            active_model, model_info, selected_voice, selected_style, custom_style,
            temporary_reference, inference_steps, cfg_value, quick_test_button,
            recommended_button, generation_status, text_input,
        ],
        queue=False,
    )
    cancel_button.click(fn=cancel_generation, outputs=generation_status, queue=False)
    save_generated_button.click(
        fn=save_generated_voice,
        inputs=[generated_voice_name, generated_take, active_model],
        outputs=[selected_voice, generated_voice_name, generated_voice_status, temporary_reference],
    )
    estimate_inputs = [
        text_input, selected_voice, selected_style, custom_style,
        temporary_reference, inference_steps, active_model,
    ]
    for control in estimate_inputs:
        control.change(fn=preview_estimate, inputs=estimate_inputs, outputs=timing_preview, queue=False)

    save_voice_event = save_voice_button.click(
        fn=lock_for_voice_save,
        inputs=[],
        outputs=[
            generate_button,
            save_voice_button,
            permanent_voice_status,
        ],
        queue=False,
    ).then(
        fn=add_permanent_voice,
        inputs=[
            new_voice_name,
            new_voice_description,
            new_voice_audio,
            new_voice_language,
            active_model,
        ],
        outputs=[
            selected_voice,
            new_voice_name,
            new_voice_description,
            new_voice_audio,
            permanent_voice_status,
        ],
        show_progress="full",
    ).then(
        fn=unlock_after_voice_save,
        inputs=[],
        outputs=[
            generate_button,
            save_voice_button,
        ],
        queue=False,
    )

    # ========================================================
    # GENERATE AUDIO
    #
    # 1. Lock buttons
    # 2. Generate
    # 3. Unlock buttons
    # ========================================================

    generate_event = generate_button.click(
        fn=lock_for_generation,
        inputs=[],
        outputs=[
            generate_button,
            save_voice_button,
            generation_status,
            cancel_button,
        ],
        queue=False,
    ).then(
        fn=generate_named_take,
        inputs=[
            text_input,
            selected_voice,
            selected_style,
            custom_style,
            temporary_reference,
            cfg_value,
            inference_steps,
            active_model,
        ],
        outputs=[
            audio_output,
            download_button,
            generation_status,
            generated_take,
        ],
        show_progress="full",
    ).then(
        fn=unlock_after_generation,
        inputs=[],
        outputs=[
            generate_button,
            save_voice_button,
            cancel_button,
        ],
        queue=False,
    ).then(
        fn=preview_estimate, inputs=estimate_inputs, outputs=timing_preview, queue=False,
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    app.queue(
        default_concurrency_limit=1,
    )

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        max_file_size="100mb",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=STUDIO_CSS,
    )
