"""Voice presets, reusable recordings, and speaker-library persistence."""

import json
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import soundfile as sf

from .config import VOICE_DIR, VOICE_DB
from .speech_models import FULL_MODEL, VOX_MODELS, model_language


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
