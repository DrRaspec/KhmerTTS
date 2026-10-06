from pathlib import Path
from uuid import uuid4

import json
import os
import platform
import re
import shutil
import time

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from voxcpm import VoxCPM


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "openbmb/VoxCPM2"

# Apple Silicon should use its GPU explicitly, rather than silently falling
# back to CPU. VOXCPM_DEVICE=cpu remains available for compatibility testing.
DEFAULT_DEVICE = (
    "mps"
    if platform.system() == "Darwin" and platform.machine() == "arm64"
    else "auto"
)
RUNTIME_DEVICE = os.environ.get("VOXCPM_DEVICE", DEFAULT_DEVICE).strip().lower()

OUTPUT_DIR = Path("outputs")
VOICE_DIR = Path("voices")
VOICE_DB = Path("voices.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Built-in Voice Design Presets
# ============================================================

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
}


# ============================================================
# Voice Database
# ============================================================

def create_voice_database_if_needed():
    if not VOICE_DB.exists():
        VOICE_DB.write_text(
            "{}",
            encoding="utf-8",
        )


def load_saved_voices():
    create_voice_database_if_needed()

    try:
        with open(
            VOICE_DB,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {}

        return data

    except json.JSONDecodeError:
        print("Warning: voices.json is invalid.")
        return {}


def save_voice_database(voices):
    with open(
        VOICE_DB,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            voices,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# Load Voice Library
# ============================================================

SAVED_VOICES = load_saved_voices()

VOICES = {
    **DEFAULT_VOICES,
    **SAVED_VOICES,
}


# ============================================================
# Load VoxCPM
# ============================================================

print("=" * 60)
print("Loading VoxCPM2...")
print("First launch may download model files.")
print("=" * 60)

if RUNTIME_DEVICE == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError(
        "Apple GPU (MPS) is unavailable in this Python environment. "
        "Run .venv/bin/python -c 'import torch; "
        "print(torch.backends.mps.is_available())' in Terminal to check. "
        "For a slower CPU compatibility test, run VOXCPM_DEVICE=cpu python app.py."
    )

model = VoxCPM.from_pretrained(
    MODEL_ID,

    device=RUNTIME_DEVICE,

    # Better compatibility outside CUDA.
    optimize=False,

    load_denoiser=False,
)

print("VoxCPM2 loaded successfully.")
print(f"Runtime: {model.tts_model.device}, dtype: {model.tts_model.config.dtype}")
print("=" * 60)

print("Available voices:")

for voice_name in VOICES:
    print(f" - {voice_name}")

print("=" * 60)


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
    """
    Example:
    Khmer Male 04 -> khmer_male_04
    """

    name = name.strip().lower()

    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name,
    )

    name = name.strip("_")

    if not name:
        name = f"voice_{uuid4().hex[:8]}"

    return name


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
        raise gr.Error(
            "Selected voice could not be found."
        )

    reference = voice.get("reference")

    if not reference:
        return None

    path = Path(reference)

    if not path.exists():
        raise gr.Error(
            f"Reference audio does not exist:\n\n"
            f"{reference}"
        )

    return str(path)


def build_voice_prompt(
    selected_voice,
    selected_style,
    custom_style,
    has_reference=False,
):
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
    """
    Disable important actions while generating.
    """

    return (
        gr.Button(
            value="⏳ Generating Audio...",
            interactive=False,
        ),

        gr.Button(
            value="💾 Save Permanent Voice",
            interactive=False,
        ),

        "⏳ **Generating Khmer audio...**",
    )


def unlock_after_generation():
    return (
        gr.Button(
            value="🎙️ Generate Khmer Audio",
            interactive=True,
        ),

        gr.Button(
            value="💾 Save Permanent Voice",
            interactive=True,
        ),
    )


def lock_for_voice_save():
    """
    Disable important actions while saving a voice.
    """

    return (
        gr.Button(
            value="🎙️ Generate Khmer Audio",
            interactive=False,
        ),

        gr.Button(
            value="⏳ Saving Voice...",
            interactive=False,
        ),

        "⏳ **Saving permanent voice...**",
    )


def unlock_after_voice_save():
    return (
        gr.Button(
            value="🎙️ Generate Khmer Audio",
            interactive=True,
        ),

        gr.Button(
            value="💾 Save Permanent Voice",
            interactive=True,
        ),
    )


# ============================================================
# Add Permanent Voice
# ============================================================

def add_permanent_voice(
    voice_name,
    voice_description,
    audio_file,
):
    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not voice_name:
        raise gr.Error(
            "Voice Name is required."
        )

    voice_name = voice_name.strip()

    if not voice_name:
        raise gr.Error(
            "Voice Name is required."
        )

    if not voice_description:
        raise gr.Error(
            "Voice Description is required."
        )

    voice_description = voice_description.strip()

    if not voice_description:
        raise gr.Error(
            "Voice Description is required."
        )

    if not audio_file:
        raise gr.Error(
            "Reference Audio is required."
        )

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if voice_name in VOICES:
        raise gr.Error(
            f'A voice named "{voice_name}" already exists.\n\n'
            "Please choose another name."
        )

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    source_path = Path(audio_file)

    if not source_path.exists():
        raise gr.Error(
            "Uploaded audio file could not be found."
        )

    # --------------------------------------------------------
    # Permanent filename
    # --------------------------------------------------------

    safe_name = safe_filename(
        voice_name
    )

    unique_id = uuid4().hex[:6]

    destination = (
        VOICE_DIR /
        f"{safe_name}_{unique_id}.wav"
    )

    # --------------------------------------------------------
    # Copy audio
    # --------------------------------------------------------

    shutil.copy2(
        source_path,
        destination,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    voice_data = {
        "reference": str(destination),
        "description": voice_description,
        "permanent": True,
    }

    VOICES[voice_name] = voice_data

    saved_voices = load_saved_voices()

    saved_voices[voice_name] = voice_data

    save_voice_database(
        saved_voices
    )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Permanent voice added")
    print(f"Name: {voice_name}")
    print(f"Audio: {destination}")
    print("=" * 60)

    # --------------------------------------------------------
    # Update dropdown + clear fields
    # --------------------------------------------------------

    return (
        gr.Dropdown(
            choices=list(VOICES.keys()),
            value=voice_name,
        ),

        "",     # clear voice name
        "",     # clear description
        None,   # clear audio

        (
            f"✅ **{voice_name}** saved permanently.\n\n"
            f"Reference: `{destination}`"
        ),
    )


# ============================================================
# Generate Audio
# ============================================================

def generate_audio(
    text,
    selected_voice,
    selected_style,
    custom_style,
    temporary_reference,
    cfg_value,
    inference_steps,
):
    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not text:
        raise gr.Error(
            "Please enter some Khmer text."
        )

    text = text.strip()

    if not text:
        raise gr.Error(
            "Please enter some Khmer text."
        )

    if len(text) > 4000:
        raise gr.Error(
            "Your text is too long. "
            "Please keep it below 4000 characters."
        )

    if not selected_voice:
        raise gr.Error(
            "Please choose a voice."
        )

    # --------------------------------------------------------
    # Determine speaker
    # --------------------------------------------------------

    reference_path = get_voice_reference(
        selected_voice,
        temporary_reference,
    )

    # --------------------------------------------------------
    # Voice + Style
    # --------------------------------------------------------

    voice_prompt = build_voice_prompt(
        selected_voice,
        selected_style,
        custom_style,
        has_reference=bool(reference_path),
    )

    generation_text = (
        f"({voice_prompt}) "
        f"{text}"
    )

    # --------------------------------------------------------
    # Generation Options
    # --------------------------------------------------------

    generation_options = {
        "text": generation_text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(
            inference_steps
        ),
        # The external normalizer targets Chinese/English, not Khmer.
        "normalize": False,
        # Avoid silently generating the entire script multiple times.
        "retry_badcase": False,
    }

    if reference_path:
        generation_options[
            "reference_wav_path"
        ] = reference_path

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Generating speech")
    print(f"Voice: {selected_voice}")
    print(f"Style: {selected_style}")
    print(f"Device: {model.tts_model.device}, dtype: {model.tts_model.config.dtype}")

    if temporary_reference:
        print(
            "Reference: Temporary uploaded voice"
        )

    elif reference_path:
        print(
            f"Reference: {reference_path}"
        )

    else:
        print(
            "Reference: Voice Design"
        )

    print(
        f"CFG: {cfg_value}"
    )

    print(
        f"Steps: {inference_steps}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generation_started = time.perf_counter()
    try:

        wav = model.generate(
            **generation_options
        )

    except Exception as error:

        print("Generation error:")
        print(error)

        raise gr.Error(
            f"VoxCPM generation failed:\n\n"
            f"{error}\n\n"
            "For an MPS compatibility test, restart with "
            "VOXCPM_DEVICE=cpu python app.py (CPU will be slower)."
        )

    # --------------------------------------------------------
    # Prepare waveform
    # --------------------------------------------------------

    wav = prepare_waveform(
        wav
    )

    generation_seconds = time.perf_counter() - generation_started
    audio_seconds = len(wav) / model.tts_model.sample_rate
    realtime_factor = generation_seconds / audio_seconds if audio_seconds else 0
    print(
        f"Generation: {generation_seconds:.1f}s; audio: {audio_seconds:.1f}s; "
        f"real-time factor: {realtime_factor:.2f} (lower is faster)"
    )
    if model.tts_model.device == "mps":
        print(
            f"MPS memory after generation: "
            f"{torch.mps.driver_allocated_memory() / 1024**3:.2f} GiB"
        )

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    safe_voice_name = safe_filename(
        selected_voice
    )

    output_name = (
        f"{safe_voice_name}_"
        f"{uuid4().hex[:8]}.wav"
    )

    output_path = (
        OUTPUT_DIR /
        output_name
    )

    sf.write(
        output_path,
        wav,
        model.tts_model.sample_rate,
    )

    print(
        f"Generated: {output_path}"
    )

    # --------------------------------------------------------
    # Determine Voice Source
    # --------------------------------------------------------

    voice_info = VOICES.get(
        selected_voice,
        {},
    )

    is_permanent = voice_info.get(
        "permanent",
        False,
    )

    if temporary_reference:

        source_description = (
            "temporary uploaded reference"
        )

    elif is_permanent:

        source_description = (
            "permanent saved voice"
        )

    else:

        source_description = (
            "Voice Design preset"
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = (
        f"✅ Generated using **{selected_voice}**  \n"
        f"Style: **{selected_style}**  \n"
        f"Voice source: **{source_description}**  \n"
        f"Generated {audio_seconds:.1f}s of audio in {generation_seconds:.1f}s "
        f"(RTF {realtime_factor:.2f})"
    )

    return (
        str(output_path),
        str(output_path),
        status,
    )


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Khmer VoxCPM2 Studio",
) as app:

    gr.Markdown(
        """
# 🇰🇭 Khmer VoxCPM2 Studio

Create Khmer speech, manage permanent voices,
preview audio, and download WAV files.

**Voice = who speaks**

**Style = how they speak**

**Text = what they say**
"""
    )

    # ========================================================
    # Text
    # ========================================================

    text_input = gr.Textbox(
        label="Khmer Text",
        placeholder=(
            "សរសេរ ឬ paste "
            "អត្ថបទខ្មែររបស់អ្នកនៅទីនេះ..."
        ),
        lines=12,
        value="""សួស្តីអ្នកទាំងអស់គ្នា។

ថ្ងៃនេះ យើងនឹងនិយាយអំពីសុវត្ថិភាពនៅលើអ៊ីនធឺណិត។

តើអ្នកធ្លាប់ភ្ជាប់វ៉ាយហ្វាយឥតគិតថ្លៃ
នៅហាងកាហ្វេ ឬកន្លែងសាធារណៈដែរឬទេ?

វាងាយស្រួលមែនទែន។
ប៉ុន្តែ វាក៏អាចមានហានិភ័យដែរ។""",
    )

    # ========================================================
    # Voice + Style
    # ========================================================

    with gr.Row():

        selected_voice = gr.Dropdown(
            choices=list(
                VOICES.keys()
            ),
            value=list(
                VOICES.keys()
            )[0],
            label="Voice",
            filterable=True,
        )

        selected_style = gr.Dropdown(
            choices=list(
                STYLES.keys()
            ),
            value="Natural",
            label="Speaking Style",
        )

    # ========================================================
    # Extra Style
    # ========================================================

    custom_style = gr.Textbox(
        label=(
            "Extra Voice Instruction "
            "(Optional)"
        ),
        placeholder=(
            "Example: warm and conversational, "
            "with gentle expression"
        ),
        lines=2,
    )

    # ========================================================
    # Temporary Reference
    # ========================================================

    with gr.Accordion(
        "🎤 Temporary Reference Voice",
        open=False,
    ):

        gr.Markdown(
            """
Upload or record a voice here if you only want
to use it temporarily.

It will **not** be added to your permanent library.

It overrides the selected voice for the current
generation.

For a consistent voice, use a clean 10–20 second recording
of one native Khmer speaker at a comfortable pace, without
music or background noise. Speaking Style controls the delivery.
"""
        )

        temporary_reference = gr.Audio(
            label="Temporary Reference Voice",
            sources=[
                "upload",
                "microphone",
            ],
            type="filepath",
            format="wav",
        )

    # ========================================================
    # Add Permanent Voice
    # ========================================================

    with gr.Accordion(
        "➕ Add Permanent Voice",
        open=False,
    ):

        gr.Markdown(
            """
Upload a reusable voice to your library.

All three fields are required.

The reference WAV is stored in `voices/`
and its information is stored in `voices.json`.
"""
        )

        new_voice_name = gr.Textbox(
            label="Voice Name *",
            placeholder=(
                "Example: Khmer Male 04"
            ),
        )

        new_voice_description = gr.Textbox(
            label="Voice Description *",
            placeholder=(
                "Example: Native Khmer male, "
                "young, warm voice, "
                "clear pronunciation"
            ),
            lines=3,
        )

        new_voice_audio = gr.Audio(
            label="Reference Audio *",
            sources=[
                "upload",
                "microphone",
            ],
            type="filepath",
            format="wav",
        )

        save_voice_button = gr.Button(
            "💾 Save Permanent Voice",
            variant="secondary",
        )

        permanent_voice_status = (
            gr.Markdown()
        )

    # ========================================================
    # Advanced Settings
    # ========================================================

    with gr.Accordion(
        "⚙️ Advanced Settings",
        open=False,
    ):

        cfg_value = gr.Slider(
            minimum=1.0,
            maximum=3.0,
            value=2.0,
            step=0.1,
            label="CFG Value",
            info=(
                "2.0 is a good starting point."
            ),
        )

        inference_steps = gr.Slider(
            minimum=4,
            maximum=30,
            value=10,
            step=1,
            label="Inference Steps",
            info=(
                "More steps may improve quality "
                "but take longer."
            ),
        )

    # ========================================================
    # Generate
    # ========================================================

    generate_button = gr.Button(
        "🎙️ Generate Khmer Audio",
        variant="primary",
        size="lg",
    )

    generation_status = (
        gr.Markdown()
    )

    # ========================================================
    # Output
    # ========================================================

    audio_output = gr.Audio(
        label="Generated Audio",
        type="filepath",
        format="wav",
        interactive=False,
    )

    download_button = gr.DownloadButton(
        label="⬇️ Download WAV",
        variant="secondary",
    )

    # ========================================================
    # SAVE PERMANENT VOICE
    #
    # 1. Lock buttons
    # 2. Save voice
    # 3. Unlock buttons
    # ========================================================

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
        ],
        queue=False,
    ).then(
        fn=generate_audio,
        inputs=[
            text_input,
            selected_voice,
            selected_style,
            custom_style,
            temporary_reference,
            cfg_value,
            inference_steps,
        ],
        outputs=[
            audio_output,
            download_button,
            generation_status,
        ],
        show_progress="full",
    ).then(
        fn=unlock_after_generation,
        inputs=[],
        outputs=[
            generate_button,
            save_voice_button,
        ],
        queue=False,
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
    )
