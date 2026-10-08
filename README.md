# Khmer VoxCPM2 Studio — Setup, Voice Configuration & Run Guide

For dataset samples and the separate model fine-tuning workflow, see
[TRAINING.md](TRAINING.md). Preparing recordings or saving a reference voice
does not train or replace the speech model.

This guide explains how to set up, run, and configure the **Khmer VoxCPM2 Studio** app.

The app lets you:

- Paste Khmer text
- Choose different Khmer voice profiles
- Choose different speaking styles
- Upload or record a custom reference voice
- Generate speech
- Preview generated audio
- Download the result as a WAV file
- Add more voices later without changing the UI

## Studio interface and live progress

### Keep a named speaker across generations

Design presets such as **Khmer Male - Young** describe a voice type, so the
speaker can change each time. To reuse a particular speaker:

1. Generate a short sample and listen to it.
2. Below the audio preview, open **Keep this speaker**.
3. Enter a name such as **Dara** or **Sophea**, then click **Save this speaker**.
4. Keep that named voice selected when generating new scripts.

The app saves up to the first 20 seconds as a permanent reference, selects
the named speaker when its language matches the active VoxCPM2 model, and
clears any temporary reference override. Saved speakers appear before design
presets and remain available after restarting. Each section of a long script
uses the same recording. This improves speaker consistency; it does not
guarantee identical delivery. You can also upload or record a specific person's
voice in **Voices**, give it a human name, and save it there.

The studio has three tabs:

- **Create** — choose a voice and style, write your script, and generate audio.
  The preview, status, and WAV download sit beside the script. Open **Voice
  options** for extra direction or a temporary reference recording.
- **Voices** — upload or record a reference and save a reusable voice.
- **Settings** — choose and apply a model, adjust detail steps, or open
  advanced voice guidance. Use **Fast · 4 steps** for a quick sample.

During generation the status shows preparation, audio creation with a live
elapsed timer, saving, and completion. Long runs display a performance hint;
generation errors leave a readable message and restore the action buttons.
The model does not provide a reliable total, so elapsed time is shown rather
than a misleading completion percentage. Terminal details remain available.
Restart the app after updating to load the new interface.

Time estimates learn from successful runs on the same hardware/model and
voice settings. After a sample completes, an approximate range appears
before generation and counts down during audio creation. Loading and
downloads are separate. Cancelled/failed runs do not train the estimate.
Very different script lengths show "Learning your speed" instead of an
unsupported extrapolation; a run exceeding its range shows "Taking longer
than estimated." Timing history stores only hashed configuration keys,
character counts, and durations in `.generation_timings.json`, not scripts
or audio. Estimates can change with speaking pace, memory pressure, and
other running apps.

The estimate sits directly above **Generate**. During a run, the progress
card shows the stage, **Elapsed**, and **Remaining** in a compact status line.
"Learning speed" means no comparable successful run is available yet.
Explanations are under **About timing & models** in Settings.

## Cancel generation and switch to a lightweight model

**Stop** requests a cooperative stop. The UI shows **Stopping**
until the current model computation finishes and the next checkpoint can
stop it. Cancelled requests discard their audio; new generation and model
switching wait until the worker has stopped. A session can only cancel its
own request. During a first-time download/model load, cancellation waits for
loading to return; it cannot interrupt an in-flight download or GPU operation.

Choose a model in **Settings → Speech model**, then press **Apply model**:

| Model | Features | Memory / use |
|---|---|---|
| VoxCPM2 | Voice design, speaking styles, reference cloning | Large 2B model; heavy on 16 GB Macs |
| MMS Khmer | One built-in Khmer speaker | Small 36.3M model, runs on CPU; noncommercial only |
| VoxCPM2 English | Four English voice presets, styles, reference cloning | Same cached weights; fresh instance when switching language |
| MMS English | One built-in English speaker | Small 36.3M model, runs on CPU; noncommercial only |

For English, open **Settings**, select **VoxCPM2 English** or **MMS English**,
and click **Apply model**. The Create tab updates the voice choices and script
placeholder. The starter sample changes to English; scripts you have edited
are preserved. VoxCPM2 English includes warm/deep male and bright/calm female
voice design presets. These are descriptive presets, not separate trained
speaker checkpoints. Use a reference recording for a consistent speaker.

When saving a reusable voice in **Voices**, choose its **Language**. Existing
saved voices default to Khmer. A saved English voice appears when using an
English model. MMS has a single built-in voice, so preset and reference controls
remain disabled for both MMS models.

MMS English downloads [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng)
on its first generation and caches it locally. It uses the existing Transformers
dependency and carries the same CC-BY-NC 4.0 noncommercial license as MMS Khmer.
VoxCPM2's [model card](https://huggingface.co/openbmb/VoxCPM2) documents English
speech generation and voice design; its English mode needs no separate download.

### Khmer pronunciation troubleshooting

The Khmer comedy/cartoon presets and their gentler Playful replacements were
removed after reports of unintelligible output. **Playful**, **Comedy**, and
**Cartoon** styles are hidden in Khmer mode.

The plain-text workaround was also removed. The default is restored to
**Khmer Male - Young** with **Natural** style. The original Khmer voice
descriptions, style descriptions, parenthesized prompt format, and generation
options are restored. Stale requests using removed experimental presets are
rejected, rather than silently changing the generation prompt. Switching
between Khmer and English unloads the previous model so inference state is
not shared between modes; downloaded weights remain cached.

Start with a short Khmer sentence at 10 inference steps and guidance 2.0.
Long VoxCPM2 scripts are now split automatically into sections of up to 180
script characters, using paragraphs, sentence punctuation, and spaces where
possible. Splitting preserves Khmer combining marks and coeng sequences.
Each section gets the same voice/style instructions and reference recording;
the audio is joined with 0.25-second pauses into one downloadable WAV. Short
scripts use the original single request. Status shows the current section,
and cancellation discards the entire result. The 180-character target is a
conservative app setting, not a documented model limit. Voice identity may
vary across sections without a reference clip; long scripts may take longer
because each section starts a new synthesis request.
If VoxCPM2 still sounds wrong, compare it with **MMS Khmer** under Settings.
MMS uses a dedicated Khmer checkpoint, with a single voice and no expressive
styles or cloning. For VoxCPM2 speaker identity, a clean Khmer reference clip
can be supplied through Voice options; verify the pronunciation by listening.

MMS Khmer uses [`facebook/mms-tts-khm`](https://huggingface.co/facebook/mms-tts-khm),
licensed CC-BY-NC 4.0. It does not support the speaker presets, cloning,
style instructions, CFG, or diffusion steps; those controls are disabled
when you switch to it. Listen to a sample before choosing it for a project.
The reduced model size lowers memory demand but does not guarantee a
particular generation time or the same expressive quality as VoxCPM2.

The app starts without loading either speech model. The first generation
loads/downloads the selected model. Switching to a different model releases
the previous model and its unused GPU cache first; the two models are never
kept loaded together. MMS model files are cached under `.cache/huggingface/hub/`.
Keep the `studio/` package beside `app.py`.

Install dependencies, including Khmer text romanization for MMS:

```bash
python -m pip install -r requirements.txt
python app.py
```

## Hardware detection and current performance settings

The app detects available PyTorch accelerators on the computer running
Python: CUDA first, then Apple GPU (`mps`), then CPU. Startup logs show
the operating system, architecture, CPU thread count, selected accelerator,
and NVIDIA GPU memory when available. The UI shows the selected accelerator;
generation logs show the actual device and dtype. If the app runs on a
server, detection describes that server, not a visitor's browser computer.
Hardware detection lives in `studio/runtime_config.py`. Restart after updating:

```bash
source .venv/bin/activate
python app.py
```

Inference Steps defaults to **10 on GPUs** and **6 on CPU**. Automatic full-script retries and
the Chinese/English text normalizer are disabled for Khmer generation.
Write numbers as the Khmer words you want spoken. The finished result
reports generation time, audio duration, and real-time factor (generation
seconds divided by audio seconds; lower is faster).

Start with one Khmer sentence at 10 steps. If generation is still slow on
an M2 Pro with 16 GB RAM, check Activity Monitor's Memory Pressure and Swap
Used while generating. The installed VoxCPM library uses float32 on MPS
for stability, which increases memory demand. Do not force float16 or
bfloat16 merely to save memory; the library warns of unstable audio.

For an intentional CPU compatibility test:

```bash
VOXCPM_DEVICE=cpu python app.py
```

CPU generation may be much slower. The older setup examples below describe
the previous `auto`, 20-step configuration; this section reflects the current
app defaults.

`VOXCPM_DEVICE=auto` is the default. You can explicitly select `cpu`, `mps`,
`cuda`, or an indexed GPU such as `cuda:1`. Invalid or unavailable explicit
choices fail clearly rather than silently changing devices. Detection uses
the backends supported by the installed PyTorch/VoxCPM environment; it does
not install GPU drivers or enable unsupported GPUs. The slider shows coarse
hardware recommendations: CPU 4–10 steps, MPS 8–15, baseline CUDA 10–20. Users can still
select 4–30 steps; generation shows a notice above the suggested range.
These are conservative starting guidelines, not per-model benchmarks or
guarantees against slowdowns. Steps primarily affect compute time; they do
not solve insufficient memory. Hardware detection does not predict current
memory pressure or the load from other apps. Compare a short sample before
increasing steps for a full script.

CUDA recommendations also consider GPU memory, multiprocessor count, and
compute capability. Lower-capacity cards start at 8 steps. Modern cards
(compute capability major 8 or newer) with at least 16 GiB and 80
multiprocessors start at 15; those with at least 24 GiB and 120
multiprocessors start at 20. Their suggested upper values are 25 and 30.
Missing specifications use the 10-step baseline. These tiers are estimates:
VRAM alone does not measure speed, and processor counts are not directly
comparable across architectures. A low-memory GPU can still run out of
memory at low steps. Apple GPUs retain the 10-step starting point because
this PyTorch interface does not provide comparable GPU capability metrics.

---

## 1. Recommended Project Structure

Create a project folder like this:

```text
khmer-voxcpm/
├── app.py                 # Launch entry point
├── studio/                # UI, voices, generation, and runtime modules
├── training/              # Dataset preparation tools
├── tests/                 # Automated tests
├── voices/
├── outputs/
├── data/                  # Downloaded datasets and training manifests
└── .venv/
```

### What each folder is for

- `app.py` — starts the Gradio application
- `studio/ui.py` — builds the interface and wires events
- `studio/voices.py` — voice presets, styles, and saved speaker references
- `studio/generation.py` — speech generation, cancellation, and timing
- `studio/speech_models.py` — model loading and inference adapters
- `studio/config.py` — shared runtime settings and data paths
- `studio/runtime_config.py` — hardware detection and step recommendations
- `studio/generation_estimates.py` — timing history and estimates
- `training/prepare_data.py` — prepares Khmer dataset samples
- `tests/` — run with `python -m unittest discover -s tests`
- `voices/` — permanent reference voice WAV files
- `outputs/` — generated WAV files
- `.venv/` — Python virtual environment

---

## 2. Create the Project

Open Terminal:

```bash
mkdir khmer-voxcpm
cd khmer-voxcpm
```

Create the folders:

```bash
mkdir voices
mkdir outputs
```

---

## 3. Create the Python Environment

For macOS / Apple Silicon, use Python 3.12:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

When it is active, your Terminal normally starts with something like:

```text
(.venv)
```

---

## 4. Install Dependencies

Run:

```bash
python -m pip install --upgrade pip setuptools
pip install voxcpm gradio soundfile numpy
```

You normally only need to install these once.

---

# 5. Run the App

Make sure you are inside your project folder:

```bash
cd khmer-voxcpm
```

Activate the environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python app.py
```

The browser should open automatically.

If it does not, open:

```text
http://127.0.0.1:7860
```

---

# 6. Do I Need to Rerun After Changing Code?

Yes.

If `app.py` is currently running and you change the Python code:

Press:

```text
Ctrl + C
```

Then run again:

```bash
python app.py
```

If the virtual environment is no longer active:

```bash
source .venv/bin/activate
python app.py
```

---

# 7. Understanding the Voice System

The app separates three things:

```text
VOICE = Who is speaking
STYLE = How they are speaking
TEXT  = What they are saying
```

Example:

```text
Voice:
Khmer Male - Young

Style:
Cybersecurity

Text:
តើអ្នកធ្លាប់ភ្ជាប់វ៉ាយហ្វាយឥតគិតថ្លៃ...
```

Another example:

```text
Voice:
Khmer Female - Young

Style:
News

Text:
សួស្តីអ្នកទាំងអស់គ្នា...
```

This makes it easy to reuse the same speaker with many different styles.

---

# 8. Configure Voices

Find this section inside `studio/voices.py`:

```python
DEFAULT_VOICES = {
    ...
}
```

A voice looks like this:

```python
"Khmer Male - Young": {
    "reference": None,
    "description": (
        "Native Khmer speaker, "
        "young Cambodian male around 20 to 25 years old, "
        "warm natural voice, "
        "medium-low pitch, "
        "clear Khmer pronunciation"
    ),
},
```

There are two ways to create voices.

---

## Method A — Voice Design

Set:

```python
"reference": None
```

Example:

```python
"Khmer Male - Deep": {
    "reference": None,
    "description": (
        "Native Khmer speaker, "
        "Cambodian male, "
        "deep mature voice, "
        "low pitch, "
        "strong and confident voice, "
        "clear Khmer pronunciation"
    ),
},
```

VoxCPM will try to create a voice based on the description.

This is useful for experimenting with many voices.

### Good description properties

You can describe:

- Gender
- Age
- Pitch
- Voice depth
- Warmth
- Energy
- Speaking speed
- Confidence
- Softness
- Seriousness

Example:

```python
"description": (
    "Native Khmer speaker, "
    "young Cambodian male around 23 years old, "
    "medium-low pitch, "
    "warm voice, "
    "clear Khmer pronunciation"
)
```

Another:

```python
"description": (
    "Native Khmer speaker, "
    "young Cambodian female around 22 years old, "
    "soft bright voice, "
    "friendly natural tone, "
    "clear Khmer pronunciation"
)
```

---

# 9. Important: Voice Design Is Not Guaranteed to Stay Identical

If you use:

```python
"reference": None
```

the generated voice may vary.

For example:

```text
young Cambodian male
```

does not guarantee the exact same person's voice every time.

Voice Design is good for:

- Testing
- Experimenting
- Finding interesting voices
- Creating temporary characters

For a permanent content voice, use a reference WAV.

---

# 10. Permanent / Consistent Voices

Suppose you have:

```text
voices/male_01.wav
```

Change:

```python
"reference": None,
```

to:

```python
"reference": "voices/male_01.wav",
```

Example:

```python
"Khmer Male 01": {
    "reference": "voices/male_01.wav",
    "description": (
        "Native Khmer speaker, "
        "young Cambodian male, "
        "clear Khmer pronunciation"
    ),
},
```

Now VoxCPM uses `male_01.wav` as the speaker identity.

This is better for a content channel because the voice remains much more consistent.

---

# 11. Adding More Voices

You are NOT limited to 2 or 3 voices.

You can add many:

```python
VOICES = {
    "Khmer Male 01": {
        "reference": "voices/male_01.wav",
        "description": (
            "Native Khmer speaker, "
            "young Cambodian male, "
            "clear Khmer pronunciation"
        ),
    },

    "Khmer Male 02": {
        "reference": "voices/male_02.wav",
        "description": (
            "Native Khmer speaker, "
            "young Cambodian male, "
            "deep voice, "
            "clear Khmer pronunciation"
        ),
    },

    "Khmer Female 01": {
        "reference": "voices/female_01.wav",
        "description": (
            "Native Khmer speaker, "
            "young Cambodian female, "
            "soft natural voice, "
            "clear Khmer pronunciation"
        ),
    },

    "Khmer Female 02": {
        "reference": "voices/female_02.wav",
        "description": (
            "Native Khmer speaker, "
            "Cambodian female, "
            "professional voice, "
            "clear Khmer pronunciation"
        ),
    },

    "Khmer Narrator": {
        "reference": "voices/narrator.wav",
        "description": (
            "Native Khmer speaker, "
            "deep mature Cambodian male voice, "
            "clear pronunciation"
        ),
    },
}
```

Your dropdown automatically reads:

```python
choices=list(VOICES.keys())
```

So if you add another voice to `VOICES`, it automatically appears in the UI after you restart the app.

---

# 12. Recommended Voice Folder

Example:

```text
voices/
├── male_01.wav
├── male_02.wav
├── male_03.wav
├── female_01.wav
├── female_02.wav
├── narrator_01.wav
└── narrator_02.wav
```

Then your app can expose:

```text
Khmer Male 01
Khmer Male 02
Khmer Male 03
Khmer Female 01
Khmer Female 02
Narrator 01
Narrator 02
```

---

# 13. Reference Voice Recording Tips

For the best Khmer result, use a clean native Khmer recording.

Recommended:

```text
10–30 seconds
```

Try to record:

- One speaker only
- No music
- No echo
- No background noise
- Normal speaking voice
- Clear Khmer pronunciation
- Consistent microphone distance

Example recording text:

```text
សួស្តីអ្នកទាំងអស់គ្នា។

ថ្ងៃនេះ យើងនឹងនិយាយអំពីបច្ចេកវិទ្យា
និងសុវត្ថិភាពនៅលើអ៊ីនធឺណិត។

ការយល់ដឹងអំពីសុវត្ថិភាពឌីជីថល
អាចជួយការពារព័ត៌មានផ្ទាល់ខ្លួនរបស់យើងបាន។
```

Use voices that you own or have permission to clone.

---

# 14. Configure Speaking Styles

Find this section in `studio/voices.py`:

```python
STYLES = {
    ...
}
```

Example:

```python
STYLES = {
    "Natural": (
        "natural conversational delivery, "
        "medium speaking speed, "
        "natural pauses"
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
}
```

---

# 15. Add Your Own Style

For example:

```python
"Scary Story": (
    "dark mysterious storytelling, "
    "slow speaking speed, "
    "dramatic pauses, "
    "serious and tense delivery"
),
```

Or:

```python
"TikTok": (
    "young energetic content creator, "
    "friendly and exciting delivery, "
    "slightly fast speaking speed"
),
```

Or:

```python
"Teacher": (
    "friendly educational teacher, "
    "very clear pronunciation, "
    "medium-slow speaking speed, "
    "calm and patient delivery"
),
```

Restart the app after modifying `STYLES`.

---

# 16. Voice + Style Combinations

The same speaker can be reused.

Example:

```text
Khmer Male 01
+
Natural
```

Then:

```text
Khmer Male 01
+
Cybersecurity
```

Then:

```text
Khmer Male 01
+
Storytelling
```

The identity stays similar because the reference WAV stays the same, but the delivery can change.

---

# 17. Custom Reference Voice in the UI

The app also includes:

```text
Custom Reference Voice
```

You can upload or record a voice there.

That voice temporarily overrides the saved voice reference.

Priority:

```text
1. Uploaded Custom Reference
2. Saved VOICES reference
3. Voice Design
```

So if you select:

```text
Khmer Male 01
```

but also upload another WAV file, the uploaded WAV is used.

This is useful for testing voices before permanently adding them to `voices/`.

---

# 18. Extra Voice Instruction

The UI also has:

```text
Extra Voice Instruction
```

You can type something temporary like:

```text
speak more slowly
```

or:

```text
sound more mysterious and use dramatic pauses
```

or:

```text
sound excited but keep the pronunciation clear
```

This gets added to the normal voice + style configuration.

---

# 19. Recommended Khmer Settings

Start with:

```text
CFG Value:
2.0

Inference Steps:
20
```

These are good general starting values.

---

# 20. CFG Value

In the UI:

```text
CFG Value
```

Start at:

```text
2.0
```

You can experiment roughly between:

```text
1.0 – 3.0
```

Do not assume higher means better.

Try:

```text
1.8
2.0
2.2
```

and compare the results.

---

# 21. Inference Steps

Start with:

```text
20
```

For quick tests:

```text
10
```

For higher-quality experiments:

```text
20–25
```

More steps usually mean slower generation.

---

# 22. Make Khmer Sound More Natural

The input text matters a lot.

Do NOT write one giant sentence.

Bad:

```text
សួស្តីអ្នកទាំងអស់គ្នាថ្ងៃនេះយើងនឹងនិយាយអំពីសុវត្ថិភាពនៅលើអ៊ីនធឺណិតហើយយើងនឹង...
```

Better:

```text
សួស្តីអ្នកទាំងអស់គ្នា។

ថ្ងៃនេះ យើងនឹងនិយាយអំពីសុវត្ថិភាព
នៅលើអ៊ីនធឺណិត។

តើអ្នកធ្លាប់ប្រើវ៉ាយហ្វាយសាធារណៈដែរឬទេ?

វាងាយស្រួលមែនទែន...
ប៉ុន្តែ វាក៏អាចមានហានិភ័យដែរ។
```

Use:

- `។` for normal sentence endings
- `?` for questions when useful
- `...` for dramatic pauses
- commas for short pauses
- separate paragraphs for larger pauses

---

# 23. Avoid Too Much English Inside Khmer

Instead of:

```text
login ចូល bank account
```

prefer:

```text
ចូលគណនីធនាគារ
```

Instead of:

```text
Free Wi-Fi
```

you can use natural Khmer wording where possible:

```text
វ៉ាយហ្វាយឥតគិតថ្លៃ
```

This can make Khmer delivery smoother and more consistent.

You can still use English technical terms when needed.

---

# 24. Example Cybersecurity Script

```text
តើអ្នកធ្លាប់ភ្ជាប់វ៉ាយហ្វាយឥតគិតថ្លៃ
នៅហាងកាហ្វេ ឬកន្លែងសាធារណៈដែរឬទេ?

វាងាយស្រួលមែនទែន...
ប៉ុន្តែ វាក៏អាចមានហានិភ័យដែរ។

អ្នកវាយប្រហារអាចបង្កើតវ៉ាយហ្វាយក្លែងក្លាយ
ដែលមានឈ្មោះស្រដៀងនឹងវ៉ាយហ្វាយពិត។

ដូច្នេះ មុនពេលភ្ជាប់
សូមពិនិត្យឈ្មោះបណ្តាញឱ្យបានច្បាស់។
```

Recommended:

```text
Voice:
Khmer Male - Young

Style:
Cybersecurity

CFG:
2.0

Steps:
20
```

---

# 25. Generated Files

Generated files are saved inside:

```text
outputs/
```

Example:

```text
outputs/
├── khmer_khmer_male_-_young_a21c03ef.wav
├── khmer_khmer_female_-_young_91fa43cd.wav
└── khmer_khmer_male_-_deep_88ab219d.wav
```

You can also download them directly from the web UI.

---

# 26. Apple Silicon / M2 Configuration

The app currently uses:

```python
device="auto"
```

This allows VoxCPM to select an available device automatically.

If you get an MPS-related error, change:

```python
device="auto",
```

to:

```python
device="cpu",
```

Then restart:

```bash
Ctrl + C
python app.py
```

CPU will normally be slower, but it can help if a particular operation has compatibility problems on MPS.

---

# 27. Model Loading

The model is loaded once when the app starts:

```python
model = VoxCPM.from_pretrained(
    MODEL_ID,
    device="auto",
    optimize=False,
    load_denoiser=False,
)
```

You do NOT want to load the model every time you press Generate.

That would make every generation unnecessarily slow.

---

# 28. First Launch

The first run may need to download VoxCPM2 model files.

So:

```bash
python app.py
```

may take longer the first time.

Future launches can reuse downloaded model files.

---

# 29. Quick Daily Workflow

When you want to create content:

```bash
cd khmer-voxcpm
source .venv/bin/activate
python app.py
```

Then in the browser:

```text
1. Paste Khmer script
2. Choose Voice
3. Choose Style
4. Optional: add extra instruction
5. Optional: upload custom reference voice
6. Click Generate Khmer Audio
7. Listen
8. Download WAV
```

---

# 30. When You Add a New Permanent Voice

Copy the WAV:

```text
voices/new_voice.wav
```

Then add:

```python
"Khmer Male 04": {
    "reference": "voices/new_voice.wav",
    "description": (
        "Native Khmer speaker, "
        "young Cambodian male, "
        "clear Khmer pronunciation"
    ),
},
```

Save `studio/voices.py`.

Stop the app:

```text
Ctrl + C
```

Run again:

```bash
python app.py
```

The new voice should appear automatically in the dropdown.

---

# 31. Suggested Voice Library for Content Creation

A good starting collection could be:

```text
Khmer Male 01 — Young / Friendly
Khmer Male 02 — Deep / Serious
Khmer Male 03 — Soft / Relaxed

Khmer Female 01 — Young / Friendly
Khmer Female 02 — Professional
Khmer Female 03 — Soft

Narrator 01 — Documentary
Narrator 02 — Storytelling
```

Then reuse those with different styles.

Example:

```text
Male 01 + Cybersecurity
Male 01 + Natural
Male 01 + Energetic

Female 01 + News
Female 01 + Storytelling

Narrator 01 + Documentary
Narrator 01 + Serious
```

You do not need a separate voice recording for every speaking style.

---

# 32. Future English Support

For now, focus on Khmer.

Later you can expand the same app for English content.

A future structure could be:

```text
Language:
Khmer
English

Voice:
...

Style:
...
```

There is no need to redesign the entire app.

---

# 33. Useful Commands

Activate environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python app.py
```

Stop:

```text
Ctrl + C
```

Exit virtual environment:

```bash
deactivate
```

Install dependencies:

```bash
pip install voxcpm gradio soundfile numpy
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools
```

---

# 34. Recommended Starting Setup

For Khmer content:

```text
Voice:
A good native Khmer reference voice

Style:
Natural or Cybersecurity

CFG:
2.0

Inference Steps:
20

Reference length:
10–30 seconds

Text:
Natural Khmer with punctuation and paragraph breaks
```

The most important rule is:

```text
REFERENCE VOICE = WHO SPEAKS
STYLE           = HOW THEY SPEAK
TEXT            = WHAT THEY SAY
```

That structure lets you build a reusable Khmer content-generation workflow with many voices and many speaking styles.
