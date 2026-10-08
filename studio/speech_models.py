"""Lazy speech models and cooperative cancellation without overlapping inference."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
import gc
import re
import unicodedata
from pathlib import Path
from threading import Event, Lock

import numpy as np
import torch


FULL_MODEL = "VoxCPM2 · Voice design & cloning"
ENGLISH_MODEL = "VoxCPM2 English · Voice design & cloning"
LITE_MODEL = "MMS Khmer · Lightweight (noncommercial)"
ENGLISH_LITE_MODEL = "MMS English · Lightweight (noncommercial)"
VOX_MODELS = (FULL_MODEL, ENGLISH_MODEL)
LITE_CHECKPOINTS = {
    LITE_MODEL: "facebook/mms-tts-khm",
    ENGLISH_LITE_MODEL: "facebook/mms-tts-eng",
}
MODEL_CHOICES = [FULL_MODEL, LITE_MODEL, ENGLISH_MODEL, ENGLISH_LITE_MODEL]
VOX_CHUNK_CHARS = 180


def split_vox_text(text, max_chars=VOX_CHUNK_CHARS):
    """Bound script sections before adding voice prompts; keep Khmer clusters intact."""
    if max_chars < 1:
        raise ValueError("Chunk size must be positive.")
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    for paragraph in re.split(r"\n+", text):
        pending = ""
        for sentence in re.split(r"(?<=[។!?])|(?<=\.)(?=\s|$)", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if pending and len(pending) + 1 + len(sentence) <= max_chars:
                pending += " " + sentence
                continue
            if pending:
                chunks.append(pending)
                pending = ""
            while len(sentence) > max_chars:
                cut = max((i for i, char in enumerate(sentence[:max_chars + 1]) if char.isspace()), default=0)
                if cut == 0:
                    cut = max_chars
                    # Khmer coeng joins the following consonant; dependent vowels
                    # and other marks belong to the preceding character.
                    while cut > 0 and (
                        unicodedata.category(sentence[cut]).startswith("M")
                        or sentence[cut - 1] in ("\u17d2", "\u200d")
                        or sentence[cut] in ("\u200d", "\u200c")
                    ):
                        cut -= 1
                if cut == 0:
                    raise ValueError("Text cannot be split safely. Add spaces between phrases.")
                chunks.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            pending = sentence
        if pending:
            chunks.append(pending)
    return chunks


def model_language(key):
    return "English" if key in (ENGLISH_MODEL, ENGLISH_LITE_MODEL) else "Khmer"


class GenerationCancelled(Exception):
    pass


@dataclass
class GenerationTask:
    owner: str
    model_key: str
    cancelled: Event = field(default_factory=Event)
    phase: str = "Loading model"
    inference_started: float | None = None

    def check(self, *args):
        if self.cancelled.is_set():
            raise GenerationCancelled()


@contextmanager
def cancellation_hooks(modules, task):
    handles = []
    try:
        for module in modules:
            if hasattr(module, "register_forward_pre_hook"):
                handles.append(module.register_forward_pre_hook(task.check))
        task.check()
        yield
        task.check()
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def readable_audio_progress(core):
    """Show steps and elapsed time: VoxCPM's limit is not a completion target."""
    module = import_module(type(core).__module__)
    original = module.tqdm
    module.tqdm = partial(
        original,
        desc="Creating audio",
        bar_format="{desc}: {n_fmt} audio steps | elapsed {elapsed}",
    )
    try:
        yield
    finally:
        module.tqdm = original


@contextmanager
def generation_executor(task):
    """Signal a stopped/disconnected generator before waiting for its worker."""
    class TrackedExecutor(ThreadPoolExecutor):
        def __init__(self):
            super().__init__(max_workers=1)
            self.futures = []

        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            self.futures.append(future)
            return future

    executor = TrackedExecutor()
    try:
        yield executor
    except BaseException:
        # A completed worker's ordinary error is a failure, not a cancel.
        if any(not future.done() for future in executor.futures):
            task.cancelled.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class VoxSpeechModel:
    def __init__(self, device):
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2", device=device, optimize=False, load_denoiser=False,
        )
        self.device = self.model.tts_model.device
        self.sample_rate = self.model.tts_model.sample_rate

    def generate(self, text, options, task):
        core = self.model.tts_model
        modules = [core]
        for name in ("base_lm", "residual_lm", "feat_decoder", "audio_vae"):
            module = getattr(core, name, None)
            if module is not None:
                modules.append(module)
                # Catch diffusion steps and VAE work as well as audio iterations.
                if hasattr(module, "children"):
                    modules.extend(module.children())
        with cancellation_hooks(modules, task), readable_audio_progress(core):
            return self.model.generate(text=text, **options)


def split_lite_text(text, max_chars=250):
    """Keep lightweight inference bounded; preserve every character in order."""
    chunks = []
    for sentence in re.findall(r"[^។!?\n]+[។!?]?|[។!?]+", text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars + 1)
            if cut <= 0:
                cut = max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


class LiteSpeechModel:
    def __init__(self, model_key=LITE_MODEL):
        from transformers import AutoTokenizer, VitsModel
        from huggingface_hub import hf_hub_download
        # CPU is intentional: this small model avoids shared GPU memory demand
        # and uses operations that are not uniformly supported on MPS.
        cache = str(Path(__file__).parent / ".cache" / "huggingface" / "hub")
        checkpoint = LITE_CHECKPOINTS[model_key]
        self.language = model_language(model_key)
        def load(component, required_file=None):
            try:
                # Model weights may be cached before the tokenizer vocabulary.
                # VitsTokenizer can raise TypeError for that incomplete cache.
                if required_file:
                    hf_hub_download(checkpoint, required_file, cache_dir=cache, local_files_only=True)
                return component.from_pretrained(
                    checkpoint, cache_dir=cache, local_files_only=True,
                )
            except OSError:
                return component.from_pretrained(checkpoint, cache_dir=cache)
        self.model = load(VitsModel).to("cpu").eval()
        self.tokenizer = load(AutoTokenizer, required_file="vocab.json")
        self.device = "cpu"
        self.sample_rate = self.model.config.sampling_rate

    def generate(self, text, options, task):
        chunks = split_lite_text(text)
        audio = []
        with cancellation_hooks(self.model.modules(), task), torch.inference_mode():
            for index, chunk in enumerate(chunks):
                task.check()
                task.phase = f"Creating sentence {index + 1} of {len(chunks)}"
                inputs = self.tokenizer(chunk, return_tensors="pt")
                if inputs["input_ids"].numel() == 0:
                    raise ValueError("No spoken text found in this sentence.")
                waveform = self.model(**inputs).waveform.squeeze().cpu().numpy()
                task.check()
                if audio:
                    audio.append(np.zeros(int(self.sample_rate * 0.2), dtype=np.float32))
                audio.append(waveform.reshape(-1))
        if not audio:
            raise ValueError(f"Please enter spoken {self.language} text.")
        return np.concatenate(audio)


class SpeechModelManager:
    def __init__(self, device, factory=None):
        self.device = device
        self.factory = factory or self._create_model
        self._lock = Lock()
        self.active_task = None
        self.model = None
        self.loaded_key = None

    def _create_model(self, key):
        return LiteSpeechModel(key) if key in LITE_CHECKPOINTS else VoxSpeechModel(self.device)

    def _unload(self):
        self.model = None
        self.loaded_key = None
        gc.collect()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def switch(self, key):
        if key not in MODEL_CHOICES:
            raise ValueError("Choose a supported speech model.")
        with self._lock:
            if self.active_task is not None:
                raise RuntimeError("Cancel or finish the current generation before switching models.")
            if self.loaded_key != key:
                self._unload()

    def begin(self, owner, key):
        if key not in MODEL_CHOICES:
            raise ValueError("Choose a supported speech model.")
        with self._lock:
            if self.active_task is not None:
                raise RuntimeError("Another generation is still running. Wait for it to stop.")
            task = GenerationTask(owner, key)
            self.active_task = task
            return task

    def cancel(self, owner):
        with self._lock:
            if self.active_task is None or self.active_task.owner != owner:
                return False
            self.active_task.cancelled.set()
            return True

    def load(self, task):
        task.check()
        if self.loaded_key != task.model_key:
            self._unload()
            loaded = self.factory(task.model_key)
            task.check()
            self.model = loaded
            self.loaded_key = task.model_key
        return self.model

    def finish(self, task):
        with self._lock:
            if self.active_task is task:
                self.active_task = None

    def commit(self, task):
        """Close the cancellation window atomically before publishing an output."""
        with self._lock:
            task.check()
            if self.active_task is task:
                self.active_task = None
