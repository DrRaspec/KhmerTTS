"""Smoke-test the UI and progress flow without loading the speech model."""

import os
import importlib
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.old_cwd)
        self.model = SimpleNamespace(
            tts_model=SimpleNamespace(
                device="cpu", config=SimpleNamespace(dtype="float32"), sample_rate=16000,
            ),
            generate=lambda **kwargs: np.zeros(1600, dtype=np.float32),
        )
        fake_voxcpm = SimpleNamespace(VoxCPM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: self.model,
        ))
        # Keep imported UI modules available for Gradio's schema inspection.
        self.patch = patch.dict("sys.modules", {"voxcpm": fake_voxcpm})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        # Refresh shared runtime state inside the temporary workspace for each test.
        modules = {
            name: importlib.reload(importlib.import_module(f"studio.{name}"))
            for name in ("config", "voices", "generation", "ui")
        }
        voices, generation, ui = (modules[name] for name in ("voices", "generation", "ui"))
        speech_models = importlib.import_module("studio.speech_models")
        self.studio = {
            **vars(speech_models), **vars(voices), **vars(generation),
            "app": ui.build_app(),
        }
        self.studio["MODEL_MANAGER"].factory = lambda key: SimpleNamespace(
            device="cpu", sample_rate=16000,
            generate=lambda text, options, task: self.model.generate(text=text, **options),
        )
        self.args = ("សួស្តីអ្នកទាំងអស់គ្នា។", "Khmer Male - Young", "Natural", "", None, 2, 4)

    def test_ui_builds_with_settings_and_player(self):
        labels = [c["props"].get("label") for c in self.studio["app"].get_config_file()["components"]]
        self.assertIn("Detail steps", labels)
        self.assertIn("Audio preview", labels)

    def test_english_switch_filters_voices_and_preserves_user_script(self):
        key = self.studio["ENGLISH_MODEL"]
        result = self.studio["switch_model"](key, "My own script.")
        self.assertTrue(result[2]["interactive"])
        self.assertTrue(all(name.startswith("English") for name in result[2]["choices"]))
        self.assertNotIn("value", result[-1])
        result = self.studio["switch_model"](key, self.args[0])
        self.assertEqual(result[-1]["value"], self.studio["SCRIPT_SAMPLES"]["English"])
        result = self.studio["switch_model"](self.studio["ENGLISH_LITE_MODEL"])
        self.assertFalse(result[2]["interactive"])

    def test_khmer_rejects_exaggerated_styles_in_ui_and_stale_requests(self):
        khmer = self.studio["switch_model"](self.studio["FULL_MODEL"])
        english = self.studio["switch_model"](self.studio["ENGLISH_MODEL"])
        for style in ("Comedy", "Cartoon"):
            self.assertNotIn(style, khmer[3]["choices"])
            self.assertIn(style, english[3]["choices"])
            with self.assertRaises(gr.Error):
                self.studio["build_voice_prompt"](
                    "Khmer Male - Young", style, "", has_reference=True,
                )
        self.assertNotIn("Playful", khmer[3]["choices"])
        self.assertFalse(any(name.startswith("Khmer Funny") for name in khmer[2]["choices"]))

    def test_original_khmer_preset_sends_original_prompt_and_options(self):
        received = []
        self.model.generate = lambda **kwargs: received.append(kwargs) or np.zeros(1600, dtype=np.float32)
        script = "សួស្តីអ្នកទាំងអស់គ្នា។"
        list(self.studio["generate_audio"](
            script, "Khmer Male - Young", "Natural", "", None, 2, 10,
        ))
        self.assertEqual(received[-1], {
            "text": "(Native Khmer speaker, young Cambodian male around 20 to 25 years old, "
                    "warm natural voice, medium-low pitch, clear Khmer pronunciation, "
                    "warm conversational tone, comfortable speaking pace, "
                    "gentle expression and natural pauses) " + script,
            "cfg_value": 2.0, "inference_timesteps": 10,
            "normalize": False, "retry_badcase": False,
        })
        self.assertEqual(self.studio["voices_for_model"](self.studio["FULL_MODEL"])[0], "Khmer Male - Young")

    def test_long_vox_script_joins_sections_and_repeats_reference_and_settings(self):
        received = []
        def generate(**kwargs):
            received.append(kwargs)
            if len(received) == 1:
                time.sleep(1.1)  # Keep a section active long enough for the live timer.
            return np.ones(1600, dtype=np.float32) * 0.1
        self.model.generate = generate
        script = ("ខ្ញុំស្រឡាញ់អ្នក។ សូមស្រឡាញ់ខ្ញុំវិញផង។\n" * 10).strip()
        reference = Path("reference.wav")
        self.studio["sf"].write(reference, np.zeros(1600), 16000)
        updates = list(self.studio["generate_audio"](
            script, "Khmer Male - Young", "Natural", "", str(reference), 2, 10,
        ))
        chunks = self.studio["split_vox_text"](script)
        self.assertEqual(len(received), len(chunks))
        for call, chunk in zip(received, chunks):
            self.assertTrue(call["text"].endswith(") " + chunk))
            self.assertEqual(call["reference_wav_path"], str(reference))
            self.assertEqual(call["inference_timesteps"], 10)
            self.assertEqual(call["cfg_value"], 2)
        audio, rate = self.studio["sf"].read(updates[-1][0])
        self.assertEqual(len(audio), 1600 * len(chunks) + 4000 * (len(chunks) - 1))
        self.assertEqual(rate, 16000)
        self.assertTrue(any("section" in update[2] for update in updates))

    def test_cancelled_long_script_does_not_generate_next_section_or_save(self):
        received = []
        manager = self.studio["MODEL_MANAGER"]
        def generate(**kwargs):
            received.append(kwargs)
            manager.cancel("local")
            return np.zeros(1600, dtype=np.float32)
        self.model.generate = generate
        updates = list(self.studio["generate_audio"](
            "សូមស្រឡាញ់ខ្ញុំវិញផង។\n" * 20, *self.args[1:],
        ))
        self.assertEqual(len(received), 1)
        self.assertIn("cancelled", updates[-1][2])
        self.assertIsNone(updates[-1][0])
        self.assertEqual(list(Path("outputs").glob("*.wav")), [])

    def test_english_vox_receives_english_voice_design(self):
        received = []
        def generate(**kwargs):
            received.append(kwargs)
            return np.zeros(1600, dtype=np.float32)
        self.model.generate = generate
        updates = list(self.studio["generate_audio"](
            "Hello everyone.", "English Male - Warm", "Natural", "", None, 2, 4,
            self.studio["ENGLISH_MODEL"],
        ))
        self.assertIn("Native English male", received[0]["text"])
        self.assertNotIn("Khmer", received[0]["text"])
        self.assertTrue(Path(updates[-1][0]).exists())

    def test_english_lite_receives_plain_text_and_english_filename(self):
        received = []
        self.model.generate = lambda **kwargs: received.append(kwargs) or np.zeros(1600, dtype=np.float32)
        updates = list(self.studio["generate_audio"](
            "Hello everyone.", "English Male - Warm", "Natural", "", None, 2, 4,
            self.studio["ENGLISH_LITE_MODEL"],
        ))
        self.assertEqual(received, [{"text": "Hello everyone."}])
        self.assertIn("mms_english_", updates[-1][0])

    def test_saved_english_voice_keeps_language_and_filters_into_english(self):
        source = Path("reference.wav")
        self.studio["sf"].write(source, np.zeros(1600), 16000)
        self.studio["add_permanent_voice"]("English custom", "Warm voice", str(source), "English")
        voices = self.studio["load_saved_voices"]()
        self.assertEqual(voices["English custom"]["language"], "English")
        self.assertIn("English custom", self.studio["voices_for_model"](self.studio["ENGLISH_MODEL"]))
        self.assertNotIn("English custom", self.studio["voices_for_model"](self.studio["FULL_MODEL"]))

    def test_named_generated_voice_reuses_reference_on_every_section(self):
        updates = list(self.studio["generate_named_take"](*self.args))
        take = updates[-1][3]
        result = self.studio["save_generated_voice"]("Dara", take, self.studio["FULL_MODEL"])
        self.assertIsNone(result[-1])  # Remove any temporary speaker override.
        saved = self.studio["load_saved_voices"]()["Dara"]
        self.assertTrue(Path(saved["reference"]).exists())
        self.assertEqual(self.studio["voices_for_model"](self.studio["FULL_MODEL"])[0], "Dara")
        calls = []
        self.model.generate = lambda **kwargs: calls.append(kwargs) or np.zeros(1600, dtype=np.float32)
        args = ("សួស្តីអ្នកទាំងអស់គ្នា។ " * 20, "Dara", "Natural", "", None, 2, 4)
        for _ in range(2):
            list(self.studio["generate_audio"](*args))
        self.assertGreater(len(calls), 2)
        self.assertTrue(all(call["reference_wav_path"] == saved["reference"] for call in calls))

    def test_generated_voice_retains_original_language_after_model_switch(self):
        take = list(self.studio["generate_named_take"](*self.args))[-1][3]
        self.studio["save_generated_voice"]("Sophea", take, self.studio["ENGLISH_MODEL"])
        self.assertEqual(self.studio["load_saved_voices"]()["Sophea"]["language"], "Khmer")
        self.assertNotIn("Sophea", self.studio["voices_for_model"](self.studio["ENGLISH_MODEL"]))

    def test_cannot_save_missing_generated_take_or_replace_named_speaker(self):
        with self.assertRaises(gr.Error):
            self.studio["save_generated_voice"]("Dara", None, self.studio["FULL_MODEL"])
        take = list(self.studio["generate_named_take"](*self.args))[-1][3]
        self.studio["save_generated_voice"]("Dara", take, self.studio["FULL_MODEL"])
        with self.assertRaises(gr.Error):
            self.studio["save_generated_voice"]("Dara", take, self.studio["FULL_MODEL"])

    def test_live_elapsed_status_and_saved_audio(self):
        def generate(**kwargs):
            time.sleep(1.1)
            return np.zeros(1600, dtype=np.float32)
        self.model.generate = generate
        updates = list(self.studio["generate_audio"](*self.args))
        statuses = [u[2] for u in updates]
        self.assertIn("Preparing", statuses[0])
        self.assertTrue(any("00:01" in s and "Remaining" in s for s in statuses))
        self.assertTrue(any("Saving your audio" in s for s in statuses))
        self.assertIn("audio is ready", statuses[-1])
        self.assertTrue(Path(updates[-1][0]).exists())
        preview = self.studio["preview_estimate"](*self.args[:5], self.args[6], self.studio["FULL_MODEL"])
        self.assertIn("Estimate:", preview)

    def test_failed_inference_reports_error_without_audio(self):
        def fail(**kwargs):
            raise RuntimeError("Test failure")
        self.model.generate = fail
        updates = list(self.studio["generate_audio"](*self.args))
        self.assertIn("couldn’t finish", updates[-1][2])
        self.assertIsNone(updates[-1][0])

    def test_failed_save_reports_error_without_download(self):
        with patch.object(self.studio["sf"], "write", side_effect=OSError("Disk full")):
            updates = list(self.studio["generate_audio"](*self.args))
        self.assertIn("could not be saved", updates[-1][2])
        self.assertIsNone(updates[-1][1])

    def test_cancel_discards_audio_and_releases_generation(self):
        def generate(text, options, task):
            while True:
                task.check()
                time.sleep(0.01)
        manager = self.studio["MODEL_MANAGER"]
        manager.factory = lambda key: SimpleNamespace(device="cpu", sample_rate=16000, generate=generate)
        iterator = self.studio["generate_audio"](*self.args)
        next(iterator)  # Preparing; owner is registered before the worker starts.
        next(iterator)  # Worker is running.
        self.assertTrue(manager.cancel("local"))
        updates = list(iterator)
        self.assertIn("cancelled", updates[-1][2])
        self.assertIsNone(updates[-1][0])
        self.assertIsNone(manager.active_task)
        self.assertEqual(list(Path("outputs").glob("*.wav")), [])
        self.assertFalse(Path(".generation_timings.json").exists())

    def test_switch_updates_capabilities_and_unloads_model(self):
        manager = self.studio["MODEL_MANAGER"]
        list(self.studio["generate_audio"](*self.args))
        self.assertIsNotNone(manager.model)
        outputs = self.studio["switch_model"](self.studio["LITE_MODEL"])
        self.assertIsNone(manager.model)
        self.assertIn("Noncommercial", outputs[1])
        self.assertFalse(outputs[2]["interactive"])

    def test_lite_generation_receives_plain_khmer_without_voice_instructions(self):
        received = []
        manager = self.studio["MODEL_MANAGER"]
        def generate(text, options, task):
            received.append((text, options))
            return np.zeros(1600, dtype=np.float32)
        manager.factory = lambda key: SimpleNamespace(device="cpu", sample_rate=16000, generate=generate)
        updates = list(self.studio["generate_audio"](*self.args, self.studio["LITE_MODEL"]))
        self.assertEqual(received, [(self.args[0], {})])
        self.assertIn("audio is ready", updates[-1][2])

    def test_cancel_during_save_removes_the_output(self):
        manager = self.studio["MODEL_MANAGER"]
        write = self.studio["sf"].write
        def write_then_cancel(*args, **kwargs):
            write(*args, **kwargs)
            manager.cancel("local")
        with patch.object(self.studio["sf"], "write", side_effect=write_then_cancel):
            updates = list(self.studio["generate_audio"](*self.args))
        self.assertIn("cancelled", updates[-1][2])
        self.assertEqual(list(Path("outputs").glob("*.wav")), [])


if __name__ == "__main__":
    unittest.main()
