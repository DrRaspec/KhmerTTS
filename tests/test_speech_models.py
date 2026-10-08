import time
import re
import unicodedata
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from studio.speech_models import (
    FULL_MODEL, LITE_MODEL, ENGLISH_MODEL, ENGLISH_LITE_MODEL, LiteSpeechModel,
    GenerationCancelled, GenerationTask, SpeechModelManager,
    cancellation_hooks, generation_executor, split_lite_text, split_vox_text,
)


class SpeechModelTests(unittest.TestCase):
    def test_vox_chunking_preserves_mixed_script_and_short_input(self):
        short = "ខ្ញុំស្រឡាញ់អ្នក។ សូមស្រឡាញ់ខ្ញុំវិញផង។"
        self.assertEqual(split_vox_text(short), [short])
        text = (short + '\n“Error 404: មិនអាចរកប៊ូតុងឈប់ស្រឡាញ់បានទេ”។\n') * 8
        chunks = split_vox_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(c) <= 180 for c in chunks))
        self.assertEqual(re.sub(r"\s", "", "".join(chunks)), re.sub(r"\s", "", text))

    def test_vox_hard_cut_does_not_split_khmer_marks_or_coeng(self):
        text = "ខ្ញុំស្រឡាញ់អ្នក" * 60
        chunks = split_vox_text(text)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c) <= 180 for c in chunks))
        for left, right in zip(chunks, chunks[1:]):
            self.assertFalse(unicodedata.category(right[0]).startswith("M"))
            self.assertNotEqual(left[-1], "\u17d2")

    def test_english_vox_does_not_share_khmer_inference_state(self):
        calls = []
        manager = SpeechModelManager("cpu", lambda key: calls.append(key) or object())
        task = manager.begin("owner", FULL_MODEL)
        engine = manager.load(task)
        manager.finish(task)
        manager.switch(ENGLISH_MODEL)
        self.assertIsNone(manager.model)
        task = manager.begin("owner", ENGLISH_MODEL)
        self.assertIsNot(manager.load(task), engine)
        self.assertEqual(calls, [FULL_MODEL, ENGLISH_MODEL])
        manager.finish(task)
        manager.switch(ENGLISH_LITE_MODEL)
        self.assertIsNone(manager.model)

    def test_english_lite_loads_english_checkpoint_for_model_and_tokenizer(self):
        model = SimpleNamespace(config=SimpleNamespace(sampling_rate=16000))
        model.to = lambda device: model
        model.eval = lambda: model
        with patch("transformers.VitsModel.from_pretrained", return_value=model) as weights, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=object()) as tokenizer:
            engine = LiteSpeechModel(ENGLISH_LITE_MODEL)
        self.assertEqual(engine.language, "English")
        self.assertEqual(engine.sample_rate, 16000)
        self.assertEqual(weights.call_args.args, ("facebook/mms-tts-eng",))
        self.assertEqual(tokenizer.call_args.args, ("facebook/mms-tts-eng",))

    def test_lite_downloads_tokenizer_when_only_weights_are_cached(self):
        model = SimpleNamespace(config=SimpleNamespace(sampling_rate=16000))
        model.to = lambda device: model
        model.eval = lambda: model
        with patch("transformers.VitsModel.from_pretrained", return_value=model), \
             patch("huggingface_hub.hf_hub_download", side_effect=OSError("Missing vocabulary")), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=object()) as tokenizer:
            LiteSpeechModel(ENGLISH_LITE_MODEL)
        self.assertEqual(tokenizer.call_args.args, ("facebook/mms-tts-eng",))
        self.assertNotIn("local_files_only", tokenizer.call_args.kwargs)

    def test_cancel_checked_between_forward_calls_and_hooks_removed(self):
        layer = torch.nn.Linear(1, 1)
        task = GenerationTask("owner", FULL_MODEL)
        with self.assertRaises(GenerationCancelled):
            with cancellation_hooks([layer], task):
                layer(torch.ones(1))
                task.cancelled.set()
                layer(torch.ones(1))
        self.assertEqual(len(layer._forward_pre_hooks), 0)

    def test_switch_releases_old_model_and_loading_is_lazy(self):
        factory_calls = []
        def factory(key):
            factory_calls.append(key)
            return SimpleNamespace(key=key)
        manager = SpeechModelManager("cpu", factory)
        manager.switch(LITE_MODEL)
        self.assertEqual(factory_calls, [])
        task = manager.begin("owner", LITE_MODEL)
        first = manager.load(task)
        self.assertIs(first, manager.load(task))
        manager.finish(task)
        manager.switch(FULL_MODEL)
        self.assertIsNone(manager.model)
        self.assertEqual(factory_calls, [LITE_MODEL])

    def test_busy_switch_and_second_generation_rejected(self):
        manager = SpeechModelManager("cpu")
        task = manager.begin("owner", FULL_MODEL)
        with self.assertRaises(RuntimeError):
            manager.switch(LITE_MODEL)
        with self.assertRaises(RuntimeError):
            manager.begin("another", FULL_MODEL)
        self.assertFalse(manager.cancel("another"))
        self.assertTrue(manager.cancel("owner"))
        with self.assertRaises(GenerationCancelled):
            manager.commit(task)
        manager.finish(task)
        self.assertIsNone(manager.active_task)

    def test_completed_task_cannot_be_cancelled(self):
        manager = SpeechModelManager("cpu")
        task = manager.begin("owner", FULL_MODEL)
        manager.commit(task)
        self.assertFalse(manager.cancel("owner"))

    def test_abandoned_executor_cancels_and_joins_worker(self):
        task = GenerationTask("owner", FULL_MODEL)
        def work():
            while True:
                task.check()
                time.sleep(0.01)
        with self.assertRaises(RuntimeError):
            with generation_executor(task) as executor:
                future = executor.submit(work)
                raise RuntimeError("Disconnected")
        self.assertTrue(future.done())
        self.assertIsInstance(future.exception(), GenerationCancelled)

    def test_successful_executor_does_not_mark_cancelled(self):
        task = GenerationTask("owner", FULL_MODEL)
        with generation_executor(task) as executor:
            self.assertEqual(executor.submit(lambda: 1).result(), 1)
        self.assertFalse(task.cancelled.is_set())

    def test_long_lite_script_preserves_text_and_bounds_chunks(self):
        text = "សួស្តីអ្នកទាំងអស់គ្នា។\n" + "ក" * 610 + "?"
        chunks = split_lite_text(text)
        self.assertTrue(all(len(chunk) <= 250 for chunk in chunks))
        self.assertEqual("".join(chunks), text.replace("\n", ""))


if __name__ == "__main__":
    unittest.main()
