import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from training.prepare_data import prepare_source, split_records


class DataPreparationTests(unittest.TestCase):
    def test_repeated_transcripts_do_not_cross_partitions(self):
        records = [{"text": text, "audio": str(i)} for i, text in enumerate(
            ["សួស្តី", " សួស្តី ", "អរគុណ", "ជម្រាបលា", "សុខសប្បាយ"]
        )]
        train, validation = split_records(records)
        self.assertTrue(train and validation)
        self.assertFalse({r["text"].strip() for r in train} & {r["text"].strip() for r in validation})
        self.assertEqual(len(train) + len(validation), len(records))

    def test_download_keeps_one_speaker_and_resamples_audio(self):
        buffer = io.BytesIO()
        signal = np.sin(np.arange(24000) * 0.1) * 0.2
        sf.write(buffer, signal, 8000, format="WAV")
        rows = []
        for i, identity in enumerate(["0308", "other", "0308", "0308"]):
            rows.append({"row_idx": i, "row": {
                "speaker_id": identity, "transcription": f"សួស្តី {i}", "duration": 3,
                "audio": [{"src": f"https://example.org/--/{'a' * 40}/--/{i}.wav"}],
            }})
        class Session:
            def get(self, url, **kwargs):
                return SimpleNamespace(
                    raise_for_status=lambda: None, json=lambda: {"rows": rows},
                    content=buffer.getvalue(),
                )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = prepare_source(Session(), "openslr42", root, 3)
            self.assertEqual(report["clips"], 3)
            records = [json.loads(line) for line in (root / "openslr42/provenance.jsonl").read_text().splitlines()]
            self.assertEqual({r["speaker_id"] for r in records}, {"0308"})
            for record in records:
                info = sf.info(record["audio"])
                self.assertEqual(info.samplerate, 16000)
                self.assertEqual(info.channels, 1)
                self.assertAlmostEqual(info.duration, 3)
                self.assertEqual(record["source_revision"], "a" * 40)
            self.assertTrue((root / "openslr42/pilot_lora.yaml").exists())

    def test_unknown_fleurs_speakers_remain_evaluation_only(self):
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(np.arange(48000) * 0.1) * 0.2, 16000, format="WAV")
        rows = [{"row_idx": i, "row": {
            "raw_transcription": f"សួស្តី {i}",
            "audio": [{"src": "https://example.org/sample.wav"}],
        }} for i in range(3)]
        session = SimpleNamespace(get=lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"rows": rows}, content=buffer.getvalue(),
        ))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = prepare_source(session, "fleurs", root, 3)
            self.assertEqual(report["partitions"], {"evaluation": 3})
            self.assertFalse((root / "fleurs/train.jsonl").exists())
            self.assertFalse((root / "fleurs/pilot_lora.yaml").exists())


if __name__ == "__main__":
    unittest.main()
