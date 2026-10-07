import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "python"))
from restoration_task import decode_source, run_task


class CopyModel:
    provenance = {"model": "fixture", "strict": True, "newTraining": False}
    def __init__(self, *_args, **_kwargs): pass
    def restore(self, image, output_size):
        assert output_size == (image.shape[1], image.shape[0])
        result = image.copy()
        result[:, :, 1] = np.minimum(result[:, :, 1].astype(np.uint16) + 2, 255)
        return result


class RestorationTaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "workspace"
        self.runtime = self.workspace / ".webui"
        for directory in [self.workspace / "data_src", *[self.runtime / "restoration" / part for part in ["staging", "outputs"]]]:
            directory.mkdir(parents=True, exist_ok=True)
        self.source = self.workspace / "data_src" / "original.png"
        image = np.full((40, 64, 3), [40, 100, 200], dtype=np.uint8)
        Image.fromarray(image).save(self.source)
        self.original_bytes = self.source.read_bytes()
        self.request = {"taskId": "rst-" + "a" * 32, "side": "src", "modelId": "swinir-psnr", "device": "cpu",
                        "inputs": [{"name": self.source.name, "sha256": hashlib.sha256(self.original_bytes).hexdigest()}]}

    def tearDown(self): self.temporary.cleanup()

    def run_task(self, model_factory=CopyModel, progress=None):
        return run_task(self.request, self.workspace, self.runtime, "unused", model_factory, progress)

    def test_complete_new_copy_preserves_rectangular_rgb_dimensions_and_original_bytes(self):
        result = self.run_task()
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        file = result["outputs"][0]
        output = self.runtime / "restoration" / "outputs" / self.request["taskId"] / "images" / file["name"]
        with Image.open(output) as image:
            self.assertEqual(image.size, (64, 40))
            self.assertEqual(image.mode, "RGB")
        self.assertEqual(file["inputSha256"], self.request["inputs"][0]["sha256"])
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), file["outputSha256"])
        self.assertFalse(file["metadataCopied"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.run_task()

    def test_mid_batch_failure_leaves_no_published_partial_images(self):
        second = self.source.with_name("second.png")
        second.write_bytes(self.original_bytes)
        self.request["inputs"].append({"name": second.name, "sha256": self.request["inputs"][0]["sha256"]})
        class FailsSecond(CopyModel):
            calls = 0
            def restore(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2: raise RuntimeError("inference failure")
                return super().restore(*args, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "inference failure"):
            self.run_task(FailsSecond)
        self.assertEqual(list((self.runtime / "restoration" / "outputs").iterdir()), [])
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_source_change_during_inference_prevents_whole_batch_publication(self):
        def progress(_): self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed since"):
            self.run_task(progress=progress)
        self.assertEqual(list((self.runtime / "restoration" / "outputs").iterdir()), [])

    def test_selection_limit_collisions_and_wrong_model_stage_fail_before_loading(self):
        for changes in [{"inputs": self.request["inputs"] * 501}, {"modelId": "gfpgan-v1.4"},
                        {"inputs": self.request["inputs"] * 2},
                        {"inputs": [{"name": "../original.png", "sha256": "a"*64}]}]:
            original = self.request
            self.request = {**original, **changes}
            with self.assertRaises(ValueError): self.run_task(lambda *_a, **_k: self.fail("Must reject before model loading"))
            self.request = original

    def test_aligned_pickle_metadata_is_rejected_without_deserialization(self):
        payload = b"not executable pickle; must never deserialize"
        chunk = struct.pack(">I", len(payload)) + b"fcWp" + payload + struct.pack(">I", zlib.crc32(b"fcWp" + payload))
        with self.assertRaisesRegex(ValueError, "Aligned metadata"):
            decode_source(self.original_bytes[:33] + chunk + self.original_bytes[33:])
        jpeg = BytesIO()
        with Image.open(self.source) as image: image.save(jpeg, format="JPEG")
        data = jpeg.getvalue()
        app15 = b"\xff\xef" + struct.pack(">H", len(payload) + 2) + payload
        with self.assertRaisesRegex(ValueError, "APP15"):
            decode_source(data[:2] + app15 + data[2:])

    def test_arbitrary_model_shape_is_refused_before_publication(self):
        class WrongShape(CopyModel):
            def restore(self, *_args, **_kwargs): return np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "same-size"):
            self.run_task(WrongShape)
        self.assertEqual(list((self.runtime / "restoration" / "outputs").iterdir()), [])


if __name__ == "__main__": unittest.main()
