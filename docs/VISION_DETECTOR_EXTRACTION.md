# Explicit optional detector extraction

The existing S3FD/manual choices remain the compatibility path. Extraction
also accepts `yolo11m-face`, `yolo12l-face` and `yolo26s-face` when the user
explicitly selects one. This is opt-in integration, not a quality winner
selection or candidate-weight redistribution.

`main.py extract --detector` passes the selected ID through
`ExtractSubprocessor` and its worker dictionary. The independent
`facelib/DetectorCandidates.py` adapter calls the shared
`webui/python/vision_detectors.py` PyTorch implementation. It preserves the
rects API, largest-face-first limiting, RGB/BGR flag and optional legacy
overlap suppression. The recorded detector settings remain input size 960,
confidence 0.25, NMS IoU 0.5, FP32 and no augmentation.

The selected model's registry entry is checked against the fixed public
catalog's hash, size, runtime, task, scale, architecture and filename. Weight bytes and installed optional
runtime version and Torch/torchvision import health are checked before `Extractor.main` creates or clears any
output. Missing assets or dependency/version mismatches fail explicitly.
Model loading also checks the one-face detection task and architecture
scale. Inference never downloads a model or substitutes S3FD for a selected
candidate. No option automatically changes a product default.

Landmark inference and geometry remain the existing native 68-point FAN
path; `head` continues to use the existing 3DFAN heatmaps. A 98-point
candidate is never written into the legacy 68-point metadata. 3DFAN heatmaps
still produce the existing XY landmark contract; this evidence does not
claim newly predicted depth coordinates.

## Actual end-to-end evidence, 2026-10-06

All three optional detectors completed separate full CLI multiprocessing
extractions on CUDA device 0. The input was a private ignored copy of the
official AdaFace `img1.jpeg` demonstration, with the original read-only.
Each job used `head`, image size 256, JPEG quality 95, at most one face and
disabled debug-image output. Each produced exactly one aligned image.

The source SHA256 is
`519346b222f4b8b9be453d5866c5e235eec4f92540afd7fb0e1fc975e20be7e1`.
The existing 3DFAN weight SHA256 is
`b50d2faf0fd4d6503aba9d19365e7aff06f2ea96bac37fc5f8f25a191a0a63a9`.
The locked detector weight hashes and output-image hashes remain in the
ignored `workspace/.vision-models/detectors/extraction-20261006/evidence.json`.

| Detector | Aligned files | Image | Source/aligned landmark shape | Source-to-aligned affine max difference |
| --- | ---: | --- | --- | ---: |
| YOLO11m-face | 1 | 256x256, head | 68x2, finite | 0.0000172611 |
| YOLO12l-face | 1 | 256x256, head | 68x2, finite | 0.0000260003 |
| YOLO26s-face | 1 | 256x256, head | 68x2, finite | 0.0000252382 |

DFL metadata was reloaded from each written JPEG. The original filename,
finite source rect, finite 2x3 affine, native 68 points and `head` face type
were checked; applying the stored affine to source landmarks reproduced
the stored aligned landmarks within 0.0001 pixels. This verifies plumbing,
real candidate inference and metadata geometry on one image. Detector
quality comparison and uncertainty remain separate from this integration
check; this single image establishes no quality ranking.

Nine targeted tests passed, covering worker propagation, unchanged default
S3FD/native 68 behavior, retained head FAN3D initialization, rect ordering,
overlap/color contracts, absent assets, fixed-catalog metadata tampering, runtime mismatch, broken native ABI and failure before
output mutation.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tools/tests -p test_extractor_detector_candidates.py -v
.venv/Scripts/python.exe _internal/DeepFaceLab/main.py extract --input-dir workspace/example/input --output-dir workspace/example/aligned --detector yolo12l-face --face-type head --image-size 256 --jpeg-quality 95 --max-faces-from-image 1 --no-output-debug --force-gpu-idxs 0
```

Prepare candidate assets and project-local optional dependencies before the
second command. Existing user aligned directories were not modified in this
acceptance check.

The media command help now describes reference frame PTS and audio timing,
and distinguishes a `.nut` RGB lossless master from MP4 CRF 0's YUV-domain
losslessness. This change updates help text only; media implementations are
handled separately.
