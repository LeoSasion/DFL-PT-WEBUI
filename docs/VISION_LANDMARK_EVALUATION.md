# Offline landmark quality comparison (2026-10-06)

This is candidate evaluation, not a production Extractor change or a default
model selection. All five outputs ran on the same twelve read-only source
images, with the existing shared xyxy face boxes and 26 locked visible eye/mouth
regions. No detector calls, training, metadata writes or model-media uploads.

The references are H3CE's locked assistant-drawn exploratory contours. They are
**not expert ground truth**, earlier outputs had been seen, and no expert NME,
official WFLW score or global quality winner is claimed. Occluded/ambiguous
regions were excluded before computing scores. 68/98-point curve densities and
their native preprocessing differ. Each image weights eyes and mouth equally
when both exist; images then receive equal weight.

| Native output | Images / regions | Eyes | Mouth | Combined |
| --- | ---: | ---: | ---: | ---: |
| TUFA98 (H3CE reference model) | 12 / 26 | 0.6852 | 0.7554 | 0.7335 |
| TUFA68 (same weights, native 68 prompt) | 12 / 26 | 0.8352 | 1.1168 | 1.0002 |
| ORFormer98 | 12 / 26 | 0.6358 | 0.9661 | 0.8029 |
| Regression without Soft-Argmax98 | 12 / 26 | 0.6853 | 0.8646 | 0.7495 |
| Existing project FAN68 compatibility path | 12 / 26 | 2.3261 | 1.8753 | 2.0576 |

Numbers are symmetric mean point-to-segment contour distances, normalized to a
common head size of 256 using `256/(1.35*max(box width, box height))`; lower is
closer to this reference. They are neither interocular NME nor a recognition
accuracy percentage. Curves are sampled uniformly in arc length 256 times.
ORFormer is numerically closer on eyes; TUFA98 is numerically closer on mouth
and combined regions in this sample. This does not preselect an upgrade.

The paired 20,000-image bootstrap (seed 20261006), conditional on these fixed
references, gives combined error differences versus TUFA98. It does not include
reference bias, anatomical ambiguity or population sampling uncertainty:

| Output minus TUFA98 | Mean difference | Conditional 95% interval |
| --- | ---: | --- |
| TUFA68 | +0.2668 | +0.1621 to +0.3745 |
| ORFormer98 | +0.0695 | -0.0788 to +0.1987 |
| Regression98 | +0.0160 | -0.1185 to +0.1388 |
| FAN68 | +1.3241 | +1.0459 to +1.5662 |

The intervals for the two independent 98-point candidates cross zero. The
small sample supports further paired review, not an overall champion.

## Interface and native topology

`webui/python/vision_landmarks.py` exposes `LandmarkPredictor(model_id,
assets_root, device).predict(image_rgb_uint8_or_path, box_xyxy)` and the
single-shot `predict_landmarks(...)` helper. Outputs include original-image
XY, model XY, original/model affine matrices, native topology/region indices,
weight identity and source revision, preprocessing, visibility status and
whether the output is actually 68 points. The CLI accepts `--image`, `--box`,
`--model`, `--device`, `--assets`, and `--output`.

Models are `tufa98`, `tufa68`, `orformer98`, `regression98`, `fan68`. TUFA's
68-point result uses its own official prompt; 98 points are never written into
a 68-point compatibility field or interpolated into another topology. The
existing FAN crop rounds endpoints before resize; its reported affine records
the actual integer crop and OpenCV resize pixel centers. None of these outputs
provides 3D or reliable point visibility. Production compatibility/3D approval
requires its own explicit implementation and acceptance. The separate optional
detector integration retains the existing FAN68/FAN3D production path.

`tools/vision-landmark-benchmark.py` accepts a local sample manifest (see its
module docstring) or explicit `--h3-hardset` plus `--reference` paths. It hashes
each source, applies the same box to all models, stores native predictions and
overlays, records excluded regions and leaves unscored quality as null. Loading
one candidate failure never skips the remaining candidates. The benchmark
does not automatically choose a model.

## Official assets and local reproduction

Research source and assets remain ignored under
`workspace/.vision-models/landmarks/{TUFA,ORFormer,regression}/`. Each folder
contains `source/`, `weights/`, and `identity.json` with source file hashes,
weight hashes, source revision, official download URL and license record.
The adapter independently pins each family/repository/revision, complete
source-file manifest plus license declaration, weight hashes and sizes,
required executable modules and license-record bytes. Rewriting a source
file and its manifest hash cannot bypass the reviewed fingerprint. Shared
asset-path resolution rejects absolute paths, traversal and reparse-point
escapes, including the entire source directory. TUFA98, ORFormer98 and
Regression98 each loaded once after tightening these contracts; their
mathematical prediction paths and final saved predictions were unchanged.
Loading verifies these records and strictly loads Torch state dictionaries
with `weights_only=True`. No H3CE application source or environment was copied.

| Model | Official source / pinned revision | License record |
| --- | --- | --- |
| TUFA | [Jiahao-UTS/TUFA](https://github.com/Jiahao-UTS/TUFA), `58b3833ebbd481aeaa72a8500d5cc781e4ab8112` | GPL-2.0; local evaluation only |
| ORFormer | [ben0919/ORFormer](https://github.com/ben0919/ORFormer), `7e77569783b677f00a71f0caa45d8663d6113167` | Repository root has no declared license; no release distribution |
| Regression | [ca-joe-yang/regression-without-softarg](https://github.com/ca-joe-yang/regression-without-softarg), `b391d21aae976fdeecbc22285aae8702faecd00e` | Apache-2.0 declaration in official README |

TUFA's official weight SHA-256 is
`3fe7071bef814af320c6e6ec7d724b4b1bb253677500e1a48a5e2a66aaf10899`.
ORFormer's original official archive hash is
`00aa35f078512124a199a126df25279fa946803a41764e02e84843842e517ef6`;
the WFLW HGNet and ORFormer member hashes are in local identity records.
The [Regression author's Hugging Face weights](https://huggingface.co/Yeh-Purdue/regression-without-softarg)
were downloaded at revision `5f47d639d868cf54afcc6226091a5178165e68be`,
`WFLW/best_model.pkl`, SHA-256
`066d150329f57bdba780ae8915a7ec7c158436c8affaba6dfcd98603cab84b1e`.

Runtime: project Python, Torch 2.9.1+cu128, torchvision 0.24.1+cu128,
NumPy 2.2.6, OpenCV 4.12.0, timm 0.4.12, yacs 0.1.8, einops 0.8.1;
FP32, eval/inference mode, TF32 disabled, no flip TTA. Regression's official
input is **BGR** normalized to [-1,1]. Its official crop matrix is used with
scale derived from the shared box, not WFLW ground-truth crop metadata.
Its output point inverse uses the official dataset's `align_corners=False`:
`((u+1)*256-1)/2`, mapping -1/0/+1 to -0.5/127.5/255.5. Crop centering and
point normalization are separate conventions.
TUFA/ORFormer use RGB ImageNet normalization and their own official crops.

The first RGB Regression probe was rejected after preprocessing review. Its
original outputs are preserved in local `evaluation-20261006-rgb-preprocessing-audit`
with a rejection explanation. A subsequent BGR run still had the wrong
coordinate inverse; its full original evidence is preserved separately in
`evaluation-20261006-bgr-decoder-audit` with
`rejected_coordinate_decoder_audit`, distinct from the RGB audit.
Only Regression98 was rerun after the decoder correction, into a fresh
`evaluation-20261006-regression-decoder-corrected-v2` directory. The final
`evaluation-20261006-final` consolidates its twelve corrected predictions
with the 48 unchanged TUFA/ORFormer/FAN predictions; it records per-model
run provenance and implementation hashes, 60 overlays and recalculated
paired uncertainty. No rejected Regression result is relabelled valid.
Private images, annotations and predictions are not
included in the public repository.

Ten contract/metric tests passed with
`python -m unittest discover -s tools/tests -p test_vision_landmarks.py`:
affine roundtrip, official Regression pixel-center inverse, invalid input rejection, native topology/3D distinctions,
fixed source/weight/license identity, source-plus-manifest tampering,
real Windows junction escapes, geometric metric properties, exclusions and reproducible
paired bootstrap. Actual GPU predictions additionally verified finite native
point counts for all sixty results.
Benchmark output must be a fresh ignored evaluation subdirectory; existing
evidence and source-image directories are rejected before writing predictions.
