# Identity descriptors and SFace PyTorch migration

Evaluation date: 2026-10-06. H3CE has no identity-descriptor model in this
comparison. The existing DFL SFace model is the compatibility baseline;
AdaFace R100/WebFace12M and MagFace iResNet100/MS1MV2 are optional candidates.
No candidate is promoted by this experiment.

## Runtime and interface

`webui/python/vision_identity.py` exposes
`IdentityPredictor(model_id, assets_root, device).predict(aligned_rgb)`.
Input is one shared, already aligned **112x112 RGB uint8** crop. Output records
the normalized descriptor, dimension, preprocessing, model/source/weight
identity, PyTorch runtime, FP32 precision and disabled test-time augmentation.
SFace produces 128 dimensions; the two candidates produce 512. Dimensions
and thresholds must not be mixed between descriptor families.
Stable IDs match the public candidate catalog: `sface`,
`adaface-ir100-webface12m`, `magface-iresnet100`. The shorter `adaface` and
`magface` names remain accepted aliases; prediction records use stable IDs.

All neural execution uses PyTorch 2.9.1+cu128. The official candidate Python
implementations and weights are loaded from ignored
`workspace/.vision-models/identity/{AdaFace,MagFace}` after hash verification.
Absolute paths, parent escapes and symlink escapes are rejected with the
shared asset-path resolver. The exact family, official source/revision,
checkpoint path/hash and required executable source path are fixed by the
adapter; executable source must appear in the verified manifest.
The complete source-file manifest and license declaration also have reviewed
canonical fingerprints pinned independently in the adapter. Updating both
an executable and its declared hash still fails this fixed source contract.
The two official local source trees and checkpoints were checked again
after this tightening, without changing or repeating the final predictions.
`identity.json` records the official URL, source revision, license, hashes of
source files and author alignment weights, and checkpoint SHA256. Candidate
checkpoints use `torch.load(..., weights_only=True)` and strict state loading.
No H3CE application source or virtual environment is used.

The product SFace adapter parses the existing pinned ONNX file with
`onnx==1.20.0` **only to read weights and graph structure**. Its 88 nodes
(Conv, BatchNormalization, PRelu, Sub, Mul, Dropout, Flatten, Gemm) run as
PyTorch operators. The exact published weight hash is required before
parsing; unknown graph operations/interfaces fail. There is no fallback
runtime or network download. OpenCV DNN runs only in the benchmark/tests as
the numerical reference for the migration.

`webui/python/role_grouping.py` now calls this CPU PyTorch adapter. Alignment,
model digest, thresholds, complete-linkage clustering, co-occurrence
constraints, group IDs, representative selection, rounded `score`, progress
events and the `sface-2021dec-complete-linkage` method remain the existing
contract. Results remain anonymous review candidates.

## Assets

| Model | Official source revision | Weight SHA256 | Source license |
| --- | --- | --- | --- |
| SFace 2021dec | Existing OpenCV Zoo asset | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |
| AdaFace R100/WebFace12M | `c60eaa786a42c03444f3df7096dbaf9d57ae010d` | `0e7a3238d2a50f3fe3860782534928ac7cb2598977cf897f6869fd5ac2493fd0` | MIT |
| MagFace iResNet100/MS1MV2 | `99bae614ac2643b9694bf18e0c5645272ff6acfa` | `cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58` | Apache-2.0 |

Official sources: [AdaFace](https://github.com/mk-minchul/AdaFace),
[MagFace](https://github.com/IrvingMeng/MagFace), and
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
AdaFace's R100 model is named `adaface_ir101_webface12m.ckpt`; the author's
`ir_101` factory builds the depth-100 backbone. Its input is **BGR [-1,1]**.
MagFace uses the author's `gen_feat.py` inference convention, **BGR [0,1]**
(ToTensor and zero mean/unit standard deviation). SFace takes RGB raw
0..255, with `(x-127.5)/128` inside the graph. The adapters keep these
different input conventions explicit.

## Numerical migration evidence

The same three official example crops plus zero, white and deterministic
random crops were fed to CPU OpenCV DNN and CPU PyTorch. All six pass:

| Measurement | Observed maximum / minimum | Required |
| --- | --- | --- |
| Raw descriptor absolute difference | 0.000016533 | <= 0.0001 |
| Normalized descriptor absolute difference | 0.0000071414 | <= 0.00001 |
| Descriptor cosine | >= 0.99999994 | >= 0.999999 |
| Pairwise cosine absolute difference | 0.0000022203 | <= 0.00001 |

Tests additionally check eight other deterministic random crops plus the
zero/white cases, production role output semantics, invalid descriptors,
shape/dtype rejection, asset tampering and exclusion of copied/unlabelled
observations. Twelve identity tests passed, including source-path escapes,
real Windows junction escapes with matching file hashes, source-plus-manifest
tampering and missing executable registration. OpenCV reference version is
4.12.0. The numerical match is bounded to tested FP32 CPU execution; it does
not claim bitwise equality or guarantee that a threshold exactly on a score
boundary cannot change from floating-point rounding.

## Small quality observation

All three models actually ran on CUDA without error. The official AdaFace
repository's three example images were aligned once using its bundled
PyTorch MTCNN five-point similarity transform. Every model used that exact
same crop; original hashes, crop hashes, five points, affine and alignment
asset identity are retained in the ignored evaluation manifest.

The author's README gives a similarity matrix but **no identity class-label
file**. The apparent first-two-same/third-different relation is recorded as
`label_status: provisional`, not independently verified identity ground
truth. There are zero independently verified identity samples here. This
table reports observations on the three original images, with the
provisional relationship shown for context:

| Model | img1 vs img2 (provisional same) | img1 vs img3 (provisional different) | img2 vs img3 (provisional different) |
| --- | ---: | ---: | ---: |
| SFace baseline | 0.807374 | 0.066678 | 0.125053 |
| AdaFace R100 | 0.732830 | -0.079463 | -0.008661 |
| MagFace iResNet100 | 0.793065 | -0.031647 | 0.059469 |

No FAR, TAR, accuracy percentage, calibrated threshold, ranking or winner
can be inferred. The JSON leaves FAR/TAR/ranking null and retains the need
for more independent images with consented, verified identity labels.

The following are mean cosines to **the same original image** after controlled
degradation of each of the three crops. They describe descriptor robustness;
they are not recognition rates on different photographs:

| Model | Gaussian blur sigma 1.5 | Downsample to 28 then resize to 112 | JPEG quality 35 |
| --- | ---: | ---: | ---: |
| SFace | 0.931487 | 0.792957 | 0.938362 |
| AdaFace | 0.947459 | 0.834323 | 0.930950 |
| MagFace | 0.916544 | 0.695469 | 0.955714 |

## Reproduce locally

Assets are optional local evaluation inputs, not bundled candidate defaults.
After acquiring the exact official sources/checkpoints and recording their
hashes in the documented local layout:

```powershell
.venv/Scripts/python.exe tools/vision-identity-benchmark.py --official-samples --output workspace/.vision-models/identity/evaluation-20261006 --device cuda
.venv/Scripts/python.exe -m unittest discover -s tools/tests -p test_vision_identity.py -v
```

For independently verified user-supplied evaluation data, `--manifest`
accepts `samples` containing `id`, read-only original `image`/`sha256`,
`aligned_image`/`aligned_sha256`, optional `identity_label`, and
`label_status: verified` plus the label evidence. Absent/provisional labels
stay provisional; duplicate original hashes do not multiply pair counts.
Unlabelled samples produce embeddings but no labelled-pair observations.
The benchmark does not choose a product model or change role thresholds.
Outputs must use a fresh ignored evaluation subdirectory; previous evidence
cannot be overwritten. The final path-contract run is retained locally as
`workspace/.vision-models/identity/evaluation-20261006-final`.
