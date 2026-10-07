# Native landmark extraction (2026-10-06)

`main.py extract --landmark-model tufa` uses the verified official TUFA network
with its native IBUG68 prompt for ordinary 2D extraction, and independently
predicts its native WFLW98 prompt for eye/mouth inspection. The network is shared
between the two prompts. No 98-to-68 conversion is performed. The legacy DFL
68-point alignment code, face padding, rotation mapping and training contract
remain unchanged. TUFA uses its official RGB crop/normalization without FAN's
optional second detection/refinement pass.

HEAD continues to use the existing **3DFAN weights and 68 XY output** for the
existing HEAD geometry. TUFA98 is auxiliary inspection data there. The FAN
adapter does not return depth; no new model or metadata claims to generate 3D
coordinates. Manual landmark editing also retains the existing FAN geometry,
with TUFA98 inspection generated once during the final export stage.

The CLI defaults to `fan` to preserve existing scripts. The WebUI selects its
own explicit new-extraction default. Existing aligned images are not migrated,
rewritten or relabelled. Native98 data does not automatically produce XSeg
masks, labels or training targets.

## Preflight

```
.venv/Scripts/python.exe _internal/DeepFaceLab/facelib/LandmarkCandidates.py --verify-assets --model tufa --face-type head
```

This emits one JSON object on stdout, with `ok`, `model`, `geometryProtocol`
and verified `assets`, or `ok:false,error` and exit status 1. TUFA source
repository/revision, full reviewed source/license fingerprint, weight size/hash
and required execution/prompt files are pinned by `vision_landmarks.py`.
Dependency imports are checked. HEAD/manual FAN weights are checked against
the pinned release hashes and the exact existing constructor search order.
There is no automatic download or fallback. Extractor repeats the checks before
creating or clearing its output directory. Missing resources preserve aligned
data. Pending work following a failed TUFA worker makes the extraction fail.

## Metadata contract

Every new extraction retains the existing `landmarks`, `source_landmarks`
68x2 fields and `image_to_face_mat`. The DFL JPEG dictionary additionally has:

- `landmark_provenance`: schemaVersion, alignmentModel (`tufa68`, `fan68` or
  `fan3d`), actual asset identity, `policy:dfl-native-68-v1`, legacy68 definition,
  manual flag, depthAvailable=false, HEAD policy, processed source SHA256 and
  processed canvas WH.
- `native_landmarks98` in TUFA mode: the original prediction record with native
  point_definition (WFLW98), all 98 `points_original`, model/source affine
  matrices, source box, source image size, asset/prompt identity and preprocessing.
  It adds all 98 `points_aligned`, `source_to_aligned_affine`,
  `aligned_canvas_wh` and an explicit inspection-only usage statement.
- `restoration_provenance` when a source map is provided: processed/source
  filenames, processed SHA256, declared original SHA256, originalHashVerified=false,
  coordinate policy and supplied restoration identity fields.

TUFA exports a `<aligned>.jpg.landmarks.json` sidecar containing the same three
namespaces, source/aligned filenames and `aligned_sha256` calculated after DFL
metadata save. Consumers must check that hash/canvas before using cached aligned
points. Later resizing or editing can invalidate the sidecar. Original source
points remain distinct from aligned coordinates.

## Restored PNG source naming

`--source-map <json>` accepts an `outputs` array with basename-only `name`
(processed PNG) and `sourceName` (original frame filename), optional
`inputSha256`, `outputSha256`, `model`, `modelId`, `assetIdentity`, `inputSize`
and `outputSize`. Every processed input must be registered. Paths, traversal,
escaped symlinks/junctions, duplicate names, wrong output hashes and declared
resizing are rejected before output mutation. The processed hash is calculated
locally; the original hash is explicitly a declaration because extraction
does not reopen the original media. Restore operations must preserve the
original canvas. DFL `source_filename` becomes `sourceName` so merging can
continue locating the original frame; the actual processed PNG remains recorded.

## Real acceptance

The bounded CLI acceptance is reproducible with
`tools/vision-production-landmark-smoke.py --input-manifest <private-json>
--output <fresh-ignored-directory>`. It requires one to three explicitly selected
hash-locked samples and executes TUFA whole_face, FAN HEAD, TUFA HEAD and one
PNG source-map fixture, at most ten CLI image inputs. It reads every resulting
DFL JPEG and verifies both coordinate topologies, original/aligned transforms,
existing float32 geometry regeneration and sidecar hashes. HEAD source68 and
affine matrices must match the separate original FAN HEAD extraction exactly.
It rechecks original source hashes and writes inspection overlays, without
claiming visual-quality scores.

Actual evidence: ignored
`workspace/.vision-evaluation/production-landmarks-20261006-v2/acceptance.json`.
Three distinct images (one official AdaFace example and the first/last locked
H3CE difficult samples) produced nine complete aligned exports plus one
source-map export. All coordinate/metadata checks passed. All three TUFA HEAD
outputs have exactly equal source68/matrices to the original FAN HEAD outputs.
The source-map PNG is a lossless container-copy QA fixture, not a restoration
network result. Original H3CE media and evidence were read-only. This acceptance
did not train any model or change the ME bridge. Independent visual review is
handled separately by the coordinating agent.

The earlier `production-landmarks-20261006-v1` attempt retains its failed
acceptance: its script incorrectly reconstructed the old float32 estimator
using float64 lists, producing tiny matrix differences. The prediction code
was unchanged; v2 restores the actual estimator dtype and passes. Old evidence
is not overwritten or relabelled.

Eight new focused tests cover asset failure before output mutation, native68
shape/color enforcement, HEAD FAN3D retention, original-canvas native98 audit,
DFL/sidecar coordinate/hash roundtrip, source-map paths/hashes/size declarations,
FAN pin rejection and failed-worker completion. The existing ten landmark
evaluation and nine detector extraction tests also pass (27 total).
