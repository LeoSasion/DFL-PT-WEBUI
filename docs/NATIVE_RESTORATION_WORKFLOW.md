# Native restoration copies

The workstation now has a local, non-training source-frame restoration task.
SwinIR-L x4 PSNR is the default based on the bounded restoration comparison;
Real-ESRGAN_x4plus is an explicit alternative. One task uses exactly one model.
GFPGANv1.4 requires separately qualified FFHQ alignment and is rejected by this
source-frame entry point, with an explanation instead of an incorrect resize
of an arbitrary frame.

## Source and output contract

Inputs are selected **top-level** PNG/JPEG frames in the active project's
`data_src` or `data_dst`. The visible batch is limited to 500 images; offset,
range and the content-based fingerprint are exposed to the UI. The submitted
selection must belong to that exact reviewed batch. A changed source, duplicate
name, PNG name collision or wrong stage prevents processing.

The Python task reads static RGB8/grayscale images from verified plain paths,
normalizes grayscale to RGB, and requires at least 32 pixels per side and at
most 16 million pixels/50 MiB. Embedded DFL JPEG APP15/PNG metadata is rejected
without deserializing pickle. High-bit-depth PNG, nontrivial orientation,
embedded ICC profiles, transparency and unsupported color modes require an
explicit preprocessing step. No aligned coordinates or masks are copied.

`RestorationModel.restore(image, output_size=(width,height))` performs frozen,
strictly hash-verified official model inference. Its existing default 512×512
benchmark behavior remains available. Source-frame tasks retain the original
width and height: native x4 SR is area-resized back to the source canvas.
Outputs are losslessly encoded RGB PNG and verified by decoding and comparing
pixels. Resolution is preserved; this is not a claim of new ground-truth detail.

Every selected source is checked again before publication. Outputs first stay
under `.webui/restoration/staging/<taskId>`. Only a fully successful batch is
renamed once into `.webui/restoration/outputs/<taskId>`, containing
`manifest.json` and `images/`. A failure, cancellation or interruption leaves
no task-visible partial result. Output folders are immutable; a new attempt
gets a new task ID. Original input files are never overwritten.

The manifest records `outputs[{name,sourceName,inputSha256,outputSha256,
width,height,mode,metadataCopied}]`, the selected model/pin and inference
policy. These source copies must be extracted again into a separate aligned
dataset. They cannot be imported directly into the existing aligned dataset.
If a source timeline is retained, its entries must be filtered and remapped
using `sourceName` to `name`; copying an entire timeline onto a selected subset
would be incorrect.

## Manager interface

`webui/server/restoration-manager.mjs` exports `RestorationManager` and
`RestorationError`. Default paths use the active project and project-local
Python/model cache. Initialization creates only the active workspace's
`.webui/restoration` directories. The root server supplies `assertCanStart`
and `onActiveChange` to prevent GPU/job conflicts, project switching and
restarts while active.

| Method | Contract |
| --- | --- |
| `initialize()` | Recover persisted history; mark previously running tasks interrupted |
| `listInputs({side,offset,limit})` | Source-frame inventory, visible range, SHA-based fingerprint and model stage eligibility |
| `createTask({side,names,fingerprint,offset,limit,modelId})` | Start one background task immediately; reject concurrent/busy operations |
| `getTask(taskId)` / `listTasks()` | Persisted task status, progress, outputs and diagnostics |
| `activeTasks()` | Synchronous task activity for server mutation/GPU locks |
| `cancelTask(taskId)` / `close()` | Cancel and wait for child exit before releasing activity |
| `resolveOutputDirectory(taskId)` | Return fixed images folder only after completed-manifest/output hash, count, color and dimensions validation |
| `resultImage(taskId,name)` | Verified PNG bytes plus path/name/MIME; HTTP handlers should send those verified bytes |

There is no hidden waiting queue or automatic restart. A task holds the local
lock from reviewed-source verification through final state persistence. The
worker does not download assets or train. Task history contains failures
instead of inventing a quality score.

## Verification

Six Python task tests and six manager tests passed: original-byte preservation,
rectangular dimensions, metadata rejection, selection limits/collisions,
source changes, mid-batch failure, invalid model output, immutable publication,
reload, output tampering, activity locking, cancellation and external busy
checks. The existing eleven restoration tests also passed.

Actual native manager → Python → official SwinIR inference completed a
two-image task in an isolated ignored workspace: a public 96×64 graphic and a
previously locked private synthetic down4 image of 128×128. Both source hashes
remained unchanged, both output canvases matched, and both PNGs passed the
complete publication checks. Private evidence is in
`workspace/.vision-evaluation/restoration/native-workbench-20261006/evidence.json`.
This proves task functionality and preservation; further visual judgment is
separate from this acceptance.

## Re-extraction publication and cancellation

`webui/server/extract-restored.mjs` consumes only a verified completed task's
fixed `images` directory and `manifest.json`. It runs extraction with debug
output disabled, preserving the original `aligned` dataset. New faces are
written to `data_SIDE/aligned_restored/.pending-TASKID-RANDOM` first.
An exit code of zero is insufficient: the batch must contain at least one
decoded DFL JPEG with finite legacy 68-point source/aligned geometry and
restoration provenance matching the original filename, original SHA and
processed PNG SHA/canvas. The default TUFA selection additionally requires
finite native 98 points, the original-to-aligned affine, a consistent aligned
canvas and a SHA-bound `.jpg.landmarks.json` that agrees with the DFL metadata.
Explicit FAN selection validates the legacy geometry and provenance without
claiming TUFA98 output. Every produced JPEG must pass before atomic rename to
`aligned_restored/TASKID`; no valid subset is published after a batch failure.
Failures retain the pending files and `extraction-failure.json` for diagnosis.
Existing published task directories are never overwritten.

Restored source-map inputs keep their complete canvas during extraction,
including odd widths/heights. Unmapped legacy extraction retains its existing
odd-pixel crop policy. `restoration-source.json` records the validated faces,
source/output hashes, counts and affine consistency. An original hash in the
DFL provenance is declared by the restoration manifest; the extractor does
not invent an independent original-hash verification claim.

The manager uses the shared process-tree terminator when cancellation,
timeout or response-size overflow occurs. On Windows this includes the venv
launcher and actual Python interpreter. Both child close and process-tree
confirmation must finish before the activity/GPU reservation is released.
Failure to confirm termination marks the task failed and retains the activity
reservation; `close()` reports the failure instead of claiming safe shutdown.

Actual private QA completed official SwinIR → source-map extraction with
YOLO26s-face and TUFA on the pinned official face sample
`519346b222f4b8b9be453d5866c5e235eec4f92540afd7fb0e1fc975e20be7e1`.
The 259×194 restored PNG kept its source canvas; one 512×512 DFL JPEG retained
legacy 68 points, native TUFA98 and its sidecar. Original frame and original
DFL aligned bytes remained unchanged. Affine mapping error was below
`1.2e-13` pixels. Evidence and the previously rejected odd-canvas pending
diagnostic are retained under the ignored private directory
`workspace/.vision-evaluation/restoration/restored-extraction-20261006`.
This proves the functional chain and provenance, not restoration quality or
keypoint accuracy. Five isolated extraction tests and eight manager tests
passed, including actual Windows venv-interpreter cancellation with delayed
termination confirmation.
