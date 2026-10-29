# DFL-PT-WEBUI

This is a new standalone repository. The previous DFL-WEBUI and ME prototype
directories are reference sources only; all implementation belongs here.

- PyTorch is the only neural-network runtime. Do not introduce TensorFlow,
  old CPython 3.7 runtimes, old project update remotes or absolute source paths.
- ME and TAEQI2.1 are parallel face-swap training architectures. On 2026-10-10
  the user replaced H3 with TAEQI2.1 and explicitly removed the H3 trainer.
  TAEQI2.1 always freezes its encoder; decoder scope is frozen (default), tail,
  or full. Independent LIAE InterAB/B and a face-mask head use tae.pt.
  Reuse native DFL data and WebUI workflows; never load TAE checkpoints as ME.
  XSeg training remains an
  auxiliary mask tool; extraction, dataset tools, diagnostics, merging and
  encoding belong to the supported pipeline.
- Preserve the existing local workstation UI and its recoverable data workflows.
- Backend source is `_internal/DeepFaceLab`; the directory name is retained as
  the internal tool API compatibility path, and does not select a TF backend.
- Python is `.venv/Scripts/python.exe`, based on the local
  `_internal/python_base`; Node and FFmpeg are project-local runtimes.
- Preserve GTX 10 / Pascal support through the official PyTorch 2.9.1 cu126
  variant; keep cu128 for Blackwell. One workstation uses one selected venv,
  with variants resolved by `core/gpu_runtime.py` and pinned in
  `release/torch-runtimes.json`. Never replace CUDA/cuDNN DLLs independently.
  Pascal uses FP32; BF16 training requires native support, excluding emulation.
  Official build/kernel evidence and simulated tests do not prove real Pascal
  training acceptance when no GTX 10 hardware is available.
  Explicit launcher runtime changes may stage and validate another official
  Torch variant, switch only its owned packages, and retain the prior variant
  for rollback. Deleting a retained variant requires a separate user choice
  after successful activation; never silently replace or delete it.
- User media and model checkpoints stay in ignored `workspace`/`workspaces`.
- Public label: TAEQI2.1; internal model family: TAE. Resolution presets are 256, 448, 512, 640, 832, with resolution-bound Inter grids (resolution/64), not H3 tiles. Decoder scope, LR multiplier or FP32/BF16 precision changes on stopped resume retain weights/iterations/data and explicitly reset AdamW/warmup for a recorded new stage. No H3 checkpoint migration or active trainer compatibility.
- On 2026-10-10 the user requested FP32/BF16 training independently of DFM export precision. ME and TAE both expose normal FP32 export and extreme export (FP8 eligible convolutions, NVFP4 aligned Inter linears, FP16 sensitive boundaries/masks, FP32 normalization reductions). Extreme export calibrates real aligned DST faces and must build and run a TensorRT engine on Blackwell before publication; never label a fake-quant-only graph as accelerated. Preserve training checkpoints. Large DFM graphs stream external weights with relative paths, without a 2GB model limit; consumers must support external data. Keep the receipt, sidecars and hardware-specific engine together. See docs/DFM_EXPORT.md.
- On 2026-10-14 the user added native 512-resolution TAE pretraining using RTM2 (facesetR2.pak), Inter 384, hidden 96, batch 4, frozen Encoder and Decoder. The supported presets now also include 512. Pretraining uses one common pool with independent samplers and a recorded source-group holdout; normal identity training still rejects shared SRC/DST files.
- Reach for CodeGraph before code search only when `.codegraph` exists.
- Ship the verified generic XSeg inference weights in both source and portable
  release archives, with pinned hashes and original source/license records.
  Never substitute random or locally trained QA checkpoints.
- Prefer native WebUI aligned viewing and similarity review over installing
  XnViewMP or VisiPics. Keep the per-pass 500-image limit and selected batch
  ranges visible; paired comparisons use at most 250 images per batch. EbSynth
  temporal keyframe propagation is a future native capability candidate;
  do not add an external installer flow for these tools.
- Validate actual training, saving, stopping, resuming and prediction when
  changing the ME bridge, using the bounded development scope below.
  Backend/frontend unit tests alone do not prove integration.

## Current ME development acceptance scope (2026-10-04)

- Long training is prohibited in the current development scope. Use an isolated
  small model with small parameters and keep the entire training acceptance,
  including save/stop/resume, within a one-hour wall-clock budget. Set an
  automatic stop and leave time for checkpoint saving and process shutdown;
  never extend or restart training to meet an old duration or quality target.
- Accept development training on its functional evidence and a finite,
  observable convergence trend. Record the configuration, timing, loss trend,
  checkpoint/save/stop/resume and prediction evidence honestly. Final visual
  quality, identity fidelity and formal demonstration cases are later work;
  they are not development or release blockers.
- Do not run dual-GPU acceptance or seek another machine for it at this stage.
  Retain the preliminary dual-GPU implementation and its unverified status,
  and wait for open-source user feedback before taking up further validation.
  Real dual-GPU results are not a development or release prerequisite.
- Use `docs/ME_SHORT_RUN_ACCEPTANCE.md` for the current bounded acceptance.
  The old 12-hour audits and real-media/visual reports remain optional
  historical evidence. Preserve their original failures and unverified
  claims; do not lower the old audit's thresholds or relabel an old failed run
  as passed. They do not impose a current 12-hour, visual-quality or dual-GPU
  gate. A later formal case requires a new explicit scope from the user.
