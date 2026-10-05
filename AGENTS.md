# DFL-PT-WEBUI

This is a new standalone repository. The previous DFL-WEBUI and ME prototype
directories are reference sources only; all implementation belongs here.

- PyTorch is the only neural-network runtime. Do not introduce TensorFlow,
  old CPython 3.7 runtimes, old project update remotes or absolute source paths.
- ME is the only face-swap training architecture. XSeg training remains an
  auxiliary mask tool; extraction, dataset tools, diagnostics, merging and
  encoding belong to the supported pipeline.
- Preserve the existing local workstation UI and its recoverable data workflows.
- Backend source is `_internal/DeepFaceLab`; the directory name is retained as
  the internal tool API compatibility path, and does not select a TF backend.
- Python is `.venv/Scripts/python.exe`, based on the local
  `_internal/python_base`; Node and FFmpeg are project-local runtimes.
- User media and model checkpoints stay in ignored `workspace`/`workspaces`.
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
