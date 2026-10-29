# Launcher interaction contract

This contract covers the launcher surfaces maintained in `src`. Visual intent and canonical tokens are recorded in the project-root `DESIGN.md`.

## Business sources

| Rule | Authoritative source | UI consequence |
|---|---|---|
| A changed environment should offer the matching CUDA package; after it is ready, the user chooses whether to remove the original | User instruction, 2026-10-29: “环境发生变化时，提示下载另一个cuda，下载完后提示让用户选择是否删除原cuda” | Detection is read-only. Download needs a click. Old-package deletion needs a separate decision after verified activation. |
| Pascal uses FP32; existing model precision is not silently changed | `AGENTS.md`, `docs/GPU_COMPATIBILITY.md` | A cu126 recommendation includes a short FP32 hint. Model and data preservation copy does not promise unchanged BF16 continuation. |
| Source/runtime maintenance requires idle tasks and failure recovery | `launcher/ui/AGENTS.md`, host maintenance guard and runtime-switch service | Disable unsafe operations while tasks run; host checks remain authoritative. Failures retain recovery information. |
| Deletion scope is the project's retained CUDA package | Runtime-switch API: transaction-bound `deleteCudaRuntime`, explicit `confirmed: true` | Name the old version and space, preserve current runtime/models/data, do not imply driver or system-CUDA removal. |

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Dialog | `src/components/Dialog.jsx` | This contract; `DESIGN.md` | Feedback, fixed-source upgrade, retained-runtime cleanup | Browser focus, Escape, inert background, async failure |
| Runtime lifecycle | `src/App.jsx` and `src/components/CudaRuntimeNotice.jsx` | Native runtime-switch state and explicit user actions | Change required, preparation, cleanup choice, retained, error, recovery, unavailable | Browser end-to-end state fixtures and host tests |
| Status | `CudaRuntimeNotice`, existing log stream | Native events and RPC confirmations | Warning, progress, success, actionable failure | Live-region and failure/retry browser checks |
| Scrollbar | `src/runtime.css` using `styles.css` tokens | `DESIGN.md` | Global theme; modal-body stable gutter | Computed-style and narrow-window checks |

The bundled deterministic auditor recognizes a fixed list of capability names including Scrollbar; Dialog, Runtime lifecycle and Status are additional project capabilities. Their complete ownership rows remain here and their executable verification is `tests/cuda-runtime.browser.mjs`, rather than being relabeled as unrelated built-in capabilities to satisfy that parser.

## Runtime workflow

The launcher reads `state.cudaRuntime`, listens for `cudaRuntime` patches and requests a lightweight check every 20 seconds while visible and idle, and again on focus. Assessments/events are scoped to their project directory. Every prepare/keep/delete/recover RPC carries the required `expectedProjectDir` captured from the displayed assessment or cleanup snapshot; the host checks it against its selected project. A background result cannot overwrite a newer runtime event, mutation or project selection, including A→B→A changes. Switching projects releases only the old check's UI slot, so its delayed completion cannot block or clear a newer project's check. Detection failure remains “无法检查”; it must not invent a required download.

`prepareCudaRuntime` runs only after the user selects the clearly named download/switch action. During preparation, mutations are deduplicated and buttons show pending state. Progress is determinate only when the host provides measurable byte counts. There is no automatic deletion.

Only `cleanup-choice` with a transaction ID triggers the old-package decision dialog. **保留旧版** is initially focused and persists the choice through `keepCudaRuntime`. **删除旧版** submits a separate `deleteCudaRuntime` call with that transaction ID and `confirmed: true`. Escape closes the dialog without deleting; the notice retains an entry to decide later. A failed deletion leaves the dialog open with retry and preservation choices.

`retained` does not reopen its dialog on each status refresh. It offers an explicit cleanup entry. Historical `retainedRuntimes` remain reachable through a project-owned list naming each version, size and state. Keep/delete uses the selected row's transaction ID and current capability. A row marked `deleting` cannot be kept or reused; its dialog initially focuses **稍后处理**, with a separate **继续删除旧版** action. The list never deletes a row directly. When `canRecover` is true, offer recovery instead of another download or cleanup. A successful recovery must be confirmed by the host before the UI claims it is ready.

Download estimates prefer `downloadRemainingBytes`, falling back to the legacy total only when that field is absent. A fully downloaded wheel cache is described as preparation awaiting validation; it is not described as an already retained runnable environment. Only an eligible retained row for the target yields the reuse label. Status and rendering never trigger a package download or hash a wheel.

When an already committed runtime has `cleanupWarning` and `canRetryCleanup`, show an explicit **重试清理** entry. It calls the same transaction-bound recovery RPC to clean that transaction's known download/staging/temporary remnants; it is not labeled as rolling back the usable runtime. The current environment stays usable. An interrupted, uncommitted switch continues to use **恢复原运行环境** instead. Neither path runs automatically.

On a cold start with `upgradePending`, the install screen offers **恢复未完成的升级** through `rollbackUpdate`, replaces ordinary bootstrap, and shows the host's `upgradeMessage`. When `upgradeCanRollback` is false, recovery and ordinary installation remain disabled rather than overwriting the incomplete upgrade. A separate `runtimeRepairPending` state uses **恢复运行环境** through `repairDependencies`; it never selects source rollback. `environmentStatus: changing` disables new environment actions throughout either surface. Failures remain visible at the recovery action and can be retried.

## Shared behavior

Modals make the background inert, trap Tab and Shift+Tab, provide a visible safe action, and restore focus to the trigger or the next useful runtime action. Escape never confirms a destructive action and does not dismiss a mutation in progress. Multiple modal workflows are not stacked.

All actions use real buttons, visible focus and explicit Chinese labels. Errors stay at the affected notice or dialog in addition to the shared log. Pending actions do not close before confirmation. The host's capabilities, transaction identity, active-task guard and path restrictions remain authoritative; a disabled button is not authorization.

The locale is `zh-CN`. Space is shown in decimal KB/MB/GB with a maximum of one decimal place; unknown space is labeled rather than replaced with a fake zero. The narrow layout preserves reachable actions and modal body scrolling. The browser preview uses synthetic states only and cannot invoke a real runtime switch.

## Isolated verification

`node --test launcher/ui/tests/cuda-runtime.test.mjs` checks capability boundaries and truthful normalization. The browser command is `node launcher/ui/tests/cuda-runtime.browser.mjs`; it requires `DFL_CUDA_UI_QA_ROOT` to name an absolute disposable evidence folder and an already installed Playwright package (normal module resolution or `NODE_PATH`). `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` can select an existing Chrome binary. No browser or global dependency installation is required. The browser runner serves only 127.0.0.1, uses preview/fake-native state, writes its Vite cache/build/screenshots/JSON under that evidence folder, and rejects page or browser-console errors. It never connects to a real native runtime host. On workstations affected by the local Node clock issue, run the local Node process with CPU affinity 1.
