---
version: alpha
name: DFL-PT-WEBUI launcher
description: A Chinese desktop workstation shell for local face-swap training and recoverable runtime maintenance.
colors:
  background: "#030706"
  surface: "#06110c"
  action: "#08130f"
  primary: "#2ce39f"
  focus: "#54f4bb"
  success: "#6bdcb0"
  warning: "#f3b83f"
  danger: "#ff5a46"
  text: "#eaf4ee"
  secondaryText: "#c0ccc5"
  mutedText: "#829087"
  border: "#163326"
typography:
  sans:
    fontFamily: '"Microsoft YaHei UI", "Noto Sans SC", system-ui, sans-serif'
  mono:
    fontFamily: '"Cascadia Code", Consolas, monospace'
rounded:
  control: "5px"
  panel: "6px"
  dialog: "9px"
spacing:
  shellGap: "8px"
  noticePadding: "15px"
  dialogPadding: "24px"
components:
  runtimeNotice:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.secondaryText}"
    rounded: "{rounded.panel}"
    padding: "{spacing.noticePadding}"
  primaryButton:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.background}"
  primaryButtonHover:
    backgroundColor: "{colors.focus}"
  destructiveButton:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.background}"
  defaultButton:
    backgroundColor: "{colors.action}"
    textColor: "{colors.secondaryText}"
  noticeTitle:
    textColor: "{colors.text}"
  warningIcon:
    textColor: "{colors.warning}"
  successIcon:
    textColor: "{colors.success}"
  secondaryHint:
    textColor: "{colors.mutedText}"
  separator:
    backgroundColor: "{colors.border}"
---

# DFL-PT-WEBUI launcher design context

## Overview

This context documents the existing Windows launcher, not a redesign of the separate WebUI. Its reference is a local engineering workstation: a forest-black terminal, emerald status and controls, compact panels, and an official brand mark. Training operators need to see what is running and make deliberate maintenance decisions without losing their work.

The register is product utility. Chinese is the maintained UI language. The primary usage scene is a Windows desktop; narrower preview windows must keep maintenance actions reachable. No Japan-market scope is inferred from the locale.

The visual signature is the restrained emerald workstation shell and terminal. Runtime-change notices reuse it with an amber left border; successful activation uses emerald. There is no new dashboard, marketing surface, or decorative hero.

Token ownership is **Model B**: `launcher/ui/src/styles.css :root` remains canonical. This file mirrors accepted values and rationale; it does not generate CSS. `ready.css` owns existing ready-page geometry. `runtime.css` consumes the canonical tokens for notices, buttons and the shared dialog. Semantic token changes must update this document and `styles.css` together.

## Colors

The background and surfaces distinguish shell, terminal and modal through small tonal changes and borders. Emerald marks safe actions and verified completion, amber marks a decision requiring attention, and red is reserved for errors and the final permanent-delete action. Text, icons and explicit labels carry state in addition to color. The product has one dark theme; forced colors retain system-operable borders and scrollbars.

## Typography

Microsoft YaHei UI and Noto Sans SC preserve readable Chinese labels without a web-font request. Cascadia Code and Consolas display versions and space figures. The existing shell uses compact 11–13 px utility text; notices use 12 px body text with a 1.65 line height. Dialog titles are 18 px on desktop and 16 px in narrow windows. Long device names and errors wrap rather than hiding the action or truncating the consequence.

## Layout

The shell retains its 46 px native-style title bar and compact footer. The workspace is one flexible content region. An environment notice sits above the existing terminal and action sidebar. It is not a modal download prompt and never starts a download on appearance.

At desktop widths the notice has icon, message and action columns. Below 980 px the actions wrap beneath the message. Below 760 px the ready view becomes one column and the content region owns scrolling; the modal remains inside the visual viewport with its actions outside the body scroller.

## Elevation & Depth

Static panels use borders and tonal surfaces. Modal surfaces alone receive a full-screen dim backdrop. Shared layer tokens are `--z-backdrop: 500` and `--z-dialog: 600`; feature code must not invent larger values. The existing version-details disclosure remains below modal layers.

## Shapes

Controls have 5 px corners, panels 6 px and dialogs 9 px. The new workflow retains this geometry. Notice emphasis is a narrow semantic left border, not an oversized rounded card.

## Components

The canonical interaction owners are documented in `launcher/ui/UX-CONTRACT.md`.

| Design value | Runtime token or owner | Consumers |
|---|---|---|
| `colors.background` | `--canvas` | Shell, safe primary-button text |
| `colors.surface` | `--surface-dialog` | Notice and Dialog |
| `colors.primary`, `colors.focus` | `--emerald`, `--emerald-bright` | Primary actions, focus ring |
| `colors.warning`, `colors.danger` | `--amber`, `--red` | Runtime decision, inline failure, confirmed delete |
| `colors.text`, `colors.secondaryText`, `colors.mutedText` | `--text`, `--text-2`, `--text-3` | Headings, copy, secondary facts |
| `typography.mono` | `--mono` | Versions and sizes |
| Scrollbar colors | `--scrollbar-thumb`, `--scrollbar-track`, `--scrollbar-hover` | All owned scroll containers |

`Dialog.jsx` owns modal naming, inert background, initial focus, Tab containment, Escape and focus restoration for feedback, application upgrades and runtime cleanup. Destructive cleanup initially focuses **保留旧版**. An asynchronous delete stays in the dialog until the host confirms success; failure stays visible there with a retry.

Runtime buttons have stable minimum sizes, explicit hover/pressed/focus/disabled treatments, and an indeterminate spinner when no trustworthy total exists. Determinate progress is shown only from measured downloaded bytes. Tabler outline icons remain the launcher icon family. Motion signals work rather than decoration, and reduced motion uses the existing global preference.

## Do's and Don'ts

- Keep one active Python environment and explain which CUDA runtime is being enabled or retained.
- Name the operation and its exact scope in plain Chinese. Preserving models does not imply automatic training-precision migration.
- Keep downloading, activation verification and old-package deletion separate in the interaction.
- Do not call native browser confirmation dialogs or delete a retained package merely because a download succeeded.
- Do not rebrand the launcher or impose this compact desktop visual system on the independent WebUI.
