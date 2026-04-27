---
title: "fix: Align packaged backend CLI contract"
type: fix
status: active
date: 2026-04-27
origin: docs/brainstorms/2026-04-19-multi-anchor-and-global-trajectory-smoothing-requirements.md
---

# fix: Align packaged backend CLI contract

## Overview

The packaged Electron app is spawning a stale bundled backend binary. The Electron main process now passes the two-anchor contract (`--anchor1-x`, `--anchor1-y`, `--anchor2-x`, `--anchor2-y`), and `src/stabilizer_cli.py` already accepts that contract, but the local `dist-py/stabilizer_arm64` binary still exposes the older single-anchor arguments. This plan fixes the release/build path so packaged apps bundle a backend built from current source and adds CLI contract checks so the mismatch is caught before another DMG ships.

---

## Problem Frame

The user-selected test run fails immediately with argparse error code 2:

- Electron passes `--anchor1-x/--anchor1-y/--anchor2-x/--anchor2-y`.
- The bundled `stabilizer_arm64` binary reports only `--anchor-x/--anchor-y`.
- Progress remains at 0%, so no frames are processed.

The source tree is internally consistent, but the packaged artifact is not. Because the production app runs `process.resourcesPath/stabilizer_arm64` or `stabilizer_x64`, rebuilding only Electron without rebuilding `dist-py` can silently ship old Python CLI code.

---

## Requirements Trace

- R1. Packaged Apple Silicon app accepts the two-anchor CLI flags emitted by `electron/main.js`.
- R2. Packaged Intel binary build path uses the same current `src/stabilizer_cli.py` source and dependency set.
- R3. Local build workflow refreshes the bundled Python backend before Electron packaging.
- R4. CI PyInstaller workflow installs all runtime dependencies required by the current backend, including `scipy`.
- R5. Automated tests or build guards fail when the backend CLI contract drops the required two-anchor flags or reintroduces the old single-anchor contract.
- R6. The immediate local app build is regenerated and reopened so the user can retry `EXPORT TEST FRAMES/`.
- R7. Generated PyInstaller binaries remain untracked source-control artifacts so stale local binaries are not accidentally committed.

**Origin flows:** two-anchor preview-to-batch flow from `docs/brainstorms/2026-04-19-multi-anchor-and-global-trajectory-smoothing-requirements.md`.
**Origin acceptance examples:** CLI contract replacement from the origin scope boundary and deferred planning question.

---

## Scope Boundaries

- Do not change the two-anchor product behavior or fallback semantics.
- Do not restore compatibility with `--anchor-x/--anchor-y`; the origin document explicitly made this a breaking contract change.
- Do not redesign the Electron UI.
- Do not tune stabilization quality, smoothing, splice detection, or anchor tracking in this fix.
- Do not commit generated DMGs or local `dist/` output.

---

## Context & Research

### Relevant Code and Patterns

- `electron/main.js` builds packaged and dev-mode spawn arguments with the four two-anchor flags.
- `src/stabilizer_cli.py` argparse already defines the four two-anchor flags and validates all four are present in batch mode.
- `electron/package.json` bundles `../dist-py/stabilizer_arm64` and `../dist-py/stabilizer_x64` as Electron extra resources.
- `.github/workflows/build.yml` builds both PyInstaller binaries before Electron packaging, but its direct `pip install` commands omit `scipy`, which current source imports through `src/trajectory_smoothing.py`.
- `scripts/build-local.sh` currently runs only `npm install` and `npm run build`, so it can package stale `dist-py` binaries.
- `dist-py/stabilizer_arm64` is dated before the two-anchor CLI changes and is not a reliable source artifact.

### Institutional Learnings

- `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md` calls out numeric and contract values crossing Electron-to-Python process boundaries as a class of silent failures that need explicit checks.
- Prior multi-anchor plan notes that every backend/Electron change requires a full DMG rebuild and opening the packaged app for verification.

### External References

- None. This is a local packaging and contract consistency issue.

---

## Key Technical Decisions

- Build the bundled Python CLI from `src/stabilizer_cli.py`, not the Tkinter app entry point. Electron depends on JSON-lines CLI behavior, not the standalone desktop script.
- Keep the two-anchor contract strict. Accepting both old and new flags would hide stale callers and contradict the origin scope boundary.
- Treat `scipy` as a packaged runtime dependency. The current backend imports trajectory smoothing helpers that require it, so both local and CI PyInstaller paths must install it.
- Add lightweight contract tests around `stabilizer_cli.main()` rather than trying to execute generated PyInstaller binaries in unit tests.
- Add a local build guard that checks the generated binary help text for `--anchor1-x` and absence of `--anchor-x` before Electron packaging.

---

## Open Questions

### Resolved During Planning

- **Root cause category:** stale bundled backend binary, not a renderer coordinate issue.
- **Compatibility direction:** new two-anchor flags remain required; old flags stay unsupported.
- **Local verification target:** Apple Silicon packaged app is enough for immediate testing on this machine.

### Deferred to Implementation

- Whether local x64 PyInstaller rebuild is possible on this machine depends on available Rosetta Python. If not available, CI still remains authoritative for x64; the local script should fail clearly or document the limitation rather than silently pretending x64 was rebuilt.

---

## Implementation Units

- U1. **Add CLI contract regression coverage**

**Goal:** Make source-level tests fail if `stabilizer_cli.py` no longer accepts the four two-anchor arguments or accidentally accepts old single-anchor flags.

**Requirements:** R1, R5

**Dependencies:** None

**Files:**
- Create: `tests/test_cli_contract.py`
- Modify: `tests/test_detection.py` only if local test organization favors extending existing CLI tests instead of a new file

**Approach:**
- Test `stabilizer_cli.main()` with a monkeypatched `stabilize_folder` and `sys.argv` containing all four two-anchor flags; assert it passes `anchor1` and `anchor2` tuples through correctly.
- Test missing one required anchor flag exits non-zero with the existing JSON error path.
- Test old `--anchor-x/--anchor-y` flags are rejected by argparse.

**Patterns to follow:**
- Existing CLI monkeypatch and capture style in `tests/test_detection.py`.

**Test scenarios:**
- Happy path: `--anchor1-x 1 --anchor1-y 2 --anchor2-x 3 --anchor2-y 4` calls `stabilize_folder(anchor1=(1,2), anchor2=(3,4))` and emits a `done` JSON line.
- Error path: omit `--anchor2-y` and confirm batch mode exits with code 1 and names the required four-anchor contract.
- Error path: pass `--anchor-x/--anchor-y` and confirm argparse exits with code 2.

**Verification:**
- Contract tests fail on the stale single-anchor source shape and pass on the current source shape.

---

- U2. **Refresh local packaging pipeline**

**Goal:** Ensure local packaged apps bundle a freshly built Python backend before Electron packaging.

**Requirements:** R1, R3, R4, R6

**Dependencies:** U1

**Files:**
- Modify: `scripts/build-local.sh`
- Modify: `.gitignore`

**Approach:**
- Build `dist-py/stabilizer_arm64` from `src/stabilizer_cli.py` before running `npm run build`.
- Install the packaged backend dependencies needed by current source: PyInstaller, OpenCV headless, NumPy, and SciPy.
- Add a build-time help-text guard on the generated host-architecture binary: required two-anchor flags must be present and the old `--anchor-x` flag must be absent.
- Keep logs in `/tmp/perfstab-build.log` as the script already does.
- Handle x64 explicitly: either rebuild when the expected x64 Python/Rosetta path exists, or leave CI as the authoritative x64 path with a clear log note. Do not silently claim both architectures were refreshed if only arm64 was rebuilt.
- Add `dist-py/` to `.gitignore`; it is a generated PyInstaller output directory used as an Electron packaging input, not source.

**Patterns to follow:**
- Existing CI PyInstaller command shape in `.github/workflows/build.yml`.
- Existing local script logging convention in `scripts/build-local.sh`.

**Test scenarios:**
- Happy path: running the local build script regenerates `dist-py/stabilizer_arm64` and Electron packages it into `dist/mac-arm64/Perforation Stabilizer.app`.
- Error path: generated `stabilizer_arm64 --help` lacks `--anchor1-x`; script fails before Electron packaging.
- Error path: dependency install/build fails; script stops with a non-zero exit and leaves the log path for diagnosis.

**Verification:**
- `dist-py/stabilizer_arm64 --help` shows `--anchor1-x` and does not show `--anchor-x`.
- Rebuilt packaged Apple Silicon app starts and no longer rejects the two-anchor flags.

---

- U3. **Fix CI binary dependency parity**

**Goal:** Prevent release builds from failing or producing incomplete PyInstaller binaries after the backend started importing SciPy.

**Requirements:** R2, R4

**Dependencies:** U1

**Files:**
- Modify: `.github/workflows/build.yml`

**Approach:**
- Add `scipy` to both arm64 and x64 PyInstaller dependency installation commands, or switch those commands to install the project dependency set from `pyproject.toml` if that remains simple.
- Keep the existing two-stage artifact flow: `python-binary` uploads `dist-py/*`, and `build` downloads those binaries before `npm run build`.

**Patterns to follow:**
- Existing PyInstaller onefile build commands in `.github/workflows/build.yml`.

**Test scenarios:**
- CI arm64 binary build has all imports required by `src/stabilizer_cli.py`.
- CI x64 binary build has the same dependency set as arm64.

**Verification:**
- A workflow run reaches Electron packaging with both `dist-py/stabilizer_arm64` and `dist-py/stabilizer_x64` executable.

---

- U4. **Rebuild and smoke-test the local app**

**Goal:** Produce a local packaged app that the user can immediately retest.

**Requirements:** R1, R6

**Dependencies:** U2

**Files:**
- Generated only: `dist-py/stabilizer_arm64`, `dist/mac-arm64/Perforation Stabilizer.app`, and related ignored `dist/` outputs

**Approach:**
- Run the local build pipeline after source changes.
- Reopen the Apple Silicon packaged app from `dist/mac-arm64/Perforation Stabilizer.app`.
- Use the app against `EXPORT TEST FRAMES/` enough to confirm the previous argparse error is gone. Full stabilization quality is outside this fix; the gate is that processing starts with the four-anchor CLI contract accepted.

**Patterns to follow:**
- Existing app opening path used for manual testing.

**Test scenarios:**
- Integration: start packaged app, choose `EXPORT TEST FRAMES/`, set both anchors, start stabilization, and confirm no stderr `unrecognized arguments: --anchor1-x ...` message appears.
- Integration: progress advances past 0% or the next failure, if any, is unrelated to argparse contract parsing.

**Verification:**
- User can retry the same workflow without the CLI usage error.

---

## System-Wide Impact

- **Interaction graph:** `electron/renderer/renderer.js` sends anchor coordinates to `electron/main.js`; `main.js` spawns the packaged backend; `src/stabilizer_cli.py` validates args and calls `stabilize_folder`.
- **Error propagation:** argparse failures appear as stderr and exit code 2 in Electron. This fix prevents the known contract error rather than changing how process errors are displayed.
- **State lifecycle risks:** Generated binaries and DMGs are build artifacts. Source commits should contain scripts/tests/workflow changes, not local `dist/` or `dist-py/` output.
- **API surface parity:** Dev mode and packaged mode must keep using the same four-anchor CLI contract.
- **Integration coverage:** Source tests cover the Python CLI contract; build guard covers the generated local binary; manual packaged-app smoke covers Electron-to-binary integration.
- **Unchanged invariants:** JSON-lines progress protocol, report schema, stabilization math, and UI anchor selection behavior remain unchanged.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Local script refreshes only arm64, leaving stale x64 for local universal builds | Log the x64 limitation explicitly and rely on CI for x64, or build x64 when the Rosetta Python path exists |
| PyInstaller dependency install needs network access | Reuse existing environment where possible; otherwise surface the dependency failure directly |
| Unit tests pass but packaged app still uses stale resources | Build guard checks generated binary help before Electron packaging, and U4 opens the packaged app |
| Generated artifacts are accidentally committed | Ignore `dist-py/` and keep generated outputs untracked; stage only source, test, workflow, and plan changes |

---

## Documentation / Operational Notes

- Update build workflow behavior through scripts/workflow files rather than prose-only docs.
- The immediate local DMG/app rebuild is an operational artifact for testing and should not be treated as the durable fix.

---

## Sources & References

- Origin document: `docs/brainstorms/2026-04-19-multi-anchor-and-global-trajectory-smoothing-requirements.md`
- Related plan: `docs/plans/2026-04-19-002-feat-multi-anchor-and-global-trajectory-smoothing-plan.md`
- Related code: `electron/main.js`, `src/stabilizer_cli.py`, `scripts/build-local.sh`, `.github/workflows/build.yml`, `electron/package.json`
- Institutional learning: `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md`
