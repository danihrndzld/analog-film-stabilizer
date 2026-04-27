#!/usr/bin/env bash
# Local CI: builds Electron DMGs after every commit.
# Runs in background (non-blocking). Logs to /tmp/perfstab-build.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/perfstab-build.log"
ELECTRON_DIR="$REPO/electron"
DIST_PY="$REPO/dist-py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_DEPS=(pyinstaller opencv-python-headless numpy scipy)
X64_REBUILT=0

echo "" >> "$LOG"
echo "=== Build triggered $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

check_cli_contract() {
  local binary="$1"
  local help_output

  help_output="$("$binary" --help 2>&1)"

  if [[ "$help_output" != *"--anchor1-x"* ]] \
    || [[ "$help_output" != *"--anchor1-y"* ]] \
    || [[ "$help_output" != *"--anchor2-x"* ]] \
    || [[ "$help_output" != *"--anchor2-y"* ]]; then
    echo "✗ $binary does not expose the two-anchor CLI contract." >> "$LOG"
    echo "$help_output" >> "$LOG"
    exit 1
  fi

  if grep -Eq '(^|[[:space:]])--anchor-x([,[:space:]]|$)' <<< "$help_output"; then
    echo "✗ $binary still exposes the legacy --anchor-x flag." >> "$LOG"
    echo "$help_output" >> "$LOG"
    exit 1
  fi
}

echo "→ building arm64 Python backend..." >> "$LOG"
mkdir -p "$DIST_PY"
"$PYTHON_BIN" -m venv /tmp/perfstab-venv-arm64 >> "$LOG" 2>&1
/tmp/perfstab-venv-arm64/bin/pip install --upgrade pip >> "$LOG" 2>&1
/tmp/perfstab-venv-arm64/bin/pip install "${PY_DEPS[@]}" >> "$LOG" 2>&1
/tmp/perfstab-venv-arm64/bin/python -m PyInstaller --onefile --name stabilizer_arm64 \
  --distpath "$DIST_PY" \
  --workpath /tmp/pyinstaller-arm64 \
  --specpath /tmp \
  "$REPO/src/stabilizer_cli.py" >> "$LOG" 2>&1
chmod +x "$DIST_PY/stabilizer_arm64"
check_cli_contract "$DIST_PY/stabilizer_arm64"

X64_PYTHON="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
if arch -x86_64 "$PYTHON_BIN" -c 'import platform; raise SystemExit(0 if platform.machine() == "x86_64" else 1)' >> "$LOG" 2>&1; then
  echo "→ building x64 Python backend..." >> "$LOG"
  arch -x86_64 "$PYTHON_BIN" -m venv /tmp/perfstab-venv-x64 >> "$LOG" 2>&1
  arch -x86_64 /tmp/perfstab-venv-x64/bin/pip install --upgrade pip >> "$LOG" 2>&1
  arch -x86_64 /tmp/perfstab-venv-x64/bin/pip install "${PY_DEPS[@]}" >> "$LOG" 2>&1
  arch -x86_64 /tmp/perfstab-venv-x64/bin/python -m PyInstaller --onefile --name stabilizer_x64 \
    --distpath "$DIST_PY" \
    --workpath /tmp/pyinstaller-x64 \
    --specpath /tmp \
    "$REPO/src/stabilizer_cli.py" >> "$LOG" 2>&1
  chmod +x "$DIST_PY/stabilizer_x64"
  check_cli_contract "$DIST_PY/stabilizer_x64"
  X64_REBUILT=1
elif [[ -x "$X64_PYTHON" ]]; then
  echo "→ building x64 Python backend..." >> "$LOG"
  arch -x86_64 "$X64_PYTHON" -m venv /tmp/perfstab-venv-x64 >> "$LOG" 2>&1
  arch -x86_64 /tmp/perfstab-venv-x64/bin/pip install --upgrade pip >> "$LOG" 2>&1
  arch -x86_64 /tmp/perfstab-venv-x64/bin/pip install "${PY_DEPS[@]}" >> "$LOG" 2>&1
  arch -x86_64 /tmp/perfstab-venv-x64/bin/python -m PyInstaller --onefile --name stabilizer_x64 \
    --distpath "$DIST_PY" \
    --workpath /tmp/pyinstaller-x64 \
    --specpath /tmp \
    "$REPO/src/stabilizer_cli.py" >> "$LOG" 2>&1
  chmod +x "$DIST_PY/stabilizer_x64"
  check_cli_contract "$DIST_PY/stabilizer_x64"
  X64_REBUILT=1
else
  echo "→ x64 Python not found locally; CI remains authoritative for x64 binary rebuilds." >> "$LOG"
  if [[ ! -x "$DIST_PY/stabilizer_x64" ]]; then
    echo "✗ $DIST_PY/stabilizer_x64 is missing; Electron packaging requires it." >> "$LOG"
    exit 1
  fi
fi

cd "$ELECTRON_DIR"

echo "→ npm install..." >> "$LOG"
npm install --silent >> "$LOG" 2>&1

if [[ "$X64_REBUILT" -eq 1 ]]; then
  echo "→ npm run build..." >> "$LOG"
  npm run build >> "$LOG" 2>&1
else
  echo "→ npm run build (arm64 only)..." >> "$LOG"
  ARM64_CONFIG="/tmp/perfstab-electron-builder-arm64.json"
  node -e 'const fs=require("fs"); const pkg=require("./package.json"); const cfg=JSON.parse(JSON.stringify(pkg.build)); cfg.mac.target=[{target:"dmg", arch:["arm64"]}]; fs.writeFileSync(process.argv[1], JSON.stringify(cfg));' "$ARM64_CONFIG" >> "$LOG" 2>&1
  npx electron-builder --mac --config "$ARM64_CONFIG" >> "$LOG" 2>&1
fi

echo "✓ Build complete. DMGs in $REPO/dist/" >> "$LOG"
if [[ "$X64_REBUILT" -eq 1 ]]; then
  ls "$REPO/dist/"*.dmg >> "$LOG" 2>&1 || true
else
  ls "$REPO/dist/"*arm64.dmg >> "$LOG" 2>&1 || true
fi
