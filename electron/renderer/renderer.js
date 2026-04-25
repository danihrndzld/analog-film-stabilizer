'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('dropZone');
const dropContent   = document.getElementById('dropContent');
const dropLoaded    = document.getElementById('dropLoaded');
const dropFolderName = document.getElementById('dropFolderName');
const inputPathEl   = document.getElementById('inputPath');
const outputPathEl  = document.getElementById('outputPath');
const runBtn        = document.getElementById('runBtn');
const progressRow   = document.getElementById('progressRow');
const progressFill  = document.getElementById('progressFill');
const progressBar   = document.getElementById('progressBar');
const progressPct   = document.getElementById('progressPct');
const logLines      = document.getElementById('logLines');
const logEmpty      = document.getElementById('logEmpty');
const paramsSection    = document.getElementById('paramsSection');
const debugFramesPathEl = document.getElementById('debugFramesPath');

// ── Preview panel refs ────────────────────────────────────────────────────────
const previewSection    = document.getElementById('previewSection');
const previewStatusEl   = document.getElementById('previewStatus');
const previewLabelEl    = document.getElementById('previewLabel');
const previewImg        = document.getElementById('previewImg');
const anchorDot1        = document.getElementById('anchorDot1');
const anchorDot2        = document.getElementById('anchorDot2');
const resetAnchorsBtn   = document.getElementById('resetAnchorsBtn');

// Min anchor separation as a fraction of min(frame_w, frame_h). Must match
// MIN_ANCHOR_SEPARATION_FRAC in src/perforation_stabilizer_app.py.
const MIN_ANCHOR_SEPARATION_FRAC = 0.25;

// ── Version badge ─────────────────────────────────────────────────────────────
const versionBadge = document.getElementById('appVersion');
if (versionBadge && window.api.version) {
  versionBadge.textContent = `v${window.api.version}`;
  versionBadge.setAttribute('aria-label', `Versión ${window.api.version}`);
}

// ── Check for update button ───────────────────────────────────────────────────
const checkUpdateBtn = document.getElementById('checkUpdateBtn');
checkUpdateBtn.addEventListener('click', async () => {
  checkUpdateBtn.disabled = true;
  checkUpdateBtn.textContent = 'Buscando…';
  await window.api.checkForUpdates();
  // Timeout fallback in case the network call hangs
  setTimeout(() => {
    if (checkUpdateBtn.disabled) {
      checkUpdateBtn.disabled = false;
      checkUpdateBtn.textContent = 'Buscar actualización';
    }
  }, 8000);
});

window.api.onUpdateNotFound(() => {
  checkUpdateBtn.disabled = false;
  checkUpdateBtn.textContent = 'Al día ✓';
  setTimeout(() => {
    checkUpdateBtn.textContent = 'Buscar actualización';
  }, 3000);
});

// ── State ─────────────────────────────────────────────────────────────────────
let isRunning     = false;
// Two-anchor selection state. step tracks which click comes next.
let anchorState   = { step: 1, anchor1: null, anchor2: null };
let folderReady   = false;  // true when a valid folder has been selected

// ── Prevent Electron from navigating when files land outside the drop zone ────
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop',     (e) => e.preventDefault());

// ── Drop zone: drag events ────────────────────────────────────────────────────
let dragCounter = 0; // track nested dragenter/dragleave

dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  dropZone.classList.add('is-over');
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

dropZone.addEventListener('dragleave', (e) => {
  dragCounter--;
  if (dragCounter === 0) {
    dropZone.classList.remove('is-over');
  }
});

dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropZone.classList.remove('is-over');

  const files = Array.from(e.dataTransfer.files);
  if (!files.length) return;

  // webUtils.getPathForFile is exposed through the preload bridge
  const filePath = window.api.getPathForFile(files[0]);
  if (filePath) {
    setInputFolder(filePath);
  }
});

// ── Drop zone: keyboard activation ───────────────────────────────────────────
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    browseInput();
  }
});

// ── Browse buttons ────────────────────────────────────────────────────────────
async function browseInput() {
  const folder = await window.api.openFolder();
  if (folder) setInputFolder(folder);
}

async function browseOutput() {
  const folder = await window.api.openFolder();
  if (folder) outputPathEl.value = folder;
}

async function browseDebug() {
  const folder = await window.api.openFolder();
  if (folder) debugFramesPathEl.value = folder;
}

document.getElementById('browseInput').addEventListener('click', browseInput);
document.getElementById('browseOutput').addEventListener('click', browseOutput);
document.getElementById('browseDebug').addEventListener('click', browseDebug);

// Clicking anywhere on the drop zone (not on buttons) also triggers browse
dropZone.addEventListener('click', (e) => {
  // Only fire browse if the click is directly on the zone, not on a child with its own handler
  if (e.target === dropZone || e.target.closest('.drop-frame')) {
    browseInput();
  }
});

// ── Set input folder ──────────────────────────────────────────────────────────
function setInputFolder(folderPath) {
  inputPathEl.value = folderPath;

  // Auto-generate sibling output folder name
  const normalized = folderPath.replace(/\\/g, '/').replace(/\/$/, '');
  const parts  = normalized.split('/');
  const name   = parts[parts.length - 1];
  const parent = parts.slice(0, -1).join('/');
  outputPathEl.value = `${parent}/${name}_ESTABILIZADO`;

  // Update drop zone visual
  dropZone.classList.remove('is-over', 'is-error');
  dropZone.classList.add('is-loaded');
  dropFolderName.textContent = name;
  dropContent.hidden = true;
  dropLoaded.hidden  = false;

  folderReady = true;
  updateRunButton();

  addLog(`Carpeta seleccionada: ${folderPath}`);
  triggerPreview();
}

// ── Run button state ──────────────────────────────────────────────────────────
function updateRunButton() {
  // Enabled only when folder selected AND both anchors set
  const canRun = folderReady
    && anchorState.anchor1 !== null
    && anchorState.anchor2 !== null
    && !isRunning;
  runBtn.disabled = !canRun;
  if (canRun) {
    runBtn.removeAttribute('aria-disabled');
  } else {
    runBtn.setAttribute('aria-disabled', 'true');
  }
}

function resetAnchors() {
  anchorState = { step: 1, anchor1: null, anchor2: null };
  anchorDot1.hidden = true;
  anchorDot2.hidden = true;
  resetAnchorsBtn.hidden = true;
  if (previewImg.naturalWidth) {
    previewStatusEl.textContent = 'Haz clic para marcar el primer punto';
  }
  updateRunButton();
}

// ── Preview panel ─────────────────────────────────────────────────────────────

async function triggerPreview() {
  const input = inputPathEl.value.trim();
  if (!input) return;

  // Reset anchor state for new folder
  resetAnchors();

  previewSection.hidden = false;
  previewStatusEl.textContent = 'Cargando…';

  const firstFrame = await window.api.listFirstFrame(input);
  if (!firstFrame) {
    previewStatusEl.textContent = 'No se encontraron imágenes en la carpeta';
    return;
  }

  let result;
  try {
    result = await window.api.previewFrame({
      framePath: firstFrame,
    });
  } catch {
    previewStatusEl.textContent = 'Error al cargar el frame';
    return;
  }

  if (result.previewPath) {
    // Cache-bust so Electron reloads even when path is reused across calls
    previewImg.src = 'file://' + result.previewPath + '?t=' + Date.now();
  }

  previewStatusEl.textContent = 'Haz clic para marcar el primer punto';
}

// Click-to-set-anchor: two guided clicks with min-separation validation
previewImg.addEventListener('click', (e) => {
  if (!previewImg.naturalWidth) return;  // image not yet loaded
  const scaleX = previewImg.naturalWidth  / previewImg.offsetWidth;
  const scaleY = previewImg.naturalHeight / previewImg.offsetHeight;
  const frameX = e.offsetX * scaleX;
  const frameY = e.offsetY * scaleY;

  if (anchorState.step === 1) {
    anchorState.anchor1 = { x: frameX, y: frameY };
    anchorDot1.style.left = e.offsetX + 'px';
    anchorDot1.style.top  = e.offsetY + 'px';
    anchorDot1.hidden = false;
    anchorState.step = 2;
    resetAnchorsBtn.hidden = false;
    previewStatusEl.textContent =
      `Punto 1: (${Math.round(frameX)}, ${Math.round(frameY)}). Ahora marca el segundo punto, bien separado.`;
  } else if (anchorState.step === 2) {
    // Validate separation against min_sep in frame coords.
    const a1 = anchorState.anchor1;
    const dx = frameX - a1.x;
    const dy = frameY - a1.y;
    const sep = Math.hypot(dx, dy);
    const minSep = MIN_ANCHOR_SEPARATION_FRAC
      * Math.min(previewImg.naturalWidth, previewImg.naturalHeight);
    if (sep < minSep) {
      previewStatusEl.textContent =
        `Muy cerca del primer punto (${Math.round(sep)}px < ${Math.round(minSep)}px). Elige uno más alejado.`;
      return;
    }
    anchorState.anchor2 = { x: frameX, y: frameY };
    anchorDot2.style.left = e.offsetX + 'px';
    anchorDot2.style.top  = e.offsetY + 'px';
    anchorDot2.hidden = false;
    anchorState.step = 'done';
    previewStatusEl.textContent =
      `Referencias listas: (${Math.round(a1.x)}, ${Math.round(a1.y)}) · (${Math.round(frameX)}, ${Math.round(frameY)})`;
    updateRunButton();
  }
});

resetAnchorsBtn.addEventListener('click', resetAnchors);

// ── Run / Cancel ──────────────────────────────────────────────────────────────
runBtn.addEventListener('click', () => {
  if (isRunning) {
    cancelProcess();
  } else {
    startProcess();
  }
});

function startProcess() {
  const input  = inputPathEl.value.trim();
  const output = outputPathEl.value.trim();
  if (!input || !output || !anchorState.anchor1 || !anchorState.anchor2) return;

  setRunning(true);
  setProgress(0);
  progressRow.hidden     = false;
  progressRow.removeAttribute('aria-hidden');
  progressFill.className = 'progress-fill';

  window.api.startProcess({
    input,
    output,
    quality:     (q => Number.isFinite(q) ? q : 95)(parseInt(document.getElementById('paramQuality').value, 10)),
    debugFrames: debugFramesPathEl.value.trim(),
    anchor1X:    anchorState.anchor1.x,
    anchor1Y:    anchorState.anchor1.y,
    anchor2X:    anchorState.anchor2.x,
    anchor2Y:    anchorState.anchor2.y,
    borderMode:  document.getElementById('paramBorderMode').value || 'replicate',
  });
}

function cancelProcess() {
  window.api.cancelProcess();
  setRunning(false);
  addLog('Proceso cancelado.', 'error');
}

function setRunning(running) {
  isRunning = running;
  runBtn.textContent = running ? 'Cancelar' : 'Estabilizar secuencia';
  runBtn.classList.toggle('is-running', running);
  updateRunButton();
}

// ── Progress helper ───────────────────────────────────────────────────────────
function setProgress(ratio) {
  const v   = Math.max(0, Math.min(1, ratio));
  const pct = Math.round(v * 100);
  progressFill.style.transform = `scaleX(${v})`;
  progressPct.textContent      = `${pct}%`;
  progressBar.setAttribute('aria-valuenow', pct);
}

function finiteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

// ── IPC listeners ─────────────────────────────────────────────────────────────
window.api.onProgress((v) => setProgress(v));

window.api.onLog((msg) => addLog(msg));

window.api.onDone((summary) => {
  setRunning(false);
  setProgress(1);
  progressFill.classList.add('is-done');
  const total = finiteNumber(summary.total_frames);
  const detectedBoth = finiteNumber(summary.detected_both_frames);
  const incomplete = finiteNumber(summary.failed_frames_both_required);
  const amb1 = finiteNumber(summary.ambiguous_frames_a1);
  const amb2 = finiteNumber(summary.ambiguous_frames_a2);
  const motion1 = finiteNumber(summary.motion_rejected_frames_a1);
  const motion2 = finiteNumber(summary.motion_rejected_frames_a2);
  const consensus1 = finiteNumber(summary.consensus_rejected_frames_a1);
  const consensus2 = finiteNumber(summary.consensus_rejected_frames_a2);
  const nanFilled = finiteNumber(summary.nan_filled_frames);
  const outliers = finiteNumber(summary.outlier_frames_replaced);
  const splices = finiteNumber(summary.splice_count);
  const calibrationSamples = finiteNumber(summary.calibration_effective_n);
  const calibrationSpacing = finiteNumber(summary.calibration_perf_spacing_px);
  const outputWidth = finiteNumber(summary.output_width);
  const outputHeight = finiteNumber(summary.output_height);
  const calibrationStatus = summary.calibration_status === 'fallback' ? 'fallback' : 'ok';
  const parts = [
    `${total} frames`,
    `${detectedBoth} con ambos anclajes`,
  ];
  if (incomplete > 0) parts.push(`${incomplete} sin detección completa`);
  if (calibrationStatus === 'fallback') {
    parts.push('calibración no verificada');
  } else if (calibrationSamples > 0) {
    const spacing = calibrationSpacing > 0 ? `, ${Math.round(calibrationSpacing)}px` : '';
    parts.push(`calibración ${calibrationSamples} muestras${spacing}`);
  }
  if (amb1 + amb2 > 0) parts.push(`${amb1 + amb2} ambiguos`);
  if (motion1 + motion2 > 0) parts.push(`${motion1 + motion2} rechazadas por movimiento`);
  if (consensus1 + consensus2 > 0) parts.push(`${consensus1 + consensus2} rechazadas por consenso`);
  if (nanFilled > 0) parts.push(`${nanFilled} rellenadas por suavizado`);
  if (outliers > 0) parts.push(`${outliers} outliers suavizados`);
  if (splices > 0) parts.push(`${splices} splice(s)`);
  parts.push(`salida: ${outputWidth}×${outputHeight} px`);
  if (calibrationStatus === 'fallback') {
    addLog('Calibración no verificada: se usó bootstrap del primer frame.', 'warning');
  }
  addLog(`Listo — ${parts.join(' · ')}`, 'success');
});

window.api.onError((msg) => {
  setRunning(false);
  progressFill.classList.add('is-error');
  addLog(`Error: ${msg}`, 'error');
  dropZone.classList.add('is-error');
});

// ── Log helper ────────────────────────────────────────────────────────────────
function addLog(msg, type = '') {
  logEmpty.hidden = true;

  const now = new Date();
  const ts  = now.toTimeString().slice(0, 8);

  const line = document.createElement('div');
  line.className = 'log-line' + (type ? ` log-line--${type}` : '');

  const tsSpan = document.createElement('span');
  tsSpan.className = 'log-ts';
  tsSpan.setAttribute('aria-hidden', 'true');
  tsSpan.textContent = ts;
  line.appendChild(tsSpan);
  line.appendChild(document.createTextNode(msg));

  logLines.appendChild(line);
  // Scroll to bottom (non-blocking)
  requestAnimationFrame(() => {
    logLines.scrollTop = logLines.scrollHeight;
  });
}

// ── Sync aria-expanded on <details> ──────────────────────────────────────────
paramsSection.addEventListener('toggle', () => {
  const summary = paramsSection.querySelector('.params-toggle');
  summary.setAttribute('aria-expanded', paramsSection.open ? 'true' : 'false');
});

// ── Cleanup on unload ─────────────────────────────────────────────────────────
window.addEventListener('beforeunload', () => {
  ['process-progress', 'process-log', 'process-done', 'process-error',
   'update-available', 'update-download-progress'].forEach(
    (ch) => window.api.off(ch)
  );
});

// ── Auto-update banner ────────────────────────────────────────────────────────
window.api.onUpdateAvailable((info) => {
  const existing = document.getElementById('updateBanner');
  if (existing) existing.remove();

  const banner = document.createElement('div');
  banner.id        = 'updateBanner';
  banner.className = 'update-banner';

  const text = document.createElement('span');
  text.className = 'update-banner__text';
  const strong = document.createElement('strong');
  strong.textContent = `v${info.version}`;
  text.appendChild(document.createTextNode('Nueva versión '));
  text.appendChild(strong);
  text.appendChild(document.createTextNode(' disponible'));
  banner.appendChild(text);

  const updateBtn = document.createElement('button');
  updateBtn.className = 'update-banner__btn';
  updateBtn.id = 'updateBtn';
  updateBtn.textContent = 'Actualizar';
  banner.appendChild(updateBtn);

  const dismissBtn = document.createElement('button');
  dismissBtn.className = 'update-banner__dismiss';
  dismissBtn.id = 'updateDismissBtn';
  dismissBtn.setAttribute('aria-label', 'Cerrar');
  dismissBtn.textContent = '\u2715';
  banner.appendChild(dismissBtn);

  document.querySelector('.app').prepend(banner);

  dismissBtn.addEventListener('click', () => banner.remove());

  updateBtn.addEventListener('click', async () => {
    updateBtn.disabled    = true;
    updateBtn.textContent = 'Descargando…';

    window.api.onUpdateDownloadProgress((p) => {
      updateBtn.textContent = `${Math.round(p * 100)}%`;
    });

    try {
      await window.api.downloadUpdate({ downloadUrl: info.downloadUrl, assetName: info.assetName });
      updateBtn.textContent = '¡Listo!';
      text.textContent = 'Abre el DMG y arrastra la app para reemplazar la versión actual.';
    } catch {
      updateBtn.disabled    = false;
      updateBtn.textContent = 'Actualizar';
      window.api.openExternal(info.releaseUrl);
    }
  });
});
