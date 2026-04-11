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
const filmFormatEl     = document.getElementById('filmFormat');
const debugFramesPathEl = document.getElementById('debugFramesPath');

// ── Preview panel refs ────────────────────────────────────────────────────────
const previewSection    = document.getElementById('previewSection');
const previewStatusEl   = document.getElementById('previewStatus');
const previewImg        = document.getElementById('previewImg');
const anchorDot         = document.getElementById('anchorDot');
const previewRefreshBtn = document.getElementById('previewRefreshBtn');
const previewResetBtn   = document.getElementById('previewResetBtn');

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
let isRunning          = false;
let previewAnchor      = null;   // null | { x, y } in frame coords (manual override)
let autoDetectedAnchor = null;   // null | { x, y } from last Python detection

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

  runBtn.disabled = false;
  runBtn.removeAttribute('aria-disabled');

  addLog(`Carpeta seleccionada: ${folderPath}`);
  triggerPreview();
}

// ── Preview panel ─────────────────────────────────────────────────────────────

async function triggerPreview() {
  const input = inputPathEl.value.trim();
  if (!input) return;

  // Reset anchor state immediately so a stale anchor from a previous folder
  // is never used if the user hits run during the async "Analizando…" window.
  previewAnchor      = null;
  autoDetectedAnchor = null;

  previewSection.hidden = false;
  previewStatusEl.textContent = 'Analizando…';
  anchorDot.hidden = true;
  previewResetBtn.hidden = true;

  const firstFrame = await window.api.listFirstFrame(input);
  if (!firstFrame) {
    previewStatusEl.textContent = 'No se encontraron imágenes en la carpeta';
    return;
  }

  let result;
  try {
    result = await window.api.previewFrame({
      framePath:  firstFrame,
      roi:        parseFloat(document.getElementById('paramRoi').value)        || 0.22,
      threshold:  parseInt(document.getElementById('paramThreshold').value, 10) || 210,
      filmFormat: filmFormatEl.value || 'super8',
    });
  } catch {
    previewStatusEl.textContent = 'Error al analizar el frame';
    return;
  }

  if (result.previewPath) {
    // Cache-bust so Electron reloads even when path is reused across calls
    previewImg.src = 'file://' + result.previewPath + '?t=' + Date.now();
  }

  if (result.detected) {
    autoDetectedAnchor = { x: result.cx, y: result.cy };
    previewAnchor      = autoDetectedAnchor;
    previewStatusEl.textContent =
      `Auto: (${Math.round(result.cx)}, ${Math.round(result.cy)})`;
  } else {
    autoDetectedAnchor = null;
    previewAnchor      = null;
    previewStatusEl.textContent = 'No detectado — haz clic para marcar manualmente';
  }
}

// Re-detect when film format changes (different format → different detection params)
filmFormatEl.addEventListener('change', () => {
  previewAnchor = null;
  triggerPreview();
});

// Actualizar: re-run preview with current advanced params
previewRefreshBtn.addEventListener('click', () => {
  previewAnchor = null;
  triggerPreview();
});

// Restablecer: revert manual override to last auto-detected anchor
previewResetBtn.addEventListener('click', () => {
  previewAnchor  = autoDetectedAnchor;
  anchorDot.hidden = true;
  previewResetBtn.hidden = true;
  previewStatusEl.textContent = autoDetectedAnchor
    ? `Auto: (${Math.round(autoDetectedAnchor.x)}, ${Math.round(autoDetectedAnchor.y)})`
    : 'No detectado — haz clic para marcar manualmente';
});

// Click-to-set-anchor: map click position back to frame coordinates
previewImg.addEventListener('click', (e) => {
  if (!previewImg.naturalWidth) return;  // image not yet loaded
  const scaleX = previewImg.naturalWidth  / previewImg.offsetWidth;
  const scaleY = previewImg.naturalHeight / previewImg.offsetHeight;
  const frameX = e.offsetX * scaleX;
  const frameY = e.offsetY * scaleY;

  previewAnchor = { x: frameX, y: frameY };

  // ROI boundary: preview shows 40% of frame, ROI is roi_ratio of frame
  // roiBoundaryX = previewWidth * (roi_ratio / 0.40)
  const roiRatio = parseFloat(document.getElementById('paramRoi').value) || 0.22;
  const roiBoundaryX = previewImg.naturalWidth * (roiRatio / 0.40);
  const isOutsideRoi = frameX > roiBoundaryX;

  anchorDot.style.left = e.offsetX + 'px';
  anchorDot.style.top  = e.offsetY + 'px';
  anchorDot.classList.toggle('is-outside-roi', isOutsideRoi);
  anchorDot.hidden = false;

  previewStatusEl.textContent = isOutsideRoi
    ? `Manual: (${Math.round(frameX)}, ${Math.round(frameY)}) — fuera de zona de detección`
    : `Manual: (${Math.round(frameX)}, ${Math.round(frameY)})`;
  previewResetBtn.hidden = autoDetectedAnchor === null;
});

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
  if (!input || !output) return;

  setRunning(true);
  setProgress(0);
  progressRow.hidden     = false;
  progressRow.removeAttribute('aria-hidden');
  progressFill.className = 'progress-fill';

  window.api.startProcess({
    input,
    output,
    roi:           parseFloat(document.getElementById('paramRoi').value)        || 0.22,
    threshold:     parseInt(document.getElementById('paramThreshold').value, 10) || 210,
    smooth:        parseInt(document.getElementById('paramSmooth').value, 10)    || 9,
    quality:       parseInt(document.getElementById('paramQuality').value, 10),  // 0 is valid (PNG)
    filmFormat:    filmFormatEl.value || 'super8',
    debugFrames:   debugFramesPathEl.value.trim(),
    manualAnchorX: previewAnchor?.x ?? null,
    manualAnchorY: previewAnchor?.y ?? null,
    borderMode:    document.getElementById('paramBorderMode').value || 'replicate',
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
  runBtn.disabled = false;
  runBtn.removeAttribute('aria-disabled');
}

// ── Progress helper ───────────────────────────────────────────────────────────
function setProgress(ratio) {
  const v   = Math.max(0, Math.min(1, ratio));
  const pct = Math.round(v * 100);
  progressFill.style.transform = `scaleX(${v})`;
  progressPct.textContent      = `${pct}%`;
  progressBar.setAttribute('aria-valuenow', pct);
}

// ── IPC listeners ─────────────────────────────────────────────────────────────
window.api.onProgress((v) => setProgress(v));

window.api.onLog((msg) => addLog(msg));

window.api.onDone((summary) => {
  setRunning(false);
  setProgress(1);
  progressFill.classList.add('is-done');
  addLog(
    `Listo — ${summary.total_frames} frames · ${summary.failed_detections} sin detección · salida: ${summary.output_width}×${summary.output_height} px`,
    'success'
  );
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
  line.innerHTML =
    `<span class="log-ts" aria-hidden="true">${ts}</span>${escapeHtml(msg)}`;

  logLines.appendChild(line);
  // Scroll to bottom (non-blocking)
  requestAnimationFrame(() => {
    logLines.scrollTop = logLines.scrollHeight;
  });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
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
  banner.innerHTML =
    `<span class="update-banner__text">
       Nueva versión <strong>v${escapeHtml(info.version)}</strong> disponible
     </span>
     <button class="update-banner__btn" id="updateBtn">Actualizar</button>
     <button class="update-banner__dismiss" id="updateDismissBtn" aria-label="Cerrar">✕</button>`;

  document.querySelector('.app').prepend(banner);

  document.getElementById('updateDismissBtn').addEventListener('click', () => banner.remove());

  document.getElementById('updateBtn').addEventListener('click', async () => {
    const btn = document.getElementById('updateBtn');
    btn.disabled    = true;
    btn.textContent = 'Descargando…';

    window.api.onUpdateDownloadProgress((p) => {
      btn.textContent = `${Math.round(p * 100)}%`;
    });

    try {
      await window.api.downloadUpdate({ downloadUrl: info.downloadUrl, assetName: info.assetName });
      btn.textContent = '¡Listo!';
      banner.querySelector('.update-banner__text').textContent =
        'Abre el DMG y arrastra la app para reemplazar la versión actual.';
    } catch {
      btn.disabled    = false;
      btn.textContent = 'Actualizar';
      window.api.openExternal(info.releaseUrl);
    }
  });
});
