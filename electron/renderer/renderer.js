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
const selectionRect     = document.getElementById('selectionRect');

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
let previewAnchor = null;   // null | { x, y } in frame coords
let previewRect   = null;   // null | { width, height } in frame coords
let isDragging    = false;
let dragStart     = null;   // { displayX, displayY, frameX, frameY }

let zoomLevel = 1;
const ZOOM_MIN  = 1;
const ZOOM_MAX  = 5;
const ZOOM_STEP = 0.25;

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

  addLog(`Carpeta seleccionada: ${folderPath}`);
  triggerPreview();
}

// ── Preview panel ─────────────────────────────────────────────────────────────

function updateRunButton() {
  const hasInput  = inputPathEl.value.trim() !== '';
  const hasAnchor = previewAnchor !== null;
  runBtn.disabled = !hasInput || !hasAnchor;
  if (runBtn.disabled) {
    runBtn.setAttribute('aria-disabled', 'true');
  } else {
    runBtn.removeAttribute('aria-disabled');
  }
}

async function triggerPreview() {
  const input = inputPathEl.value.trim();
  if (!input) return;

  previewAnchor = null;
  previewRect   = null;
  zoomLevel = 1;
  previewImg.style.transform = '';
  previewImg.style.width     = '';
  previewImg.style.height    = '';
  updateRunButton();

  previewSection.hidden = false;
  previewStatusEl.textContent = 'Cargando…';
  anchorDot.hidden = true;
  selectionRect.hidden = true;

  const firstFrame = await window.api.listFirstFrame(input);
  if (!firstFrame) {
    previewStatusEl.textContent = 'No se encontraron imágenes en la carpeta';
    return;
  }

  previewImg.src = 'file://' + firstFrame + '?t=' + Date.now();
  previewImg.onload = () => {
    previewStatusEl.textContent = 'Haz clic y arrastra para seleccionar el área de referencia';
  };
  previewImg.onerror = () => {
    previewStatusEl.textContent = 'Error al cargar el frame';
  };
}

// ── Preview zoom ──────────────────────────────────────────────────────────────

const previewImgWrap = document.getElementById('previewImgWrap');
previewImgWrap.addEventListener('wheel', (e) => {
  if (!previewImg.naturalWidth) return;
  e.preventDefault();
  const oldZoom = zoomLevel;
  zoomLevel = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoomLevel + (e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP)));
  if (zoomLevel !== oldZoom) {
    previewImg.style.transform = `scale(${zoomLevel})`;
    previewImg.style.width  = previewImg.naturalWidth  + 'px';
    previewImg.style.height = previewImg.naturalHeight + 'px';
  }
}, { passive: false });

// ── Rectangle selection ───────────────────────────────────────────────────────

function eventToFrameCoords(e) {
  const rect     = previewImg.getBoundingClientRect();
  const displayX = e.clientX - rect.left;
  const displayY = e.clientY - rect.top;
  // getBoundingClientRect returns scaled dimensions, so divide by zoom
  // to get CSS-pixel position, then map to natural image coordinates
  const frameX   = (displayX / zoomLevel) * (previewImg.naturalWidth  / previewImg.offsetWidth);
  const frameY   = (displayY / zoomLevel) * (previewImg.naturalHeight / previewImg.offsetHeight);
  return { displayX, displayY, frameX, frameY };
}

previewImg.addEventListener('mousedown', (e) => {
  if (!previewImg.naturalWidth || e.button !== 0) return;
  e.preventDefault();

  const coords = eventToFrameCoords(e);
  isDragging = true;
  dragStart  = coords;
  previewRect = null;

  anchorDot.style.left = coords.displayX + 'px';
  anchorDot.style.top  = coords.displayY + 'px';
  anchorDot.hidden = false;

  selectionRect.hidden = true;
  selectionRect.style.left   = coords.displayX + 'px';
  selectionRect.style.top    = coords.displayY + 'px';
  selectionRect.style.width  = '0px';
  selectionRect.style.height = '0px';

  previewStatusEl.textContent = `Ancla: (${Math.round(coords.frameX)}, ${Math.round(coords.frameY)}) — arrastra para definir área`;
});

document.addEventListener('mousemove', (e) => {
  if (!isDragging || !dragStart) return;

  const rect     = previewImg.getBoundingClientRect();
  const displayX = e.clientX - rect.left;
  const displayY = e.clientY - rect.top;

  const left   = Math.min(dragStart.displayX, displayX);
  const top    = Math.min(dragStart.displayY, displayY);
  const width  = Math.abs(displayX - dragStart.displayX);
  const height = Math.abs(displayY - dragStart.displayY);

  selectionRect.style.left   = left   + 'px';
  selectionRect.style.top    = top    + 'px';
  selectionRect.style.width  = width  + 'px';
  selectionRect.style.height = height + 'px';
  selectionRect.hidden = false;
});

document.addEventListener('mouseup', (e) => {
  if (!isDragging || !dragStart) return;
  isDragging = false;

  const coords = eventToFrameCoords(e);

  const natW = previewImg.naturalWidth;
  const natH = previewImg.naturalHeight;

  // Clamp frame coordinates to image bounds
  const clamp = (v, max) => Math.max(0, Math.min(v, max));
  const x0 = clamp(Math.min(dragStart.frameX, coords.frameX), natW);
  const y0 = clamp(Math.min(dragStart.frameY, coords.frameY), natH);
  const x1 = clamp(Math.max(dragStart.frameX, coords.frameX), natW);
  const y1 = clamp(Math.max(dragStart.frameY, coords.frameY), natH);

  const rectWidth  = x1 - x0;
  const rectHeight = y1 - y0;

  const anchorX = x0;
  const anchorY = y0;

  if (rectWidth < 20 || rectHeight < 20) {
    previewAnchor = { x: dragStart.frameX, y: dragStart.frameY };
    previewRect   = null;
    selectionRect.hidden = true;
    previewStatusEl.textContent = `Ancla: (${Math.round(dragStart.frameX)}, ${Math.round(dragStart.frameY)}) — arrastra para definir área`;
    updateRunButton();
    return;
  }

  previewAnchor = { x: anchorX, y: anchorY };
  previewRect   = { width: rectWidth, height: rectHeight };

  previewStatusEl.textContent = `Ancla: (${Math.round(anchorX)}, ${Math.round(anchorY)}) · Área: ${Math.round(rectWidth)}×${Math.round(rectHeight)} px`;
  updateRunButton();
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
    rectWidth:     previewRect?.width  ?? null,
    rectHeight:    previewRect?.height ?? null,
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
