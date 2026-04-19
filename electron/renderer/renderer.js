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
const previewImg        = document.getElementById('previewImg');
const anchorDot         = document.getElementById('anchorDot');

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
  // Enabled only when both folder is selected AND anchor is set
  const canRun = folderReady && previewAnchor !== null && !isRunning;
  runBtn.disabled = !canRun;
  if (canRun) {
    runBtn.removeAttribute('aria-disabled');
  } else {
    runBtn.setAttribute('aria-disabled', 'true');
  }
}

// ── Preview panel ─────────────────────────────────────────────────────────────

async function triggerPreview() {
  const input = inputPathEl.value.trim();
  if (!input) return;

  // Reset anchor state for new folder
  previewAnchor = null;
  anchorDot.hidden = true;
  updateRunButton();

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

  previewStatusEl.textContent = 'Haz clic para seleccionar referencia';
}

// Click-to-set-anchor: map click position back to frame coordinates
previewImg.addEventListener('click', (e) => {
  if (!previewImg.naturalWidth) return;  // image not yet loaded
  const scaleX = previewImg.naturalWidth  / previewImg.offsetWidth;
  const scaleY = previewImg.naturalHeight / previewImg.offsetHeight;
  const frameX = e.offsetX * scaleX;
  const frameY = e.offsetY * scaleY;

  previewAnchor = { x: frameX, y: frameY };

  anchorDot.style.left = e.offsetX + 'px';
  anchorDot.style.top  = e.offsetY + 'px';
  anchorDot.hidden = false;

  previewStatusEl.textContent =
    `Referencia: (${Math.round(frameX)}, ${Math.round(frameY)})`;

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
  if (!input || !output || !previewAnchor) return;

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
    anchorX:     previewAnchor.x,
    anchorY:     previewAnchor.y,
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

// ── IPC listeners ─────────────────────────────────────────────────────────────
window.api.onProgress((v) => setProgress(v));

window.api.onLog((msg) => addLog(msg));

window.api.onDone((summary) => {
  setRunning(false);
  setProgress(1);
  progressFill.classList.add('is-done');
  const ambiguous = summary.ambiguous_frames ?? 0;
  const rejected = summary.motion_rejected_frames ?? 0;
  const parts = [
    `${summary.total_frames} frames`,
    `${summary.failed_detections} sin detección`,
  ];
  if (ambiguous > 0) parts.push(`${ambiguous} ambiguos`);
  if (rejected > 0 && rejected !== ambiguous) parts.push(`${rejected} rechazados por movimiento`);
  parts.push(`salida: ${summary.output_width}×${summary.output_height} px`);
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
