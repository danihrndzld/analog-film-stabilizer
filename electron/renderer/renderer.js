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
const paramsSection = document.getElementById('paramsSection');
const filmFormatEl  = document.getElementById('filmFormat');

// ── Version badge ─────────────────────────────────────────────────────────────
const versionBadge = document.getElementById('appVersion');
if (versionBadge && window.api.version) {
  versionBadge.textContent = `v${window.api.version}`;
  versionBadge.setAttribute('aria-label', `Versión ${window.api.version}`);
}

// ── State ─────────────────────────────────────────────────────────────────────
let isRunning = false;

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

document.getElementById('browseInput').addEventListener('click', browseInput);
document.getElementById('browseOutput').addEventListener('click', browseOutput);

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
}

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
    roi:       parseFloat(document.getElementById('paramRoi').value)       || 0.22,
    threshold: parseInt(document.getElementById('paramThreshold').value, 10) || 210,
    smooth:    parseInt(document.getElementById('paramSmooth').value, 10)    || 9,
    quality:    parseInt(document.getElementById('paramQuality').value, 10),  // 0 is valid (PNG)
    filmFormat: filmFormatEl.value || 'super8',
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
  ['process-progress', 'process-log', 'process-done', 'process-error'].forEach(
    (ch) => window.api.off(ch)
  );
});
