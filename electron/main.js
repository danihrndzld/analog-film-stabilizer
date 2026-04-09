'use strict';

const { app, BrowserWindow, ipcMain, dialog, shell, net } = require('electron');
const path  = require('path');
const https = require('https');
const fs    = require('fs');
const os    = require('os');
const { spawn } = require('child_process');

const GITHUB_REPO = 'danihrndzld/analog-film-stabilizer';

let mainWindow = null;
let pyProcess  = null;

// ── Auto-update: compare semver strings ──────────────────────────────────────
function isNewer(candidate, current) {
  const a = candidate.split('.').map(Number);
  const b = current.split('.').map(Number);
  for (let i = 0; i < 3; i++) {
    if ((a[i] || 0) > (b[i] || 0)) return true;
    if ((a[i] || 0) < (b[i] || 0)) return false;
  }
  return false;
}

// ── Auto-update: poll GitHub releases API ────────────────────────────────────
function checkForUpdates(manual = false) {
  const req = net.request({
    url:    `https://api.github.com/repos/${GITHUB_REPO}/releases/latest`,
    method: 'GET',
  });
  req.setHeader('User-Agent', 'Perforation-Stabilizer-Updater');
  req.setHeader('Accept',     'application/vnd.github+json');

  let body = '';
  req.on('response', (res) => {
    res.on('data',  (chunk) => { body += chunk; });
    res.on('end',   () => {
      try {
        const release   = JSON.parse(body);
        const latest    = (release.tag_name || '').replace(/^v/, '');
        const current   = app.getVersion();
        if (!latest || !isNewer(latest, current)) {
          if (manual) mainWindow?.webContents.send('update-not-found');
          return;
        }

        // Pick the right DMG for this arch
        const isArm = process.arch === 'arm64';
        const asset = (release.assets || []).find(a =>
          isArm ? a.name.includes('arm64') : (a.name.endsWith('.dmg') && !a.name.includes('arm64'))
        );

        mainWindow?.webContents.send('update-available', {
          version:     latest,
          releaseUrl:  release.html_url,
          downloadUrl: asset?.browser_download_url || release.html_url,
          assetName:   asset?.name || `Perforation.Stabilizer-${latest}.dmg`,
        });
      } catch { /* silent */ }
    });
  });
  req.on('error', () => {});
  req.end();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 780,
    height: 700,
    minWidth: 620,
    minHeight: 560,
    backgroundColor: '#181612',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // Prevent navigation when files are dropped on the window chrome
  mainWindow.webContents.on('will-navigate', (e) => e.preventDefault());

  // Check for updates 4 s after the UI is ready
  mainWindow.webContents.once('did-finish-load', () => {
    setTimeout(checkForUpdates, 4000);
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ── IPC: Manual update check ─────────────────────────────────────────────────
ipcMain.handle('check-updates', () => checkForUpdates());

// ── IPC: Open folder dialog ──────────────────────────────────────────────────
ipcMain.handle('open-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Selecciona carpeta',
  });
  return result.canceled ? null : result.filePaths[0];
});

// ── IPC: Start Python process ────────────────────────────────────────────────
ipcMain.handle('start-process', (event, opts) => {
  if (pyProcess) return; // already running

  // In production use the bundled standalone binary (no Python required on user machine).
  // In dev fall back to python3 + script.
  let executable, args;
  if (app.isPackaged) {
    const binaryName = process.arch === 'arm64' ? 'stabilizer_arm64' : 'stabilizer_x64';
    executable = path.join(process.resourcesPath, binaryName);
    args = [
      '--input',       opts.input,
      '--output',      opts.output,
      '--roi',         String(opts.roi),
      '--threshold',   String(opts.threshold),
      '--smooth',      String(opts.smooth),
      '--quality',     String(opts.quality),
      '--film-format', String(opts.filmFormat || 'super8'),
    ];
    if (opts.debugFrames) args.push('--debug-frames', opts.debugFrames);
    if (opts.manualAnchorX != null && opts.manualAnchorY != null) {
      args.push('--manual-anchor-x', String(opts.manualAnchorX),
                '--manual-anchor-y', String(opts.manualAnchorY));
    }
    if (opts.borderMode) args.push('--border-mode', String(opts.borderMode));
  } else {
    const scriptPath = path.join(__dirname, '..', 'src', 'stabilizer_cli.py');
    executable = 'python3';
    args = [
      scriptPath,
      '--input',       opts.input,
      '--output',      opts.output,
      '--roi',         String(opts.roi),
      '--threshold',   String(opts.threshold),
      '--smooth',      String(opts.smooth),
      '--quality',     String(opts.quality),
      '--film-format', String(opts.filmFormat || 'super8'),
    ];
    if (opts.debugFrames) args.push('--debug-frames', opts.debugFrames);
    if (opts.manualAnchorX != null && opts.manualAnchorY != null) {
      args.push('--manual-anchor-x', String(opts.manualAnchorX),
                '--manual-anchor-y', String(opts.manualAnchorY));
    }
    if (opts.borderMode) args.push('--border-mode', String(opts.borderMode));
  }

  pyProcess = spawn(executable, args);

  // Parse JSON-lines from stdout
  let buffer = '';
  pyProcess.stdout.on('data', (chunk) => {
    buffer += chunk.toString('utf8');
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete last line

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const msg = JSON.parse(trimmed);
        switch (msg.type) {
          case 'progress':
            mainWindow?.webContents.send('process-progress', msg.value);
            break;
          case 'log':
            mainWindow?.webContents.send('process-log', msg.msg);
            break;
          case 'done':
            mainWindow?.webContents.send('process-done', msg.summary);
            pyProcess = null;
            break;
          case 'error':
            mainWindow?.webContents.send('process-error', msg.msg);
            pyProcess = null;
            break;
        }
      } catch {
        // Malformed JSON — forward as raw log
        mainWindow?.webContents.send('process-log', trimmed);
      }
    }
  });

  // Forward stderr as log lines (Python tracebacks, dependency warnings, etc.)
  pyProcess.stderr.on('data', (chunk) => {
    for (const line of chunk.toString('utf8').split('\n')) {
      const t = line.trim();
      if (t) mainWindow?.webContents.send('process-log', `[stderr] ${t}`);
    }
  });

  // M4: handle spawn failure (python3 not found in PATH, etc.)
  pyProcess.on('error', (err) => {
    mainWindow?.webContents.send(
      'process-error',
      `No se pudo iniciar Python: ${err.message}. Asegúrate de que python3 está instalado y en tu PATH.`
    );
    pyProcess = null;
  });

  pyProcess.on('exit', (code) => {
    if (code !== 0 && pyProcess !== null) {
      mainWindow?.webContents.send('process-error', `El proceso terminó con código ${code}`);
    }
    pyProcess = null;
  });
});

// ── IPC: Cancel Python process ───────────────────────────────────────────────
ipcMain.handle('cancel-process', () => {
  if (pyProcess) {
    pyProcess.kill('SIGTERM');
    pyProcess = null;
  }
});

// ── IPC: Single-frame preview ─────────────────────────────────────────────────
// Runs --mode preview on one frame; returns { detected, cx, cy, previewPath }.
ipcMain.handle('preview-frame', (event, opts) => {
  const previewOut = path.join(os.tmpdir(), 'stabilizer_preview.jpg');

  let executable, args;
  if (app.isPackaged) {
    const binaryName = process.arch === 'arm64' ? 'stabilizer_arm64' : 'stabilizer_x64';
    executable = path.join(process.resourcesPath, binaryName);
    args = [
      '--mode',        'preview',
      '--frame-path',  opts.framePath,
      '--preview-out', previewOut,
      '--roi',         String(opts.roi        || 0.22),
      '--threshold',   String(opts.threshold  || 210),
      '--film-format', String(opts.filmFormat || 'super8'),
    ];
  } else {
    const scriptPath = path.join(__dirname, '..', 'src', 'stabilizer_cli.py');
    executable = 'python3';
    args = [
      scriptPath,
      '--mode',        'preview',
      '--frame-path',  opts.framePath,
      '--preview-out', previewOut,
      '--roi',         String(opts.roi        || 0.22),
      '--threshold',   String(opts.threshold  || 210),
      '--film-format', String(opts.filmFormat || 'super8'),
    ];
  }

  return new Promise((resolve) => {
    let stdout = '';
    const proc = spawn(executable, args);
    proc.stdout.on('data', (chunk) => { stdout += chunk.toString('utf8'); });
    proc.on('error', (err) => {
      resolve({ detected: false, error: err.message });
    });
    proc.on('close', () => {
      try {
        const line = stdout.trim().split('\n').find(l => l.trim());
        const msg  = JSON.parse(line);
        resolve({
          detected:    msg.detected    || false,
          cx:          msg.cx          ?? null,
          cy:          msg.cy          ?? null,
          previewPath: msg.previewPath ?? null,
        });
      } catch {
        resolve({ detected: false, error: 'Failed to parse preview result' });
      }
    });
  });
});

// ── IPC: First image file in a folder ────────────────────────────────────────
// Returns the full path of the lexicographically first image file, or null.
ipcMain.handle('list-first-frame', (event, folderPath) => {
  const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']);
  try {
    const entries = fs.readdirSync(folderPath).sort();
    for (const name of entries) {
      if (IMAGE_EXTS.has(path.extname(name).toLowerCase())) {
        return path.join(folderPath, name);
      }
    }
    return null;
  } catch {
    return null;
  }
});

// ── IPC: Open external URL in browser ────────────────────────────────────────
ipcMain.handle('open-external', (_, url) => shell.openExternal(url));

// ── IPC: Download update DMG → ~/Downloads, then open it ─────────────────────
ipcMain.handle('download-update', (event, { downloadUrl, assetName }) => {
  const dest = path.join(os.homedir(), 'Downloads', assetName);
  const file = fs.createWriteStream(dest);

  return new Promise((resolve, reject) => {
    function fetch(url) {
      https.get(url, { headers: { 'User-Agent': 'Perforation-Stabilizer-Updater' } }, (res) => {
        // Follow redirects (GitHub assets redirect to CDN)
        if (res.statusCode === 301 || res.statusCode === 302) {
          return fetch(res.headers.location);
        }
        const total = parseInt(res.headers['content-length'] || '0', 10);
        let received = 0;
        res.on('data', (chunk) => {
          received += chunk.length;
          if (total > 0) event.sender.send('update-download-progress', received / total);
        });
        res.pipe(file);
        file.on('finish', () => file.close(() => {
          shell.openPath(dest);
          resolve({ path: dest });
        }));
      }).on('error', (err) => {
        fs.unlink(dest, () => {});
        reject(err.message);
      });
    }
    fetch(downloadUrl);
  });
});
