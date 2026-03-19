'use strict';

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let pyProcess = null;

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
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

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

  // In production the script lives in extraResources/src/
  const scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, 'src', 'stabilizer_cli.py')
    : path.join(__dirname, '..', 'src', 'stabilizer_cli.py');

  const args = [
    scriptPath,
    '--input',     opts.input,
    '--output',    opts.output,
    '--roi',       String(opts.roi),
    '--threshold', String(opts.threshold),
    '--smooth',    String(opts.smooth),
    '--quality',   String(opts.quality),
  ];

  pyProcess = spawn('python3', args);

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
