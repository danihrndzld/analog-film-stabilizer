'use strict';

const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('api', {
  /** Resolve a File object (from drag-drop) to its filesystem path */
  getPathForFile: (file) => webUtils.getPathForFile(file),

  /** Open a native folder-picker dialog; resolves to path string or null */
  openFolder: () => ipcRenderer.invoke('open-folder'),

  /** Start the Python stabilisation process */
  startProcess: (opts) => ipcRenderer.invoke('start-process', opts),

  /** Kill the running Python process */
  cancelProcess: () => ipcRenderer.invoke('cancel-process'),

  // ── Event listeners (renderer → subscribe) ──────────────────────────────
  onProgress: (cb) => ipcRenderer.on('process-progress', (_e, v)  => cb(v)),
  onLog:      (cb) => ipcRenderer.on('process-log',      (_e, m)  => cb(m)),
  onDone:     (cb) => ipcRenderer.on('process-done',     (_e, s)  => cb(s)),
  onError:    (cb) => ipcRenderer.on('process-error',    (_e, err) => cb(err)),

  /** Remove all listeners for a given channel (call on page unload) */
  off: (channel) => ipcRenderer.removeAllListeners(channel),
});
