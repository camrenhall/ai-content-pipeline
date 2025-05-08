// This file should be placed in the src/electron directory
// It is responsible for securely exposing Electron's IPC functionality to the renderer process

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'electronAPI', {
    // Python script execution - now handling stdout/stderr callbacks directly
    runPythonScript: (options) => {
      // Store callback references
      const onStdout = options.onStdout;
      const onStderr = options.onStderr;
      
      // Remove callbacks from options before sending to main process
      // (they can't be serialized for IPC)
      const ipcOptions = { ...options };
      delete ipcOptions.onStdout;
      delete ipcOptions.onStderr;
      
      // Generate a unique callback ID for this request
      const callbackId = Date.now().toString();
      
      // Set up temporary listeners for this specific call
      if (onStdout) {
        const stdoutListener = (event, data) => {
          if (data.callbackId === callbackId) {
            onStdout(data.data);
          }
        };
        ipcRenderer.on('python-stdout-direct', stdoutListener);
        
        // Clean up listener after process completes
        setTimeout(() => {
          ipcRenderer.removeListener('python-stdout-direct', stdoutListener);
        }, 10000); // Remove after 10 seconds to avoid memory leaks
      }
      
      if (onStderr) {
        const stderrListener = (event, data) => {
          if (data.callbackId === callbackId) {
            onStderr(data.data);
          }
        };
        ipcRenderer.on('python-stderr-direct', stderrListener);
        
        // Clean up listener after process completes
        setTimeout(() => {
          ipcRenderer.removeListener('python-stderr-direct', stderrListener);
        }, 10000); // Remove after 10 seconds to avoid memory leaks
      }
      
      // Send the request to main process with callbackId
      return ipcRenderer.invoke('run-python-script', { ...ipcOptions, callbackId });
    },
    cancelPythonProcess: (processId) => {
      return ipcRenderer.invoke('cancel-python-process', processId);
    },
    onPythonStdout: (callback) => {
      ipcRenderer.on('python-stdout', (event, data) => callback(data));
    },
    onPythonStderr: (callback) => {
      ipcRenderer.on('python-stderr', (event, data) => callback(data));
    },
    
    // File system operations
    readFile: (filePath) => {
      return ipcRenderer.invoke('read-file', filePath);
    },
    writeFile: (filePath, content) => {
      return ipcRenderer.invoke('write-file', { filePath, content });
    },
    ensureDirectory: (dirPath) => {
      return ipcRenderer.invoke('ensure-directory', dirPath);
    }
  }
);