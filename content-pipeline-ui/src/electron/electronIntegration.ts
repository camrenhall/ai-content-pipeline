// This file should be placed in the src/electron directory
// It contains the main process code for Electron integration

import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import * as isDev from 'electron-is-dev';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';

// Store active Python processes
const activePythonProcesses: Map<string, ChildProcess> = new Map();

// Create the main application window
function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  // Load the React app
  if (isDev) {
    // Load from dev server in development
    mainWindow.loadURL('http://localhost:3000');
    // Open DevTools
    mainWindow.webContents.openDevTools();
  } else {
    // Load from build directory in production
    mainWindow.loadFile(path.join(__dirname, '../build/index.html'));
  }
}

// Initialize app when ready
app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    // On macOS, re-create a window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// Generate a unique process ID
function generateProcessId(): string {
  return `python-process-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
}

// Run Python script and return a process ID
ipcMain.handle('run-python-script', async (event, options: {
  scriptPath: string;
  args?: string[];
  env?: Record<string, string>;
  callbackId?: string;
}) => {
  try {
    const { scriptPath, args = [], env = {}, callbackId } = options;
    const processId = generateProcessId();
    
    console.log(`Starting Python script: ${scriptPath} with args:`, args);
    
    // Combine process env with custom env
    const processEnv = { ...process.env, ...env };
    
    // Resolve the Python script path based on development or production mode
    const appPath = app.getAppPath();
    const pythonScriptPath = isDev 
      ? path.join(appPath, '..', scriptPath.replace('../', '')) // Dev mode - direct parent access
      : path.join(process.resourcesPath, 'python', scriptPath.replace('../', '')); // Production - from resources
    
    // Spawn the Python process
    const pythonProcess = spawn('python', [pythonScriptPath, ...args], {
      env: processEnv
    });
    
    // Store the process for later reference
    activePythonProcesses.set(processId, pythonProcess);
    
    // Set up event handlers
    let stdoutData = '';
    let stderrData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      stdoutData += output;
      
      // Send stdout data to renderer via both channels
      event.sender.send('python-stdout', { processId, data: output });
      
      // Send to direct callback channel if callbackId is provided
      if (callbackId) {
        event.sender.send('python-stdout-direct', { callbackId, data: output });
      }
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString();
      stderrData += output;
      
      // Send stderr data to renderer via both channels
      event.sender.send('python-stderr', { processId, data: output });
      
      // Send to direct callback channel if callbackId is provided
      if (callbackId) {
        event.sender.send('python-stderr-direct', { callbackId, data: output });
      }
    });
    
    // Handle process exit
    return new Promise((resolve) => {
      pythonProcess.on('close', (code) => {
        console.log(`Python process ${processId} exited with code ${code}`);
        // Clean up process reference
        activePythonProcesses.delete(processId);
        // Send result back to renderer
        resolve({
          processId,
          exitCode: code,
          stdout: stdoutData,
          stderr: stderrData
        });
      });
    });
  } catch (error) {
    console.error('Error running Python script:', error);
    throw error;
  }
});

// Cancel a running Python process
ipcMain.handle('cancel-python-process', async (event, processId: string) => {
  try {
    const process = activePythonProcesses.get(processId);
    
    if (!process) {
      return { success: false, error: 'Process not found' };
    }
    
    // Kill the process
    process.kill();
    activePythonProcesses.delete(processId);
    
    return { success: true };
  } catch (error) {
    console.error('Error cancelling Python process:', error);
    return { success: false, error: String(error) };
  }
});

// File system operations
ipcMain.handle('read-file', async (event, filePath: string) => {
  try {
    const data = await fs.promises.readFile(filePath, 'utf-8');
    return { success: true, data };
  } catch (error) {
    console.error('Error reading file:', error);
    return { success: false, error: String(error) };
  }
});

ipcMain.handle('write-file', async (event, options: { filePath: string; content: string }) => {
  try {
    const { filePath, content } = options;
    // Ensure directory exists
    const dirname = path.dirname(filePath);
    await fs.promises.mkdir(dirname, { recursive: true });
    
    await fs.promises.writeFile(filePath, content, 'utf-8');
    return { success: true };
  } catch (error) {
    console.error('Error writing file:', error);
    return { success: false, error: String(error) };
  }
});

ipcMain.handle('ensure-directory', async (event, dirPath: string) => {
  try {
    await fs.promises.mkdir(dirPath, { recursive: true });
    return { success: true };
  } catch (error) {
    console.error('Error creating directory:', error);
    return { success: false, error: String(error) };
  }
});