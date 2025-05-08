// pythonRunner.ts
// This utility handles communication with Python scripts via Electron's IPC mechanism

import { useState, useCallback } from 'react';

// Define the Electron API interface for TypeScript
declare global {
    interface Window {
      electronAPI: {
        runPythonScript: (options: {
          scriptPath: string;
          args?: string[];
          env?: Record<string, string>;
          onStdout?: (data: string) => void;  // Add this line
          onStderr?: (data: string) => void;  // Add this line
        }) => Promise<{
          processId: string;
          exitCode: number;
          stdout: string;
          stderr: string;
        }>;
        cancelPythonProcess: (processId: string) => Promise<{
          success: boolean;
          error?: string;
        }>;
        onPythonStdout: (callback: (data: { processId: string; data: string }) => void) => void;
        onPythonStderr: (callback: (data: { processId: string; data: string }) => void) => void;
        readFile: (filePath: string) => Promise<{
          success: boolean;
          data?: string;
          error?: string;
        }>;
        writeFile: (filePath: string, content: string) => Promise<{
          success: boolean;
          error?: string;
        }>;
        ensureDirectory: (dirPath: string) => Promise<{
          success: boolean;
          error?: string;
        }>;
      };
    }
  }

interface PythonRunnerOptions {
  scriptPath: string;
  args?: string[];
  env?: Record<string, string>;
  onStdout?: (data: string) => void;
  onStderr?: (data: string) => void;
  onError?: (error: Error) => void;
  onExit?: (code: number) => void;
}

interface PipelineOptions {
  input?: string;
  output: string;
  prompt?: string;
  script?: string;
  profile?: string;
  cacheDir?: string;
  avatarId?: string;
  talkingPhotoId?: string;
  voiceId?: string;
  backgroundUrl?: string;
  backgroundColor?: string;
  landscapeAvatar?: boolean;
  postProcessingSteps?: string[];
}

// Store active process IDs
let activeProcessId: string | null = null;

// Check if we're running in Electron
const isElectron = !!window.electronAPI;

// Run any Python script
export async function runPythonScript(options: PythonRunnerOptions): Promise<{ exitCode: number }> {
  if (!isElectron) {
    console.warn('Running in browser mode - Python script execution is mocked');
    return mockRunPythonScript(options);
  }
  
  try {
    const { scriptPath, args = [], env = {}, onStdout, onStderr, onExit } = options;
    
    // Set up stdout/stderr listeners if provided
    if (onStdout) {
      const stdoutHandler = (data: { processId: string; data: string }) => {
        if (data.processId === activeProcessId) {
          onStdout(data.data);
        }
      };
      
      window.electronAPI.onPythonStdout(stdoutHandler);
    }
    
    if (onStderr) {
      const stderrHandler = (data: { processId: string; data: string }) => {
        if (data.processId === activeProcessId) {
          onStderr(data.data);
        }
      };
      
      window.electronAPI.onPythonStderr(stderrHandler);
    }
    
    // Run the Python script
    const result = await window.electronAPI.runPythonScript({
      scriptPath,
      args,
      env
    });
    
    activeProcessId = result.processId;
    
    // Call exit handler if provided
    if (onExit) {
      onExit(result.exitCode);
    }
    
    return { exitCode: result.exitCode };
  } catch (error) {
    console.error('Error running Python script:', error);
    if (options.onError) {
      options.onError(error as Error);
    }
    throw error;
  }
}

// Generate a script from a prompt
// Update generateScript in pythonRunner.ts

export async function generateScript(prompt: string, duration: number, output: string): Promise<string> {
    if (!isElectron) {
      console.warn('Running in browser mode - Script generation is mocked');
      return mockGenerateScript(prompt, duration, output);
    }
    
    try {
      // Create scripts directory if it doesn't exist
      const scriptDir = output.substring(0, output.lastIndexOf('/'));
      await window.electronAPI.ensureDirectory(scriptDir);
      
      // Run the script_generator.py script
      const result = await window.electronAPI.runPythonScript({
        scriptPath: '../script_generator.py',
        args: [
          prompt,
          '--output', output,
          '--duration', duration.toString(),
        ],
        env: {
          // Add any necessary environment variables, like API keys
          LLM_API_KEY: process.env.REACT_APP_LLM_API_KEY || '',
        }
      });
      
      if (result.exitCode !== 0) {
        throw new Error('Script generation failed');
      }
      
      // Read the generated script file
      const fileResult = await window.electronAPI.readFile(output);
      if (!fileResult.success) {
        throw new Error(`Failed to read script file: ${fileResult.error}`);
      }
      
      return fileResult.data || '';
    } catch (error) {
      console.error('Error generating script:', error);
      throw error;
    }
  }

// Generate a video from a script using HeyGen
// Update generateVideo in pythonRunner.ts
export async function generateVideo(
    scriptText: string, 
    output: string,
    avatarId?: string,
    talkingPhotoId?: string,
    voiceId?: string,
    options?: {
      backgroundUrl?: string;
      backgroundColor?: string;
      voiceEmotion?: string;
      voiceSpeed?: number;
      avatarScale?: number;
      avatarOffsetX?: number;
      avatarOffsetY?: number;
      avatarStyle?: string;
      landscapeAvatar?: boolean;
      elevenLabsEnabled?: boolean;
      elevenLabsModel?: string;
      elevenLabsStability?: number;
      elevenLabsSimilarity?: number;
      elevenLabsStyle?: number;
    }
  ): Promise<string> {
    if (!isElectron) {
      console.warn('Running in browser mode - Video generation is mocked');
      return mockGenerateVideo(scriptText, output, avatarId, talkingPhotoId, voiceId, options);
    }
    
    try {
      // Create a temporary script file
      const scriptPath = '../cache/scripts/temp_script.txt';
      await window.electronAPI.ensureDirectory('../cache/scripts');
      await window.electronAPI.writeFile(scriptPath, scriptText);
      
      // Create videos directory if it doesn't exist
      const videoDir = output.substring(0, output.lastIndexOf('/'));
      await window.electronAPI.ensureDirectory(videoDir);
      
      // Prepare arguments for heygen_client.py
      const args = [
        '--script', scriptPath,
        '--output', output,
      ];
      
      // Add avatar or talking photo ID
      if (avatarId) {
        args.push('--avatar-id', avatarId);
      } else if (talkingPhotoId) {
        args.push('--talking-photo-id', talkingPhotoId);
      }
      
      // Add voice ID
      if (voiceId) {
        args.push('--voice-id', voiceId);
      }
      
      // Add optional parameters
      if (options) {
        if (options.backgroundUrl) {
          args.push('--background-url', options.backgroundUrl);
        }
        
        if (options.backgroundColor) {
          args.push('--background-color', options.backgroundColor);
        }
        
        if (options.voiceEmotion) {
          args.push('--voice-emotion', options.voiceEmotion);
        }
        
        if (options.voiceSpeed) {
          args.push('--voice-speed', options.voiceSpeed.toString());
        }
        
        if (options.avatarScale) {
          args.push('--avatar-scale', options.avatarScale.toString());
        }
        
        if (options.avatarOffsetX) {
          args.push('--avatar-offset-x', options.avatarOffsetX.toString());
        }
        
        if (options.avatarOffsetY) {
          args.push('--avatar-offset-y', options.avatarOffsetY.toString());
        }
        
        if (options.avatarStyle) {
          args.push('--avatar-style', options.avatarStyle);
        }
        
        if (options.landscapeAvatar) {
          args.push('--landscape-avatar');
        }
        
        if (options.elevenLabsEnabled) {
          args.push('--use-elevenlabs');
          
          if (options.elevenLabsModel) {
            args.push('--elevenlabs-model', options.elevenLabsModel);
          }
          
          if (options.elevenLabsStability) {
            args.push('--elevenlabs-stability', options.elevenLabsStability.toString());
          }
          
          if (options.elevenLabsSimilarity) {
            args.push('--elevenlabs-similarity', options.elevenLabsSimilarity.toString());
          }
          
          if (options.elevenLabsStyle) {
            args.push('--elevenlabs-style', options.elevenLabsStyle.toString());
          }
        }
      }
      
      // Run heygen_client.py
      const result = await window.electronAPI.runPythonScript({
        scriptPath: '../heygen_client.py',
        args,
        env: {
          HEYGEN_API_KEY: process.env.REACT_APP_HEYGEN_API_KEY || '',
          ELEVENLABS_API_KEY: options?.elevenLabsEnabled ? (process.env.REACT_APP_ELEVENLABS_API_KEY || '') : '',
        }
      });
      
      if (result.exitCode !== 0) {
        throw new Error('Video generation failed');
      }
      
      return output;
    } catch (error) {
      console.error('Error generating video:', error);
      throw error;
    }
  }

// Run the complete pipeline
// Replace the runPipeline function in pythonRunner.ts

export async function runPipeline(
    options: PipelineOptions,
    onProgress?: (step: string, progress: number) => void,
    onLog?: (message: string) => void
  ): Promise<{ success: boolean; outputPath: string }> {
    if (!isElectron) {
      console.warn('Running in browser mode - Pipeline execution is mocked');
      return mockRunPipeline(options, onProgress, onLog);
    }
    
    try {
      // Create output directory if it doesn't exist
      const outputDir = options.output.substring(0, options.output.lastIndexOf('/'));
      await window.electronAPI.ensureDirectory(outputDir);
      
      // Prepare arguments for pipeline_orchestrator.py
      const args = [
        '--output', options.output,
      ];
      
      if (options.input) {
        args.push('--input', options.input);
      }
      
      if (options.script) {
        // Create a temporary script file
        const scriptPath = '../cache/scripts/temp_script.txt';
        await window.electronAPI.ensureDirectory('../cache/scripts');
        await window.electronAPI.writeFile(scriptPath, options.script);
        args.push('--script', scriptPath);
      }
      
      if (options.profile) {
        args.push('--profile', options.profile);
      }
      
      if (options.cacheDir) {
        args.push('--cache-dir', options.cacheDir);
      }
      
      if (options.avatarId) {
        args.push('--avatar-id', options.avatarId);
      }
      
      if (options.talkingPhotoId) {
        args.push('--talking-photo-id', options.talkingPhotoId);
      }
      
      if (options.voiceId) {
        args.push('--voice-id', options.voiceId);
      }
      
      if (options.backgroundUrl) {
        args.push('--background-url', options.backgroundUrl);
      }
      
      if (options.backgroundColor) {
        args.push('--background-color', options.backgroundColor);
      }
      
      if (options.landscapeAvatar) {
        args.push('--landscape-avatar');
      }
      
      if (options.postProcessingSteps && options.postProcessingSteps.length > 0) {
        args.push('--post-process-steps', options.postProcessingSteps.join(','));
      }
      
      // Set up stdout handler to parse progress updates
      const stdoutHandler = (data: string) => {
        // Log all output
        if (onLog) {
          onLog(`[${new Date().toLocaleTimeString()}] ${data.trim()}`);
        }
        
        // Parse progress information
        // Assuming the Python script outputs progress in a format like:
        // [PROGRESS] Step: Generating script... (20%)
        const progressMatch = data.match(/\[PROGRESS\]\s+Step:\s+(.*?)\s+\((\d+)%\)/);
        if (progressMatch && progressMatch.length >= 3) {
          const step = progressMatch[1];
          const progress = parseInt(progressMatch[2], 10);
          
          if (onProgress) {
            onProgress(step, progress);
          }
        }
      };
      
      // Run pipeline_orchestrator.py
      const result = await window.electronAPI.runPythonScript({
        scriptPath: '../pipeline_orchestrator.py',
        args,
        env: {
          ASSEMBLYAI_API_KEY: process.env.REACT_APP_ASSEMBLYAI_API_KEY || '',
          LLM_API_KEY: process.env.REACT_APP_LLM_API_KEY || '',
          PEXELS_API_KEY: process.env.REACT_APP_PEXELS_API_KEY || '',
          HEYGEN_API_KEY: process.env.REACT_APP_HEYGEN_API_KEY || '',
          ELEVENLABS_API_KEY: process.env.REACT_APP_ELEVENLABS_API_KEY || '',
        },
        onStdout: stdoutHandler,
        onStderr: (data: string) => {
          if (onLog) {
            onLog(`[ERROR] ${data.trim()}`);
          }
        }
      });
      
      if (result.exitCode !== 0) {
        throw new Error('Pipeline execution failed');
      }
      
      return {
        success: true,
        outputPath: options.output
      };
    } catch (error) {
      console.error('Error running pipeline:', error);
      return {
        success: false,
        outputPath: ''
      };
    }
  }

// Cancel a running pipeline
export async function cancelPipeline(): Promise<boolean> {
  if (!isElectron) {
    console.warn('Running in browser mode - Pipeline cancellation is mocked');
    return true;
  }
  
  try {
    if (!activeProcessId) {
      return false;
    }
    
    const result = await window.electronAPI.cancelPythonProcess(activeProcessId);
    return result.success;
  } catch (error) {
    console.error('Error cancelling pipeline:', error);
    return false;
  }
}

// Mock implementations for browser testing
async function mockRunPythonScript(options: PythonRunnerOptions): Promise<{ exitCode: number }> {
  console.log('Mock: Running Python script:', options.scriptPath, 'with args:', options.args);
  
  return new Promise((resolve) => {
    // Simulate script execution time
    setTimeout(() => {
      // Simulate stdout data
      if (options.onStdout) {
        options.onStdout(`Running ${options.scriptPath}...`);
        options.onStdout('Processing...');
        options.onStdout('Completed successfully.');
      }
      
      // Simulate successful exit
      if (options.onExit) {
        options.onExit(0);
      }
      
      resolve({ exitCode: 0 });
    }, 1000);
  });
}

async function mockGenerateScript(prompt: string, duration: number, output: string): Promise<string> {
    console.log('Mock: Generating script for prompt:', prompt, 'with duration:', duration);
    
    return new Promise((resolve) => {
      setTimeout(() => {
        const mockScript = `Here's a script about ${prompt} that's ${duration} seconds long. This would be generated by the actual Python script in a real implementation.`;
        resolve(mockScript);
      }, 1500);
    });
  }

  async function mockGenerateVideo(
    scriptText: string, 
    output: string,
    avatarId?: string,
    talkingPhotoId?: string,
    voiceId?: string,
    options?: any
  ): Promise<string> {
    console.log('Mock: Generating video for script:', scriptText.substring(0, 50) + '...');
    
    return new Promise((resolve) => {
      setTimeout(() => {
        // Use correct relative path
        const mockVideoPath = output || '../cache/videos/generated_video.mp4';
        resolve(mockVideoPath);
      }, 2000);
    });
  }

  async function mockRunPipeline(
    options: PipelineOptions,
    onProgress?: (step: string, progress: number) => void,
    onLog?: (message: string) => void
  ): Promise<{ success: boolean; outputPath: string }> {
    console.log('Mock: Running pipeline with options:', options);
    
    const steps = [
      'Initializing pipeline...',
      'Generating script...',
      'Creating avatar video...',
      'Analyzing script for B-roll opportunities...',
      'Extracting keywords...',
      'Retrieving video assets...',
      'Transforming videos...',
      'Assembling video...',
      'Applying post-processing...',
      'Pipeline completed successfully!'
    ];
    
    return new Promise((resolve) => {
      let stepIndex = 0;
      
      const processStep = () => {
        if (stepIndex < steps.length) {
          const step = steps[stepIndex];
          const progress = (stepIndex / (steps.length - 1)) * 100;
          
          if (onProgress) {
            onProgress(step, progress);
          }
          
          if (onLog) {
            onLog(`[${new Date().toLocaleTimeString()}] ${step}`);
          }
          
          stepIndex++;
          setTimeout(processStep, 1000);
        } else {
          resolve({
            success: true,
            outputPath: options.output || '../output/final_video.mp4' // Use correct relative path
          });
        }
      };
      
      processStep();
    });
  }

// Hook for using the Python bridge


export function usePythonBridge() {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  
  const runPipelineWithProgress = useCallback(async (options: PipelineOptions) => {
    setIsRunning(true);
    setProgress(0);
    setCurrentStep('');
    setLogs([]);
    
    try {
      const result = await runPipeline(
        options,
        (step, progress) => {
          setCurrentStep(step);
          setProgress(progress);
        },
        (log) => {
          setLogs((prevLogs) => [...prevLogs, log]);
        }
      );
      
      return result;
    } catch (error) {
      console.error('Pipeline error:', error);
      setLogs((prevLogs) => [...prevLogs, `Error: ${error}`]);
      return { success: false, outputPath: '' };
    } finally {
      setIsRunning(false);
    }
  }, []);
  
  const cancel = useCallback(async () => {
    const result = await cancelPipeline();
    if (result) {
      setIsRunning(false);
      setLogs((prevLogs) => [...prevLogs, 'Pipeline cancelled by user.']);
    }
    return result;
  }, []);
  
  return {
    isRunning,
    progress,
    currentStep,
    logs,
    runPipeline: runPipelineWithProgress,
    cancelPipeline: cancel,
    generateScript,
    generateVideo
  };
}