import React, { createContext, useContext, useState, useEffect } from 'react';

// Types for our pipeline state
export interface PipelineState {
  // Script Generation
  prompt: string;
  scriptText: string;
  duration: number;
  temperature: number;
  
  // Avatar Selection
  avatarId: string | null;
  talkingPhotoId: string | null;
  voiceId: string | null;
  voiceEmotion: string;
  voiceSpeed: number;
  avatarScale: number;
  avatarOffsetX: number;
  avatarOffsetY: number;
  avatarStyle: 'normal' | 'closeUp' | 'circle';
  landscapeAvatar: boolean;
  backgroundUrl: string;
  backgroundColor: string;
  caption: boolean;
  elevenLabsEnabled: boolean;
  elevenLabsModel: string;
  elevenLabsStability: number;
  elevenLabsSimilarity: number;
  elevenLabsStyle: number;
  
  // Pipeline Configuration
  selectedProfile: string;
  configSections: {
    [key: string]: boolean; // Whether each section is enabled
  };
  
  // Input/Output Paths
  inputVideoPath: string | null;
  outputVideoPath: string | null;
  cacheDir: string;
  
  // Execution
  isRunning: boolean;
  progress: number;
  currentStep: string;
  logOutput: string[];
}

// Define the initial state
const initialState: PipelineState = {
  // Script Generation
  prompt: '',
  scriptText: '',
  duration: 20,
  temperature: 0.7,
  
  // Avatar Selection
  avatarId: null,
  talkingPhotoId: null,
  voiceId: null,
  voiceEmotion: 'Friendly',
  voiceSpeed: 1.0,
  avatarScale: 1.0,
  avatarOffsetX: 0,
  avatarOffsetY: 0,
  avatarStyle: 'normal',
  landscapeAvatar: false,
  backgroundUrl: '',
  backgroundColor: '#f6f6fc',
  caption: false,
  elevenLabsEnabled: false,
  elevenLabsModel: 'eleven_turbo_v2',
  elevenLabsStability: 0.5,
  elevenLabsSimilarity: 0.75,
  elevenLabsStyle: 0.0,
  
  // Pipeline Configuration
  selectedProfile: 'default',
  configSections: {
    'sound_effects': true,
    'background_music': true,
    'camera_movements': false,
    'transitions': false,
    'captioning': false
  },
  
  // Input/Output Paths
  inputVideoPath: null,
  outputVideoPath: null,
  cacheDir: './cache',
  
  // Execution
  isRunning: false,
  progress: 0,
  currentStep: '',
  logOutput: []
};

// Create the context
interface PipelineContextType {
  state: PipelineState;
  updateState: (updates: Partial<PipelineState>) => void;
  resetState: () => void;
  runPipeline: () => Promise<void>;
  stopPipeline: () => void;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

// Provider component
export const PipelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<PipelineState>(() => {
    // Try to load saved state from localStorage
    const savedState = localStorage.getItem('pipelineState');
    return savedState ? { ...initialState, ...JSON.parse(savedState) } : initialState;
  });

  // Save state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('pipelineState', JSON.stringify(state));
  }, [state]);

  // Update state partially
  const updateState = (updates: Partial<PipelineState>) => {
    setState(prevState => ({ ...prevState, ...updates }));
  };

  // Reset to initial state
  const resetState = () => {
    setState(initialState);
  };

  // Function to run the pipeline
  const runPipeline = async () => {
    try {
      updateState({ isRunning: true, progress: 0, logOutput: [] });
      
      // In a real implementation, this would call the Python scripts via Electron's IPC
      // For now, we'll simulate the pipeline execution
      
      // Mock pipeline steps
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
      
      // Simulate pipeline execution
      for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        updateState({ 
          currentStep: step, 
          progress: (i / (steps.length - 1)) * 100,
          logOutput: [...state.logOutput, step]
        });
        
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      // Set final state
      updateState({ 
        isRunning: false, 
        progress: 100,
        outputVideoPath: './output/final_video.mp4'  // This would be the actual output path
      });
      
    } catch (error) {
      console.error("Pipeline execution failed:", error);
      updateState({ 
        isRunning: false,
        logOutput: [...state.logOutput, `Error: ${error}`]
      });
    }
  };

  // Function to stop the pipeline
  const stopPipeline = () => {
    // In a real implementation, this would signal the Python process to stop
    updateState({ isRunning: false });
  };

  return (
    <PipelineContext.Provider value={{ 
      state, 
      updateState, 
      resetState,
      runPipeline,
      stopPipeline
    }}>
      {children}
    </PipelineContext.Provider>
  );
};

// Custom hook to use the pipeline context
export const usePipeline = () => {
  const context = useContext(PipelineContext);
  if (context === undefined) {
    throw new Error('usePipeline must be used within a PipelineProvider');
  }
  return context;
};