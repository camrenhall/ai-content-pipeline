import React, { useState, useEffect, useRef } from 'react';
import { usePipeline } from '../context/PipelineContext';
import Card from '../components/Card';
import Button from '../components/Button';
import ProgressBar from '../components/ProgressBar';
import { usePythonBridge } from '../utils/pythonRunner';

interface ExecutionSectionProps {
  onBack: () => void;
}

const ExecutionSection: React.FC<ExecutionSectionProps> = ({ onBack }) => {
  const { state, updateState } = usePipeline();
  const pythonBridge = usePythonBridge();
  const [activeTab, setActiveTab] = useState<'execution' | 'logs' | 'preview'>('execution');
  const logEndRef = useRef<HTMLDivElement>(null);
  
  // Scroll to bottom of logs when they update
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.logOutput]);

  // Function to run the pipeline
  const handleRunPipeline = async () => {
    // Set up the post-processing steps based on enabled sections
    const postProcessingSteps = Object.entries(state.configSections)
      .filter(([_, enabled]) => enabled)
      .map(([key]) => key);
    
    try {
      updateState({ isRunning: true });
      
      // Prepare options for the pipeline
      const pipelineOptions = {
        input: state.inputVideoPath || undefined,
        output: state.outputVideoPath || `${state.cacheDir}/output/final_video.mp4`,
        script: state.scriptText,
        avatarId: state.avatarId || undefined,
        talkingPhotoId: state.talkingPhotoId || undefined,
        voiceId: state.voiceId || undefined,
        backgroundUrl: state.backgroundUrl || undefined,
        backgroundColor: state.backgroundColor,
        profile: state.selectedProfile,
        postProcessingSteps: postProcessingSteps,
        cacheDir: state.cacheDir
      };
      
      // Run the pipeline using the Python bridge
      const result = await pythonBridge.runPipeline(pipelineOptions);
      
      if (result.success) {
        updateState({ 
          isRunning: false,
          outputVideoPath: result.outputPath
        });
      } else {
        updateState({ isRunning: false });
      }
    } catch (error) {
      console.error('Failed to run pipeline:', error);
      updateState({ isRunning: false });
    }
  };

  // Function to stop the pipeline
  const handleStopPipeline = async () => {
    try {
      await pythonBridge.cancelPipeline();
      updateState({ isRunning: false });
    } catch (error) {
      console.error('Failed to stop pipeline:', error);
    }
  };

  // Create a pipeline overview list
  const pipelineSteps = [
    { 
      id: 'script', 
      name: 'Script Generation', 
      completed: !!state.scriptText,
      status: !!state.scriptText ? 'complete' : 'pending'
    },
    { 
      id: 'avatar', 
      name: 'Avatar Configuration', 
      completed: !!state.avatarId || !!state.talkingPhotoId,
      status: (!!state.avatarId || !!state.talkingPhotoId) ? 'complete' : 'pending'
    },
    {
      id: 'pipeline',
      name: 'Pipeline Configuration',
      completed: true, // Always considered completed as default values are provided
      status: 'complete'
    },
    {
      id: 'execution',
      name: 'Pipeline Execution',
      completed: !!state.outputVideoPath,
      status: state.isRunning ? 'in-progress' : (state.outputVideoPath ? 'complete' : 'pending')
    }
  ];

  // Format the render time
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Estimate the render time based on duration and enabled post-processing steps
  const estimatedRenderTime = () => {
    // Base time: 5 seconds + 2x the video duration
    let baseTime = 5 + (state.duration * 2);
    
    // Add time for each enabled post-processing step
    Object.entries(state.configSections).forEach(([key, enabled]) => {
      if (enabled) {
        switch (key) {
          case 'sound_effects':
            baseTime += 3;
            break;
          case 'background_music':
            baseTime += 2;
            break;
          case 'camera_movements':
            baseTime += 5;
            break;
          case 'transitions':
            baseTime += 4;
            break;
          case 'captioning':
            baseTime += 10;
            break;
        }
      }
    });
    
    return formatTime(baseTime);
  };

  return (
    <div className="max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Pipeline Execution</h2>
      
      {/* Tab navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('execution')}
            className={`${
              activeTab === 'execution'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
          >
            Pipeline Execution
          </button>
          <button
            onClick={() => setActiveTab('logs')}
            className={`${
              activeTab === 'logs'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
          >
            Logs
          </button>
          <button
            onClick={() => setActiveTab('preview')}
            className={`${
              activeTab === 'preview'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
            disabled={!state.outputVideoPath}
          >
            Preview Result
          </button>
        </nav>
      </div>
      
      {/* Execution tab */}
      {activeTab === 'execution' && (
        <div className="space-y-6">
          <Card title="Pipeline Overview">
            <div className="space-y-6">
              <ul className="space-y-3">
                {pipelineSteps.map((step, index) => (
                  <li key={step.id} className="relative flex items-start group">
                    <div className="flex items-center">
                      <span className="relative z-10 w-8 h-8 flex items-center justify-center bg-white border-2 border-gray-300 rounded-full group-hover:border-gray-400">
                        {step.status === 'complete' ? (
                          <svg className="w-5 h-5 text-indigo-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        ) : step.status === 'in-progress' ? (
                          <svg className="w-5 h-5 text-indigo-600 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                        ) : (
                          <span className="text-sm text-gray-500 font-medium">{index + 1}</span>
                        )}
                      </span>
                    </div>
                    <div className="ml-4 min-w-0 flex-1">
                      <div className="text-sm font-medium text-gray-900">{step.name}</div>
                      <div className="text-sm text-gray-500">
                        {step.status === 'complete' ? 'Completed' : step.status === 'in-progress' ? 'In progress' : 'Pending'}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
              
              <div className="flex justify-between items-center pt-4 border-t border-gray-200">
                <div>
                  <p className="text-sm text-gray-500">
                    Estimated processing time: <span className="font-medium">{estimatedRenderTime()}</span>
                  </p>
                </div>
                
                <div>
                  {state.isRunning ? (
                    <Button variant="danger" onClick={handleStopPipeline}>
                      Stop Pipeline
                    </Button>
                  ) : (
                    <Button 
                      variant="primary" 
                      onClick={handleRunPipeline}
                      disabled={!state.scriptText || !state.voiceId || (!state.avatarId && !state.talkingPhotoId)}
                    >
                      Run Pipeline
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </Card>
          
          {state.isRunning && (
            <Card title="Current Progress">
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-1">Current Step</h3>
                  <p className="text-sm text-gray-900">{state.currentStep || 'Initializing...'}</p>
                </div>
                
                <ProgressBar 
                  progress={state.progress} 
                  size="lg" 
                  color="indigo"
                />
              </div>
            </Card>
          )}
          
          {state.outputVideoPath && !state.isRunning && (
            <Card title="Pipeline Result">
              <div className="space-y-4">
                <div>
                  <p className="text-green-600 font-medium">
                    <svg className="inline-block w-5 h-5 mr-1 -mt-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Pipeline completed successfully!
                  </p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-1">Output Video Path</h3>
                  <p className="text-sm text-gray-900 bg-gray-50 p-2 rounded border border-gray-200 font-mono">
                    {state.outputVideoPath}
                  </p>
                </div>
                
                <div className="flex space-x-4 pt-2">
                  <Button variant="outline" onClick={() => setActiveTab('preview')}>
                    Preview Video
                  </Button>
                  <Button variant="primary">
                    Download Video
                  </Button>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}
      
      {/* Logs tab */}
      {activeTab === 'logs' && (
        <Card title="Pipeline Logs">
          <div className="h-96 overflow-y-auto bg-gray-50 p-4 rounded border border-gray-200 font-mono text-sm">
            {state.logOutput.length === 0 ? (
              <p className="text-gray-500">No logs available yet. Run the pipeline to see logs.</p>
            ) : (
              state.logOutput.map((log, index) => (
                <div key={index} className="pb-1">
                  <span className="text-gray-500">[{index + 1}]</span> {log}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </Card>
      )}
      
      {/* Preview tab */}
      {activeTab === 'preview' && (
        <Card title="Video Preview">
          {state.outputVideoPath ? (
            <div className="space-y-4">
              <div className="aspect-w-16 aspect-h-9 bg-black rounded overflow-hidden">
                <video 
                  controls 
                  className="w-full h-full"
                  src={state.outputVideoPath}
                >
                  Your browser does not support the video tag.
                </video>
              </div>
              
              <div className="flex justify-end space-x-4">
                <Button variant="outline">
                  Download Video
                </Button>
                <Button variant="primary">
                  Create Another Video
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900">No video available</h3>
              <p className="mt-1 text-sm text-gray-500">Run the pipeline to generate a video.</p>
              <div className="mt-6">
                <Button variant="primary" onClick={() => setActiveTab('execution')}>
                  Go to Pipeline Execution
                </Button>
              </div>
            </div>
          )}
        </Card>
      )}
      
      {/* Navigation buttons */}
      <div className="mt-8 flex justify-between">
        <Button variant="outline" onClick={onBack}>
          Back to Configuration
        </Button>
      </div>
    </div>
  );
};

export default ExecutionSection;