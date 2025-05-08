import React, { useState } from 'react';
import { usePipeline } from '../context/PipelineContext';
import Card from '../components/Card';
import Button from '../components/Button';
import TextInput from '../components/TextInput';
import Slider from '../components/Slider';
import Dropdown from '../components/Dropdown';
import { generateScript } from '../utils/pythonRunner';

interface ScriptInputProps {
  onNext: () => void;
}

const ScriptInput: React.FC<ScriptInputProps> = ({ onNext }) => {
  const { state, updateState } = usePipeline();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedScript, setGeneratedScript] = useState('');
  const [mode, setMode] = useState<'prompt' | 'direct' | 'file'>('prompt');

  // For prompt-based generation
  const durationOptions = [
    { value: '10', label: '10 seconds' },
    { value: '15', label: '15 seconds' },
    { value: '20', label: '20 seconds' },
    { value: '25', label: '25 seconds' },
    { value: '30', label: '30 seconds' },
    { value: '35', label: '35 seconds' },
    { value: '40', label: '40 seconds' },
  ];

  // Generate script from prompt
  const handleGenerateScript = async () => {
    try {
      setIsGenerating(true);
      const script = await generateScript(
        state.prompt,
        state.duration,
        `${state.cacheDir}/scripts/generated_script.txt`
      );
      setGeneratedScript(script);
      updateState({ scriptText: script });
    } catch (error) {
      console.error('Error generating script:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  // Load script from file
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setGeneratedScript(content);
      updateState({ scriptText: content });
    };
    reader.readAsText(file);
  };

  const handleNext = () => {
    // Save current state and proceed to next step
    updateState({ 
      scriptText: mode === 'prompt' ? generatedScript : state.scriptText
    });
    onNext();
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Video Script</h2>
      
      {/* Mode selection */}
      <div className="mb-6">
        <div className="flex space-x-4">
          <Button 
            variant={mode === 'prompt' ? 'primary' : 'outline'} 
            onClick={() => setMode('prompt')}
          >
            Generate from Prompt
          </Button>
          <Button 
            variant={mode === 'direct' ? 'primary' : 'outline'} 
            onClick={() => setMode('direct')}
          >
            Write Directly
          </Button>
          <Button 
            variant={mode === 'file' ? 'primary' : 'outline'} 
            onClick={() => setMode('file')}
          >
            Upload Script File
          </Button>
        </div>
      </div>
      
      {/* Prompt-based generation */}
      {mode === 'prompt' && (
        <Card title="Generate Script from Prompt">
          <div className="space-y-4">
            <TextInput
              id="prompt"
              label="Prompt"
              value={state.prompt}
              onChange={(value) => updateState({ prompt: value })}
              placeholder="Enter a natural language prompt describing your video content"
              multiline
              rows={4}
            />
            
            <div className="grid grid-cols-2 gap-4">
              <Dropdown
                id="duration"
                label="Duration"
                value={state.duration.toString()}
                onChange={(value) => updateState({ duration: parseInt(value) })}
                options={durationOptions}
              />
              
              <Slider
                id="temperature"
                label="Creativity"
                value={state.temperature}
                onChange={(value) => updateState({ temperature: value })}
                min={0.1}
                max={1.0}
                step={0.1}
                formatValue={(value) => `${value.toFixed(1)}`}
              />
            </div>
            
            <div className="flex justify-end">
              <Button 
                variant="primary" 
                onClick={handleGenerateScript}
                disabled={isGenerating || !state.prompt}
              >
                {isGenerating ? 'Generating...' : 'Generate Script'}
              </Button>
            </div>
            
            {generatedScript && (
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900 mb-2">Generated Script</h3>
                <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
                  <p className="whitespace-pre-wrap">{generatedScript}</p>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}
      
      {/* Direct script writing */}
      {mode === 'direct' && (
        <Card title="Write Script Directly">
          <div className="space-y-4">
            <TextInput
              id="script-text"
              label="Script"
              value={state.scriptText}
              onChange={(value) => updateState({ scriptText: value })}
              placeholder="Enter your script text here"
              multiline
              rows={8}
            />
          </div>
        </Card>
      )}
      
      {/* File upload */}
      {mode === 'file' && (
        <Card title="Upload Script File">
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-md p-6">
              <div className="text-center">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                </svg>
                <div className="mt-2">
                  <label htmlFor="file-upload" className="cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none">
                    <span>Upload a script file</span>
                    <input 
                      id="file-upload" 
                      name="file-upload" 
                      type="file" 
                      className="sr-only" 
                      accept=".txt,.md" 
                      onChange={handleFileUpload}
                    />
                  </label>
                </div>
                <p className="mt-1 text-sm text-gray-500">.txt or .md file</p>
              </div>
            </div>
            
            {state.scriptText && (
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900 mb-2">Uploaded Script</h3>
                <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
                  <p className="whitespace-pre-wrap">{state.scriptText}</p>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}
      
      {/* Navigation buttons */}
      <div className="mt-8 flex justify-end">
        <Button 
          variant="primary" 
          onClick={handleNext}
          disabled={!state.scriptText && !generatedScript}
        >
          Continue to Avatar Selection
        </Button>
      </div>
    </div>
  );
};

export default ScriptInput;