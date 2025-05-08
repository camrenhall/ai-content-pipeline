import React, { useState } from 'react';
import { usePipeline } from '../context/PipelineContext';
import Card from '../components/Card';
import Button from '../components/Button';
import Slider from '../components/Slider';
import Dropdown from '../components/Dropdown';
import TextInput from '../components/TextInput';

interface ConfigurationSectionProps {
  onBack: () => void;
  onNext: () => void;
}

// Pipeline profile options
const profileOptions = [
  { value: 'default', label: 'Default' },
  { value: 'development', label: 'Development' },
  { value: 'production', label: 'Production' },
  { value: 'high_quality', label: 'High Quality' },
  { value: 'fast', label: 'Fast' },
];

// Transition type options
const transitionOptions = [
  { value: 'fade', label: 'Fade' },
  { value: 'dissolve', label: 'Dissolve' },
  { value: 'wipe', label: 'Wipe' },
  { value: 'slide', label: 'Slide' },
  { value: 'zoom', label: 'Zoom' },
];

const ConfigurationSection: React.FC<ConfigurationSectionProps> = ({ onBack, onNext }) => {
  const { state, updateState } = usePipeline();
  const [activeTab, setActiveTab] = useState<string>('profile');

  // Toggle section enabled/disabled
  const toggleSection = (sectionKey: string) => {
    updateState({
      configSections: {
        ...state.configSections,
        [sectionKey]: !state.configSections[sectionKey]
      }
    });
  };

  // Sound effects settings
  const [soundEffectsVolume, setSoundEffectsVolume] = useState<number>(0.5);
  const [enableTransitions, setEnableTransitions] = useState<boolean>(true);
  const [enableAmbient, setEnableAmbient] = useState<boolean>(true);

  // Background music settings
  const [musicVolume, setMusicVolume] = useState<number>(0.3);
  const [musicFadeIn, setMusicFadeIn] = useState<number>(2.0);
  const [musicFadeOut, setMusicFadeOut] = useState<number>(2.0);
  const [smartDucking, setSmartDucking] = useState<boolean>(true);

  // Camera movements settings
  const [zoomFactor, setZoomFactor] = useState<number>(1.2);
  const [shakeIntensity, setShakeIntensity] = useState<number>(0);
  const [punchinFactor, setPunchinFactor] = useState<number>(1.1);

  // Transitions settings
  const [transitionType, setTransitionType] = useState<string>('fade');
  const [transitionDuration, setTransitionDuration] = useState<number>(1.0);
  const [randomizeTransitions, setRandomizeTransitions] = useState<boolean>(false);

  // Captioning settings
  const [captioningApiUrl, setCaptioningApiUrl] = useState<string>('https://api.assemblyai.com/v2');

  // Render the active tab content
  const renderTabContent = () => {
    switch (activeTab) {
      case 'profile':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Pipeline Profile</h3>
              <p className="text-sm text-gray-500 mb-4">
                Choose a predefined profile to apply optimized settings for different use cases.
              </p>
              <Dropdown
                id="pipeline-profile"
                label="Profile"
                value={state.selectedProfile}
                onChange={(value) => updateState({ selectedProfile: value })}
                options={profileOptions}
              />
            </div>
          </div>
        );
      case 'sound_effects':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Sound Effects</h3>
              <div className="flex items-center">
                <span className="mr-3 text-sm text-gray-500">
                  {state.configSections.sound_effects ? 'Enabled' : 'Disabled'}
                </span>
                <button
                  type="button"
                  className={`${
                    state.configSections.sound_effects ? 'bg-indigo-600' : 'bg-gray-200'
                  } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
                  onClick={() => toggleSection('sound_effects')}
                >
                  <span
                    className={`${
                      state.configSections.sound_effects ? 'translate-x-5' : 'translate-x-0'
                    } inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                  />
                </button>
              </div>
            </div>
            
            {state.configSections.sound_effects && (
              <div className="space-y-4">
                <Slider
                  id="sound-effects-volume"
                  label="Volume"
                  value={soundEffectsVolume}
                  onChange={setSoundEffectsVolume}
                  min={0}
                  max={1}
                  step={0.1}
                  formatValue={(value) => `${Math.round(value * 100)}%`}
                />
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="flex items-center">
                    <input
                      id="enable-transitions"
                      type="checkbox"
                      checked={enableTransitions}
                      onChange={(e) => setEnableTransitions(e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    <label htmlFor="enable-transitions" className="ml-2 block text-sm text-gray-700">
                      Enable transition sound effects
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      id="enable-ambient"
                      type="checkbox"
                      checked={enableAmbient}
                      onChange={(e) => setEnableAmbient(e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    <label htmlFor="enable-ambient" className="ml-2 block text-sm text-gray-700">
                      Enable ambient sound effects
                    </label>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      case 'background_music':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Background Music</h3>
              <div className="flex items-center">
                <span className="mr-3 text-sm text-gray-500">
                  {state.configSections.background_music ? 'Enabled' : 'Disabled'}
                </span>
                <button
                  type="button"
                  className={`${
                    state.configSections.background_music ? 'bg-indigo-600' : 'bg-gray-200'
                  } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
                  onClick={() => toggleSection('background_music')}
                >
                  <span
                    className={`${
                      state.configSections.background_music ? 'translate-x-5' : 'translate-x-0'
                    } inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                  />
                </button>
              </div>
            </div>
            
            {state.configSections.background_music && (
              <div className="space-y-4">
                <Slider
                  id="music-volume"
                  label="Volume"
                  value={musicVolume}
                  onChange={setMusicVolume}
                  min={0}
                  max={1}
                  step={0.1}
                  formatValue={(value) => `${Math.round(value * 100)}%`}
                />
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <Slider
                    id="music-fade-in"
                    label="Fade In Duration"
                    value={musicFadeIn}
                    onChange={setMusicFadeIn}
                    min={0}
                    max={5}
                    step={0.5}
                    formatValue={(value) => `${value.toFixed(1)}s`}
                  />
                  
                  <Slider
                    id="music-fade-out"
                    label="Fade Out Duration"
                    value={musicFadeOut}
                    onChange={setMusicFadeOut}
                    min={0}
                    max={5}
                    step={0.5}
                    formatValue={(value) => `${value.toFixed(1)}s`}
                  />
                </div>
                
                <div className="flex items-center">
                  <input
                    id="smart-ducking"
                    type="checkbox"
                    checked={smartDucking}
                    onChange={(e) => setSmartDucking(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <label htmlFor="smart-ducking" className="ml-2 block text-sm text-gray-700">
                    Smart Ducking (reduce music volume during speech)
                  </label>
                </div>
              </div>
            )}
          </div>
        );
      case 'camera_movements':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Camera Movements</h3>
              <div className="flex items-center">
                <span className="mr-3 text-sm text-gray-500">
                  {state.configSections.camera_movements ? 'Enabled' : 'Disabled'}
                </span>
                <button
                  type="button"
                  className={`${
                    state.configSections.camera_movements ? 'bg-indigo-600' : 'bg-gray-200'
                  } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
                  onClick={() => toggleSection('camera_movements')}
                >
                  <span
                    className={`${
                      state.configSections.camera_movements ? 'translate-x-5' : 'translate-x-0'
                    } inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                  />
                </button>
              </div>
            </div>
            
            {state.configSections.camera_movements && (
              <div className="space-y-4">
                <Slider
                  id="zoom-factor"
                  label="Zoom Factor"
                  value={zoomFactor}
                  onChange={setZoomFactor}
                  min={1}
                  max={1.5}
                  step={0.05}
                  formatValue={(value) => `${value.toFixed(2)}x`}
                />
                
                <Slider
                  id="shake-intensity"
                  label="Shake Intensity"
                  value={shakeIntensity}
                  onChange={setShakeIntensity}
                  min={0}
                  max={0.1}
                  step={0.01}
                  formatValue={(value) => value === 0 ? 'Off' : `${value.toFixed(2)}`}
                />
                
                <Slider
                  id="punchin-factor"
                  label="Punch-In Factor"
                  value={punchinFactor}
                  onChange={setPunchinFactor}
                  min={1}
                  max={1.3}
                  step={0.05}
                  formatValue={(value) => `${value.toFixed(2)}x`}
                />
              </div>
            )}
          </div>
        );
      case 'transitions':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Transitions</h3>
              <div className="flex items-center">
                <span className="mr-3 text-sm text-gray-500">
                  {state.configSections.transitions ? 'Enabled' : 'Disabled'}
                </span>
                <button
                  type="button"
                  className={`${
                    state.configSections.transitions ? 'bg-indigo-600' : 'bg-gray-200'
                  } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
                  onClick={() => toggleSection('transitions')}
                >
                  <span
                    className={`${
                      state.configSections.transitions ? 'translate-x-5' : 'translate-x-0'
                    } inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                  />
                </button>
              </div>
            </div>
            
            {state.configSections.transitions && (
              <div className="space-y-4">
                <Dropdown
                  id="transition-type"
                  label="Default Transition Type"
                  value={transitionType}
                  onChange={setTransitionType}
                  options={transitionOptions}
                />
                
                <Slider
                  id="transition-duration"
                  label="Transition Duration"
                  value={transitionDuration}
                  onChange={setTransitionDuration}
                  min={0.5}
                  max={2}
                  step={0.1}
                  formatValue={(value) => `${value.toFixed(1)}s`}
                />
                
                <div className="flex items-center">
                  <input
                    id="randomize-transitions"
                    type="checkbox"
                    checked={randomizeTransitions}
                    onChange={(e) => setRandomizeTransitions(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <label htmlFor="randomize-transitions" className="ml-2 block text-sm text-gray-700">
                    Randomize transition types
                  </label>
                </div>
              </div>
            )}
          </div>
        );
      case 'captioning':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Captioning</h3>
              <div className="flex items-center">
                <span className="mr-3 text-sm text-gray-500">
                  {state.configSections.captioning ? 'Enabled' : 'Disabled'}
                </span>
                <button
                  type="button"
                  className={`${
                    state.configSections.captioning ? 'bg-indigo-600' : 'bg-gray-200'
                  } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
                  onClick={() => toggleSection('captioning')}
                >
                  <span
                    className={`${
                      state.configSections.captioning ? 'translate-x-5' : 'translate-x-0'
                    } inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                  />
                </button>
              </div>
            </div>
            
            {state.configSections.captioning && (
              <div className="space-y-4">
                <TextInput
                  id="captioning-api-url"
                  label="Captioning API URL"
                  value={captioningApiUrl}
                  onChange={setCaptioningApiUrl}
                  placeholder="https://api.assemblyai.com/v2"
                />
              </div>
            )}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Pipeline Configuration</h2>
      
      <div className="flex">
        {/* Tab sidebar */}
        <div className="w-48 flex-shrink-0">
          <div className="space-y-1">
            <button
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'profile'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('profile')}
            >
              Pipeline Profile
            </button>
            
            <button
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'sound_effects'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('sound_effects')}
            >
              Sound Effects
            </button>
            
            <button
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'background_music'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('background_music')}
            >
              Background Music
            </button>
            
            <button
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'camera_movements'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('camera_movements')}
            >
              Camera Movements
            </button>
            
            <button
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'transitions'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('transitions')}
            >
              Transitions
            </button>
            
            <button
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'captioning'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }`}
              onClick={() => setActiveTab('captioning')}
            >
              Captioning
            </button>
          </div>
        </div>
        
        {/* Tab content */}
        <div className="flex-1 ml-6">
          <Card>
            {renderTabContent()}
          </Card>
        </div>
      </div>
      
      {/* Navigation buttons */}
      <div className="mt-8 flex justify-between">
        <Button variant="outline" onClick={onBack}>
          Back to Avatar Selection
        </Button>
        <Button variant="primary" onClick={onNext}>
          Continue to Execution
        </Button>
      </div>
    </div>
  );
};

export default ConfigurationSection;