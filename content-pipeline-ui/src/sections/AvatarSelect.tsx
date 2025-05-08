import React, { useState, useEffect } from 'react';
import { usePipeline } from '../context/PipelineContext';
import Card from '../components/Card';
import Button from '../components/Button';
import Slider from '../components/Slider';
import Dropdown from '../components/Dropdown';
import TextInput from '../components/TextInput';

interface AvatarSelectProps {
  onBack: () => void;
  onNext: () => void;
}

// Mock avatar data - in a real implementation, this would come from the HeyGen API
const mockAvatars = [
  { id: 'avatar1', name: 'Sophia', type: 'digital', image: 'https://via.placeholder.com/150?text=Sophia' },
  { id: 'avatar2', name: 'Jackson', type: 'digital', image: 'https://via.placeholder.com/150?text=Jackson' },
  { id: 'avatar3', name: 'Aisha', type: 'digital', image: 'https://via.placeholder.com/150?text=Aisha' },
  { id: 'avatar4', name: 'Carlos', type: 'digital', image: 'https://via.placeholder.com/150?text=Carlos' },
  { id: 'avatar5', name: 'Emma', type: 'digital', image: 'https://via.placeholder.com/150?text=Emma' },
  { id: 'avatar6', name: 'Michael', type: 'digital', image: 'https://via.placeholder.com/150?text=Michael' },
];

// Mock talking photos data
const mockTalkingPhotos = [
  { id: 'photo1', name: 'Business Woman', type: 'photo', image: 'https://via.placeholder.com/150?text=Business+Woman' },
  { id: 'photo2', name: 'Business Man', type: 'photo', image: 'https://via.placeholder.com/150?text=Business+Man' },
  { id: 'photo3', name: 'Casual Woman', type: 'photo', image: 'https://via.placeholder.com/150?text=Casual+Woman' },
  { id: 'photo4', name: 'Casual Man', type: 'photo', image: 'https://via.placeholder.com/150?text=Casual+Man' },
];

// Mock voice data
const mockVoices = [
  { id: 'voice1', name: 'Female Voice 1', gender: 'female' },
  { id: 'voice2', name: 'Male Voice 1', gender: 'male' },
  { id: 'voice3', name: 'Female Voice 2', gender: 'female' },
  { id: 'voice4', name: 'Male Voice 2', gender: 'male' },
  { id: 'voice5', name: 'Neutral Voice', gender: 'neutral' },
];

const voiceEmotions = [
  { value: 'Excited', label: 'Excited' },
  { value: 'Friendly', label: 'Friendly' },
  { value: 'Serious', label: 'Serious' },
  { value: 'Soothing', label: 'Soothing' },
  { value: 'Broadcaster', label: 'Broadcaster' },
];

const avatarStyles = [
  { value: 'normal', label: 'Normal' },
  { value: 'closeUp', label: 'Close Up' },
  { value: 'circle', label: 'Circle' },
];

const elevenLabsModels = [
  { value: 'eleven_turbo_v2', label: 'Eleven Turbo v2' },
  { value: 'eleven_multilingual_v2', label: 'Eleven Multilingual v2' },
  { value: 'eleven_monolingual_v1', label: 'Eleven Monolingual v1' },
];

const AvatarSelect: React.FC<AvatarSelectProps> = ({ onBack, onNext }) => {
  const { state, updateState } = usePipeline();
  const [selectedType, setSelectedType] = useState<'digital' | 'photo'>('digital');
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState<'avatar' | 'voice' | 'appearance'>('avatar');

  // Reset selected IDs when changing types
  useEffect(() => {
    if (selectedType === 'digital') {
      updateState({ talkingPhotoId: null });
    } else {
      updateState({ avatarId: null });
    }
  }, [selectedType, updateState]);

  // Filter avatars/photos based on search term
  const filteredAvatars = selectedType === 'digital'
    ? mockAvatars.filter(a => a.name.toLowerCase().includes(searchTerm.toLowerCase()))
    : mockTalkingPhotos.filter(p => p.name.toLowerCase().includes(searchTerm.toLowerCase()));

  const handleSelectAvatar = (id: string) => {
    if (selectedType === 'digital') {
      updateState({ avatarId: id, talkingPhotoId: null });
    } else {
      updateState({ talkingPhotoId: id, avatarId: null });
    }
  };

  const isAvatarSelected = !!state.avatarId || !!state.talkingPhotoId;
  const selectedId = state.avatarId || state.talkingPhotoId;

  return (
    <div className="max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Avatar Selection & Configuration</h2>
      
      {/* Tab navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('avatar')}
            className={`${
              activeTab === 'avatar'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
          >
            Avatar Selection
          </button>
          <button
            onClick={() => setActiveTab('voice')}
            className={`${
              activeTab === 'voice'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
            disabled={!isAvatarSelected}
          >
            Voice Configuration
          </button>
          <button
            onClick={() => setActiveTab('appearance')}
            className={`${
              activeTab === 'appearance'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
            disabled={!isAvatarSelected}
          >
            Appearance Settings
          </button>
        </nav>
      </div>
      
      {/* Avatar selection tab */}
      {activeTab === 'avatar' && (
        <Card>
          <div className="space-y-6">
            {/* Avatar type toggle */}
            <div className="flex space-x-4">
              <Button 
                variant={selectedType === 'digital' ? 'primary' : 'outline'} 
                onClick={() => setSelectedType('digital')}
              >
                Digital Avatars
              </Button>
              <Button 
                variant={selectedType === 'photo' ? 'primary' : 'outline'} 
                onClick={() => setSelectedType('photo')}
              >
                Talking Photos
              </Button>
            </div>
            
            {/* Search */}
            <TextInput
              id="avatar-search"
              placeholder={`Search ${selectedType === 'digital' ? 'avatars' : 'photos'}...`}
              value={searchTerm}
              onChange={setSearchTerm}
            />
            
            {/* Avatar grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
              {filteredAvatars.map((avatar) => (
                <div 
                  key={avatar.id}
                  className={`
                    cursor-pointer rounded-lg overflow-hidden border-2
                    ${selectedId === avatar.id ? 'border-indigo-500 ring-2 ring-indigo-500' : 'border-gray-200 hover:border-gray-300'}
                  `}
                  onClick={() => handleSelectAvatar(avatar.id)}
                >
                  <div className="aspect-w-1 aspect-h-1">
                    <img 
                      src={avatar.image} 
                      alt={avatar.name} 
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="p-2 text-center text-sm font-medium text-gray-900">
                    {avatar.name}
                  </div>
                </div>
              ))}
            </div>
            
            {filteredAvatars.length === 0 && (
              <div className="text-center py-4 text-gray-500">
                No {selectedType === 'digital' ? 'avatars' : 'photos'} found matching "{searchTerm}"
              </div>
            )}
          </div>
        </Card>
      )}
      
      {/* Voice configuration tab */}
      {activeTab === 'voice' && (
        <Card>
          <div className="space-y-6">
            {/* Voice selection */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Voice Selection</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {mockVoices.map((voice) => (
                  <div 
                    key={voice.id}
                    className={`
                      cursor-pointer rounded-lg p-4 border-2
                      ${state.voiceId === voice.id ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200 hover:border-gray-300'}
                    `}
                    onClick={() => updateState({ voiceId: voice.id })}
                  >
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900">{voice.name}</p>
                        <p className="text-xs text-gray-500">{voice.gender}</p>
                      </div>
                      {state.voiceId === voice.id && (
                        <div className="ml-auto">
                          <svg className="h-5 w-5 text-indigo-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Voice settings */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Voice Settings</h3>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Dropdown
                  id="voice-emotion"
                  label="Voice Emotion"
                  value={state.voiceEmotion}
                  onChange={(value) => updateState({ voiceEmotion: value })}
                  options={voiceEmotions}
                />
                
                <Slider
                  id="voice-speed"
                  label="Voice Speed"
                  value={state.voiceSpeed}
                  onChange={(value) => updateState({ voiceSpeed: value })}
                  min={0.5}
                  max={1.5}
                  step={0.1}
                  formatValue={(value) => `${value.toFixed(1)}x`}
                />
              </div>
            </div>
            
            {/* ElevenLabs Integration */}
            <div className="border-t border-gray-200 pt-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">ElevenLabs Voice Integration</h3>
                <div className="flex items-center">
                  <span className="mr-3 text-sm text-gray-500">{state.elevenLabsEnabled ? 'Enabled' : 'Disabled'}</span>
                  <button
                    type="button"
                    className={`${
                      state.elevenLabsEnabled ? 'bg-indigo-600' : 'bg-gray-200'
                    } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
                    onClick={() => updateState({ elevenLabsEnabled: !state.elevenLabsEnabled })}
                  >
                    <span
                      className={`${
                        state.elevenLabsEnabled ? 'translate-x-5' : 'translate-x-0'
                      } inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                    />
                  </button>
                </div>
              </div>
              
              {state.elevenLabsEnabled && (
                <div className="mt-4 space-y-4">
                  <Dropdown
                    id="elevenlabs-model"
                    label="Model"
                    value={state.elevenLabsModel}
                    onChange={(value) => updateState({ elevenLabsModel: value })}
                    options={elevenLabsModels}
                  />
                  
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <Slider
                      id="elevenlabs-stability"
                      label="Stability"
                      value={state.elevenLabsStability}
                      onChange={(value) => updateState({ elevenLabsStability: value })}
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      formatValue={(value) => value.toFixed(1)}
                    />
                    
                    <Slider
                      id="elevenlabs-similarity"
                      label="Similarity"
                      value={state.elevenLabsSimilarity}
                      onChange={(value) => updateState({ elevenLabsSimilarity: value })}
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      formatValue={(value) => value.toFixed(1)}
                    />
                    
                    <Slider
                      id="elevenlabs-style"
                      label="Style"
                      value={state.elevenLabsStyle}
                      onChange={(value) => updateState({ elevenLabsStyle: value })}
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      formatValue={(value) => value.toFixed(1)}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>
      )}
      
      {/* Appearance settings tab */}
      {activeTab === 'appearance' && (
        <Card>
          <div className="space-y-6">
            {/* Avatar positioning */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Avatar Positioning</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Slider
                  id="avatar-scale"
                  label="Size"
                  value={state.avatarScale}
                  onChange={(value) => updateState({ avatarScale: value })}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  formatValue={(value) => `${value.toFixed(1)}x`}
                />
                
                <Slider
                  id="avatar-offset-x"
                  label="Horizontal Position"
                  value={state.avatarOffsetX}
                  onChange={(value) => updateState({ avatarOffsetX: value })}
                  min={-1.0}
                  max={1.0}
                  step={0.1}
                  formatValue={(value) => value.toFixed(1)}
                />
                
                <Slider
                  id="avatar-offset-y"
                  label="Vertical Position"
                  value={state.avatarOffsetY}
                  onChange={(value) => updateState({ avatarOffsetY: value })}
                  min={-1.0}
                  max={1.0}
                  step={0.1}
                  formatValue={(value) => value.toFixed(1)}
                />
              </div>
            </div>
            
            {/* Avatar style */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Avatar Style</h3>
              <div className="grid grid-cols-3 gap-4">
                {avatarStyles.map((style) => (
                  <div 
                    key={style.value}
                    className={`
                      cursor-pointer rounded-lg p-4 border-2 text-center
                      ${state.avatarStyle === style.value ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200 hover:border-gray-300'}
                    `}
                    onClick={() => updateState({ avatarStyle: style.value as any })}
                  >
                    {style.label}
                  </div>
                ))}
              </div>
              
              <div className="mt-4 flex items-center">
                <div className="flex items-center h-5">
                  <input
                    id="landscape-avatar"
                    type="checkbox"
                    checked={state.landscapeAvatar}
                    onChange={(e) => updateState({ landscapeAvatar: e.target.checked })}
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor="landscape-avatar" className="font-medium text-gray-700">
                    Landscape Avatar
                  </label>
                  <p className="text-gray-500">Optimize settings for a landscape avatar in portrait video</p>
                </div>
              </div>
            </div>
            
            {/* Background settings */}
            <div className="border-t border-gray-200 pt-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Background</h3>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Background Color
                  </label>
                  <div className="flex items-center">
                    <input
                      type="color"
                      value={state.backgroundColor}
                      onChange={(e) => updateState({ backgroundColor: e.target.value })}
                      className="h-10 w-10 border-0 rounded-md cursor-pointer"
                    />
                    <input
                      type="text"
                      value={state.backgroundColor}
                      onChange={(e) => updateState({ backgroundColor: e.target.value })}
                      className="ml-2 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                    />
                  </div>
                </div>
                
                <TextInput
                  id="background-url"
                  label="Background Image/Video URL"
                  value={state.backgroundUrl}
                  onChange={(value) => updateState({ backgroundUrl: value })}
                  placeholder="https://example.com/background.jpg"
                />
              </div>
            </div>
            
            {/* Caption toggle */}
            <div className="border-t border-gray-200 pt-6">
              <div className="flex items-center">
                <div className="flex items-center h-5">
                  <input
                    id="caption"
                    type="checkbox"
                    checked={state.caption}
                    onChange={(e) => updateState({ caption: e.target.checked })}
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor="caption" className="font-medium text-gray-700">
                    Add Captions
                  </label>
                  <p className="text-gray-500">Show subtitles for the spoken content</p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
      
      {/* Navigation buttons */}
      <div className="mt-8 flex justify-between">
        <Button variant="outline" onClick={onBack}>
          Back to Script Input
        </Button>
        <Button 
          variant="primary" 
          onClick={onNext}
          disabled={!isAvatarSelected || !state.voiceId}
        >
          Continue to Configuration
        </Button>
      </div>
    </div>
  );
};

export default AvatarSelect;