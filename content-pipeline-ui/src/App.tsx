import React, { useState } from 'react';
import { PipelineProvider } from './context/PipelineContext';
import Header from './components/Header';
import Footer from './components/Footer';
import Sidebar from './components/Sidebar';
import ScriptInput from './sections/ScriptInput';
import AvatarSelect from './sections/AvatarSelect';
import ConfigurationSection from './sections/ConfigurationSection';
import ExecutionSection from './sections/ExecutionSection';

// Define the possible workflow steps
export type WorkflowStep = 
  | 'scriptInput'
  | 'avatarSelect'
  | 'configuration'
  | 'execution';

const App: React.FC = () => {
  // Current active step in the workflow
  const [activeStep, setActiveStep] = useState<WorkflowStep>('scriptInput');

  // Function to render the current step
  const renderStep = () => {
    switch (activeStep) {
      case 'scriptInput':
        return <ScriptInput onNext={() => setActiveStep('avatarSelect')} />;
      case 'avatarSelect':
        return (
          <AvatarSelect
            onBack={() => setActiveStep('scriptInput')}
            onNext={() => setActiveStep('configuration')}
          />
        );
      case 'configuration':
        return (
          <ConfigurationSection
            onBack={() => setActiveStep('avatarSelect')}
            onNext={() => setActiveStep('execution')}
          />
        );
      case 'execution':
        return (
          <ExecutionSection
            onBack={() => setActiveStep('configuration')}
          />
        );
      default:
        return <ScriptInput onNext={() => setActiveStep('avatarSelect')} />;
    }
  };

  return (
    <PipelineProvider>
      <div className="flex flex-col h-screen bg-gray-50">
        <Header />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar activeStep={activeStep} setActiveStep={setActiveStep} />
          <main className="flex-1 overflow-y-auto p-6">
            {renderStep()}
          </main>
        </div>
        <Footer />
      </div>
    </PipelineProvider>
  );
};

export default App;