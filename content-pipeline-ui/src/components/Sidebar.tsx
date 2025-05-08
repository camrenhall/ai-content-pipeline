import React from 'react';
import { WorkflowStep } from '../App';

interface SidebarProps {
  activeStep: WorkflowStep;
  setActiveStep: (step: WorkflowStep) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeStep, setActiveStep }) => {
  // Define the workflow steps
  const steps: { id: WorkflowStep; label: string; icon: string }[] = [
    { id: 'scriptInput', label: 'Script Generation', icon: '📝' },
    { id: 'avatarSelect', label: 'Avatar Selection', icon: '👤' },
    { id: 'configuration', label: 'Configuration', icon: '⚙️' },
    { id: 'execution', label: 'Execution', icon: '▶️' },
  ];

  return (
    <div className="w-64 bg-gray-800 text-white">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-medium">Workflow</h2>
      </div>
      <nav className="mt-5">
        <ul>
          {steps.map((step) => (
            <li key={step.id}>
              <button
                onClick={() => setActiveStep(step.id)}
                className={`w-full flex items-center px-4 py-3 ${
                  activeStep === step.id
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                <span className="mr-3">{step.icon}</span>
                <span>{step.label}</span>
                {activeStep === step.id && (
                  <span className="ml-auto">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;