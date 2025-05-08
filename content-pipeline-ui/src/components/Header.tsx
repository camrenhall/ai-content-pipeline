import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-sm py-4 px-6 flex items-center">
      <div className="flex items-center">
        <svg className="h-8 w-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h18M3 16h18" />
        </svg>
        <h1 className="ml-3 text-xl font-semibold text-gray-800">Automated Content Pipeline</h1>
      </div>
    </header>
  );
};

export default Header;