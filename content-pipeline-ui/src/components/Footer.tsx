import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-gray-200 py-4 px-6 text-center text-sm text-gray-500">
      <p>Content Pipeline MVP v1.0 © {new Date().getFullYear()}</p>
    </footer>
  );
};

export default Footer;