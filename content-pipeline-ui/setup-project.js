// Save this file in your content-pipeline-ui directory and run with:
// node setup-project.js

const fs = require('fs');
const path = require('path');

// Define all the directories we need to create
const directories = [
  'src/components',
  'src/context',
  'src/electron',
  'src/sections',
  'src/utils',
  'public'
];

// Create directories if they don't exist
directories.forEach(dir => {
  const fullPath = path.join(__dirname, dir);
  if (!fs.existsSync(fullPath)) {
    console.log(`Creating directory: ${dir}`);
    fs.mkdirSync(fullPath, { recursive: true });
  } else {
    console.log(`Directory already exists: ${dir}`);
  }
});

// Check if all the required files exist
const requiredFiles = [
  'src/App.tsx',
  'src/index.tsx',
  'src/index.css',
  'public/index.html',
  'public/manifest.json',
  'tailwind.config.js',
  'package.json',
  'src/context/PipelineContext.tsx',
  'src/components/Button.tsx',
  'src/components/Card.tsx',
  'src/components/Dropdown.tsx',
  'src/components/Footer.tsx',
  'src/components/Header.tsx',
  'src/components/ProgressBar.tsx',
  'src/components/Sidebar.tsx',
  'src/components/Slider.tsx',
  'src/components/TextInput.tsx',
  'src/sections/ScriptInput.tsx',
  'src/sections/AvatarSelect.tsx',
  'src/sections/ConfigurationSection.tsx',
  'src/sections/ExecutionSection.tsx',
  'src/utils/pythonRunner.ts',
  'src/electron/main.ts',
  'src/electron/electronIntegration.ts',
  'src/electron/preload.js'
];

// Check if each file exists
requiredFiles.forEach(file => {
  const fullPath = path.join(__dirname, file);
  if (fs.existsSync(fullPath)) {
    console.log(`File exists: ${file}`);
  } else {
    console.log(`WARNING: Missing file: ${file}`);
  }
});

console.log('\nSetup complete. Make sure to install all required dependencies:');
console.log('npm install');
console.log('\nTo start the development server:');
console.log('npm run electron:dev');