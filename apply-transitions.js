const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');

// Configuration
const clipsDir = path.join(__dirname, './output_clips');
const outputFile = path.join(__dirname, 'output.mp4');
const tempDir = path.join(os.tmpdir(), 'video-assembly-' + Date.now());

// Create temp directory
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

console.log('===== VIDEO ASSEMBLY SCRIPT =====');
console.log('Looking for clips in:', clipsDir);

// Get all video clips
let videoFiles = [];
try {
  videoFiles = fs.readdirSync(clipsDir)
    .filter(file => file.endsWith('.mp4') || file.endsWith('.mov'))
    .map(file => path.join(clipsDir, file))
    .sort();
    
  console.log(`Found ${videoFiles.length} video clips:`, videoFiles);
} catch (err) {
  console.error('Error reading clips directory:', err);
  process.exit(1);
}

if (videoFiles.length === 0) {
  console.error('No video clips found in ./output_clips/');
  process.exit(1);
}

// Generate FFmpeg command
let args = ['-y']; // Add -y flag to automatically overwrite files

// Add input files
videoFiles.forEach(file => {
  args.push('-i', file);
});

if (videoFiles.length === 1) {
  // If only one video, just copy it
  args.push('-c:v', 'copy', '-c:a', 'copy', outputFile);
} else {
  // Build complex filter for multiple videos with transitions
  let filterComplex = [];
  
  for (let i = 1; i < videoFiles.length; i++) {
    const prevOutput = i === 1 ? '[0:v]' : `[v${i-1}]`;
    const currentStream = `[${i}:v]`;
    const outputLabel = `[v${i}]`;
    
    // Add a crossfade transition (0.5s duration)
    filterComplex.push(`${prevOutput}${currentStream}xfade=transition=fade:duration=0.5:offset=2.5${outputLabel}`);
  }
  
  args.push('-filter_complex', filterComplex.join(';'));
  args.push('-map', `[v${videoFiles.length-1}]`);
  
  // Try to map audio from first clip
  args.push('-map', '0:a');
  args.push('-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac');
  args.push(outputFile);
}

console.log('\nExecuting FFmpeg with arguments:');
console.log('ffmpeg', args.join(' '));

// Spawn FFmpeg process
const ffmpeg = spawn('ffmpeg', args);

// Handle output
ffmpeg.stdout.on('data', (data) => {
  console.log(`stdout: ${data}`);
});

ffmpeg.stderr.on('data', (data) => {
  process.stdout.write('.');
});

ffmpeg.on('close', (code) => {
  console.log('\n');
  if (code === 0) {
    console.log(`\nSuccess! Video created at: ${outputFile}`);
  } else {
    console.error(`\nFFmpeg process exited with code ${code}`);
  }
  
  // Clean up
  try {
    fs.rmdirSync(tempDir, { recursive: true });
    console.log('Temp directory cleaned up');
  } catch (err) {
    console.error('Error cleaning up temp directory:', err);
  }
});

// Handle errors
ffmpeg.on('error', (err) => {
  console.error('Failed to start FFmpeg process:', err);
});

console.log('\nFFmpeg process started. Processing video (this might take a while)...');
console.log('Progress: ');