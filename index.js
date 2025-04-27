import fs from 'fs';
import path from 'path';
import editly from 'editly';

async function run() {
  const clipsDir = path.resolve('./clips');
  const files = fs.readdirSync(clipsDir)
    .filter(f => f.endsWith('.mp4'))
    .sort();

  if (files.length < 2) {
    console.error('Need at least two .mp4 files in /clips to apply transitions.');
    process.exit(1);
  }

  // Build clip entries with layers
  const clips = files.map(f => ({
    duration: null, // let Editly infer full duration
    layers: [
      { type: 'video', path: path.join(clipsDir, f) }
    ],
  }));

  // Prepare output path and ensure it's not an existing directory
  const outPath = path.resolve('./output.mp4');
  if (fs.existsSync(outPath)) {
    const stats = fs.lstatSync(outPath);
    if (stats.isDirectory()) {
      console.log(`Removing existing directory at ${outPath}`);
      fs.rmSync(outPath, { recursive: true, force: true });
    } else {
      // Remove existing file to avoid conflicts
      fs.unlinkSync(outPath);
    }
  }

  // 1s crossfade between each clip
  const transitions = Array(files.length - 1).fill({
    name: 'crossfade',
    duration: 1,
  });

  const editSpec = {
    outPath,
    width: 1080,
    height: 1920,
    fps: 30,
    clips,
    streamChunkSize: 4096,
    transitions,
    threads: 0,
    ffmpeg: {
      codec: 'h264_nvenc',
      preset: 'ultrafast',   // NVENC’s “low-latency high-quality” preset
      crf: 23
    }
  };

  console.log('Running Editly with spec:', editSpec);
  await editly(editSpec);
  console.log('✅ Video rendered to', outPath);
}

run().catch(err => {
  console.error('Error running Editly:', err);
  process.exit(1);
});