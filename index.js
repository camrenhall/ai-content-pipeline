import fs from 'fs';
import path from 'path';
import editly from 'editly';

async function run() {
  const clipsDir = path.resolve('./output_clips');

  // Read and parse clip files with indices
  const files = fs.readdirSync(clipsDir)
    .filter(f => f.endsWith('.mp4') && /_clip_(\d+)\.mp4$/.test(f))
    .map(f => ({
      filename: f,
      clipNumber: parseInt(f.match(/_clip_(\d+)\.mp4$/)[1], 10),
    }))
    .sort((a, b) => a.clipNumber - b.clipNumber)
    .map(obj => obj.filename);

  if (files.length < 2) {
    console.error('Need at least two .mp4 files in /output_clips to apply transitions.');
    process.exit(1);
  }

  // Build clip entries with layers
  const clips = files.map(f => ({
    duration: null, // let Editly infer full duration
    layers: [
      { type: 'video', path: path.join(clipsDir, f) }
    ],
  }));

  // Clean up any existing output.mp4
  const outPath = path.resolve('./output.mp4');
  if (fs.existsSync(outPath)) {
    const stats = fs.lstatSync(outPath);
    if (stats.isDirectory()) {
      fs.rmSync(outPath, { recursive: true, force: true });
    } else {
      fs.unlinkSync(outPath);
    }
  }

  // 1s crossfade between each clip
  const transitions = Array(clips.length - 1).fill({ name: 'crossfade', duration: 1 });

  const editSpec = {
    outPath,
    width: 1080,
    height: 1920,
    fps: 30,
    clips,
    transitions,
    streamChunkSize: 4096,
    threads: 0,
    ffmpeg: {
      preset: 'ultrafast',
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
