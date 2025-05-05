import os
import subprocess
import random
import math
import sys
from tempfile import TemporaryDirectory

# ========= CLI INPUT =========
if len(sys.argv) < 2:
    print("Usage: python punchshake.py <input_video.mp4>")
    sys.exit(1)

INPUT_CLIP = sys.argv[1]
SEGMENT_DURATION = 2  # seconds
OUTPUT_FILE = "output_punchshake_final.mp4"
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
MIN_VALID_SIZE = 100_000  # bytes

# ========= HELPERS =========
def get_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout.decode().strip())

def is_valid_mp4(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) < MIN_VALID_SIZE:
        return False
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=nw=1:nk=1", filepath
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return b"video" in result.stdout

def generate_effect_filter(effect, duration):
    if effect == "zoom_in_slow":
        # Smooth zoom-in simulated with scale over time
        return (
            f"scale=iw*1.05:ih*1.05,"
            f"crop={CANVAS_WIDTH}:{CANVAS_HEIGHT},"
            f"trim=duration={duration},setpts=PTS-STARTPTS"
        )
    elif effect == "zoom_in_hard":
        return (
            f"scale=iw*1.08:ih*1.08,"
            f"crop={CANVAS_WIDTH}:{CANVAS_HEIGHT},"
            f"trim=duration={duration},setpts=PTS-STARTPTS"
        )
    elif effect == "shake":
        amt = random.randint(3, 6)
        seed = random.randint(100, 1000)
        return (
            f"crop=iw-{amt}:ih-{amt}:"
            f"x='random({seed})*{amt}':y='random({seed+1})*{amt}',"
            f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT},"
            f"trim=duration={duration},setpts=PTS-STARTPTS"
        )
    else:
        return (
            f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT},"
            f"trim=duration={duration},setpts=PTS-STARTPTS"
        )

def export_segment(infile, start, duration, effect, outfile):
    vf = generate_effect_filter(effect, duration)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", infile,
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-movflags", "+faststart",
        outfile
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def join_segments_with_audio(segment_paths, original_audio, output_path):
    list_path = os.path.join(os.path.dirname(segment_paths[0]), "segments.txt")
    with open(list_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-i", original_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd)

# ========= MAIN WORKFLOW =========
if not os.path.exists(INPUT_CLIP):
    print(f"ERROR: File '{INPUT_CLIP}' not found.")
    sys.exit(1)

duration = get_duration(INPUT_CLIP)
segment_count = math.ceil(duration / SEGMENT_DURATION)

with TemporaryDirectory() as tmpdir:
    segment_paths = []
    for i in range(segment_count):
        start = i * SEGMENT_DURATION
        seg_path = os.path.join(tmpdir, f"seg_{i}.mp4")
        effect = random.choices(
            ["zoom_in_slow", "zoom_in_hard", "shake", "none"],
            weights=[0.3, 0.2, 0.3, 0.2],
            k=1
        )[0]
        export_segment(INPUT_CLIP, start, SEGMENT_DURATION, effect, seg_path)

        if is_valid_mp4(seg_path):
            print(f"[OK] Segment {i} ({effect}) valid.")
            segment_paths.append(seg_path)
        else:
            print(f"[SKIP] Segment {i} ({effect}) is invalid or corrupt.")

    if not segment_paths:
        print("❌ No valid segments to join. Aborting.")
        sys.exit(1)

    join_segments_with_audio(segment_paths, INPUT_CLIP, OUTPUT_FILE)

print(f"\n✅ DONE: Final video written to {OUTPUT_FILE}")
