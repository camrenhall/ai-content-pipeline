#!/usr/bin/env python3
import json
import os
import sys
import argparse
import ffmpeg

def split_video(input_video, json_payload, output_dir=None, verbose=False):
    """
    Split a video into subclips based on a JSON payload of B-roll cuts.
    
    :param input_video: Path to the input video file
    :param json_payload: Path to the JSON file containing cut instructions
    :param output_dir: Directory to save output clips (defaults to input video directory)
    :param verbose: Enable verbose logging
    :return: List of created clip paths
    """
    # If no output directory specified, use input video's directory
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(input_video)), 
            'output_clips'
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON payload
    try:
        with open(json_payload, 'r') as f:
            payload = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON payload: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract broll cuts
    broll_cuts = payload.get('broll_cuts', [])
    
    # Get video duration
    try:
        probe = ffmpeg.probe(input_video)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(probe['format']['duration'])
    except Exception as e:
        print(f"Error probing video: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare clip timestamps
    clip_timestamps = [0]  # Start with 0
    
    # Add timestamps from broll cuts
    for cut in broll_cuts:
        clip_timestamps.append(cut['timestamp'])
        clip_timestamps.append(cut['timestamp'] + cut['duration'])
    
    clip_timestamps.append(duration)  # Add video end timestamp
    
    # Remove duplicates and sort
    clip_timestamps = sorted(set(clip_timestamps))
    
    # Store created clip paths
    created_clips = []
    
    # Generate clips
    for i in range(len(clip_timestamps) - 1):
        start_time = clip_timestamps[i]
        end_time = clip_timestamps[i + 1]
        
        # Skip zero-duration clips
        if start_time >= end_time:
            continue
        
        # Create output filename
        base_filename = os.path.splitext(os.path.basename(input_video))[0]
        output_path = os.path.join(output_dir, f'{base_filename}_clip_{i}.mp4')
        
        try:
            if verbose:
                print(f"Creating clip {i}: {start_time} - {end_time} seconds")
            
            (
                ffmpeg
                .input(input_video, ss=start_time, t=end_time-start_time)
                .output(output_path, c='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            created_clips.append(output_path)
            
            if verbose:
                print(f"Created clip: {output_path}")
        
        except ffmpeg.Error as e:
            print(f"Error creating clip {i}: {e.stderr.decode()}", file=sys.stderr)
    
    return created_clips

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Split a video into subclips based on a JSON payload of B-roll cuts.',
        epilog='Example: python video_splitter.py input.mp4 broll_cuts.json -o output_clips -v'
    )
    
    # Positional arguments
    parser.add_argument('input_video', 
                        help='Path to the input video file')
    parser.add_argument('json_payload', 
                        help='Path to the JSON file containing cut instructions')
    
    # Optional arguments
    parser.add_argument('-o', '--output-dir', 
                        help='Directory to save output clips (default: ./output_clips relative to input video)')
    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Enable verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call split_video function
    try:
        clips = split_video(
            args.input_video, 
            args.json_payload, 
            output_dir=args.output_dir, 
            verbose=args.verbose
        )
        
        # Print summary
        print(f"\nVideo splitting complete. Created {len(clips)} clips:")
        for clip in clips:
            print(f"  {clip}")
    
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

# Requirements:
# 1. Install FFmpeg: 
#    - On Ubuntu/Debian: sudo apt-get install ffmpeg
#    - On macOS with Homebrew: brew install ffmpeg
#    - On Windows: Download from FFmpeg official website
# 2. Install Python dependencies:
#    pip install ffmpeg-python