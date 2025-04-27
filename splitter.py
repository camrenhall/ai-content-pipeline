#!/usr/bin/env python3
import json
import os
import sys
import argparse
import ffmpeg
from decimal import Decimal, getcontext

# Set decimal precision to ensure accuracy
getcontext().prec = 28

def split_video(input_video, json_payload, output_dir=None, verbose=False, ms_offset=0.001):
    """
    Split a video into subclips based on a JSON payload of B-roll cuts with millisecond precision.
    
    :param input_video: Path to the input video file
    :param json_payload: Path to the JSON file containing cut instructions
    :param output_dir: Directory to save output clips (defaults to input video directory)
    :param verbose: Enable verbose logging
    :param ms_offset: Millisecond offset to use between adjacent clips (default: 0.001)
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
    
    # Get video duration with high precision
    try:
        probe = ffmpeg.probe(input_video)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        # Use Decimal for high precision
        duration = Decimal(probe['format']['duration'])
        if verbose:
            print(f"Video duration: {duration} seconds")
    except Exception as e:
        print(f"Error probing video: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Derive segments
    segments = []
    
    # First, create a list of all points where cuts happen
    cut_points = []
    
    # Add beginning of video
    cut_points.append(Decimal('0'))
    
    # Add all timestamps from the JSON
    for cut in broll_cuts:
        # Start of cut
        start_time = Decimal(str(cut['timestamp']))
        cut_points.append(start_time)
        
        # End of cut
        end_time = start_time + Decimal(str(cut['duration']))
        cut_points.append(end_time)
    
    # Add end of video
    cut_points.append(duration)
    
    # Sort and de-duplicate
    cut_points = sorted(set(cut_points))
    
    # Create segments from adjacent cut points
    for i in range(len(cut_points) - 1):
        start_time = cut_points[i]
        # If this isn't the first segment, add a small offset to prevent overlap
        if i > 0:
            start_time = start_time + Decimal(str(ms_offset))
        
        end_time = cut_points[i + 1]
        
        # Only add the segment if it has a positive duration
        if end_time > start_time:
            segments.append((start_time, end_time))
    
    # Store created clip paths
    created_clips = []
    
    # Generate clips
    for i, (start_time, end_time) in enumerate(segments):
        # Create output filename
        base_filename = os.path.splitext(os.path.basename(input_video))[0]
        output_path = os.path.join(output_dir, f'{base_filename}_clip_{i}.mp4')
        
        try:
            if verbose:
                print(f"Creating clip {i}: {start_time} - {end_time} seconds (duration: {end_time-start_time}s)")
            
            # Use float(str()) to preserve full decimal precision when passing to ffmpeg
            (
                ffmpeg
                .input(input_video, ss=float(str(start_time)), t=float(str(end_time-start_time)))
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
        description='Split a video into subclips with millisecond precision based on a JSON payload of B-roll cuts.',
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
    parser.add_argument('--ms-offset', 
                       type=float, 
                       default=0.001,
                       help='Millisecond offset between adjacent clips (default: 0.001)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call split_video function
    try:
        clips = split_video(
            args.input_video, 
            args.json_payload, 
            output_dir=args.output_dir, 
            verbose=args.verbose,
            ms_offset=args.ms_offset
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