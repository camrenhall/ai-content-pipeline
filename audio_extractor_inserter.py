#!/usr/bin/env python3
"""
Audio Extractor/Inserter Component

This script extracts audio from an MP4 file or inserts audio into an MP4 file.
It's designed to be lightweight, performant and follow software engineering best practices.

Usage:
    extract mode: python audio_handler.py --mode extract --input input.mp4 --output audio.aac
    insert mode: python audio_handler.py --mode insert --video video.mp4 --audio audio.aac --output output.mp4
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
import logging
import time


def setup_logger():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract or insert audio in MP4 files')
    parser.add_argument('--mode', choices=['extract', 'insert'], required=True,
                        help='Operation mode: extract audio from video or insert audio into video')
    
    # Extract mode arguments
    parser.add_argument('--input', help='Input MP4 file (for extract mode)')
    parser.add_argument('--output', help='Output file path')
    
    # Insert mode arguments
    parser.add_argument('--video', help='Video file (for insert mode)')
    parser.add_argument('--audio', help='Audio file (for insert mode)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'extract':
        if not args.input or not args.output:
            parser.error('Extract mode requires --input and --output arguments')
    elif args.mode == 'insert':
        if not args.video or not args.audio or not args.output:
            parser.error('Insert mode requires --video, --audio, and --output arguments')
            
    return args


def check_ffmpeg():
    """Check if FFmpeg is installed and available."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def extract_audio(input_file, output_file, logger):
    """Extract audio from an MP4 file without re-encoding."""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    os.makedirs(output_path, exist_ok=True)
    
    # Extract audio stream without re-encoding
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-vn',              # Disable video
        '-acodec', 'copy',  # Copy audio codec without re-encoding
        '-y',               # Overwrite output file if it exists
        output_file
    ]
    
    logger.info(f"Extracting audio from {input_file}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        logger.error(f"Error extracting audio: {result.stderr.decode()}")
        sys.exit(1)
    
    duration = time.time() - start_time
    logger.info(f"Audio extraction completed in {duration:.2f} seconds")
    
    # Verify the output file exists and has content
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info(f"Audio saved to {output_file}")
    else:
        logger.error("Audio extraction failed: Output file is empty or missing")
        sys.exit(1)


def insert_audio(video_file, audio_file, output_file, logger):
    """Insert audio into a video file without re-encoding video."""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    os.makedirs(output_path, exist_ok=True)
    
    # Create a temporary file for intermediate processing
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Combine video and audio streams without re-encoding
        cmd = [
            'ffmpeg',
            '-i', video_file,     # Input video file
            '-i', audio_file,     # Input audio file
            '-c:v', 'copy',       # Copy video stream without re-encoding
            '-c:a', 'aac',        # Use AAC for audio (widely compatible)
            '-map', '0:v:0',      # Use the first video stream from the first input
            '-map', '1:a:0',      # Use the first audio stream from the second input
            '-shortest',          # Finish encoding when the shortest input stream ends
            '-y',                 # Overwrite output file if it exists
            temp_filename
        ]
        
        logger.info(f"Inserting audio from {audio_file} into {video_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            logger.error(f"Error inserting audio: {result.stderr.decode()}")
            os.unlink(temp_filename)
            sys.exit(1)
        
        # Move the temp file to the output location
        os.replace(temp_filename, output_file)
        
        duration = time.time() - start_time
        logger.info(f"Audio insertion completed in {duration:.2f} seconds")
        logger.info(f"Output saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during audio insertion: {str(e)}")
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        sys.exit(1)


def main():
    """Main entry point of the script."""
    logger = setup_logger()
    
    # Check for FFmpeg dependency
    if not check_ffmpeg():
        logger.error("FFmpeg is not installed or not in PATH. Please install FFmpeg and try again.")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        if args.mode == 'extract':
            extract_audio(args.input, args.output, logger)
        elif args.mode == 'insert':
            insert_audio(args.video, args.audio, args.output, logger)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()