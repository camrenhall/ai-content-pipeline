#!/usr/bin/env python3
"""
Audio Transfer Script for MP4 Videos

This script takes the audio from one MP4 file and applies it to another MP4 file.
It handles edge cases such as duration mismatches and provides logging.

Usage:
    python audio_transfer.py --video source_video.mp4 --audio source_audio.mp4 --output output.mp4
"""

import argparse
import os
import logging
from moviepy import VideoFileClip

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transfer_audio(video_source_path, audio_source_path, output_path):
    """
    Extract audio from one MP4 file and apply it to another MP4 file.
    
    Args:
        video_source_path: Path to the MP4 file that will keep its video track
        audio_source_path: Path to the MP4 file that will provide the audio track
        output_path: Path where the resulting video will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Input validation
        if not os.path.exists(video_source_path):
            logger.error(f"Video source file does not exist: {video_source_path}")
            return False
        
        if not os.path.exists(audio_source_path):
            logger.error(f"Audio source file does not exist: {audio_source_path}")
            return False
        
        # Load the video that will provide the video track
        logger.info(f"Loading video from: {video_source_path}")
        video_clip = VideoFileClip(video_source_path)
        
        # Load the video that will provide the audio track
        logger.info(f"Loading audio from: {audio_source_path}")
        audio_clip = VideoFileClip(audio_source_path, audio=True)
        
        if audio_clip.audio is None:
            logger.error(f"No audio track found in: {audio_source_path}")
            video_clip.close()
            audio_clip.close()
            return False
        
        # Handle duration mismatch
        video_duration = video_clip.duration
        audio_duration = audio_clip.audio.duration
        
        logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
        
        # If audio is longer than video, trim it using subclipped (not subclip)
        if audio_duration > video_duration:
            logger.info(f"Trimming audio to match video duration ({video_duration:.2f}s)")
            audio_clip = audio_clip.subclipped(0, video_duration)
        
        # Set the audio of the video clip
        logger.info("Applying audio to video")
        final_clip = video_clip.with_audio(audio_clip.audio)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Write the output file
        logger.info(f"Writing output to: {output_path}")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        logger.info("Audio transfer completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during audio transfer: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Transfer audio from one MP4 file to another')
    parser.add_argument('--video', required=True, help='Path to the MP4 file that will keep its video track')
    parser.add_argument('--audio', required=True, help='Path to the MP4 file that will provide the audio track')
    parser.add_argument('--output', required=True, help='Path where the resulting video will be saved')
    
    args = parser.parse_args()
    
    result = transfer_audio(args.video, args.audio, args.output)
    
    if not result:
        logger.error("Audio transfer failed")
        exit(1)

if __name__ == "__main__":
    main()