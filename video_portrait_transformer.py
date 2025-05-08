# video_transformer.py
import ffmpeg
import os
import sys
import argparse
from pathlib import Path
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def is_portrait_1080x1920(video_path):
    """Check if video is already in 1080x1920 portrait format"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] 
                            if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            return False
            
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        return width == 1080 and height == 1920
    except Exception as e:
        logger.error(f"Error checking video dimensions: {str(e)}")
        return False

def crop_to_portrait(input_path, output_path, width=1080, height=1920):
    """Convert video to portrait (1080x1920)"""
    try:
        # Check if already in target format
        if is_portrait_1080x1920(input_path):
            logger.info(f"Video already in 1080x1920 format, skipping: {input_path}")
            # Create a symbolic link or copy the file to output_path if needed
            if input_path != output_path:
                import shutil
                shutil.copy2(input_path, output_path)
                logger.info(f"Copied original file to: {output_path}")
            return output_path
        
        # Get video information
        logger.info(f"Analyzing video: {input_path}")
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] 
                            if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            logger.error("Error: No video stream found")
            return None
            
        orig_width = int(video_stream['width'])
        orig_height = int(video_stream['height'])
        logger.info(f"Original dimensions: {orig_width}x{orig_height}")
        
        # Check for audio stream
        audio_stream = next((stream for stream in probe['streams'] 
                        if stream['codec_type'] == 'audio'), None)
        
        # Calculate aspect ratios
        original_aspect = orig_width / orig_height
        target_aspect = 9 / 16  # Portrait aspect ratio
        
        # Build the ffmpeg pipeline
        pipeline = ffmpeg.input(input_path)
        
        # Adjust the video based on its original dimensions
        if original_aspect > target_aspect:
            # Video is too wide - need to crop width
            target_width = int(orig_height * target_aspect)
            crop_x = int((orig_width - target_width) / 2)
            logger.info(f"Video too wide, cropping: width={target_width}, x={crop_x}")
            pipeline = pipeline.crop(crop_x, 0, target_width, orig_height)
        elif original_aspect < target_aspect:
            # Video is too tall - need to crop height or pad width
            # Option 1: Crop height
            # target_height = int(orig_width / target_aspect)
            # crop_y = int((orig_height - target_height) / 2)
            # logger.info(f"Video too tall, cropping: height={target_height}, y={crop_y}")
            # pipeline = pipeline.crop(0, crop_y, orig_width, target_height)
            
            # Option 2: Pad width (this avoids losing vertical content)
            target_width = int(orig_height * target_aspect)
            pad_x = int((target_width - orig_width) / 2)
            logger.info(f"Video too tall, padding: adding {pad_x} pixels to each side")
            pipeline = pipeline.filter('pad', target_width, orig_height, pad_x, 0)
        
        # Scale to final dimensions
        pipeline = pipeline.filter('scale', width, height)
        
        # Add proper audio handling
        if audio_stream:
            pipeline = pipeline.output(output_path, acodec='aac', vcodec='libx264')
        else:
            pipeline = pipeline.output(output_path, vcodec='libx264')
        
        # Execute the ffmpeg command
        pipeline.overwrite_output().run()
        
        logger.info(f"Successfully converted video to portrait: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None

def process_videos(input_videos, output_dir=None, width=1080, height=1920):
    """
    Process multiple videos, converting them to portrait format
    
    Args:
        input_videos (list): List of video file paths to process
        output_dir (str, optional): Directory for output videos. If None, uses same directory as input
        width (int): Target width
        height (int): Target height
    
    Returns:
        list: Paths to successfully processed videos
    """
    successful_outputs = []
    
    for input_video in input_videos:
        input_path = Path(input_video)
        
        if not input_path.exists():
            logger.error(f"Error: Input video not found at {input_path}")
            continue
            
        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}_portrait{input_path.suffix}"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_portrait{input_path.suffix}")
            
        # Process the video
        result = crop_to_portrait(str(input_path), str(output_path), width, height)
        
        if result:
            successful_outputs.append(result)
            logger.info(f"Video processing complete. Output saved to: {result}")
        else:
            logger.error(f"Video processing failed for: {input_path}")
    
    return successful_outputs

def main():
    # Create a parser without adding the height argument initially
    parser = argparse.ArgumentParser(description='Convert videos to portrait format (1080x1920)')
    parser.add_argument('input', nargs='+', help='Input video file(s)')
    parser.add_argument('-o', '--output-dir', help='Output directory (optional)')
    parser.add_argument('-w', '--width', type=int, default=1080, help='Target width (default: 1080)')
    parser.add_argument('--height', type=int, default=1920, help='Target height (default: 1920)')
    
    args = parser.parse_args()
    
    successful_outputs = process_videos(args.input, args.output_dir, args.width, args.height)
    
    logger.info(f"Completed processing {len(successful_outputs)} of {len(args.input)} videos")
    
    if not successful_outputs:
        logger.error("All video processing operations failed")
        sys.exit(1)

if __name__ == "__main__":
    main()