# test_ffmpeg_python.py
import ffmpeg
import os
import sys

def crop_to_portrait(input_path, output_path, width=1080, height=1920):
    """Convert video to portrait (1080x1920)"""
    try:
        # Get video information
        print(f"Analyzing video: {input_path}")
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] 
                            if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            print("Error: No video stream found")
            return None
            
        orig_width = int(video_stream['width'])
        orig_height = int(video_stream['height'])
        print(f"Original dimensions: {orig_width}x{orig_height}")
        
        # Calculate crop width to maintain 9:16 aspect ratio
        target_width = int(orig_height * 9/16)
        crop_x = int((orig_width - target_width) / 2)
        print(f"Crop parameters: width={target_width}, x={crop_x}")
        
        # Process using ffmpeg
        print(f"Processing video to {output_path}...")
        
        # Check if audio stream exists
        audio_stream = next((stream for stream in probe['streams'] 
                            if stream['codec_type'] == 'audio'), None)
        
        # Build the ffmpeg pipeline
        pipeline = (
            ffmpeg
            .input(input_path)
            .crop(crop_x, 0, target_width, orig_height)
            .filter('scale', width, height)
        )
        
        # Add proper audio handling
        if audio_stream:
            pipeline = pipeline.output(output_path, acodec='aac', vcodec='libx264')
        else:
            pipeline = pipeline.output(output_path, vcodec='libx264')
        
        # Execute the ffmpeg command
        pipeline.overwrite_output().run()
        
        print(f"Successfully converted video to portrait: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        input_video = "test_video2.mp4"
        
    if len(sys.argv) > 2:
        output_video = sys.argv[2]
    else:
        output_video = "test_video_portrait2.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        sys.exit(1)
    
    result = crop_to_portrait(input_video, output_video)
    
    if result:
        print(f"Video processing complete. Output saved to: {result}")
    else:
        print("Video processing failed.")
        sys.exit(1)