import argparse
import os
import sys

# Import directly from moviepy instead of moviepy.editor
from moviepy import VideoFileClip, concatenate_videoclips


def insert_broll(main_video_path, broll_video_path, timestamp, broll_duration):
    """
    Insert a B-roll clip into a main video at a specified timestamp.
    Preserves the original audio of the main video.
    
    Parameters:
    main_video_path (str): Path to the main video file
    broll_video_path (str): Path to the B-roll video file
    timestamp (float): Time in seconds where the B-roll should be inserted
    broll_duration (float): Duration in seconds for how long the B-roll should play
    
    Returns:
    VideoFileClip: The final edited video with B-roll inserted
    """
    try:
        print(f"Loading main video: {main_video_path}")
        # Load the main video
        main_video = VideoFileClip(main_video_path)
        
        print(f"Loading B-roll video: {broll_video_path}")
        # Load the B-roll video without audio
        broll_video = VideoFileClip(broll_video_path).without_audio()
        
        # If B-roll is longer than the specified duration, trim it
        if broll_video.duration > broll_duration:
            print(f"Trimming B-roll to {broll_duration} seconds")
            # In v2, method renamed from subclip to subclipped
            broll_video = broll_video.subclipped(0, broll_duration)
        
        print(f"Splitting main video at timestamp {timestamp}")
        # Split the main video at the timestamp
        clip_before = main_video.subclipped(0, timestamp)
        clip_after = main_video.subclipped(timestamp + broll_duration)
        
        # Extract the original audio from the main video
        print("Extracting audio from main video")
        original_audio = main_video.audio
        
        # Create the middle part with B-roll video but maintain original audio
        print("Creating B-roll segment with original audio")
        audio_for_broll = original_audio.subclipped(timestamp, timestamp + broll_duration)
        broll_with_original_audio = broll_video.with_audio(audio_for_broll)
        
        print("Concatenating clips")
        # Concatenate the clips in the desired order
        final_clip = concatenate_videoclips([clip_before, broll_with_original_audio, clip_after])
        
        return final_clip
        
    except Exception as e:
        print(f"Error processing videos: {e}")
        return None

def save_edited_video(final_clip, output_path):
    """
    Save the edited video to the specified output path.
    
    Parameters:
    final_clip (VideoFileClip): The edited video clip
    output_path (str): Path where the edited video should be saved
    """
    if final_clip is not None:
        try:
            print(f"Writing output video to {output_path}")
            # In v2, write_videofile may have different parameters
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print(f"Video saved successfully to {output_path}")
            
            # Close the clips to free resources
            final_clip.close()
        except Exception as e:
            print(f"Error saving video: {e}")
    else:
        print("No video to save. Please check for errors above.")

def main():
    parser = argparse.ArgumentParser(description='Insert B-roll into a video at a specified timestamp')
    parser.add_argument('--main', required=True, help='Path to the main video file')
    parser.add_argument('--broll', required=True, help='Path to the B-roll video file')
    parser.add_argument('--timestamp', type=float, required=True, help='Timestamp (in seconds) to insert the B-roll')
    parser.add_argument('--duration', type=float, required=True, help='Duration (in seconds) for the B-roll')
    parser.add_argument('--output', required=True, help='Path for the output video file')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.main):
        print(f"Error: Main video file not found at {args.main}")
        return
        
    if not os.path.exists(args.broll):
        print(f"Error: B-roll file not found at {args.broll}")
        return
    
    # Process the videos
    final_clip = insert_broll(args.main, args.broll, args.timestamp, args.duration)
    
    # Save the result
    save_edited_video(final_clip, args.output)

if __name__ == "__main__":
    try:
        print(f"Starting B-roll insertion process")
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")