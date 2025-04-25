import argparse
import os
import sys
import json

# Import directly from moviepy instead of moviepy.editor
from moviepy import VideoFileClip, concatenate_videoclips


def insert_multiple_brolls(main_video_path, broll_cuts):
    """
    Insert multiple B-roll clips into a main video at specified timestamps.
    Preserves the original audio of the main video.
    
    Parameters:
    main_video_path (str): Path to the main video file
    broll_cuts (list): List of dictionaries containing broll info:
                      [{"path": str, "timestamp": float, "duration": float}, ...]
    
    Returns:
    VideoFileClip: The final edited video with B-rolls inserted
    """
    try:
        print(f"Loading main video: {main_video_path}")
        # Load the main video
        main_video = VideoFileClip(main_video_path)
        main_duration = main_video.duration
        
        # Sort broll_cuts by timestamp to process them in chronological order
        broll_cuts.sort(key=lambda x: x["timestamp"])
        
        # Validate timestamps and durations
        for i, cut in enumerate(broll_cuts):
            if cut["timestamp"] < 0 or cut["timestamp"] >= main_duration:
                print(f"Warning: B-roll #{i+1} timestamp {cut['timestamp']} is outside the main video duration. Skipping.")
                broll_cuts[i] = None
            elif cut["duration"] <= 0:
                print(f"Warning: B-roll #{i+1} has invalid duration {cut['duration']}. Skipping.")
                broll_cuts[i] = None
        
        # Remove invalid cuts
        broll_cuts = [cut for cut in broll_cuts if cut is not None]
        
        if not broll_cuts:
            print("No valid B-roll cuts to insert.")
            return main_video
        
        # Adjust overlapping timestamps
        for i in range(1, len(broll_cuts)):
            prev_end = broll_cuts[i-1]["timestamp"] + broll_cuts[i-1]["duration"]
            if broll_cuts[i]["timestamp"] < prev_end:
                print(f"Warning: B-roll #{i+1} overlaps with previous B-roll. Adjusting timestamp.")
                broll_cuts[i]["timestamp"] = prev_end
        
        # Build final video segment by segment
        segments = []
        current_time = 0
        
        # Original audio from main video
        original_audio = main_video.audio
        
        for cut in broll_cuts:
            # Add segment of main video before the current B-roll
            if current_time < cut["timestamp"]:
                clip_before = main_video.subclipped(current_time, cut["timestamp"])
                segments.append(clip_before)
            
            # Load and process B-roll
            print(f"Loading B-roll video: {cut['path']}")
            try:
                broll_video = VideoFileClip(cut["path"]).without_audio()
                
                # If B-roll is longer than the specified duration, trim it
                if broll_video.duration > cut["duration"]:
                    print(f"Trimming B-roll to {cut['duration']} seconds")
                    broll_video = broll_video.subclipped(0, cut["duration"])
                
                # Create B-roll segment with original audio
                print(f"Creating B-roll segment at {cut['timestamp']} with original audio")
                audio_for_broll = original_audio.subclipped(cut["timestamp"], 
                                                           min(cut["timestamp"] + cut["duration"], main_duration))
                broll_with_original_audio = broll_video.with_audio(audio_for_broll)
                segments.append(broll_with_original_audio)
                
                # Update current time
                current_time = cut["timestamp"] + cut["duration"]
            except Exception as e:
                print(f"Error processing B-roll {cut['path']}: {e}")
                # If B-roll processing fails, use the original video segment instead
                clip_original = main_video.subclipped(cut["timestamp"], 
                                                     min(cut["timestamp"] + cut["duration"], main_duration))
                segments.append(clip_original)
                current_time = cut["timestamp"] + cut["duration"]
        
        # Add the final segment of the main video after the last B-roll
        if current_time < main_duration:
            final_segment = main_video.subclipped(current_time, main_duration)
            segments.append(final_segment)
        
        print("Concatenating all segments")
        # Concatenate all segments to create the final video
        final_clip = concatenate_videoclips(segments)
        
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
    parser = argparse.ArgumentParser(description='Insert multiple B-roll clips into a video')
    parser.add_argument('--main', required=True, help='Path to the main video file')
    parser.add_argument('--output', required=True, help='Path for the output video file')
    
    # Options for handling multiple B-rolls
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json', help='JSON file containing B-roll cuts information')
    group.add_argument('--broll', action='append', help='Path to a B-roll video file (can be used multiple times)')
    
    # When using individual parameters instead of JSON
    parser.add_argument('--timestamp', type=float, action='append', help='Timestamp (in seconds) to insert the B-roll')
    parser.add_argument('--duration', type=float, action='append', help='Duration (in seconds) for the B-roll')
    
    args = parser.parse_args()
    
    # Check if main video exists
    if not os.path.exists(args.main):
        print(f"Error: Main video file not found at {args.main}")
        return
    
    broll_cuts = []
    
    # Load B-roll cuts from JSON file if provided
    if args.json:
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found at {args.json}")
            return
        
        try:
            with open(args.json, 'r') as f:
                broll_cuts = json.load(f)
                print(f"Loaded {len(broll_cuts)} B-roll cuts from JSON file")
                
                # Validate JSON structure
                for i, cut in enumerate(broll_cuts):
                    if not all(key in cut for key in ["path", "timestamp", "duration"]):
                        print(f"Error: B-roll cut #{i+1} in JSON is missing required fields")
                        return
                    
                    # Check if B-roll file exists
                    if not os.path.exists(cut["path"]):
                        print(f"Error: B-roll file not found at {cut['path']}")
                        return
        except json.JSONDecodeError:
            print("Error: Invalid JSON format")
            return
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return
    
    # Or build broll_cuts from individual parameters
    elif args.broll:
        if not args.timestamp or not args.duration:
            print("Error: When using --broll, you must also provide --timestamp and --duration for each B-roll")
            return
        
        if len(args.broll) != len(args.timestamp) or len(args.broll) != len(args.duration):
            print("Error: Number of B-rolls, timestamps, and durations must match")
            return
        
        for i, (broll_path, timestamp, duration) in enumerate(zip(args.broll, args.timestamp, args.duration)):
            if not os.path.exists(broll_path):
                print(f"Error: B-roll file #{i+1} not found at {broll_path}")
                return
            
            broll_cuts.append({
                "path": broll_path,
                "timestamp": timestamp,
                "duration": duration
            })
    
    # Process the videos
    final_clip = insert_multiple_brolls(args.main, broll_cuts)
    
    # Save the result
    save_edited_video(final_clip, args.output)

if __name__ == "__main__":
    try:
        print(f"Starting multiple B-roll insertion process")
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")