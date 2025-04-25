# integrated_broll_pipeline.py
import argparse
import os
import sys
import json
import tempfile

# Import from our existing modules
from enhanced_broll_inserter import insert_multiple_brolls, save_edited_video
from video_portrait_transformer import crop_to_portrait

def process_and_insert_brolls(main_video_path, broll_cuts, output_path):
    """
    Main pipeline function that:
    1. Transforms each B-roll to portrait format
    2. Inserts the transformed B-rolls into the main video
    3. Saves the final output
    
    Parameters:
    main_video_path (str): Path to the main video file
    broll_cuts (list): List of dictionaries containing broll info:
                      [{"path": str, "timestamp": float, "duration": float}, ...]
    output_path (str): Path for the output video file
    """
    # Create a temporary directory for transformed B-rolls
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for transformed B-rolls: {temp_dir}")
        
        # Transform each B-roll to portrait format
        transformed_broll_cuts = []
        for i, cut in enumerate(broll_cuts):
            original_path = cut["path"]
            
            # Skip if the B-roll file doesn't exist
            if not os.path.exists(original_path):
                print(f"Warning: B-roll file not found at {original_path}. Skipping.")
                continue
                
            # Define the path for the transformed B-roll
            transformed_path = os.path.join(temp_dir, f"transformed_broll_{i}.mp4")
            
            print(f"Transforming B-roll {i+1}/{len(broll_cuts)}: {original_path}")
            # Transform the B-roll to portrait mode
            transformed_path = crop_to_portrait(original_path, transformed_path)
            
            if transformed_path:
                # Update the cut with the transformed path
                cut_copy = cut.copy()  # Create a copy to avoid modifying the original
                cut_copy["path"] = transformed_path
                transformed_broll_cuts.append(cut_copy)
                print(f"Successfully transformed B-roll {i+1}")
            else:
                print(f"Failed to transform B-roll {i+1}. Skipping.")
        
        if not transformed_broll_cuts:
            print("No valid transformed B-rolls to insert.")
            return False
        
        # Insert the transformed B-rolls into the main video
        print(f"Inserting {len(transformed_broll_cuts)} transformed B-rolls into main video")
        final_clip = insert_multiple_brolls(main_video_path, transformed_broll_cuts)
        
        # Save the final video
        if final_clip:
            save_edited_video(final_clip, output_path)
            return True
        else:
            print("Failed to create the final video.")
            return False

def main():
    parser = argparse.ArgumentParser(description='Transform and insert multiple B-roll clips into a video')
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
            broll_cuts.append({
                "path": broll_path,
                "timestamp": timestamp,
                "duration": duration
            })
    
    # Process the videos through the integrated pipeline
    process_and_insert_brolls(args.main, broll_cuts, args.output)

if __name__ == "__main__":
    try:
        print(f"Starting integrated B-roll pipeline process")
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")