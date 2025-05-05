import cv2
import numpy as np
import argparse
import random

# Define states for our effect state machine
ZOOMING = 0
SHAKING = 1
PUNCHIN = 2
NORMAL = 3

def apply_video_effects(input_video, output_video, zoom_factor=1.1, shake_intensity=2, 
                      punchin_factor=1.08, frame_rate=30, zoom_duration=100, 
                      shake_duration=60, punchin_duration=50, normal_duration=40,
                      shake_interval=450):
    """
    Apply alternating video effects to a video: zoom, shake, and punch-in.
    
    Parameters:
    -----------
    input_video : str
        Path to the input video file
    output_video : str
        Path to save the output video file
    zoom_factor : float
        Maximum zoom factor for slow zoom (default: 1.1)
    shake_intensity : float
        Maximum pixel displacement for camera shake (default: 2)
    punchin_factor : float
        Maximum zoom factor for punch-in effect (default: 1.08)
    frame_rate : int
        Frame rate for the output video
    zoom_duration : int
        Number of frames for the zoom-in effect
    shake_duration : int
        Number of frames for the camera shake effect
    punchin_duration : int
        Number of frames for the punch-in effect
    normal_duration : int
        Number of frames to remain in normal state between effects
    shake_interval : int
        Minimum frames between shake effects (~15 seconds at 30fps)
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer to output the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # State machine variables
    current_state = ZOOMING  # Start with zooming
    frame_count = 0
    zoom_scale = 1.0
    
    # Track the last effect to create proper rotation
    last_effect = None
    
    # Track when the last shake occurred
    last_shake_frame = -shake_interval  # Allow a shake at the start
    
    # Center of the frame
    center_x, center_y = width // 2, height // 2

    print(f"Processing video with effects: zoom={zoom_factor}, shake={shake_intensity}, punch-in={punchin_factor}")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame based on current state
        if current_state == ZOOMING:
            # Apply zoom-in effect gradually
            progress = min(1.0, frame_count / zoom_duration)
            zoom_scale = 1 + (zoom_factor - 1) * progress
            
            # Apply the zoom to the frame
            matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
            processed_frame = cv2.warpAffine(frame, matrix, (width, height))
            
            # Increment frame counter
            frame_count += 1
            
            # Transition to NORMAL state when zoom duration is complete
            if frame_count >= zoom_duration:
                current_state = NORMAL
                frame_count = 0
                last_effect = ZOOMING
                print(f"Frame {i}: Zoom effect complete, transitioning to Normal state")
                
        elif current_state == SHAKING:
            # Apply camera shake effect - random displacement but more subtle
            dx = random.uniform(-shake_intensity, shake_intensity)
            dy = random.uniform(-shake_intensity, shake_intensity)
            
            # Create translation matrix
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            processed_frame = cv2.warpAffine(frame, translation_matrix, (width, height))
            
            # Increment frame counter
            frame_count += 1
            
            # Transition to NORMAL state when shake duration is complete
            if frame_count >= shake_duration:
                current_state = NORMAL
                frame_count = 0
                last_effect = SHAKING
                last_shake_frame = i  # Record when we finished shaking
                print(f"Frame {i}: Shake effect complete, transitioning to Normal state")
        
        elif current_state == PUNCHIN:
            # Apply punch-in effect - completely abrupt zoom
            # Just two states: zoomed in for first half, normal for second half
            progress = frame_count / punchin_duration
            
            if progress < 0.5:
                # First half: fully zoomed in (abrupt punch in)
                zoom_scale = punchin_factor
            else:
                # Second half: back to normal (abrupt return)
                zoom_scale = 1.0
            
            # Apply the zoom to the frame
            matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
            processed_frame = cv2.warpAffine(frame, matrix, (width, height))
            
            # Increment frame counter
            frame_count += 1
            
            # Transition to NORMAL state when punch-in duration is complete
            if frame_count >= punchin_duration:
                current_state = NORMAL
                frame_count = 0
                last_effect = PUNCHIN
                print(f"Frame {i}: Punch-in effect complete, transitioning to Normal state")
                
        elif current_state == NORMAL:
            # No effect in normal state
            processed_frame = frame.copy()
            
            # Increment frame counter
            frame_count += 1
            
            # Transition to next effect when normal duration is complete
            if frame_count >= normal_duration:
                # Choose the next effect based on what we just did and shake interval
                if last_effect == ZOOMING:
                    # Only apply shake if enough time has passed since last shake
                    if (i - last_shake_frame) >= shake_interval:
                        current_state = SHAKING
                        print(f"Frame {i}: Transitioning to Shake state")
                    else:
                        current_state = PUNCHIN
                        print(f"Frame {i}: Skipping Shake (too soon), transitioning to Punch-in state")
                elif last_effect == SHAKING:
                    current_state = PUNCHIN
                    print(f"Frame {i}: Transitioning to Punch-in state")
                else:  # last_effect == PUNCHIN
                    current_state = ZOOMING
                    print(f"Frame {i}: Transitioning to Zoom state")
                frame_count = 0
        
        # Write the processed frame to the output video
        out.write(processed_frame)

    # Clean up
    cap.release()
    out.release()
    print(f"Video with effects saved to {output_video}")

    # Clean up
    cap.release()
    out.release()
    print(f"Video with effects saved to {output_video}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply alternating video effects: zoom, shake and punch-in.")
    parser.add_argument("input_video", help="Path to the input video file (e.g., input_video.mp4)")
    parser.add_argument("output_video", help="Path to the output video file (e.g., processed_video.mp4)")
    parser.add_argument("--zoom_factor", type=float, default=1.1, help="Zoom factor for slow zoom (default: 1.1)")
    parser.add_argument("--shake_intensity", type=float, default=2, help="Maximum pixel displacement for camera shake (default: 2)")
    parser.add_argument("--punchin_factor", type=float, default=1.08, help="Zoom factor for punch-in effect (default: 1.08)")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for the output video (default: 30)")
    parser.add_argument("--zoom_duration", type=int, default=100, help="Number of frames for the zoom-in effect (default: 100)")
    parser.add_argument("--shake_duration", type=int, default=60, help="Number of frames for the camera shake effect (default: 60)")
    parser.add_argument("--punchin_duration", type=int, default=50, help="Number of frames for the punch-in effect (default: 50)")
    parser.add_argument("--normal_duration", type=int, default=40, help="Number of frames between effects (default: 40)")
    parser.add_argument("--shake_interval", type=int, default=450, help="Minimum frames between shake effects (default: 450, ~15 sec at 30fps)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    apply_video_effects(
        args.input_video, 
        args.output_video,
        args.zoom_factor,
        args.shake_intensity,
        args.punchin_factor,
        args.frame_rate,
        args.zoom_duration,
        args.shake_duration,
        args.punchin_duration,
        args.normal_duration,
        args.shake_interval
    )

if __name__ == "__main__":
    main()