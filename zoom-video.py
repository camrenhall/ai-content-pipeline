import cv2
import numpy as np
import argparse
import random

# Define states for our effect state machine
ZOOMING = 0
SHAKING = 1
NORMAL = 2

def apply_video_effects(input_video, output_video, zoom_factor=1.1, shake_intensity=5, 
                      frame_rate=30, zoom_duration=100, shake_duration=60, 
                      normal_duration=40):
    """
    Apply alternating zoom and camera shake effects to a video.
    
    Parameters:
    -----------
    input_video : str
        Path to the input video file
    output_video : str
        Path to save the output video file
    zoom_factor : float
        Maximum zoom factor (default: 1.1)
    shake_intensity : float
        Maximum pixel displacement for camera shake (default: 5)
    frame_rate : int
        Frame rate for the output video
    zoom_duration : int
        Number of frames for the zoom-in effect
    shake_duration : int
        Number of frames for the camera shake effect
    normal_duration : int
        Number of frames to remain in normal state between effects
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
    
    # Track the previous effect to alternate properly
    previous_effect_was_zoom = False
    
    # Center of the frame
    center_x, center_y = width // 2, height // 2

    print(f"Processing video with effects: zoom={zoom_factor}, shake={shake_intensity}")

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
                previous_effect_was_zoom = True
                print(f"Frame {i}: Zoom effect complete, transitioning to Normal state")
                
        elif current_state == SHAKING:
            # Apply camera shake effect - random displacement
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
                previous_effect_was_zoom = False
                print(f"Frame {i}: Shake effect complete, transitioning to Normal state")
                
        elif current_state == NORMAL:
            # No effect in normal state
            processed_frame = frame.copy()
            
            # Increment frame counter
            frame_count += 1
            
            # Transition to next effect when normal duration is complete
            if frame_count >= normal_duration:
                # Alternate between zoom and shake based on the previous effect
                if previous_effect_was_zoom:
                    current_state = SHAKING
                    print(f"Frame {i}: Transitioning to Shake state")
                else:
                    current_state = ZOOMING
                    print(f"Frame {i}: Transitioning to Zoom state")
                frame_count = 0
        
        # Write the processed frame to the output video
        out.write(processed_frame)

    # Clean up
    cap.release()
    out.release()
    print(f"Video with effects saved to {output_video}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply alternating zoom and camera shake effects to video.")
    parser.add_argument("input_video", help="Path to the input video file (e.g., input_video.mp4)")
    parser.add_argument("output_video", help="Path to the output video file (e.g., processed_video.mp4)")
    parser.add_argument("--zoom_factor", type=float, default=1.1, help="Zoom factor (default: 1.1)")
    parser.add_argument("--shake_intensity", type=float, default=5, help="Maximum pixel displacement for camera shake (default: 5)")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for the output video (default: 30)")
    parser.add_argument("--zoom_duration", type=int, default=100, help="Number of frames for the zoom-in effect (default: 100)")
    parser.add_argument("--shake_duration", type=int, default=60, help="Number of frames for the camera shake effect (default: 60)")
    parser.add_argument("--normal_duration", type=int, default=40, help="Number of frames between effects (default: 40)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    apply_video_effects(
        args.input_video, 
        args.output_video,
        args.zoom_factor,
        args.shake_intensity,
        args.frame_rate,
        args.zoom_duration,
        args.shake_duration,
        args.normal_duration
    )

if __name__ == "__main__":
    main()