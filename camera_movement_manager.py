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
                      punchin_factor=1.08, frame_rate=None, zoom_duration=100, 
                      shake_duration=60, punchin_duration=50, normal_duration=40,
                      shake_interval=450):
    """
    Apply alternating video effects to a video: zoom, shake, and punch-in.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_rate is None:
        frame_rate = input_fps

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
    print(f"Input FPS: {input_fps}, Output FPS: {frame_rate}")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame based on current state
        if current_state == ZOOMING:
            progress = min(1.0, frame_count / zoom_duration)
            zoom_scale = 1 + (zoom_factor - 1) * progress
            matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
            processed_frame = cv2.warpAffine(frame, matrix, (width, height))
            frame_count += 1
            if frame_count >= zoom_duration:
                current_state = NORMAL
                frame_count = 0
                last_effect = ZOOMING
                print(f"Frame {i}: Zoom effect complete, transitioning to Normal state")

        elif current_state == SHAKING:
            dx = random.uniform(-shake_intensity, shake_intensity)
            dy = random.uniform(-shake_intensity, shake_intensity)
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            processed_frame = cv2.warpAffine(frame, translation_matrix, (width, height))
            frame_count += 1
            if frame_count >= shake_duration:
                current_state = NORMAL
                frame_count = 0
                last_effect = SHAKING
                last_shake_frame = i
                print(f"Frame {i}: Shake effect complete, transitioning to Normal state")

        elif current_state == PUNCHIN:
            progress = frame_count / punchin_duration
            zoom_scale = punchin_factor if progress < 0.5 else 1.0
            matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
            processed_frame = cv2.warpAffine(frame, matrix, (width, height))
            frame_count += 1
            if frame_count >= punchin_duration:
                current_state = NORMAL
                frame_count = 0
                last_effect = PUNCHIN
                print(f"Frame {i}: Punch-in effect complete, transitioning to Normal state")

        elif current_state == NORMAL:
            processed_frame = frame.copy()
            frame_count += 1
            if frame_count >= normal_duration:
                if last_effect == ZOOMING:
                    if (i - last_shake_frame) >= shake_interval:
                        current_state = SHAKING
                        print(f"Frame {i}: Transitioning to Shake state")
                    else:
                        current_state = PUNCHIN
                        print(f"Frame {i}: Skipping Shake (too soon), transitioning to Punch-in state")
                elif last_effect == SHAKING:
                    current_state = PUNCHIN
                    print(f"Frame {i}: Transitioning to Punch-in state")
                else:
                    current_state = ZOOMING
                    print(f"Frame {i}: Transitioning to Zoom state")
                frame_count = 0

        out.write(processed_frame)
        i += 1

    # Final logging
    print(f"Finished processing. Input frames: {total_frames}, Output frames: {i}")

    # Clean up
    cap.release()
    out.release()
    print(f"Video with effects saved to {output_video}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply alternating video effects: zoom, shake and punch-in.")
    parser.add_argument("input_video", help="Path to the input video file (e.g., input_video.mp4)")
    parser.add_argument("output_video", help="Path to the output video file (e.g., processed_video.mp4)")
    parser.add_argument("--zoom_factor", type=float, default=1.1)
    parser.add_argument("--shake_intensity", type=float, default=2)
    parser.add_argument("--punchin_factor", type=float, default=1.08)
    parser.add_argument("--frame_rate", type=int, default=None, help="Frame rate for output video (default: match input)")
    parser.add_argument("--zoom_duration", type=int, default=100)
    parser.add_argument("--shake_duration", type=int, default=60)
    parser.add_argument("--punchin_duration", type=int, default=50)
    parser.add_argument("--normal_duration", type=int, default=40)
    parser.add_argument("--shake_interval", type=int, default=450)
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
