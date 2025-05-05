import cv2
import numpy as np
import argparse

def apply_subtle_zoom(input_video, output_video, zoom_factor=1.1, frame_rate=30, zoom_duration=100, snap_back_duration=1, delay_duration=100):
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer to output the zoomed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Variables for zoom
    zoom_scale = 1
    zoom_in = True  # Flag to alternate between zoom-in and normal
    zoom_in_frame_count = 0
    delay_frame_count = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if zoom_in:
            # Apply zoom-in effect gradually
            zoom_scale = 1 + (zoom_factor - 1) * (zoom_in_frame_count / zoom_duration)
            zoom_in_frame_count += 1

            # When zoom-in duration is complete, snap back to normal
            if zoom_in_frame_count >= zoom_duration:
                zoom_in = False  # Start snapping back to normal
                zoom_in_frame_count = 0  # Reset zoom-in counter
        else:
            # Snap back to normal zoom level
            zoom_scale = 1
            delay_frame_count += 1

            # After the snap back duration, start zooming in again
            if delay_frame_count >= delay_duration:
                zoom_in = True
                delay_frame_count = 0  # Reset delay counter

        # Get the center of the frame
        center_x, center_y = width // 2, height // 2

        # Apply the zoom to the frame
        matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_scale)
        zoomed_frame = cv2.warpAffine(frame, matrix, (width, height))

        # Write the zoomed frame to the output video
        out.write(zoomed_frame)

    cap.release()
    out.release()
    print(f"Zoomed video saved to {output_video}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply subtle zoom-in effects with hard snap back to normal.")
    parser.add_argument("input_video", help="Path to the input video file (e.g., input_video.mp4)")
    parser.add_argument("output_video", help="Path to the output video file (e.g., zoom_in_video.mp4)")
    parser.add_argument("--zoom_factor", type=float, default=1.1, help="Zoom factor (default: 1.1)")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for the output video (default: 30)")
    parser.add_argument("--zoom_duration", type=int, default=100, help="Number of frames for the zoom-in effect (default: 100)")
    parser.add_argument("--snap_back_duration", type=int, default=1, help="Duration for snapping back to normal (default: 1 frame)")
    parser.add_argument("--delay_duration", type=int, default=100, help="Frames of delay before the next zoom-in effect (default: 100)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    apply_subtle_zoom(args.input_video, args.output_video, args.zoom_factor, args.frame_rate, args.zoom_duration, args.snap_back_duration, args.delay_duration)

if __name__ == "__main__":
    main()
