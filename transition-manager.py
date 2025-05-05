import cv2
import numpy as np
import os
from tqdm import tqdm

class VideoTransitionEffects:
    def __init__(self, input_path, output_path, transition_duration=30):
        """
        Initialize the video transition effects processor.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the output video file
            transition_duration: Duration of transition in frames
        """
        self.input_path = input_path
        self.output_path = output_path
        self.transition_duration = transition_duration
        
        # Open the input video
        self.video = cv2.VideoCapture(input_path)
        if not self.video.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize the output video writer
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    
    def detect_scene_changes(self, threshold=45, min_scene_length=90, max_scenes=None):
        """
        Detect scene changes in the video to apply transitions at those points.
        
        Args:
            threshold: Threshold for scene change detection (higher = fewer detections)
            min_scene_length: Minimum length of a scene in frames (higher = fewer transitions)
            max_scenes: Maximum number of scenes to detect (None for unlimited)
            
        Returns:
            List of frame indices where scene changes occur
        """
        print("Detecting scene changes...")
        scene_changes = []
        prev_frame = None
        frame_idx = 0
        
        # Calculate target number of transitions based on video length
        # Aiming for about 4-5 transitions per 15 seconds
        video_duration_seconds = self.total_frames / self.fps
        target_transitions = int((video_duration_seconds / 15) * 4.5)
        if max_scenes is None:
            max_scenes = target_transitions
        
        print(f"Video duration: {video_duration_seconds:.2f} seconds")
        print(f"Target number of transitions: {target_transitions}")
        
        # Reset video to the beginning
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Store differences for adaptive thresholding
        differences = []
        frames_analyzed = 0
        
        # First pass: collect frame differences to determine adaptive threshold
        with tqdm(total=self.total_frames, desc="Analyzing frame differences") as pbar:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                
                # Convert frame to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate absolute difference between current and previous frame
                    diff = cv2.absdiff(gray, prev_frame)
                    non_zero = np.count_nonzero(diff > threshold)
                    percent_changed = non_zero / (self.width * self.height)
                    differences.append(percent_changed)
                
                prev_frame = gray
                frames_analyzed += 1
                pbar.update(1)
                
                # Analyze only a subset of frames for long videos to speed up processing
                if frames_analyzed > min(1000, self.total_frames):
                    break
        
        # Determine adaptive threshold based on collected differences
        if differences:
            differences = np.array(differences)
            # Sort differences and take a high percentile to determine significant changes
            # Using 95th percentile to identify significant scene changes
            adaptive_threshold = np.percentile(differences, 95) 
            print(f"Adaptive threshold determined: {adaptive_threshold:.6f}")
        else:
            adaptive_threshold = 0.05  # Fallback default
        
        # Reset for actual scene detection
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_frame = None
        frame_idx = 0
        
        with tqdm(total=self.total_frames, desc="Detecting scenes") as pbar:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                
                # Skip frames at the beginning (often contain intros/titles)
                if frame_idx < self.fps * 1:  # Skip first second
                    frame_idx += 1
                    pbar.update(1)
                    continue
                
                # Convert frame to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate absolute difference between current and previous frame
                    diff = cv2.absdiff(gray, prev_frame)
                    non_zero = np.count_nonzero(diff > threshold)
                    percent_changed = non_zero / (self.width * self.height)
                    
                    # If the difference is significant based on adaptive threshold
                    if percent_changed > adaptive_threshold:
                        # Check if the new scene change is far enough from the previous one
                        if not scene_changes or (frame_idx - scene_changes[-1]) >= min_scene_length:
                            scene_changes.append(frame_idx)
                            print(f"Scene change detected at frame {frame_idx} ({frame_idx/self.fps:.2f}s)")
                            
                            # Check if we've reached the maximum number of scenes
                            if max_scenes and len(scene_changes) >= max_scenes:
                                print(f"Reached maximum number of scenes ({max_scenes})")
                                break
                
                prev_frame = gray
                frame_idx += 1
                pbar.update(1)
        
        # If we detected too many scenes, prioritize the most significant changes
        if len(scene_changes) > max_scenes:
            # Calculate the importance of each scene change (can be based on difference magnitude)
            scene_importances = []
            for scene_idx in scene_changes:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, max(0, scene_idx - 1))
                ret1, frame1 = self.video.read()
                
                self.video.set(cv2.CAP_PROP_POS_FRAMES, scene_idx)
                ret2, frame2 = self.video.read()
                
                if ret1 and ret2:
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray1, gray2)
                    importance = np.sum(diff) / (self.width * self.height * 255)
                    scene_importances.append((scene_idx, importance))
                else:
                    scene_importances.append((scene_idx, 0))
            
            # Sort by importance and take the top max_scenes
            scene_importances.sort(key=lambda x: x[1], reverse=True)
            scene_changes = [idx for idx, _ in scene_importances[:max_scenes]]
            scene_changes.sort()  # Re-sort by frame index
        
        # Ensure more even distribution of transitions
        if len(scene_changes) > 1:
            # Calculate the average frame distance between transitions
            avg_distance = (scene_changes[-1] - scene_changes[0]) / (len(scene_changes) - 1)
            min_allowed_distance = avg_distance * 0.6  # Allow some variation
            
            # Filter out transitions that are too close to each other
            filtered_scenes = [scene_changes[0]]
            for i in range(1, len(scene_changes)):
                if scene_changes[i] - filtered_scenes[-1] >= min_allowed_distance:
                    filtered_scenes.append(scene_changes[i])
            
            scene_changes = filtered_scenes
        
        print(f"Final detected scene changes: {len(scene_changes)}")
        return scene_changes
    
    def fade_transition(self, frame1, frame2, progress):
        """
        Apply a fade transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
    
    def wipe_transition(self, frame1, frame2, progress, direction="left"):
        """
        Apply a wipe transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            progress: Transition progress (0 to 1)
            direction: Direction of the wipe ("left", "right", "up", "down")
            
        Returns:
            Blended frame
        """
        result = frame1.copy()
        h, w = frame1.shape[:2]
        
        if direction == "left":
            edge = int(w * progress)
            result[:, :edge] = frame2[:, :edge]
        elif direction == "right":
            edge = int(w * (1 - progress))
            result[:, edge:] = frame2[:, edge:]
        elif direction == "up":
            edge = int(h * progress)
            result[:edge, :] = frame2[:edge, :]
        elif direction == "down":
            edge = int(h * (1 - progress))
            result[edge:, :] = frame2[edge:, :]
        
        return result
    
    def zoom_transition(self, frame1, frame2, progress):
        """
        Apply a zoom transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        
        # Zoom out frame1
        scale1 = 1 + 0.2 * progress
        zoomed_w1 = int(w * scale1)
        zoomed_h1 = int(h * scale1)
        zoomed_frame1 = cv2.resize(frame1, (zoomed_w1, zoomed_h1))
        
        # Extract the center portion
        start_x1 = (zoomed_w1 - w) // 2
        start_y1 = (zoomed_h1 - h) // 2
        zoomed_frame1 = zoomed_frame1[start_y1:start_y1+h, start_x1:start_x1+w]
        
        # Zoom in frame2
        scale2 = 1.2 - 0.2 * progress
        zoomed_w2 = int(w * scale2)
        zoomed_h2 = int(h * scale2)
        
        # Ensure minimum size
        zoomed_w2 = max(zoomed_w2, 1)
        zoomed_h2 = max(zoomed_h2, 1)
        
        zoomed_frame2 = cv2.resize(frame2, (zoomed_w2, zoomed_h2))
        
        # Pad the frame if needed
        if zoomed_w2 < w or zoomed_h2 < h:
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            start_x2 = (w - zoomed_w2) // 2
            start_y2 = (h - zoomed_h2) // 2
            padded[start_y2:start_y2+zoomed_h2, start_x2:start_x2+zoomed_w2] = zoomed_frame2
            zoomed_frame2 = padded
        else:
            # Extract the center portion
            start_x2 = (zoomed_w2 - w) // 2
            start_y2 = (zoomed_h2 - h) // 2
            zoomed_frame2 = zoomed_frame2[start_y2:start_y2+h, start_x2:start_x2+w]
        
        # Blend the frames
        return cv2.addWeighted(zoomed_frame1, 1 - progress, zoomed_frame2, progress, 0)
    
    def rotate_transition(self, frame1, frame2, progress):
        """
        Apply a rotate transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotate frame1 out
        angle1 = 90 * progress
        rotation_matrix1 = cv2.getRotationMatrix2D(center, angle1, 1.0 - 0.5 * progress)
        rotated_frame1 = cv2.warpAffine(frame1, rotation_matrix1, (w, h))
        
        # Rotate frame2 in
        angle2 = 90 * (1 - progress)
        rotation_matrix2 = cv2.getRotationMatrix2D(center, -angle2, 0.5 + 0.5 * progress)
        rotated_frame2 = cv2.warpAffine(frame2, rotation_matrix2, (w, h))
        
        # Blend the frames
        return cv2.addWeighted(rotated_frame1, 1 - progress, rotated_frame2, progress, 0)
    
    def blur_transition(self, frame1, frame2, progress):
        """
        Apply a blur transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        # Calculate blur amount (max at progress = 0.5)
        blur_amount = int(50 * (1 - abs(2 * progress - 1)))
        blur_amount = max(1, blur_amount)  # Ensure odd kernel size
        if blur_amount % 2 == 0:
            blur_amount += 1
        
        # Apply blur to both frames
        blurred_frame1 = cv2.GaussianBlur(frame1, (blur_amount, blur_amount), 0)
        blurred_frame2 = cv2.GaussianBlur(frame2, (blur_amount, blur_amount), 0)
        
        # Crossfade between original frame1 and blurred frame1
        if progress <= 0.5:
            normalized_progress = progress * 2
            temp1 = cv2.addWeighted(frame1, 1 - normalized_progress, blurred_frame1, normalized_progress, 0)
            result = temp1
        # Crossfade between blurred frame2 and original frame2
        else:
            normalized_progress = (progress - 0.5) * 2
            temp2 = cv2.addWeighted(blurred_frame2, 1 - normalized_progress, frame2, normalized_progress, 0)
            result = temp2
        
        return result
    
    def process_video(self, transition_type="fade", custom_scenes=None):
        """
        Process the video and apply transitions at scene changes.
        
        Args:
            transition_type: Type of transition to apply ("fade", "wipe", "zoom", "rotate", "blur")
            custom_scenes: Custom list of frame indices for scene changes (optional)
        """
        # Detect scene changes if custom scenes are not provided
        scene_changes = custom_scenes or self.detect_scene_changes()
        
        if not scene_changes:
            print("No scene changes detected. Using equally spaced transitions.")
            # Create equally spaced transitions every 3 seconds if no scenes detected
            target_transitions = int((self.total_frames / self.fps / 15) * 4.5)  # 4-5 per 15 seconds
            scene_spacing = int(self.total_frames / (target_transitions + 1))
            scene_changes = list(range(scene_spacing, self.total_frames - self.transition_duration, scene_spacing))
        
        # Map transition type to function
        transition_functions = {
            "fade": self.fade_transition,
            "wipe_left": lambda f1, f2, p: self.wipe_transition(f1, f2, p, "left"),
            "wipe_right": lambda f1, f2, p: self.wipe_transition(f1, f2, p, "right"),
            "wipe_up": lambda f1, f2, p: self.wipe_transition(f1, f2, p, "up"),
            "wipe_down": lambda f1, f2, p: self.wipe_transition(f1, f2, p, "down"),
            "zoom": self.zoom_transition,
            "rotate": self.rotate_transition,
            "blur": self.blur_transition
        }
        
        if transition_type not in transition_functions:
            raise ValueError(f"Unsupported transition type: {transition_type}")
        
        transition_func = transition_functions[transition_type]
        
        # Create output video writer
        writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
        
        # Reset video to the beginning
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Pre-calculate transition ranges
        transition_ranges = []
        for scene_idx in scene_changes:
            # Ensure transitions don't start too close to the beginning or end
            start = max(0, scene_idx - self.transition_duration // 2)
            end = min(self.total_frames - 1, start + self.transition_duration)
            transition_ranges.append((start, end))
        
        # Pre-load frames for all transitions
        transition_frames = {}
        for start, end in transition_ranges:
            # Get frame before transition
            self.video.set(cv2.CAP_PROP_POS_FRAMES, max(0, start - 1))
            ret, frame_before = self.video.read()
            if not ret:
                continue
            
            # Get frame after transition
            self.video.set(cv2.CAP_PROP_POS_FRAMES, min(self.total_frames - 1, end))
            ret, frame_after = self.video.read()
            if not ret:
                continue
            
            # Store frames for this transition
            transition_frames[start] = {
                'before': frame_before,
                'after': frame_after
            }
        
        # Check if frame is in a transition
        def in_transition(idx):
            for start, end in transition_ranges:
                if start <= idx < end:
                    return start, end
            return None
        
        # Reset video to beginning to start processing
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("Processing video with transitions...")
        frame_idx = 0
        with tqdm(total=self.total_frames) as pbar:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                
                transition_info = in_transition(frame_idx)
                
                if transition_info:
                    # We're in a transition
                    start, end = transition_info
                    
                    # Only apply transition if we have the frames for it
                    if start in transition_frames:
                        # Calculate progress for this transition frame (0 to 1)
                        progress = (frame_idx - start) / (end - start)
                        
                        # Apply transition
                        frame_before = transition_frames[start]['before']
                        frame_after = transition_frames[start]['after']
                        transition_frame = transition_func(frame_before, frame_after, progress)
                        writer.write(transition_frame)
                    else:
                        # If we don't have transition frames for this point, write the original frame
                        writer.write(frame)
                else:
                    # No transition, write frame as is
                    writer.write(frame)
                
                frame_idx += 1
                pbar.update(1)
        
        # Release resources
        self.video.release()
        writer.release()
        print(f"Video with {transition_type} transitions saved to {self.output_path}")
    
    def close(self):
        """Release video resources"""
        if self.video.isOpened():
            self.video.release()


def main():
    """Main function to demonstrate the usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add transition effects to videos")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--transition", "-t", default="fade", choices=[
        "fade", "wipe_left", "wipe_right", "wipe_up", "wipe_down", "zoom", "rotate", "blur"
    ], help="Transition effect type")
    parser.add_argument("--duration", "-d", type=int, default=30, 
                        help="Transition duration in frames")
    parser.add_argument("--scenes", "-s", type=str, default=None,
                        help="Comma-separated list of frame indices for custom scene changes")
    parser.add_argument("--max-scenes", "-m", type=int, default=None,
                        help="Maximum number of scene changes to detect")
    
    args = parser.parse_args()
    
    # Parse custom scenes if provided
    custom_scenes = None
    if args.scenes:
        custom_scenes = [int(idx) for idx in args.scenes.split(",")]
    
    # Process video
    processor = VideoTransitionEffects(args.input, args.output, args.duration)
    try:
        processor.process_video(args.transition, custom_scenes)
    finally:
        processor.close()


if __name__ == "__main__":
    main()