import cv2
import numpy as np
import os
import json
import random
from tqdm import tqdm

class VideoTransitionEffects:
    def __init__(self, input_path, output_path, transition_duration=20):
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
        
        # Initialize transition functions dictionary
        self.transition_functions = {
            # Basic transitions
            "cross_fade": self.cross_fade_transition,
            "slide_left": self.slide_transition_left,
            "slide_right": self.slide_transition_right,
            "slide_up": self.slide_transition_up,
            "slide_down": self.slide_transition_down,
            "zoom_blend": self.zoom_blend_transition,
            "whip_pan": self.whip_pan_transition,
            "push_pull": self.push_pull_transition,
            "overlay_dissolve": self.overlay_dissolve_transition,
            
            # Advanced transitions
            "kaleidoscope": self.kaleidoscope_transition,
            "glitch": self.glitch_transition,
            "particle": self.particle_transition,
            "shatter": self.shatter_transition,
            "ripple": self.ripple_transition,
            "pixelate": self.pixelate_transition,
            "cube_rotate": self.cube_rotate_transition,
            "swirl": self.swirl_transition
        }
    
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
    
    # =================== BASIC TRANSITIONS ===================
    
    def cross_fade_transition(self, frame1, frame2, progress):
        """
        Smooth cross-fade that maintains motion in both clips.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

    def slide_transition_left(self, frame1, frame2, progress):
        """
        Slide transition from right to left while maintaining motion.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Calculate the slide position
        slide_pos = int(w * progress)
        
        # Copy portions of each frame based on slide position
        result[:, :w-slide_pos] = frame1[:, slide_pos:]  # First clip slides out
        result[:, w-slide_pos:] = frame2[:, :slide_pos]  # Second clip slides in
        
        return result

    def slide_transition_right(self, frame1, frame2, progress):
        """
        Slide transition from left to right while maintaining motion.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Calculate the slide position
        slide_pos = int(w * progress)
        
        # FIXED: Correctly copy portions of each frame
        # Left side shows incoming clip
        result[:, :slide_pos] = frame2[:, :slide_pos]
        # Right side shows original clip
        result[:, slide_pos:] = frame1[:, slide_pos:]
        
        return result

    def slide_transition_up(self, frame1, frame2, progress):
        """
        Slide transition from bottom to top while maintaining motion.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Calculate the slide position
        slide_pos = int(h * progress)
        
        # Copy portions of each frame based on slide position
        result[:h-slide_pos, :] = frame1[slide_pos:, :]  # First clip slides out
        result[h-slide_pos:, :] = frame2[:slide_pos, :]  # Second clip slides in
        
        return result

    def slide_transition_down(self, frame1, frame2, progress):
        """
        Slide transition from top to bottom while maintaining motion.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Calculate the slide position
        slide_pos = int(h * progress)
        
        # Copy portions of each frame based on slide position
        result[slide_pos:, :] = frame1[:h-slide_pos, :]  # First clip slides out
        result[:slide_pos, :] = frame2[h-slide_pos:, :]  # Second clip slides in
        
        return result

    def zoom_blend_transition(self, frame1, frame2, progress):
        """
        Zoom transition that maintains motion in both clips.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        
        # Create zoom effect on first clip
        scale1 = 1 + (0.2 * progress)
        new_h1 = int(h * scale1)
        new_w1 = int(w * scale1)
        frame1_zoomed = cv2.resize(frame1, (new_w1, new_h1))
        # Crop to original size from center
        y_start = (new_h1 - h) // 2
        x_start = (new_w1 - w) // 2
        frame1_zoomed = frame1_zoomed[y_start:y_start+h, x_start:x_start+w]
        
        # Create zoom effect on second clip (zooming from small to original)
        scale2 = 0.8 + (0.2 * progress)
        new_h2 = int(h * scale2)
        new_w2 = int(w * scale2)
        frame2_zoomed = cv2.resize(frame2, (new_w2, new_h2))
        # Pad to original size in center
        frame2_result = np.zeros_like(frame1)
        y_start = (h - new_h2) // 2
        x_start = (w - new_w2) // 2
        frame2_result[y_start:y_start+new_h2, x_start:x_start+new_w2] = frame2_zoomed
        
        # Blend the two frames based on progress
        result = cv2.addWeighted(frame1_zoomed, 1 - progress, frame2_result, progress, 0)
        return result

    def whip_pan_transition(self, frame1, frame2, progress):
        """
        Whip pan transition effect (motion blur + cross fade).
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        # Apply motion blur based on progress (max at middle point)
        blur_amount = int(30 * (1 - abs(2 * progress - 1)))
        blur_amount = max(1, blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1  # Ensure odd kernel size
        
        # Apply horizontal motion blur
        if progress < 0.5:
            # First clip getting blurrier
            frame1_blurred = cv2.GaussianBlur(frame1, (blur_amount, 1), 0)
            return frame1_blurred
        else:
            # Second clip getting clearer
            frame2_blurred = cv2.GaussianBlur(frame2, (blur_amount, 1), 0)
            normalized_progress = (progress - 0.5) * 2  # 0 to 1 for second half
            result = cv2.addWeighted(frame2_blurred, 1 - normalized_progress, frame2, normalized_progress, 0)
            return result

    def push_pull_transition(self, frame1, frame2, progress):
        """
        Push-pull effect (one clip pushes the other out) while maintaining motion.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Calculate push amount
        push_amount = int(w * progress)
        
        # Divide the frame into vertical strips
        strip_width = max(1, w // 20)
        for i in range(0, w, strip_width):
            end = min(i + strip_width, w)
            # Alternate direction for push effect
            if (i // strip_width) % 2 == 0:
                # Push right
                offset = push_amount
                src_start = max(0, i - offset)
                src_end = min(w, end - offset)
                dst_start = max(0, i)
                dst_end = min(w, end)
                
                if src_start < src_end and dst_start < dst_end:
                    # Width of the actual slice we're copying
                    copy_width = min(src_end - src_start, dst_end - dst_start)
                    result[:, dst_start:dst_start+copy_width] = frame2[:, src_start:src_start+copy_width]
            else:
                # Push left
                offset = push_amount
                src_start = max(0, i + offset)
                src_end = min(w, end + offset)
                dst_start = max(0, i)
                dst_end = min(w, end)
                
                if src_start < src_end and dst_start < dst_end:
                    # Width of the actual slice we're copying
                    copy_width = min(src_end - src_start, dst_end - dst_start)
                    result[:, dst_start:dst_start+copy_width] = frame1[:, src_start:src_start+copy_width]
        
        # Blend for smoother transition
        alpha = 0.2
        result = cv2.addWeighted(result, 1-alpha, cv2.addWeighted(frame1, 1-progress, frame2, progress, 0), alpha, 0)
        
        return result

    def overlay_dissolve_transition(self, frame1, frame2, progress):
        """
        Overlay dissolve transition that preserves motion.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip (already advanced to maintain motion)
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        # Create a noise pattern for the dissolve
        h, w = frame1.shape[:2]
        noise = np.random.random((h, w))
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Threshold the noise based on progress
        mask[noise < progress] = 255
        
        # Create 3-channel mask
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # Apply the mask
        frame1_masked = cv2.bitwise_and(frame1, cv2.bitwise_not(mask_3ch))
        frame2_masked = cv2.bitwise_and(frame2, mask_3ch)
        
        # Combine the masked frames
        result = cv2.add(frame1_masked, frame2_masked)
        
        # Add a slight cross-fade for smoothness
        alpha = 0.2
        result = cv2.addWeighted(result, 1-alpha, cv2.addWeighted(frame1, 1-progress, frame2, progress, 0), alpha, 0)
        
        return result
    
    # =================== ADVANCED TRANSITIONS ===================
    
    def kaleidoscope_transition(self, frame1, frame2, progress):
        """
        Kaleidoscope transition effect.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame
        """
        h, w = frame1.shape[:2]
        
        # Blend frames based on progress
        blended = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        
        # Create kaleidoscope effect
        center_x, center_y = w // 2, h // 2
        segments = 8  # Number of kaleidoscope segments
        
        # Create a triangular segment mask
        mask = np.zeros((h, w), dtype=np.uint8)
        segment_angle = 360 / segments
        
        # Define triangle points for first segment
        pts = np.array([
            [center_x, center_y],
            [center_x + int(w * 0.5 * np.cos(0)), center_y + int(h * 0.5 * np.sin(0))],
            [center_x + int(w * 0.5 * np.cos(np.radians(segment_angle))), 
             center_y + int(h * 0.5 * np.sin(np.radians(segment_angle)))]
        ], np.int32)
        
        # Draw the first segment
        cv2.fillPoly(mask, [pts], 255)
        
        # Create kaleidoscope result
        result = np.zeros_like(blended)
        
        # Get the first segment
        segment = cv2.bitwise_and(blended, blended, mask=mask)
        
        # Rotate and copy the segment to create kaleidoscope
        for i in range(segments):
            angle = i * segment_angle
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated = cv2.warpAffine(segment, M, (w, h))
            result = cv2.add(result, rotated)
        
        # Apply kaleidoscope effect based on progress
        # More pronounced in the middle of the transition
        kaleidoscope_strength = 1 - abs(2 * progress - 1)  # 0->1->0
        return cv2.addWeighted(blended, 1 - kaleidoscope_strength, result, kaleidoscope_strength, 0)
    
    def glitch_transition(self, frame1, frame2, progress):
        """
        Digital glitch effect transition.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Glitched frame
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Choose frame based on progress
        base_frame = frame1 if progress < 0.5 else frame2
        other_frame = frame2 if progress < 0.5 else frame1
        
        # Copy the base frame
        result = base_frame.copy()
        
        # Split into BGR channels
        b, g, r = cv2.split(result)
        b2, g2, r2 = cv2.split(other_frame)
        
        # Glitch intensity is highest in the middle of the transition
        intensity = 1 - abs(2 * progress - 1)  # 0->1->0
        
        # Apply different effects based on glitch intensity
        if intensity > 0.2:
            # RGB shift
            shift_amount = max(1, int(10 * intensity))
            if shift_amount > 0:
                # Shift red channel
                r_shifted = np.zeros_like(r)
                r_shifted[:, shift_amount:] = r[:, :-shift_amount]
                r = r_shifted
                
                # Shift blue channel the other way
                b_shifted = np.zeros_like(b)
                b_shifted[:, :-shift_amount] = b[:, shift_amount:]
                b = b_shifted
        
        # Horizontal line glitches
        num_glitches = max(1, int(20 * intensity))
        for _ in range(num_glitches):
            y_pos = np.random.randint(0, h)
            # Fix the error by ensuring high > low
            height = np.random.randint(1, max(2, int(10 * intensity) + 1))
            offset = np.random.randint(-max(1, int(20 * intensity)), max(2, int(20 * intensity)))
            
            # Apply offset to a horizontal slice
            y_end = min(y_pos + height, h)
            if offset != 0 and y_pos < y_end:
                # Shift for RGB channels
                for channel in [b, g, r]:
                    temp = channel[y_pos:y_end, :].copy()
                    if offset > 0:
                        channel[y_pos:y_end, offset:] = temp[:, :-offset]
                        channel[y_pos:y_end, :offset] = temp[:, -offset:]
                    else:
                        offset_abs = abs(offset)
                        channel[y_pos:y_end, :-offset_abs] = temp[:, offset_abs:]
                        channel[y_pos:y_end, -offset_abs:] = temp[:, :offset_abs]
        
        # Occasionally mix in some blocks from the other frame
        if intensity > 0.3:
            num_blocks = max(1, int(10 * intensity))
            for _ in range(num_blocks):
                x_pos = np.random.randint(0, max(1, w - 50))
                y_pos = np.random.randint(0, max(1, h - 50))
                block_w = np.random.randint(20, max(21, min(100, w - x_pos)))
                block_h = np.random.randint(10, max(11, min(30, h - y_pos)))
                
                # Ensure we stay within bounds
                x_end = min(x_pos + block_w, w)
                y_end = min(y_pos + block_h, h)
                
                # Choose a channel to replace
                channel_idx = np.random.randint(0, 3)
                if channel_idx == 0 and x_pos < x_end and y_pos < y_end:
                    b[y_pos:y_end, x_pos:x_end] = b2[y_pos:y_end, x_pos:x_end]
                elif channel_idx == 1 and x_pos < x_end and y_pos < y_end:
                    g[y_pos:y_end, x_pos:x_end] = g2[y_pos:y_end, x_pos:x_end]
                elif x_pos < x_end and y_pos < y_end:
                    r[y_pos:y_end, x_pos:x_end] = r2[y_pos:y_end, x_pos:x_end]
        
        # Merge channels
        result = cv2.merge([b, g, r])
        
        # Add noise
        if intensity > 0.1:
            noise = np.random.randint(0, max(1, int(30 * intensity)), result.shape, dtype=np.uint8)
            result = cv2.add(result, noise)
            
            # Add static/dead pixels
            dead_pixels = np.random.random(result.shape[:2]) < (0.01 * intensity)
            if np.any(dead_pixels):
                result[dead_pixels] = [255, 255, 255]
        
        # Blend with normal transition at start and end for smoother effect
        normal_blend = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        blend_factor = intensity  # Use more of the glitch in the middle of the transition
        return cv2.addWeighted(normal_blend, 1 - blend_factor, result, blend_factor, 0)
    
    def particle_transition(self, frame1, frame2, progress):
        """
        Particle effect transition.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame with particle effect
        """
        h, w = frame1.shape[:2]
        
        # Start with a blend of the two frames
        result = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        
        # Create a mask for the particles
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Generate particles
        num_particles = int(2000 * (1 - abs(2 * progress - 1)))  # Max particles in the middle
        particle_size_max = int(5 + 10 * progress)  # Particles grow as we progress
        
        # Draw particles on the mask
        for _ in range(num_particles):
            # Random position
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # Determine direction based on progress
            direction = -1 if progress < 0.5 else 1
            
            # Move particles based on progress and direction
            offset = int(100 * abs(progress - 0.5) * direction)
            x_offset = x + offset
            
            # Keep within bounds
            if 0 <= x_offset < w:
                # Random particle size
                size = np.random.randint(1, particle_size_max)
                # Draw the particle
                cv2.circle(mask, (x_offset, y), size, 255, -1)
        
        # Create 3-channel mask
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # Determine which frame to apply particles to
        source_frame = frame1 if progress < 0.5 else frame2
        
        # Apply the mask
        particles = cv2.bitwise_and(source_frame, mask_3ch)
        
        # Add particles to the result
        result = cv2.add(result, particles)
        
        # Add motion blur to particles for better effect
        if progress > 0.1 and progress < 0.9:
            blur_size = int(20 * (1 - abs(2 * progress - 1)))
            if blur_size % 2 == 0:
                blur_size += 1
            
            # Only blur the particles
            particles_blurred = cv2.GaussianBlur(particles, (blur_size, 1), 0)
            
            # Blend original particles with blurred ones
            particles = cv2.addWeighted(particles, 0.3, particles_blurred, 0.7, 0)
            
            # Add to result
            result = cv2.add(result, particles)
        
        return result
    
    def shatter_transition(self, frame1, frame2, progress):
            """
            Glass shatter transition effect.
            
            Args:
                frame1: Frame from the first clip
                frame2: Frame from the second clip
                progress: Transition progress (0 to 1)
                
            Returns:
                Blended frame with shatter effect
            """
            h, w = frame1.shape[:2]
            
            # Start with second frame as base
            result = frame2.copy()
            
            # Only show shatter during the transition progress
            if progress < 0.7:  # Normalize progress to 0-1 range for the shatter effect
                shatter_progress = min(1.0, progress / 0.7)
                
                # Create shattered triangles
                num_triangles = 50
                triangles = []
                
                # Generate random triangles
                np.random.seed(42)  # For reproducible results
                for i in range(num_triangles):
                    # Create a triangle with random vertices
                    pts = np.array([
                        [np.random.randint(0, w), np.random.randint(0, h)],
                        [np.random.randint(0, w), np.random.randint(0, h)],
                        [np.random.randint(0, w), np.random.randint(0, h)]
                    ], np.int32)
                    
                    # Move triangles based on progress (simulating shattering)
                    center = np.mean(pts, axis=0).astype(int)
                    direction = center - np.array([w//2, h//2])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1, 0])
                    
                    # Scale motion by progress and randomness
                    motion = direction * shatter_progress * np.random.randint(10, 200)
                    pts = pts + motion.astype(int)
                    
                    triangles.append(pts)
                
                # Draw each triangle using the first frame
                for pts in triangles:
                    # Create a mask for this triangle
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    
                    # Apply the mask to frame1
                    mask_3ch = cv2.merge([mask, mask, mask])
                    masked_src = cv2.bitwise_and(frame1, mask_3ch)
                    
                    # Create inverse mask for frame2
                    inv_mask_3ch = cv2.bitwise_not(mask_3ch)
                    masked_dst = cv2.bitwise_and(result, inv_mask_3ch)
                    
                    # Combine the two
                    result = cv2.add(masked_src, masked_dst)
            
            # Apply a flash effect during shattering
            if 0.1 < progress < 0.5:
                # Flash intensity peaks around progress 0.3
                flash_intensity = max(0, 0.7 - abs(progress - 0.3) * 4)
                flash = np.ones_like(result) * 255
                result = cv2.addWeighted(result, 1 - flash_intensity, flash, flash_intensity, 0)
            
            return result
        
    def ripple_transition(self, frame1, frame2, progress):
        """
        Water ripple transition effect.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame with ripple effect
        """
        h, w = frame1.shape[:2]
        
        # Blend frames based on progress
        result = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        
        # Apply ripple effect, strongest in the middle of the transition
        ripple_strength = 1 - abs(2 * progress - 1)  # 0->1->0
        
        if ripple_strength > 0.1:
            # Create displacement maps for ripple effect
            center_x, center_y = w // 2, h // 2
            map_x = np.zeros((h, w), np.float32)
            map_y = np.zeros((h, w), np.float32)
            
            # Fill maps with pixel coordinates
            for y in range(h):
                for x in range(w):
                    map_x[y, x] = x
                    map_y[y, x] = y
            
            # Add ripple effect
            frequency = 20.0
            amplitude = 10.0 * ripple_strength
            speed = 10.0
            
            for y in range(h):
                for x in range(w):
                    # Distance from center
                    dx = x - center_x
                    dy = y - center_y
                    distance = np.sqrt(dx * dx + dy * dy)
                    
                    # Skip points too close to center
                    if distance > 5:
                        # Calculate ripple effect
                        phase = progress * speed
                        offset = amplitude * np.sin(distance / frequency - phase)
                        ratio = offset / distance
                        
                        # Apply offset
                        map_x[y, x] = x + dx * ratio
                        map_y[y, x] = y + dy * ratio
            
            # Remap the image using the displacement maps
            result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR)
        
        return result
    
    def pixelate_transition(self, frame1, frame2, progress):
        """
        Pixelate transition effect.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame with pixelation effect
        """
        h, w = frame1.shape[:2]
        
        # Determine which frame to use based on progress
        if progress < 0.5:
            # First half - pixelate first frame
            frame = frame1
            pixel_progress = progress * 2  # 0->1 in first half
        else:
            # Second half - pixelate second frame and gradually reveal
            frame = frame2
            pixel_progress = (1 - progress) * 2  # 1->0 in second half
        
        # Calculate pixel size based on progress
        min_pixel_size = 1
        max_pixel_size = 40
        
        # Apply more pixelation in the middle of each half
        pixel_size = int(min_pixel_size + (max_pixel_size - min_pixel_size) * pixel_progress)
        pixel_size = max(1, pixel_size)  # Ensure at least 1
        
        # Apply pixelation
        if pixel_size > 1:
            # Resize down then up to create pixelation
            small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            pixelated = frame
        
        # Crossfade between frame1 and pixelated in first half
        if progress < 0.5:
            result = cv2.addWeighted(frame1, 1 - pixel_progress, pixelated, pixel_progress, 0)
        else:
            # Crossfade between pixelated and frame2 in second half
            result = cv2.addWeighted(pixelated, pixel_progress, frame2, 1 - pixel_progress, 0)
        
        return result
    
    def cube_rotate_transition(self, frame1, frame2, progress):
        """
        3D cube rotation transition effect.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame with 3D cube rotation
        """
        h, w = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        # Calculate rotation angle based on progress (90 degrees)
        angle = progress * 90
        
        # Perspective transformation for 3D effect
        # Calculate vanishing point
        center_x, center_y = w // 2, h // 2
        
        # When angle is 0, show frame1 fully
        if angle < 1:
            return frame1
        
        # When angle is 90, show frame2 fully
        if angle > 89:
            return frame2
        
        # Calculate perspective transforms for both faces
        # Frame1 rotates out of view
        src_points1 = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Calculate destination points based on rotation angle
        angle_rad = np.radians(angle)
        scale_x = np.cos(angle_rad)
        
        dst_points1 = np.array([
            [center_x - (center_x * scale_x), 0],
            [center_x + (center_x * scale_x), 0],
            [center_x + (center_x * scale_x), h],
            [center_x - (center_x * scale_x), h]
        ], dtype=np.float32)
        
        # Frame2 rotates into view (starts from 90 degrees)
        src_points2 = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Calculate perpendicular face transformation
        angle_rad2 = np.radians(90 - angle)
        scale_x2 = np.cos(angle_rad2)
        
        dst_points2 = np.array([
            [center_x - (center_x * scale_x2), 0],
            [center_x + (center_x * scale_x2), 0],
            [center_x + (center_x * scale_x2), h],
            [center_x - (center_x * scale_x2), h]
        ], dtype=np.float32)
        
        # Apply perspective transformation
        M1 = cv2.getPerspectiveTransform(src_points1, dst_points1)
        warped1 = cv2.warpPerspective(frame1, M1, (w, h))
        
        M2 = cv2.getPerspectiveTransform(src_points2, dst_points2)
        warped2 = cv2.warpPerspective(frame2, M2, (w, h))
        
        # Combine the two faces
        result = cv2.add(warped1, warped2)
        
        return result
    
    def swirl_transition(self, frame1, frame2, progress):
        """
        Swirl/whirlpool transition effect.
        
        Args:
            frame1: Frame from the first clip
            frame2: Frame from the second clip
            progress: Transition progress (0 to 1)
            
        Returns:
            Blended frame with swirl effect
        """
        h, w = frame1.shape[:2]
        
        # Blend frames based on progress
        blend = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
        
        # Swirl effect intensity, strongest in the middle of the transition
        swirl_strength = 1 - abs(2 * progress - 1)  # 0->1->0
        
        if swirl_strength > 0.1:
            # Create swirl effect using mapping
            center_x, center_y = w // 2, h // 2
            map_x = np.zeros((h, w), np.float32)
            map_y = np.zeros((h, w), np.float32)
            
            max_radius = np.sqrt(w * w + h * h) / 2
            swirl_amount = 10.0 * swirl_strength
            
            # Fill maps with transformed coordinates
            for y in range(h):
                for x in range(w):
                    # Calculate distance from center
                    dx = x - center_x
                    dy = y - center_y
                    distance = np.sqrt(dx * dx + dy * dy)
                    
                    # Calculate swirl angle based on distance and progress
                    if distance < max_radius:
                        # Normalize distance
                        dist_factor = distance / max_radius
                        # Swirl more toward the center
                        angle = swirl_amount * (1.0 - dist_factor) * 2 * np.pi
                        
                        # Apply rotation
                        sin_val = np.sin(angle)
                        cos_val = np.cos(angle)
                        
                        map_x[y, x] = cos_val * dx - sin_val * dy + center_x
                        map_y[y, x] = sin_val * dx + cos_val * dy + center_y
                    else:
                        map_x[y, x] = x
                        map_y[y, x] = y
            
            # Apply the swirl effect
            result = cv2.remap(blend, map_x, map_y, cv2.INTER_LINEAR)
        else:
            result = blend
        
        return result
    
    def load_transitions_from_file(self, file_path):
        """
        Load available transitions from a JSON file.
        
        Args:
            file_path: Path to the JSON file with transition configurations
            
        Returns:
            Dictionary mapping transition names to functions
        """
        try:
            if not os.path.exists(file_path):
                print(f"Transitions file {file_path} not found. Creating default file...")
                create_default_transitions_file(file_path)
            
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Create a dictionary of available transitions
            available_transitions = {}
            for name in config.get("enabled_transitions", []):
                if name in self.transition_functions:
                    available_transitions[name] = self.transition_functions[name]
            
            if not available_transitions:
                print("No valid transitions found in file. Using default transitions.")
                return {"cross_fade": self.transition_functions["cross_fade"]}
            
            return available_transitions
        except Exception as e:
            print(f"Error loading transitions file: {e}")
            print("Using default cross_fade transition.")
            return {"cross_fade": self.transition_functions["cross_fade"]}
    
    def save_transitions_to_file(self, file_path, enabled_transitions=None):
        """
        Save transitions configuration to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
            enabled_transitions: List of transition names to enable (None for all)
        """
        if enabled_transitions is None:
            enabled_transitions = list(self.transition_functions.keys())
        
        config = {
            "all_available_transitions": list(self.transition_functions.keys()),
            "enabled_transitions": enabled_transitions
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Transitions configuration saved to {file_path}")
        except Exception as e:
            print(f"Error saving transitions file: {e}")
    
    def process_video_with_motion_transitions(self, transition_type="cross_fade", custom_scenes=None, randomize=False, transitions_file=None):
        """
        Process the video with motion-preserving transitions that don't freeze frames.
        
        Args:
            transition_type: Type of transition to apply
            custom_scenes: Custom list of frame indices for scene changes (optional)
            randomize: Whether to randomize transitions for each scene
            transitions_file: Path to a JSON file with transition configurations
        """
        # Load transitions if file is provided
        if transitions_file and os.path.exists(transitions_file):
            available_transitions = self.load_transitions_from_file(transitions_file)
            print(f"Loaded {len(available_transitions)} transitions from {transitions_file}")
        else:
            available_transitions = self.transition_functions
        
        if not available_transitions:
            raise ValueError("No transitions available. Check your transitions file.")
        
        # Detect scene changes if custom scenes are not provided
        scene_changes = custom_scenes or self.detect_scene_changes()
        
        if not scene_changes:
            print("No scene changes detected. Using equally spaced transitions.")
            # Create equally spaced transitions every 3 seconds if no scenes detected
            target_transitions = int((self.total_frames / self.fps / 15) * 4.5)  # 4-5 per 15 seconds
            scene_spacing = int(self.total_frames / (target_transitions + 1))
            scene_changes = list(range(scene_spacing, self.total_frames - self.transition_duration, scene_spacing))
        
        # Determine which transition function to use for each scene change
        transition_funcs = []
        
        if randomize:
            # Randomly select a transition for each scene change
            available_names = list(available_transitions.keys())
            print(f"Randomly selecting from transitions: {', '.join(available_names)}")
            for _ in scene_changes:
                random_name = random.choice(available_names)
                transition_funcs.append((random_name, available_transitions[random_name]))
        else:
            # Use the specified transition type for all scene changes
            if transition_type not in available_transitions:
                raise ValueError(f"Unsupported transition type: {transition_type}")
            for _ in scene_changes:
                transition_funcs.append((transition_type, available_transitions[transition_type]))
        
        # Create output video writer
        writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
        
        # Load all frames from the video into memory
        print("Loading video frames...")
        all_frames = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        with tqdm(total=self.total_frames) as pbar:
            for _ in range(self.total_frames):
                ret, frame = self.video.read()
                if not ret:
                    break
                all_frames.append(frame.copy())
                pbar.update(1)
        
        if not all_frames:
            raise ValueError("Failed to load video frames")
        
        print(f"Loaded {len(all_frames)} frames")
        
        # Calculate transition ranges
        transition_ranges = []
        for scene_idx in scene_changes:
            start = max(0, scene_idx - self.transition_duration // 2)
            end = min(len(all_frames) - 1, start + self.transition_duration)
            transition_ranges.append((start, end))
        
        # Pre-compute frame mapping for each transition
        # This solves the rewind problem by ensuring continuous progress
        frame_mapping = {}
        for i, (start, end) in enumerate(transition_ranges):
            # For each transition frame, calculate which frame from clip B to use
            for j in range(start, end):
                progress = (j - start) / (end - start)
                
                # CRITICAL FIX: Use frames from clip B that maintain proper continuity
                # We use the frame at position "end" as our reference point
                # and work backwards to determine which frames to use during the transition
                
                # Calculate frames backward from the end point
                # This ensures a smooth continuation when the transition finishes
                offset = (1.0 - progress) * self.transition_duration
                next_clip_idx = max(0, min(len(all_frames) - 1, end - int(offset)))
                
                # Store the mapping
                frame_mapping[(i, j)] = next_clip_idx
        
        # Process frames with transitions
        print("Applying transitions...")
        
        with tqdm(total=len(all_frames)) as pbar:
            for current_idx in range(len(all_frames)):
                # Check if current frame is in a transition
                in_transition = False
                transition_idx = -1
                
                for idx, (start, end) in enumerate(transition_ranges):
                    if start <= current_idx < end:
                        in_transition = True
                        transition_idx = idx
                        break
                
                if in_transition:
                    # We're in a transition
                    start, end = transition_ranges[transition_idx]
                    
                    # Calculate transition progress (0 to 1)
                    progress = (current_idx - start) / (end - start)
                    
                    # Get current frame from first clip
                    current_frame = all_frames[current_idx]
                    
                    # Get the right frame from the second clip using our mapping
                    next_frame_idx = frame_mapping[(transition_idx, current_idx)]
                    next_frame = all_frames[next_frame_idx]
                    
                    # Apply transition
                    transition_name, transition_func = transition_funcs[transition_idx]
                    output_frame = transition_func(current_frame, next_frame, progress)
                    
                    # Write the output frame
                    writer.write(output_frame)
                    
                    # Log transition start
                    if current_idx == start:
                        print(f"Applying {transition_name} transition at frame {current_idx} ({current_idx/self.fps:.2f}s)")
                else:
                    # Not in a transition, write the frame as is
                    writer.write(all_frames[current_idx])
                
                pbar.update(1)
        
        writer.release()
        print(f"Video with transitions saved to {self.output_path}")
    
    def close(self):
        """Release video resources"""
        if self.video.isOpened():
            self.video.release()


def create_default_transitions_file(file_path):
    """
    Create a default transitions configuration file.
    
    Args:
        file_path: Path to save the transitions file
    """
    all_transitions = [
        "cross_fade", "slide_left", "slide_right", "slide_up", "slide_down", 
        "zoom_blend", "whip_pan", "push_pull", "overlay_dissolve",
        "kaleidoscope", "glitch", "particle", "shatter", "ripple", 
        "pixelate", "cube_rotate", "swirl"
    ]
    
    # Default enabled transitions (a subset that works well for most videos)
    default_enabled = [
        "cross_fade", "slide_left", "slide_right", "zoom_blend", 
        "push_pull", "overlay_dissolve", "glitch", "ripple"
    ]
    
    config = {
        "all_available_transitions": all_transitions,
        "enabled_transitions": default_enabled
    }
    
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Default transitions configuration created at {file_path}")
    except Exception as e:
        print(f"Error creating transitions file: {e}")


def main():
    """Main function to demonstrate the usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add motion-preserving transition effects to videos")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--transition", "-t", default="cross_fade", 
                    help="Transition effect type (ignored if random is used)")
    parser.add_argument("--duration", "-d", type=int, default=20, 
                        help="Transition duration in frames")
    parser.add_argument("--scenes", "-s", type=str, default=None,
                        help="Comma-separated list of frame indices for custom scene changes")
    parser.add_argument("--max-scenes", "-m", type=int, default=None,
                        help="Maximum number of scene changes to detect")
    parser.add_argument("--random", "-r", action="store_true", 
                        help="Randomize transitions for each scene change")
    parser.add_argument("--transitions-file", "-f", type=str, default=None,
                        help="Path to JSON file with enabled transitions")
    parser.add_argument("--create-transitions-file", "-c", action="store_true",
                        help="Create a default transitions configuration file")
    parser.add_argument("--transitions-file-path", "-p", type=str, default="transitions.json",
                        help="Path to save the transitions configuration file")
    
    args = parser.parse_args()
    
    # Create default transitions file if requested
    if args.create_transitions_file:
        create_default_transitions_file(args.transitions_file_path)
        return
    
    # Parse custom scenes if provided
    custom_scenes = None
    if args.scenes:
        custom_scenes = [int(idx) for idx in args.scenes.split(",")]
    
    # Process video
    processor = VideoTransitionEffects(args.input, args.output, args.duration)
    try:
        processor.process_video_with_motion_transitions(
            args.transition, 
            custom_scenes=custom_scenes,
            randomize=args.random,
            transitions_file=args.transitions_file
        )
    finally:
        processor.close()


if __name__ == "__main__":
    main()