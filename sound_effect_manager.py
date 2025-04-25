# sound_effect_manager.py
import os
import json
import random
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SoundEffect:
    """Represents a single sound effect with metadata."""
    name: str
    path: str
    category: str
    duration: float
    energy_level: str  # 'low', 'medium', 'high'
    tags: List[str] = field(default_factory=list)
    
    def load_audio(self) -> AudioFileClip:
        """Load the audio file as a MoviePy AudioFileClip."""
        return AudioFileClip(self.path)
    
    def load_pydub(self) -> AudioSegment:
        """Load the audio file as a PyDub AudioSegment for advanced processing."""
        return AudioSegment.from_file(self.path)

@dataclass
class TransitionPoint:
    """Represents a point in the video where a sound effect should be added."""
    timestamp: float  # seconds from start of video
    duration: float   # duration of the B-roll segment
    transition_type: str  # 'in', 'out', or 'both'
    context: Dict     # contains additional info about the transition point
    
    @property
    def exit_point(self) -> float:
        """Calculate the exit point (when B-roll ends)."""
        return self.timestamp + self.duration


class SoundEffectManager:
    """
    Manages the loading, selection, and application of sound effects 
    for B-roll transitions in videos.
    """
    
    def __init__(
        self,
        sound_effects_dir: str = "./assets/sound_effects",
        cache_dir: Optional[str] = "./cache/sound_effects",
        fade_duration: float = 0.3,  # seconds
        volume_adjustment: float = 0.7,  # relative to original audio
        max_sounds_per_video: int = 5,  # avoid repetitive sound usage
        min_gap_between_effects: float = 3.0,  # minimum seconds between sound effects
    ):
        """
        Initialize the SoundEffectManager.
        
        Args:
            sound_effects_dir: Directory containing sound effect files
            cache_dir: Directory to cache processed audio
            fade_duration: Duration in seconds for fading sound effects
            volume_adjustment: Volume level for sound effects relative to original audio
            max_sounds_per_video: Maximum number of times to use the same sound in a video
            min_gap_between_effects: Minimum seconds between sound effects
        """
        self.sound_effects_dir = sound_effects_dir
        self.cache_dir = cache_dir
        self.fade_duration = fade_duration
        self.volume_adjustment = volume_adjustment
        self.max_sounds_per_video = max_sounds_per_video
        self.min_gap_between_effects = min_gap_between_effects
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary to store loaded sound effects by category
        self.sound_effects: Dict[str, List[SoundEffect]] = {}
        
        # Load sound effects
        self._load_sound_effects()
        
        # Keep track of used sounds to prevent repetition
        self.used_sounds = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _load_sound_effects(self) -> None:
        """
        Load sound effects from the specified directory.
        
        The expected structure is:
        - sound_effects_dir/
        - whooshes/
            - swoosh_fast_bright.mp3
        - impacts/
            - impact_heavy_reverb.mp3
        - stingers/
            - stinger_upbeat.mp3
        - ui/
            - click_clean.mp3
        - foley/
            - footstep_concrete.mp3
        
        Each directory name is treated as a category.
        """
        base_dir = Path(self.sound_effects_dir)
        
        if not base_dir.exists():
            self.logger.warning(f"Sound effects directory not found: {base_dir}")
            
            # Create base directory and example structure with README
            os.makedirs(base_dir, exist_ok=True)
            
            # Create example category directories with our new structure
            categories = ["whooshes", "impacts", "stingers", "ui", "foley"]
            for category in categories:
                os.makedirs(base_dir / category, exist_ok=True)
            
            # Create a README file explaining the directory structure
            readme_path = base_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write("# Sound Effects Directory\n\n")
                f.write("Place your sound effect files in the appropriate subdirectories:\n\n")
                f.write("- `whooshes/`: Transition sounds, movement effects, and swooshes\n")
                f.write("- `impacts/`: Sounds for ending transitions and emphasis points\n")
                f.write("- `stingers/`: Short musical accents and transitions\n")
                f.write("- `ui/`: Interface sounds, clicks, and notifications\n")
                f.write("- `foley/`: Real-world sounds for added realism\n")
                f.write("\nFile naming convention: `type_attribute1_attribute2.mp3`\n")
                f.write("Example: `swoosh_fast_bright.mp3`, `impact_heavy_reverb.mp3`\n")
            
            self.logger.info(f"Created sound effects directory structure at {base_dir}")
            self.logger.info(f"Please add sound effect files to the appropriate subdirectories.")
            return
        
        # Discover categories (subdirectories)
        for category_dir in base_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            self.sound_effects[category_name] = []
            
            # Load sound effects from this category
            for sound_file in category_dir.glob("*.mp3"):
                try:
                    # Load basic info about the sound file
                    audio = AudioSegment.from_file(str(sound_file))
                    duration = len(audio) / 1000.0  # convert ms to seconds
                    
                    # Estimate energy level based on RMS
                    rms = audio.rms
                    if rms < 2000:
                        energy_level = "low"
                    elif rms < 5000:
                        energy_level = "medium"
                    else:
                        energy_level = "high"
                    
                    # Extract tags from filename (e.g., "swoosh_fast_bright.mp3" -> ["fast", "bright"])
                    name_parts = sound_file.stem.split('_')
                    name = name_parts[0]
                    tags = name_parts[1:] if len(name_parts) > 1 else []
                    
                    # Create sound effect object
                    sound_effect = SoundEffect(
                        name=name,
                        path=str(sound_file),
                        category=category_name,
                        duration=duration,
                        energy_level=energy_level,
                        tags=tags
                    )
                    
                    self.sound_effects[category_name].append(sound_effect)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load sound effect {sound_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.sound_effects[category_name])} sound effects in category '{category_name}'")
    
    def detect_transition_points(self, broll_data_path: str) -> List[TransitionPoint]:
        """
        Analyze broll cuts data to identify transition points for sound effects.
        
        Args:
            broll_data_path: Path to the B-roll cuts JSON file
            
        Returns:
            List of TransitionPoint objects
        """
        transitions = []
        
        try:
            # Load B-roll data
            with open(broll_data_path, 'r') as f:
                data = json.load(f)
            
            broll_cuts = data.get("broll_cuts", [])
            
            # Get video duration if available
            video_duration = data.get("metadata", {}).get("video_duration", 0)
            
            for i, cut in enumerate(broll_cuts):
                # Skip cuts without a valid path
                if not cut.get("path"):
                    continue
                
                timestamp = cut.get("timestamp", 0)
                duration = cut.get("duration", 0)
                
                # Create context with additional information
                context = {
                    "keywords": cut.get("keywords", []),
                    "transcript_segment": cut.get("transcript_segment", ""),
                    "reason": cut.get("reason", ""),
                    "is_first_cut": i == 0,
                    "is_last_cut": i == len(broll_cuts) - 1,
                    "video_duration": video_duration
                }
                
                # Add transition IN point (start of B-roll)
                transitions.append(TransitionPoint(
                    timestamp=timestamp,
                    duration=duration,
                    transition_type="in",
                    context=context
                ))
                
                # Add transition OUT point (end of B-roll)
                transitions.append(TransitionPoint(
                    timestamp=timestamp + duration,
                    duration=0,  # No duration for out transition
                    transition_type="out",
                    context=context
                ))
            
            # Sort transitions by timestamp
            transitions.sort(key=lambda t: t.timestamp)
            
            # Filter transitions that are too close to each other
            filtered_transitions = []
            last_timestamp = -self.min_gap_between_effects
            
            for transition in transitions:
                if transition.timestamp - last_timestamp >= self.min_gap_between_effects:
                    filtered_transitions.append(transition)
                    last_timestamp = transition.timestamp
                else:
                    self.logger.debug(f"Skipping transition at {transition.timestamp}s (too close to previous)")
            
            self.logger.info(f"Detected {len(filtered_transitions)} transition points from {len(broll_cuts)} B-roll cuts")
            return filtered_transitions
            
        except Exception as e:
            self.logger.error(f"Error detecting transition points: {e}")
            return []
        
    def _select_sound_by_context(self, transition: TransitionPoint, available_effects: List[SoundEffect]) -> SoundEffect:
        """Select a sound effect based on transition context."""
        # Extract context information
        context = transition.context
        keywords = context.get("keywords", [])
        transcript = context.get("transcript_segment", "")
        is_first_cut = context.get("is_first_cut", False)
        is_last_cut = context.get("is_last_cut", False)
        
        # Default score for each effect
        scores = {effect: 1.0 for effect in available_effects}
        
        # Score based on energy level
        energy_preference = "high" if any(word in " ".join(keywords).lower() 
                                        for word in ["action", "fast", "energetic", "dynamic"]) else "medium"
        
        if "calm" in " ".join(keywords).lower() or "gentle" in " ".join(keywords).lower():
            energy_preference = "low"
        
        # Adjust scores based on energy match
        for effect in available_effects:
            if effect.energy_level == energy_preference:
                scores[effect] *= 2.0
            elif (energy_preference == "high" and effect.energy_level == "medium") or \
                (energy_preference == "medium" and effect.energy_level in ["high", "low"]):
                scores[effect] *= 1.5
        
        # Prioritize certain sounds for first/last cuts
        if is_first_cut:
            for effect in available_effects:
                if "intro" in effect.tags or "opening" in effect.tags:
                    scores[effect] *= 2.5
        
        if is_last_cut:
            for effect in available_effects:
                if "outro" in effect.tags or "closing" in effect.tags:
                    scores[effect] *= 2.5
        
        # Check for keyword matches in sound effect tags
        for effect in available_effects:
            for tag in effect.tags:
                if any(keyword.lower() in tag.lower() for keyword in keywords):
                    scores[effect] *= 1.8
        
        # Select based on weighted random
        total_score = sum(scores.values())
        if total_score == 0:
            return random.choice(available_effects)
        
        r = random.uniform(0, total_score)
        upto = 0
        for effect, score in scores.items():
            if upto + score >= r:
                return effect
            upto += score
        
        # Fallback
        return random.choice(available_effects)
    
    def select_sound_effect(self, transition: TransitionPoint) -> Optional[SoundEffect]:
        """
        Select an appropriate sound effect for a given transition point.
        
        Args:
            transition: The transition point
            
        Returns:
            Selected SoundEffect or None if no suitable effect found
        """
        # Determine appropriate category based on transition type
        if transition.transition_type == "in":
            # For transition into B-roll, primarily use whooshes
            categories = ["whooshes"]
            # Secondary options if whooshes aren't available
            fallback_categories = ["stingers", "ui"]
        else:  # "out"
            # For transition out of B-roll, primarily use impacts
            categories = ["impacts"] 
            # Secondary options if impacts aren't available
            fallback_categories = ["stingers", "foley"]
        
        # Filter categories to those that actually exist
        available_categories = [c for c in categories if c in self.sound_effects and self.sound_effects[c]]
        
        if not available_categories:
            # Try fallback categories if primary aren't available
            available_categories = [c for c in fallback_categories if c in self.sound_effects and self.sound_effects[c]]
            
            if not available_categories:
                # Fall back to any available category
                available_categories = [c for c in self.sound_effects if self.sound_effects[c]]
                
                if not available_categories:
                    self.logger.warning("No sound effects available")
                    return None
        
        # Choose a random category from available ones
        category = random.choice(available_categories)
        
        # Get list of sound effects in this category
        effects = self.sound_effects[category]
        
        # Filter out overused sounds
        available_effects = [
            effect for effect in effects
            if self.used_sounds.get(effect.path, 0) < self.max_sounds_per_video
        ]
        
        if not available_effects:
            # If all sounds are overused, reset counts for this category
            available_effects = effects
            for effect in effects:
                self.used_sounds[effect.path] = 0
        
        # With context-aware selection:
        selected_effect = self._select_sound_by_context(transition, available_effects)
        
        # Update usage count
        self.used_sounds[selected_effect.path] = self.used_sounds.get(selected_effect.path, 0) + 1
        
        return selected_effect
    
    def apply_sound_effects(
        self,
        video_path: str,
        broll_data_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Apply sound effects to the video at B-roll transition points.
        
        Args:
            video_path: Path to the input video
            broll_data_path: Path to the B-roll cuts JSON file
            output_path: Path for the output video (if None, will use input path with "_with_sfx" suffix)
            
        Returns:
            Path to the output video with sound effects
        """
        try:
            # Determine output path
            if output_path is None:
                input_path = Path(video_path)
                output_path = str(input_path.with_stem(f"{input_path.stem}_with_sfx"))
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Detect transition points
            transition_points = self.detect_transition_points(broll_data_path)
            
            if not transition_points:
                self.logger.warning("No valid transition points found. Copying original video.")
                import shutil
                shutil.copy2(video_path, output_path)
                return output_path
            
            # Load video
            self.logger.info(f"Loading video: {video_path}")
            video = VideoFileClip(video_path)
            
            # Get original audio
            original_audio = video.audio
            if original_audio is None:
                self.logger.warning("Video has no audio track. Creating silent audio.")
                from moviepy.audio.AudioClip import AudioClip
                import numpy as np
                silence = lambda t: 0
                original_audio = AudioClip(make_frame=silence, duration=video.duration)
            
            # Prepare list for all audio clips
            audio_clips = [original_audio]
            
            # Generate unique cache key based on input files and parameters
            cache_key = hashlib.md5(
                f"{video_path}_{broll_data_path}_{self.fade_duration}_{self.volume_adjustment}".encode()
            ).hexdigest()
            
            # Check if we have a cached result
            cache_path = None
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"{cache_key}_audio.mp3")
                
                if os.path.exists(cache_path):
                    self.logger.info(f"Using cached audio from {cache_path}")
                    
                    # Create final video with cached audio
                    final_audio = AudioFileClip(cache_path)
                    final_video = video.set_audio(final_audio)
                    
                    # Write final video
                    self.logger.info(f"Writing video with cached audio to {output_path}")
                    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
                    
                    # Clean up
                    final_video.close()
                    final_audio.close()
                    video.close()
                    
                    return output_path
            
            # Apply sound effects at each transition point
            for transition in transition_points:
                # Select appropriate sound effect
                sound_effect = self.select_sound_effect(transition)
                
                if not sound_effect:
                    self.logger.warning(f"No suitable sound effect found for transition at {transition.timestamp}s")
                    continue
                
                self.logger.info(f"Adding {sound_effect.category}/{sound_effect.name} at {transition.timestamp}s")
                
                # Load sound effect audio
                effect_audio = sound_effect.load_audio()
                
                # Apply fade in/out to sound effect
                effect_audio = effect_audio.audio_fadeout(self.fade_duration)
                effect_audio = effect_audio.audio_fadein(self.fade_duration)
                
                # Adjust volume
                effect_audio = effect_audio.volumex(self.volume_adjustment)
                
                # Position the sound effect at the transition point
                # For "in" transitions, start slightly before the transition
                # For "out" transitions, center the sound on the transition
                if transition.transition_type == "in":
                    # Start effect slightly before transition point
                    start_time = max(0, transition.timestamp - (effect_audio.duration / 3))
                else:  # "out"
                    # Center effect on transition point
                    start_time = max(0, transition.timestamp - (effect_audio.duration / 2))
                
                # Create positioned audio clip
                positioned_effect = effect_audio.set_start(start_time)
                
                # Add to list of audio clips
                audio_clips.append(positioned_effect)
            
            # Combine all audio clips
            self.logger.info(f"Combining {len(audio_clips)} audio clips")
            final_audio = CompositeAudioClip(audio_clips)
            
            # Cache the combined audio if needed
            if cache_path:
                self.logger.info(f"Caching combined audio to {cache_path}")
                final_audio.write_audiofile(cache_path, codec="mp3")
            
            # Apply combined audio to video
            final_video = video.set_audio(final_audio)
            
            # Write final video
            self.logger.info(f"Writing final video to {output_path}")
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            # Clean up
            final_video.close()
            final_audio.close()
            video.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error applying sound effects: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return original video path if processing failed
            return video_path
    
    def get_available_categories(self) -> List[str]:
        """Get a list of available sound effect categories."""
        return list(self.sound_effects.keys())
    
    def get_effects_in_category(self, category: str) -> List[SoundEffect]:
        """Get a list of sound effects in a specific category."""
        return self.sound_effects.get(category, [])


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply sound effects to B-roll transitions')
    parser.add_argument('--video', required=True, help='Path to the input video')
    parser.add_argument('--broll-data', required=True, help='Path to the B-roll cuts JSON file')
    parser.add_argument('--output', help='Path for the output video (optional)')
    parser.add_argument('--sound-effects-dir', default='./assets/sound_effects', 
                        help='Directory containing sound effect files')
    parser.add_argument('--cache-dir', default='./cache/sound_effects', 
                        help='Directory to cache processed audio')
    parser.add_argument('--fade-duration', type=float, default=0.3, 
                        help='Duration in seconds for fading sound effects')
    parser.add_argument('--volume', type=float, default=0.7, 
                        help='Volume adjustment for sound effects (0.0-1.0)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Initialize sound effect manager
        manager = SoundEffectManager(
            sound_effects_dir=args.sound_effects_dir,
            cache_dir=args.cache_dir,
            fade_duration=args.fade_duration,
            volume_adjustment=args.volume
        )
        
        # Apply sound effects
        output_path = manager.apply_sound_effects(
            video_path=args.video,
            broll_data_path=args.broll_data,
            output_path=args.output
        )
        
        print(f"Successfully applied sound effects. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)