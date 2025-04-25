# background_music_manager.py
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
class MusicTrack:
    """Represents a background music track with metadata."""
    name: str
    path: str
    genre: str
    mood: str  # 'energetic', 'calm', 'inspirational', etc.
    duration: float
    loop_friendly: bool  # Whether the track is designed for seamless looping
    tags: List[str] = field(default_factory=list)
    
    def load_audio(self) -> AudioFileClip:
        """Load the audio file as a MoviePy AudioFileClip."""
        return AudioFileClip(self.path)
    
    def load_pydub(self) -> AudioSegment:
        """Load the audio file as a PyDub AudioSegment for advanced processing."""
        return AudioSegment.from_file(self.path)


class BackgroundMusicManager:
    """
    Manages background music for videos including selection, mixing, and ducking.
    """
    
    def __init__(
        self,
        music_dir: str = "./assets/background_music",
        cache_dir: Optional[str] = "./cache/background_music",
        fade_in_duration: float = 2.0,  # seconds
        fade_out_duration: float = 3.0,  # seconds
        base_volume: float = 0.15,  # 0.0-1.0 relative to original audio
        ducking_amount: float = 0.5,  # 0.0-1.0 how much to reduce during speech
        smart_ducking: bool = True,  # Whether to use dynamic ducking based on speech
    ):
        """
        Initialize the BackgroundMusicManager.
        
        Args:
            music_dir: Directory containing music files
            cache_dir: Directory to cache processed audio
            fade_in_duration: Duration for fading in music at start
            fade_out_duration: Duration for fading out music at end
            base_volume: Base volume level for background music
            ducking_amount: How much to reduce volume during speech
            smart_ducking: Whether to use dynamic ducking based on speech detection
        """
        self.music_dir = music_dir
        self.cache_dir = cache_dir
        self.fade_in_duration = fade_in_duration
        self.fade_out_duration = fade_out_duration
        self.base_volume = base_volume
        self.ducking_amount = ducking_amount
        self.smart_ducking = smart_ducking
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary to store loaded music tracks by genre
        self.music_tracks: Dict[str, List[MusicTrack]] = {}
        
        # Load music tracks
        self._load_music_tracks()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _load_music_tracks(self) -> None:
        """
        Load music tracks from the specified directory.
        
        The expected structure is:
        - music_dir/
          - energetic/
            - track1.mp3
            - track2.mp3
          - calm/
            - track3.mp3
          ...
        
        Each directory name is treated as a genre/mood.
        """
        base_dir = Path(self.music_dir)
        
        if not base_dir.exists():
            self.logger.warning(f"Music directory not found: {base_dir}")
            
            # Create base directory and example structure with README
            os.makedirs(base_dir, exist_ok=True)
            
            # Create example genre directories
            genres = ["energetic", "calm", "inspirational", "dramatic", "ambient"]
            for genre in genres:
                os.makedirs(base_dir / genre, exist_ok=True)
            
            # Create a README file explaining the directory structure
            readme_path = base_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write("# Background Music Directory\n\n")
                f.write("Place your music files in the appropriate subdirectories by mood/genre:\n\n")
                for genre in genres:
                    f.write(f"- `{genre}/`: Music with {genre} mood\n")
                f.write("\nThe BackgroundMusicManager will automatically categorize music based on these directories.\n")
                f.write("\nNaming convention for files: artist_title_bpm_key.mp3\n")
                f.write("Example: johndoe_inspiration_120_Cmaj.mp3\n")
            
            self.logger.info(f"Created music directory structure at {base_dir}")
            self.logger.info(f"Please add music files to the appropriate subdirectories.")
            return
        
        # Discover genres (subdirectories)
        for genre_dir in base_dir.iterdir():
            if not genre_dir.is_dir():
                continue
                
            genre_name = genre_dir.name
            self.music_tracks[genre_name] = []
            
            # Load music tracks from this genre
            for music_file in genre_dir.glob("*.mp3"):
                try:
                    # Load basic info about the music file
                    audio = AudioSegment.from_file(str(music_file))
                    duration = len(audio) / 1000.0  # convert ms to seconds
                    
                    # Extract metadata from filename
                    # Expected format: artist_title_bpm_key.mp3
                    name_parts = music_file.stem.split('_')
                    name = name_parts[1] if len(name_parts) > 1 else music_file.stem
                    
                    # Extract tags from filename
                    tags = []
                    if len(name_parts) > 2:
                        tags = name_parts[2:]
                    
                    # Determine if track is loop-friendly
                    # This is a heuristic - for actual implementation, might need manual tagging
                    loop_friendly = "loop" in music_file.stem.lower()
                    
                    # Create music track object
                    music_track = MusicTrack(
                        name=name,
                        path=str(music_file),
                        genre=genre_name,
                        mood=genre_name,  # Use genre name as mood for simplicity
                        duration=duration,
                        loop_friendly=loop_friendly,
                        tags=tags
                    )
                    
                    self.music_tracks[genre_name].append(music_track)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load music track {music_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.music_tracks[genre_name])} music tracks in genre '{genre_name}'")
    
    def select_music_by_content(self, broll_data_path: str) -> Optional[MusicTrack]:
        """
        Select appropriate background music based on video content analysis.
        
        Args:
            broll_data_path: Path to the B-roll cuts JSON file, which contains context info
            
        Returns:
            Selected MusicTrack or None if no suitable track found
        """
        try:
            # Load B-roll data
            with open(broll_data_path, 'r') as f:
                data = json.load(f)
            
            # Analyze content to determine mood
            broll_cuts = data.get("broll_cuts", [])
            
            # Collect all keywords from all B-roll cuts
            all_keywords = []
            for cut in broll_cuts:
                all_keywords.extend(cut.get("keywords", []))
                all_keywords.extend(cut.get("original_keywords", []))
                
                # Also check abstract concepts if available
                all_keywords.extend(cut.get("abstract_concepts", []))
            
            # Define mood mappings based on keywords
            mood_keywords = {
                "energetic": ["fast", "action", "energy", "dynamic", "exciting", "movement", "sports"],
                "calm": ["slow", "peaceful", "relaxing", "gentle", "quiet", "nature", "meditation"],
                "inspirational": ["inspire", "motivation", "success", "achievement", "growth", "journey"],
                "dramatic": ["tension", "dramatic", "suspense", "conflict", "serious", "challenge"],
                "ambient": ["background", "subtle", "atmosphere", "space", "environment", "minimal"]
            }
            
            # Count keyword matches for each mood
            mood_scores = {mood: 0 for mood in mood_keywords}
            
            for keyword in all_keywords:
                keyword = keyword.lower()
                for mood, related_keywords in mood_keywords.items():
                    for related in related_keywords:
                        if related in keyword:
                            mood_scores[mood] += 1
            
            # Find the mood with the highest score
            best_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
            
            # If no clear winner, default to calm
            if mood_scores[best_mood] == 0:
                best_mood = "calm"
            
            self.logger.info(f"Selected mood for background music: {best_mood}")
            
            # Find available tracks for this mood
            available_tracks = self.music_tracks.get(best_mood, [])
            
            # If no tracks in the selected mood, fall back to any available tracks
            if not available_tracks:
                for tracks in self.music_tracks.values():
                    if tracks:
                        available_tracks = tracks
                        break
            
            if not available_tracks:
                self.logger.warning("No background music tracks available")
                return None
            
            # Select a random track from the available ones
            # For more sophisticated selection, could add criteria like duration matching
            selected_track = random.choice(available_tracks)
            
            return selected_track
            
        except Exception as e:
            self.logger.error(f"Error selecting music track: {e}")
            
            # Fall back to a random track from any available genre
            all_tracks = []
            for tracks in self.music_tracks.values():
                all_tracks.extend(tracks)
                
            if all_tracks:
                return random.choice(all_tracks)
                
            return None
    
    def apply_background_music(
        self,
        video_path: str,
        broll_data_path: str,
        output_path: Optional[str] = None,
        selected_track: Optional[MusicTrack] = None,
        custom_genre: Optional[str] = None
    ) -> str:
        """
        Apply background music to the video.
        
        Args:
            video_path: Path to the input video
            broll_data_path: Path to the B-roll cuts JSON file
            output_path: Path for the output video (if None, will use input path with "_music" suffix)
            selected_track: Specific music track to use (if None, will auto-select)
            custom_genre: Genre/mood to select from (if None, will auto-detect)
            
        Returns:
            Path to the output video with background music
        """
        try:
            # Determine output path
            if output_path is None:
                input_path = Path(video_path)
                output_path = str(input_path.with_stem(f"{input_path.stem}_music"))
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Select music track if not provided
            if selected_track is None:
                if custom_genre and custom_genre in self.music_tracks and self.music_tracks[custom_genre]:
                    # Use custom genre if specified and available
                    selected_track = random.choice(self.music_tracks[custom_genre])
                else:
                    # Auto-select based on content
                    selected_track = self.select_music_by_content(broll_data_path)
            
            if not selected_track:
                self.logger.warning("No suitable music track found. Using original audio.")
                import shutil
                shutil.copy2(video_path, output_path)
                return output_path
            
            # Generate unique cache key
            cache_key = hashlib.md5(
                f"{video_path}_{selected_track.path}_{self.base_volume}_{self.ducking_amount}".encode()
            ).hexdigest()
            
            # Check for cached result
            cache_path = None
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"{cache_key}_music.mp3")
                
                if os.path.exists(cache_path):
                    self.logger.info(f"Using cached music track from {cache_path}")
                    
                    # Create final video with cached audio
                    video = VideoFileClip(video_path)
                    final_audio = AudioFileClip(cache_path)
                    final_video = video.set_audio(final_audio)
                    
                    # Write final video
                    self.logger.info(f"Writing video with cached music to {output_path}")
                    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
                    
                    # Clean up
                    final_video.close()
                    final_audio.close()
                    video.close()
                    
                    return output_path
            
            # Load video
            self.logger.info(f"Loading video: {video_path}")
            video = VideoFileClip(video_path)
            
            # Get original audio
            original_audio = video.audio
            if original_audio is None:
                self.logger.warning("Video has no audio track. Creating silent audio.")
                from moviepy.audio.AudioClip import AudioClip
                silence = lambda t: 0
                original_audio = AudioClip(make_frame=silence, duration=video.duration)
            
            # Load music track
            self.logger.info(f"Loading music track: {selected_track.name} ({selected_track.genre})")
            music_audio = selected_track.load_audio()
            
            # Check if we need to loop the music to match video duration
            if music_audio.duration < video.duration:
                if selected_track.loop_friendly:
                    # For loop-friendly tracks, we can seamlessly loop
                    self.logger.info(f"Looping music track to match video duration")
                    
                    # Create a list to hold the looped segments
                    looped_segments = []
                    
                    # Calculate how many loops we need
                    total_time = 0
                    while total_time < video.duration:
                        start_time = total_time
                        looped_segment = music_audio.set_start(start_time)
                        looped_segments.append(looped_segment)
                        total_time += music_audio.duration
                    
                    # Combine the looped segments
                    music_audio = CompositeAudioClip(looped_segments)
                else:
                    # For non-loop-friendly tracks, we need to crossfade
                    self.logger.info(f"Creating crossfaded loops of music track")
                    
                    # Use PyDub for more precise audio manipulation
                    music_segment = selected_track.load_pydub()
                    
                    # Calculate how many loops we need
                    num_loops = int(video.duration / selected_track.duration) + 1
                    
                    # Create crossfaded loops
                    crossfade_duration = min(3000, len(music_segment) // 4)  # in milliseconds
                    looped_segment = music_segment
                    
                    for _ in range(num_loops - 1):
                        looped_segment = looped_segment.append(
                            music_segment, 
                            crossfade=crossfade_duration
                        )
                    
                    # Export to temporary file
                    temp_path = os.path.join(self.cache_dir, f"temp_looped_{cache_key}.mp3")
                    looped_segment.export(temp_path, format="mp3")
                    
                    # Load back into MoviePy
                    music_audio = AudioFileClip(temp_path)
            
            # If the music is longer than the video, trim it
            if music_audio.duration > video.duration:
                music_audio = music_audio.subclip(0, video.duration)
            
            # Apply fades
            music_audio = music_audio.audio_fadein(self.fade_in_duration)
            music_audio = music_audio.audio_fadeout(self.fade_out_duration)
            
            # Apply base volume adjustment
            music_audio = music_audio.volumex(self.base_volume)
            
            # Apply dynamic ducking if enabled
            if self.smart_ducking:
                self.logger.info("Applying smart audio ducking")
                
                # This is a simplified approach to ducking
                # For production, you might want to use a more sophisticated
                # approach with speech detection and RMS analysis
                
                # Load B-roll data to identify speech segments
                with open(broll_data_path, 'r') as f:
                    data = json.load(f)
                
                broll_cuts = data.get("broll_cuts", [])
                
                # Create a dynamic volume curve
                # The idea is to reduce volume during speech segments
                # and keep it higher during B-roll segments
                
                # Create a default volume curve (all at base volume)
                fps = 100  # Resolution of volume curve
                volume_curve = [1.0] * int(video.duration * fps)
                
                # Identify B-roll segments (where we keep higher volume)
                for cut in broll_cuts:
                    if not cut.get("path"):
                        continue
                        
                    start_time = cut.get("timestamp", 0)
                    duration = cut.get("duration", 0)
                    
                    # Calculate start and end frames
                    start_frame = int(start_time * fps)
                    end_frame = int((start_time + duration) * fps)
                    
                    # Ensure frame indices are within bounds
                    start_frame = max(0, min(start_frame, len(volume_curve) - 1))
                    end_frame = max(0, min(end_frame, len(volume_curve) - 1))
                    
                    # Set higher volume during B-roll
                    for i in range(start_frame, end_frame):
                        volume_curve[i] = 1.0
                
                # Reduce volume during non-B-roll segments (assumed to be speech)
                for i in range(len(volume_curve)):
                    if volume_curve[i] < 1.0:
                        volume_curve[i] = 1.0 - self.ducking_amount
                
                # Smooth the volume curve to avoid abrupt changes
                smoothed_curve = volume_curve.copy()
                smoothing_window = int(0.5 * fps)  # 0.5 second smoothing window
                
                for i in range(smoothing_window, len(volume_curve) - smoothing_window):
                    window = volume_curve[i-smoothing_window:i+smoothing_window]
                    smoothed_curve[i] = sum(window) / len(window)
                
                # Create a dynamic volume function
                def dynamic_volume(t):
                    frame = int(t * fps)
                    if frame < len(smoothed_curve):
                        return smoothed_curve[frame]
                    return 1.0 - self.ducking_amount
                
                # Apply the dynamic volume
                music_audio = music_audio.fx(
                    lambda clip: clip.fl(
                        lambda gf, t: gf(t) * dynamic_volume(t)
                    )
                )
            
            # Mix with original audio
            final_audio = CompositeAudioClip([original_audio, music_audio])
            
            # Cache the final audio if needed
            if cache_path:
                self.logger.info(f"Caching music mix to {cache_path}")
                final_audio.write_audiofile(cache_path, codec="mp3")
            
            # Apply to video
            final_video = video.set_audio(final_audio)
            
            # Write final video
            self.logger.info(f"Writing final video with music to {output_path}")
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            # Clean up
            final_video.close()
            final_audio.close()
            video.close()
            music_audio.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error applying background music: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return original video path if processing failed
            return video_path
    
    def get_available_genres(self) -> List[str]:
        """Get a list of available music genres."""
        return list(self.music_tracks.keys())
    
    def get_tracks_in_genre(self, genre: str) -> List[MusicTrack]:
        """Get a list of music tracks in a specific genre."""
        return self.music_tracks.get(genre, [])


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply background music to a video')
    parser.add_argument('--video', required=True, help='Path to the input video')
    parser.add_argument('--broll-data', required=True, help='Path to the B-roll cuts JSON file')
    parser.add_argument('--output', help='Path for the output video (optional)')
    parser.add_argument('--music-dir', default='./assets/background_music', 
                        help='Directory containing music files')
    parser.add_argument('--genre', help='Specific genre/mood to use')
    parser.add_argument('--track', help='Specific track to use (full path)')
    parser.add_argument('--volume', type=float, default=0.15, 
                        help='Base volume for background music (0.0-1.0)')
    parser.add_argument('--ducking', type=float, default=0.5, 
                        help='Amount to reduce volume during speech (0.0-1.0)')
    parser.add_argument('--no-smart-ducking', action='store_true', 
                        help='Disable smart audio ducking')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Initialize background music manager
        manager = BackgroundMusicManager(
            music_dir=args.music_dir,
            cache_dir="./cache/background_music",
            base_volume=args.volume,
            ducking_amount=args.ducking,
            smart_ducking=not args.no_smart_ducking
        )
        
        # Handle specific track if provided
        selected_track = None
        if args.track:
            if not os.path.exists(args.track):
                print(f"Error: Specified track not found: {args.track}")
                sys.exit(1)
                
            # Create a simple MusicTrack object for the specified file
            track_path = Path(args.track)
            audio = AudioSegment.from_file(args.track)
            duration = len(audio) / 1000.0
            
            selected_track = MusicTrack(
                name=track_path.stem,
                path=args.track,
                genre="custom",
                mood="custom",
                duration=duration,
                loop_friendly=False
            )
        
        # Apply background music
        output_path = manager.apply_background_music(
            video_path=args.video,
            broll_data_path=args.broll_data,
            output_path=args.output,
            selected_track=selected_track,
            custom_genre=args.genre
        )
        
        print(f"Successfully applied background music. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)