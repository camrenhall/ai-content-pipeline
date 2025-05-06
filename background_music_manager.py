# background_music_manager.py
import os
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
import subprocess
import math

import ffmpeg
from pydub import AudioSegment
from pydub.utils import make_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BackgroundMusicManager')


class BackgroundMusicManager:
    """
    Manages the application of background music to videos.
    Uses FFmpeg for audio mixing with precise control.
    """

    def __init__(self, 
                 music_dir: str = "./assets/background_music",
                 cache_dir: Optional[str] = "./cache/background_music",
                 base_volume: float = 0.15,
                 fade_in_duration: float = 0.1,
                 fade_out_duration: float = 0.1):
        """
        Initialize the BackgroundMusicManager.

        Args:
            music_dir: Directory containing background music files
            cache_dir: Directory to cache processed audio files
            base_volume: Base volume for background music relative to original audio (0.0-1.0)
            fade_in_duration: Duration of fade-in effect in seconds
            fade_out_duration: Duration of fade-out effect in seconds
        """
        self.music_dir = music_dir
        self.cache_dir = cache_dir
        self.base_volume = max(0.0, min(1.0, base_volume))  # Clamp between 0.0-1.0
        self.fade_in_duration = fade_in_duration
        self.fade_out_duration = fade_out_duration
        self.temp_files = []
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        
    def apply_background_music(self,
                              video_path: str,
                              broll_data_path: Optional[str] = None,
                              music_data_path: Optional[str] = None,
                              output_path: Optional[str] = None,
                              volume_scale: Optional[float] = None) -> str:
        """
        Apply background music to a video.

        Args:
            video_path: Path to the input video
            broll_data_path: Path to B-roll data JSON (for metadata)
            music_data_path: Path to music selection JSON (if None, will be extracted from broll_data)
            output_path: Path for the output video (if None, will generate one)
            volume_scale: Optional override for music volume scaling factor (0.0-2.0)
            
        Returns:
            Path to the output video
        """
        try:
            # Validate input video
            if not os.path.exists(video_path):
                raise ValueError(f"Input video not found: {video_path}")
                
            # Determine output path if not provided
            if not output_path:
                input_path = Path(video_path)
                output_path = str(input_path.with_stem(f"{input_path.stem}_music"))
                
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Get music selection - either directly or from B-roll data
            music_selection = self._get_music_selection(broll_data_path, music_data_path)
            
            if not music_selection:
                self.logger.warning("No music selection found. Copying input to output.")
                # If no processing needed, just copy the input to output
                ffmpeg.input(video_path).output(output_path, c='copy').run(quiet=True, overwrite_output=True)
                return output_path
            
            # Get music file path
            music_file_path = music_selection.get("file_path", "")
            if not music_file_path or not os.path.exists(music_file_path):
                if not os.path.exists(music_file_path):
                    self.logger.warning(f"Music file not found: {music_file_path}")
                
                # Try to resolve relative path
                base_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(base_dir, music_file_path)
                
                if os.path.exists(resolved_path):
                    music_file_path = resolved_path
                else:
                    self.logger.warning("No valid music file found. Copying input to output.")
                    ffmpeg.input(video_path).output(output_path, c='copy').run(quiet=True, overwrite_output=True)
                    return output_path
            
            # Analyze video audio to determine volume levels
            video_info = self._analyze_video_audio(video_path)
            
            # Apply volume scale if provided, otherwise use base volume
            effective_volume = volume_scale if volume_scale is not None else self.base_volume
            
            # Prepare music with appropriate parameters
            processed_music = self._prepare_music(
                music_file_path,
                video_info["duration"],
                video_info,
                effective_volume
            )
            
            if not processed_music:
                self.logger.warning("Failed to prepare music. Copying input to output.")
                ffmpeg.input(video_path).output(output_path, c='copy').run(quiet=True, overwrite_output=True)
                return output_path
            
            # Apply music to video
            self._apply_music_to_video(video_path, processed_music, output_path)
            
            self.logger.info(f"Successfully applied background music to {output_path}")
            
            # Clean up temp files
            self._cleanup_temp_files()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error applying background music: {e}")
            self._cleanup_temp_files()
            
            # In case of error, try to return the original video
            if os.path.exists(video_path) and output_path:
                try:
                    import shutil
                    shutil.copy2(video_path, output_path)
                    return output_path
                except:
                    pass
                    
            return video_path
    
    def _get_music_selection(self, 
                           broll_data_path: Optional[str] = None,
                           music_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get music selection from either music data or B-roll data file.
        
        Args:
            broll_data_path: Path to B-roll data JSON
            music_data_path: Path to music selection JSON
            
        Returns:
            Dictionary with music selection information
        """
        # Try dedicated music data file first
        if music_data_path and os.path.exists(music_data_path):
            try:
                with open(music_data_path, 'r') as f:
                    music_data = json.load(f)
                    
                self.logger.info(f"Loaded music selection from {music_data_path}")
                return music_data
                
            except Exception as e:
                self.logger.warning(f"Failed to load music data from {music_data_path}: {e}")
        
        # Try to extract from B-roll data
        if broll_data_path and os.path.exists(broll_data_path):
            try:
                with open(broll_data_path, 'r') as f:
                    broll_data = json.load(f)
                
                # Check if there's a background_music field in the data
                if "background_music" in broll_data:
                    self.logger.info(f"Found music selection in B-roll data")
                    return broll_data["background_music"]
                    
                # Look for associated background_music.json file
                broll_dir = os.path.dirname(broll_data_path)
                broll_name = os.path.splitext(os.path.basename(broll_data_path))[0]
                music_path = os.path.join(broll_dir, f"{broll_name}_background_music.json")
                
                if os.path.exists(music_path):
                    with open(music_path, 'r') as f:
                        music_data = json.load(f)
                        
                    self.logger.info(f"Loaded music selection from associated file: {music_path}")
                    return music_data
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract music data from B-roll data: {e}")
        
        # No music selection found
        return {}
    
    def _analyze_video_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video to get duration and audio characteristics,
        including volume levels for intelligent mixing.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information including audio analysis
        """
        self.logger.info(f"Analyzing video audio: {video_path}")
        
        try:
            # Get video info with FFmpeg
            probe = ffmpeg.probe(video_path)
            
            # Extract video stream info
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            
            # Extract audio stream info
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            # Get duration from container or video stream
            duration = float(probe['format'].get('duration', 0))
            if not duration and video_stream and 'duration' in video_stream:
                duration = float(video_stream['duration'])
            
            # Initialize with basic info
            video_info = {
                "duration": duration,
                "has_audio": audio_stream is not None,
                "audio_channels": int(audio_stream.get('channels', 0)) if audio_stream else 0,
                "peak_volume": -30,  # Default value if we can't analyze
                "rms_volume": -40,   # Default value if we can't analyze
                "volume_analysis": "default"  # Flag to indicate if we used defaults
            }
            
            # If no audio, just return basic info
            if not audio_stream:
                self.logger.info(f"Video has no audio: {duration:.2f}s")
                return video_info
                
            # Extract audio to analyze volume levels
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.temp_files.append(temp_audio.name)
            temp_audio.close()
            
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(temp_audio.name, format='wav', acodec='pcm_s16le', ac=2)
                .run(quiet=True, overwrite_output=True)
            )
            
            # Analyze audio with PyDub
            try:
                # Load audio for analysis
                audio = AudioSegment.from_file(temp_audio.name)
                
                # Get RMS volume (overall loudness)
                rms_volume = audio.rms
                if rms_volume > 0:
                    # Convert to dBFS (dB Full Scale)
                    rms_volume_db = 20 * math.log10(rms_volume / 32768)  # 16-bit audio reference
                else:
                    rms_volume_db = -100  # Effectively silent
                
                # Analyze in chunks to find peak volume
                chunk_size = 1000  # 1 second chunks
                chunks = make_chunks(audio, chunk_size)
                
                max_chunk_db = max((chunk.dBFS for chunk in chunks if len(chunk) > 0), default=-100)
                
                # Update video info with volume analysis
                video_info.update({
                    "peak_volume": max_chunk_db,
                    "rms_volume": rms_volume_db,
                    "volume_analysis": "detailed"
                })
                
                self.logger.info(f"Audio volume analysis: Peak={max_chunk_db:.2f}dB, RMS={rms_volume_db:.2f}dB")
                
            except Exception as e:
                self.logger.warning(f"Detailed volume analysis failed: {e}. Using basic analysis.")
                
                # Fallback to basic analysis with ffmpeg's volumedetect filter
                temp_stats = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
                self.temp_files.append(temp_stats.name)
                temp_stats.close()
                
                try:
                    # Run ffmpeg volumedetect filter
                    cmd = [
                        'ffmpeg',
                        '-i', video_path,
                        '-filter:a', 'volumedetect',
                        '-f', 'null',
                        '/dev/null'
                    ]
                    
                    process = subprocess.run(
                        cmd, 
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False  # Don't raise error on non-zero exit
                    )
                    
                    # Parse the output
                    output = process.stderr
                    
                    # Extract peak and mean volume
                    peak_volume = -30  # Default
                    mean_volume = -40  # Default
                    
                    for line in output.split('\n'):
                        if 'max_volume' in line:
                            try:
                                peak_volume = float(line.split(':')[1].strip().split()[0])
                            except:
                                pass
                        elif 'mean_volume' in line:
                            try:
                                mean_volume = float(line.split(':')[1].strip().split()[0])
                            except:
                                pass
                    
                    # Update video info
                    video_info.update({
                        "peak_volume": peak_volume,
                        "rms_volume": mean_volume,
                        "volume_analysis": "basic"
                    })
                    
                    self.logger.info(f"Basic audio analysis: Peak={peak_volume:.2f}dB, Mean={mean_volume:.2f}dB")
                    
                except Exception as e2:
                    self.logger.warning(f"Basic volume analysis also failed: {e2}. Using defaults.")
            
            return video_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            return {
                "duration": 0, 
                "has_audio": False, 
                "audio_channels": 0,
                "peak_volume": -30,
                "rms_volume": -40,
                "volume_analysis": "none"
            }
    
    def _prepare_music(self, 
                     music_file_path: str, 
                     video_duration: float,
                     video_info: Dict[str, Any],
                     volume_scale: float) -> Optional[str]:
        """
        Prepare music file for mixing with video:
        - Adjust length to match video duration (loop or trim)
        - Apply intelligent volume adjustment based on video audio levels
        - Add fade in/out effects
        
        Args:
            music_file_path: Path to the music file
            video_duration: Duration of the video in seconds
            video_info: Dictionary with video info including audio levels
            volume_scale: User-specified volume scale factor (0.0-2.0)
            
        Returns:
            Path to the prepared music file
        """
        self.logger.info(f"Preparing music file: {music_file_path}")
        
        try:
            # Create a temporary file for the processed music
            temp_music = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.temp_files.append(temp_music.name)
            temp_music.close()
            
            # Load music with PyDub
            music = AudioSegment.from_file(music_file_path)
            music_duration_ms = len(music)
            music_duration_sec = music_duration_ms / 1000
            
            # Check if we need to adjust duration
            if abs(music_duration_sec - video_duration) > 1.0:  # More than 1 second difference
                if music_duration_sec < video_duration:
                    # Need to loop the music
                    self.logger.info(f"Looping music to match video duration: {music_duration_sec:.2f}s -> {video_duration:.2f}s")
                    
                    # Calculate how many loops we need
                    loops_needed = int(video_duration / music_duration_sec) + 1
                    
                    # Create looped audio
                    looped_music = music * loops_needed
                    
                    # Trim to exact duration
                    target_duration_ms = int(video_duration * 1000)
                    music = looped_music[:target_duration_ms]
                    
                else:
                    # Music is longer than video, trim it
                    self.logger.info(f"Trimming music to match video duration: {music_duration_sec:.2f}s -> {video_duration:.2f}s")
                    target_duration_ms = int(video_duration * 1000)
                    music = music[:target_duration_ms]
            
            # Get music volume level
            music_volume = music.dBFS
            
            # Calculate intelligent volume adjustment based on video audio levels
            if video_info.get("has_audio", False):
                # Get reference levels from video
                video_peak = video_info.get("peak_volume", -30)
                video_rms = video_info.get("rms_volume", -40)
                
                # For videos with very quiet audio, adjust our reference level
                if video_peak < -30:
                    reference_level = max(video_peak, -50)  # Don't go below -50dB
                else:
                    reference_level = video_peak
                
                # Calculate target level for music based on video level and user scale
                # We want music to be volume_scale times quieter than the video's peak level
                target_level = reference_level - (20 * math.log10(1/volume_scale))
                
                # Calculate required adjustment in dB
                adjustment_db = target_level - music_volume
                
                self.logger.info(f"Intelligent volume adjustment:")
                self.logger.info(f"  Video peak: {video_peak:.2f}dB, Music level: {music_volume:.2f}dB")
                self.logger.info(f"  Target level: {target_level:.2f}dB, Adjustment: {adjustment_db:.2f}dB")
                
                # Apply the adjustment
                music = music.apply_gain(adjustment_db)
            else:
                # No video audio, use a conservative approach
                # Set music volume to a moderate level (-18dB is a good default)
                target_level = -18
                adjustment_db = target_level - music_volume
                
                # Scale this by user volume setting
                adjustment_db += (20 * math.log10(volume_scale))
                
                self.logger.info(f"Basic volume adjustment (no video audio):")
                self.logger.info(f"  Target level: {target_level:.2f}dB, Adjustment: {adjustment_db:.2f}dB")
                
                # Apply the adjustment
                music = music.apply_gain(adjustment_db)
            
            # Apply fade in/out (millisecond level)
            fade_in_ms = int(self.fade_in_duration * 1000)
            fade_out_ms = int(self.fade_out_duration * 1000)
            
            if fade_in_ms > 0:
                music = music.fade_in(fade_in_ms)
                
            if fade_out_ms > 0:
                music = music.fade_out(fade_out_ms)
            
            # Export to temporary file
            self.logger.info(f"Exporting processed music to: {temp_music.name}")
            music.export(temp_music.name, format="wav")
            
            return temp_music.name
            
        except Exception as e:
            self.logger.error(f"Error preparing music: {e}")
            return None
    
    def _apply_music_to_video(self, 
                           video_path: str, 
                           music_path: str, 
                           output_path: str) -> bool:
        """
        Apply the prepared music to the video using FFmpeg.
        
        Args:
            video_path: Path to the input video
            music_path: Path to the prepared music file
            output_path: Path for the output video
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Applying music to video: {video_path} -> {output_path}")
        
        try:
            # Check if video has audio
            probe = ffmpeg.probe(video_path)
            has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
            
            # Build the FFmpeg command
            video_input = ffmpeg.input(video_path)
            music_input = ffmpeg.input(music_path)
            
            if has_audio:
                # Mix original audio with music
                # The amix filter is set to average inputs (normalize=0) to maintain relative volumes
                mixed_audio = ffmpeg.filter([video_input.audio, music_input], 'amix', inputs=2, duration='longest', normalize=0)
                
                # Combine video with mixed audio
                output = ffmpeg.output(
                    video_input.video, 
                    mixed_audio, 
                    output_path,
                    vcodec='copy',  # Copy video stream to avoid re-encoding
                    acodec='aac',    # Use AAC for audio
                    audio_bitrate='192k'  # High quality audio
                )
            else:
                # Video has no audio, just use the music
                output = ffmpeg.output(
                    video_input.video, 
                    music_input, 
                    output_path,
                    vcodec='copy',  # Copy video stream to avoid re-encoding
                    acodec='aac',    # Use AAC for audio
                    audio_bitrate='192k'  # High quality audio
                )
            
            # Execute the command
            self.logger.info("Running FFmpeg command to mix audio")
            output.overwrite_output().run(quiet=True)
            
            self.logger.info(f"Successfully created output video with background music: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying music to video: {e}")
            return False
    
    def _cleanup_temp_files(self) -> None:
        """Clean up any temporary files created during processing."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        
        self.temp_files = []


def parse_args():
    """Parse command line arguments for standalone usage."""
    parser = argparse.ArgumentParser(description='Apply background music to a video')
    
    parser.add_argument('--video', '-v', required=True, help='Path to the input video file')
    parser.add_argument('--output', '-o', required=True, help='Path for the output video file')
    parser.add_argument('--music-data', '-m', help='Path to the JSON file with music selection')
    parser.add_argument('--broll-data', '-b', help='Path to the JSON file with B-roll data')
    parser.add_argument('--volume', '-vol', type=float, default=0.15, 
                        help='Volume scale for background music (0.0-2.0, default: 0.15)')
    parser.add_argument('--music-dir', default='./assets/background_music', 
                        help='Directory containing background music files')
    parser.add_argument('--cache-dir', default='./cache/background_music', 
                        help='Directory to cache processed files')
    parser.add_argument('--fade-in', type=float, default=0.1, 
                        help='Duration of fade-in effect in seconds (default: 0.1)')
    parser.add_argument('--fade-out', type=float, default=0.1, 
                        help='Duration of fade-out effect in seconds (default: 0.1)')
    
    return parser.parse_args()


def main():
    """Main entry point for CLI usage."""
    args = parse_args()
    
    logger.info(f"Background Music Manager starting with video: {args.video}")
    
    manager = BackgroundMusicManager(
        music_dir=args.music_dir,
        cache_dir=args.cache_dir,
        base_volume=args.volume,
        fade_in_duration=args.fade_in,
        fade_out_duration=args.fade_out
    )
    
    output_path = manager.apply_background_music(
        video_path=args.video,
        music_data_path=args.music_data,
        broll_data_path=args.broll_data,
        output_path=args.output,
        volume_scale=args.volume
    )
    
    if output_path and os.path.exists(output_path):
        logger.info(f"Background music successfully applied. Output saved to: {output_path}")
        return 0
    else:
        logger.error("Failed to apply background music")
        return 1


if __name__ == "__main__":
    exit(main())