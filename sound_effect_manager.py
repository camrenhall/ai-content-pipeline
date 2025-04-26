# SoundEffectManager.py
import json
import os
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

import ffmpeg
from pydub import AudioSegment
from pydub.utils import make_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SoundEffectManager')


@dataclass
class SoundEffect:
    """Represents a sound effect to be added to the video."""
    timestamp: float  # When to insert the sound (seconds)
    duration: float   # Duration of the sound effect (seconds)
    type: str         # Type of sound effect (transition_in, transition_out, ambient)
    sound_file_path: str  # Path to the sound effect file
    volume_scale: float = 1.0  # Scale factor for volume


class SoundEffectManager:
    """
    Manages the application of sound effects to videos based on opportunities detected.
    Uses FFmpeg for precise audio mixing and PyDub for advanced audio manipulation.
    """

    def __init__(self, 
                 video_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 opportunities_path: Optional[str] = None,
                 global_volume_scale: float = 1.0):
        """
        Initialize the SoundEffectManager.

        Args:
            video_path: Path to the video file to enhance
            output_path: Path where the enhanced video will be saved
            opportunities_path: Path to the JSON file with sound effect opportunities
            global_volume_scale: Global volume scale for all sound effects (0.0-2.0)
        """
        self.video_path = video_path
        self.output_path = output_path
        self.opportunities_path = opportunities_path
        self.global_volume_scale = global_volume_scale
        self.sound_effects: List[SoundEffect] = []
        self.temp_files: List[str] = []
        self.video_info: Dict[str, Any] = {}

    def load_opportunities(self, opportunities_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load sound effect opportunities from a JSON file.

        Args:
            opportunities_path: Path to the opportunities JSON file (overrides constructor path)

        Returns:
            List of sound effect opportunities
        """
        path = opportunities_path or self.opportunities_path
        if not path:
            raise ValueError("No opportunities path provided")

        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Extract opportunities from the JSON structure
            opportunities = data.get('sound_effect_opportunities', [])
            logger.info(f"Loaded {len(opportunities)} sound effect opportunities from {path}")
            return opportunities
        
        except Exception as e:
            logger.error(f"Failed to load opportunities: {e}")
            raise

    def analyze_video_audio(self, video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the video's audio track to determine volume levels at different timestamps.
        This is used to adaptively scale sound effect volumes.

        Args:
            video_path: Path to the video file (overrides constructor path)

        Returns:
            Dictionary with audio analysis data
        """
        path = video_path or self.video_path
        if not path:
            raise ValueError("No video path provided")

        logger.info(f"Analyzing audio for {path}")
        
        try:
            # Extract basic video information
            probe = ffmpeg.probe(path)
            video_info = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            if not video_info:
                raise ValueError("No video stream found in the input file")
            
            # Store video information for later use
            self.video_info = {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'has_audio': audio_info is not None
            }
            
            # If no audio track, return early with default values
            if not audio_info:
                logger.warning("No audio stream found in the video. Using default volume levels.")
                return {
                    'avg_volume': 0,
                    'peak_volume': 0,
                    'has_audio': False,
                    'segments': []
                }
            
            # Extract audio for analysis using PyDub
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.temp_files.append(temp_audio.name)
            
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(path)
                .output(temp_audio.name, format='wav', acodec='pcm_s16le', ac=1)
                .run(quiet=True, overwrite_output=True)
            )
            
            # Load audio for analysis
            audio = AudioSegment.from_file(temp_audio.name)
            
            # Analyze overall audio levels
            overall_dBFS = audio.dBFS
            
            # Analyze audio in 1-second chunks
            chunk_length_ms = 1000  # 1 second
            chunks = make_chunks(audio, chunk_length_ms)
            
            # Calculate volume level for each chunk
            segments = []
            for i, chunk in enumerate(chunks):
                # Only include if chunk has content
                if len(chunk) > 0:
                    segment_data = {
                        'timestamp': i,
                        'duration': len(chunk) / 1000,
                        'volume_dBFS': chunk.dBFS if chunk.dBFS > float('-inf') else -80
                    }
                    segments.append(segment_data)
            
            # Calculate peak volume
            peak_volume = max((s['volume_dBFS'] for s in segments), default=-80)
            
            # Clean up temp files if needed
            if os.path.exists(temp_audio.name):
                try:
                    os.unlink(temp_audio.name)
                    self.temp_files.remove(temp_audio.name)
                except:
                    pass
                
            analysis = {
                'avg_volume': overall_dBFS,
                'peak_volume': peak_volume,
                'has_audio': True,
                'segments': segments
            }
            
            logger.info(f"Audio analysis complete. Average volume: {overall_dBFS:.2f} dBFS, Peak: {peak_volume:.2f} dBFS")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze video audio: {e}")
            self._cleanup_temp_files()
            raise

    def prepare_sound_effects(self, opportunities: List[Dict[str, Any]], audio_analysis: Dict[str, Any]) -> List[SoundEffect]:
        """
        Prepare sound effects based on opportunities and audio analysis.
        Adjusts volume and duration based on the video's existing audio.

        Args:
            opportunities: List of sound effect opportunities
            audio_analysis: Audio analysis data from analyze_video_audio

        Returns:
            List of prepared SoundEffect objects
        """
        sound_effects = []
        
        # Define volume scales based on sound effect type
        type_volume_scales = {
            'transition_in': 0.85,   # Louder for transitions
            'transition_out': 0.85,  # Louder for transitions
            'ambient': 0.5           # Quieter for ambient/background sounds
        }
        
        for opp in opportunities:
            try:
                # Verify required fields
                required_fields = ['timestamp', 'duration', 'type', 'sound_file_path']
                if not all(field in opp for field in required_fields):
                    logger.warning(f"Skipping opportunity due to missing fields: {opp}")
                    continue
                
                # Verify sound file exists
                if not os.path.exists(opp['sound_file_path']):
                    logger.warning(f"Sound file not found: {opp['sound_file_path']}")
                    continue
                
                # Calculate adaptive volume scale based on audio analysis and effect type
                volume_scale = self._calculate_volume_scale(
                    opp['timestamp'], 
                    opp['duration'], 
                    opp['type'],
                    audio_analysis
                )
                
                # Apply base volume scale for effect type
                volume_scale *= type_volume_scales.get(opp['type'], 0.8)
                
                # Apply global volume scale
                volume_scale *= self.global_volume_scale
                
                # Create SoundEffect object
                sound_effect = SoundEffect(
                    timestamp=opp['timestamp'],
                    duration=opp['duration'],
                    type=opp['type'],
                    sound_file_path=opp['sound_file_path'],
                    volume_scale=volume_scale
                )
                
                sound_effects.append(sound_effect)
                logger.info(f"Prepared sound effect at {sound_effect.timestamp}s: {os.path.basename(sound_effect.sound_file_path)} (volume scale: {volume_scale:.2f})")
                
            except Exception as e:
                logger.warning(f"Failed to prepare sound effect: {e}")
                continue
                
        return sound_effects

    def _calculate_volume_scale(self, 
                              timestamp: float, 
                              duration: float, 
                              effect_type: str,
                              audio_analysis: Dict[str, Any]) -> float:
        """
        Calculate adaptive volume scale based on the video's audio level at the specified timestamp.

        Args:
            timestamp: Timestamp in seconds
            duration: Duration in seconds
            effect_type: Type of sound effect
            audio_analysis: Audio analysis data

        Returns:
            Volume scale factor (0.0-1.0)
        """
        # Default scale if no audio or incomplete analysis
        default_scale = {
            'transition_in': 0.8,
            'transition_out': 0.8,
            'ambient': 0.5
        }.get(effect_type, 0.7)
        
        if not audio_analysis.get('has_audio', False) or not audio_analysis.get('segments'):
            return default_scale
            
        # Find segments that overlap with our effect
        effect_end = timestamp + duration
        overlapping_segments = [
            s for s in audio_analysis['segments'] 
            if (s['timestamp'] <= effect_end and s['timestamp'] + s['duration'] >= timestamp)
        ]
        
        if not overlapping_segments:
            return default_scale
            
        # Calculate average volume of overlapping segments
        segment_volumes = [s['volume_dBFS'] for s in overlapping_segments]
        avg_segment_volume = sum(segment_volumes) / len(segment_volumes)
        
        # Reference levels (can be adjusted)
        reference_quiet = -30  # dBFS
        reference_loud = -10   # dBFS
        
        # Scale based on audio level (louder when video is quiet, quieter when video is loud)
        if avg_segment_volume <= reference_quiet:
            # Video is very quiet, use higher scale
            scale = 0.9
        elif avg_segment_volume >= reference_loud:
            # Video is very loud, use lower scale
            scale = 0.3
        else:
            # Linear interpolation between quiet and loud references
            # Normalize the range from reference_quiet to reference_loud to 0.3-0.9
            normalized_pos = (avg_segment_volume - reference_quiet) / (reference_loud - reference_quiet)
            scale = 0.9 - (normalized_pos * 0.6)  # Scale from 0.9 down to 0.3
            
        # Further adjust based on effect type
        if effect_type == 'ambient':
            # Ambient sounds should be quieter
            scale *= 0.6
            
        return min(max(scale, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    def process_sound_effects(self, sound_effects: Optional[List[SoundEffect]] = None) -> bool:
        """
        Process and apply sound effects to the video.

        Args:
            sound_effects: List of sound effects to apply (uses self.sound_effects if None)

        Returns:
            True if successful, False otherwise
        """
        effects = sound_effects or self.sound_effects
        if not effects:
            logger.warning("No sound effects to process")
            return False
            
        if not self.video_path or not self.output_path:
            raise ValueError("Video path and output path must be set")
            
        logger.info(f"Processing {len(effects)} sound effects for {self.video_path}")
        
        try:
            # Normalize sound effects (adjust duration and format)
            prepared_effects = self._prepare_effect_files(effects)
            
            # Create a complex filter for FFmpeg
            filter_complex = self._create_ffmpeg_filter(prepared_effects)
            
            # Execute FFmpeg command
            self._execute_ffmpeg_command(filter_complex)
            
            logger.info(f"Successfully added sound effects to {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process sound effects: {e}")
            return False
        finally:
            self._cleanup_temp_files()

    def _prepare_effect_files(self, sound_effects: List[SoundEffect]) -> List[Dict[str, Any]]:
        """
        Prepare sound effect files by adjusting duration and format if needed.

        Args:
            sound_effects: List of sound effects to prepare

        Returns:
            List of prepared effect dictionaries
        """
        prepared_effects = []
        
        for i, effect in enumerate(sound_effects):
            try:
                # Load sound file with PyDub
                sound = AudioSegment.from_file(effect.sound_file_path)
                
                # Check duration
                original_duration = len(sound) / 1000  # Convert ms to seconds
                
                if original_duration > effect.duration:
                    # Trim sound to match duration
                    logger.info(f"Trimming sound effect {i+1} from {original_duration:.2f}s to {effect.duration:.2f}s")
                    sound = sound[:int(effect.duration * 1000)]
                elif original_duration < effect.duration:
                    # For shorter sounds, we'll just use them as-is and let FFmpeg handle the timing
                    # No looping - just use the original sound
                    logger.info(f"Sound effect {i+1} is shorter than specified duration ({original_duration:.2f}s < {effect.duration:.2f}s). Using as-is.")
                
                # Adjust volume
                if effect.volume_scale != 1.0:
                    # Convert scale factor to dB change
                    db_change = 20 * (effect.volume_scale - 1)  # This converts linear scale to dB
                    sound = sound.apply_gain(db_change)
                
                # Save to temporary file
                temp_effect = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                self.temp_files.append(temp_effect.name)
                
                sound.export(temp_effect.name, format='wav')
                
                prepared_effects.append({
                    'index': i,
                    'effect': effect,
                    'temp_path': temp_effect.name,
                    'original_duration': original_duration
                })
                
            except Exception as e:
                logger.warning(f"Failed to prepare sound effect {i+1}: {e}")
                continue
                
        return prepared_effects

    def _create_ffmpeg_filter(self, prepared_effects: List[Dict[str, Any]]) -> str:
        """
        Create a complex filter string for FFmpeg with all sound effects.

        Args:
            prepared_effects: List of prepared sound effect dictionaries

        Returns:
            FFmpeg complex filter string
        """
        # Start with filter parts for individual effects
        filter_parts = []
        
        # Create sound effect parts
        for i, effect in enumerate(prepared_effects):
            # Input index starts at 1 (0 is the main video)
            input_idx = i + 1
            sound_idx = effect['index']
            timestamp = effect['effect'].timestamp
            
            # Add filter for this sound effect
            filter_parts.append(f"[{input_idx}:a]adelay={int(timestamp*1000)}|{int(timestamp*1000)}[a{sound_idx}]")
        
        # Now handle the mixing
        if not self.video_info.get('has_audio', False):
            # No original audio, create silence
            filter_parts.append(f"anullsrc=r=44100:cl=stereo:d={self.video_info['duration']}[silence]")
            
            # If there are sound effects, mix them with the silence
            if prepared_effects:
                # Create the mix command
                mix_inputs = ["[silence]"]
                for i in range(len(prepared_effects)):
                    mix_inputs.append(f"[a{i}]")
                
                # Use amerge instead of amix for better control
                filter_parts.append(f"{' '.join(mix_inputs)}amerge=inputs={len(mix_inputs)}[aout]")
            else:
                # No sound effects either, just use silence
                filter_parts.append("[silence]asetpts=PTS-STARTPTS[aout]")
        else:
            # We have original audio, preserve its volume
            if prepared_effects:
                # Create multiple mix steps to preserve original audio volume
                
                # First step: Create a named reference to the original audio
                filter_parts.append("[0:a]asetpts=PTS-STARTPTS[original]")
                
                # For each sound effect, mix it with either the original or the previous mix
                for i in range(len(prepared_effects)):
                    input_label = "[original]" if i == 0 else f"[mix{i-1}]"
                    output_label = f"[mix{i}]" if i < len(prepared_effects) - 1 else "[aout]"
                    
                    # Use volume filter to preserve the original audio volume
                    # The goal is to layer the sound effects on top without changing the original audio
                    filter_parts.append(
                        f"{input_label}[a{i}]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0{output_label}"
                    )
            else:
                # No sound effects, just pass through the original audio
                filter_parts.append("[0:a]asetpts=PTS-STARTPTS[aout]")
        
        # Complete filter string
        return ';'.join(filter_parts)

    def _execute_ffmpeg_command(self, filter_complex: str) -> None:
        """
        Execute the FFmpeg command to apply sound effects.

        Args:
            filter_complex: FFmpeg complex filter string
        """
        # Prepare the base command
        cmd = ['ffmpeg', '-y', '-i', self.video_path]
        
        # Add each sound effect file as input
        for effect in self.temp_files:
            if effect.endswith('.wav'):
                cmd.extend(['-i', effect])
        
        # Add filter complex
        cmd.extend(['-filter_complex', filter_complex])
        
        # Add output settings
        cmd.extend([
            '-map', '0:v',         # Map original video
            '-map', '[aout]',      # Map mixed audio output
            '-c:v', 'copy',        # Copy video codec to avoid re-encoding
            '-c:a', 'aac',         # Audio codec
            '-b:a', '192k',        # Audio bitrate
            self.output_path
        ])
        
        # Execute command
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def _cleanup_temp_files(self) -> None:
        """Clean up any temporary files created during processing."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        
        self.temp_files = []

    def process(self, 
               video_path: Optional[str] = None,
               output_path: Optional[str] = None,
               opportunities_path: Optional[str] = None,
               global_volume_scale: Optional[float] = None) -> bool:
        """
        Full pipeline to process sound effects from opportunities to final video.

        Args:
            video_path: Path to the video file (overrides constructor path)
            output_path: Path for the output video (overrides constructor path)
            opportunities_path: Path to opportunities JSON (overrides constructor path)
            global_volume_scale: Global volume scale for all sound effects (overrides constructor value)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Set paths if provided
            self.video_path = video_path or self.video_path
            self.output_path = output_path or self.output_path
            self.opportunities_path = opportunities_path or self.opportunities_path
            
            # Set global volume scale if provided
            if global_volume_scale is not None:
                self.global_volume_scale = global_volume_scale
            
            # Validate paths
            if not self.video_path or not os.path.exists(self.video_path):
                raise ValueError(f"Invalid or missing video path: {self.video_path}")
                
            if not self.output_path:
                raise ValueError("Output path is required")
                
            if not self.opportunities_path or not os.path.exists(self.opportunities_path):
                raise ValueError(f"Invalid or missing opportunities path: {self.opportunities_path}")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            
            # Load opportunities
            opportunities = self.load_opportunities()
            
            # Skip if no opportunities
            if not opportunities:
                logger.info("No sound effect opportunities found. Copying input to output.")
                
                # If no processing needed, just copy the input to output
                ffmpeg.input(self.video_path).output(self.output_path, c='copy').run(quiet=True, overwrite_output=True)
                return True
                
            # Analyze video audio
            audio_analysis = self.analyze_video_audio()
            
            # Prepare sound effects
            self.sound_effects = self.prepare_sound_effects(opportunities, audio_analysis)
            
            # Process sound effects
            success = self.process_sound_effects()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to process sound effects: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._cleanup_temp_files()
            return False


def parse_args():
    """Parse command line arguments for CLI usage."""
    parser = argparse.ArgumentParser(description='Apply sound effects to a video based on opportunities')
    
    parser.add_argument('--video', '-v', required=True, help='Path to the input video file')
    parser.add_argument('--output', '-o', required=True, help='Path for the output video file')
    parser.add_argument('--opportunities', '-j', required=True, help='Path to the JSON file with sound effect opportunities')
    parser.add_argument('--volume', '-vol', type=float, default=1.0, help='Global volume scale for all sound effects (0.1-2.0, default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate volume scale
    if args.volume < 0.1 or args.volume > 2.0:
        parser.error("Volume scale must be between 0.1 and 2.0")
    
    return args


def main():
    """Main entry point for CLI usage."""
    args = parse_args()
    
    logger.info(f"Sound Effect Manager starting with video: {args.video}")
    
    manager = SoundEffectManager(
        video_path=args.video,
        output_path=args.output,
        opportunities_path=args.opportunities,
        global_volume_scale=args.volume
    )
    
    success = manager.process()
    
    if success:
        logger.info(f"Sound effects successfully applied. Output saved to: {args.output}")
        return 0
    else:
        logger.error("Failed to apply sound effects")
        return 1


if __name__ == "__main__":
    exit(main())