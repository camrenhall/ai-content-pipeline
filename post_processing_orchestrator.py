# post_processing_orchestrator.py
import os
import sys
import json
import logging
import argparse
import time
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml

# Set up logging with a colorful formatter
import colorama
from colorama import Fore, Style
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("post_processing")

# Import post-processing components
try:
    from sound_effect_manager import SoundEffectManager
    from sound_effect_opportunity_detector import SoundEffectOpportunityDetector
    from background_music_manager import BackgroundMusicManager
    from background_music_opportunity_detector import BackgroundMusicOpportunityDetector
    from transition_manager import VideoTransitionEffects
    from sound_reapplier import transfer_audio
    from caption_manager import upload_and_download
    from camera_movement_manager import apply_video_effects as apply_camera_movements
except ImportError as e:
    logger.error(f"Failed to import post-processing components: {e}")
    logger.error("Please ensure all components are installed and in the Python path.")
    sys.exit(1)

@dataclass
class ProcessingStep:
    """Represents a step in the post-processing pipeline with metadata."""
    name: str
    description: str
    enabled: bool = True
    completed: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output_path: Optional[str] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


class PostProcessingOrchestrator:
    """
    Coordinates all post-processing steps after the main pipeline completes.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache_dir: str = "./cache/post_processing",
        sound_effects_dir: str = "./assets/sound_effects",
        music_dir: str = "./assets/background_music",
    ):
        """
        Initialize the post-processing orchestrator.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory to store cached files
            sound_effects_dir: Directory containing sound effect files
            music_dir: Directory containing background music files
        """
        self.config = config
        self.cache_dir = cache_dir
        self.sound_effects_dir = sound_effects_dir
        self.music_dir = music_dir
        
        # Create cache directory and subdirectories if needed
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "sound_effects"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "background_music"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "transitions"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "camera_movements"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "captions"), exist_ok=True)
        
        # Initialize temporary files list for cleanup
        self.temp_files = []
        
        # Initialize pipeline steps
        self.steps = {
            # Define the steps in the processing order
            "sound_effects": ProcessingStep(
                name="sound_effects",
                description="Add sound effects to the video"
            ),
            "background_music": ProcessingStep(
                name="background_music",
                description="Add background music to the video"
            ),
            "camera_movements": ProcessingStep(
                name="camera_movements",
                description="Apply camera movement effects (zoom, shake, punch-in)"
            ),
            "transitions": ProcessingStep(
                name="transitions",
                description="Apply transition effects between scenes"
            ),
            "sound_reapplier": ProcessingStep(
                name="sound_reapplier",
                description="Reapply audio to the video after effects"
            ),
            "captioning": ProcessingStep(
                name="captioning",
                description="Add captions to the video"
            )
        }
        
        # Enable/disable steps based on configuration
        self._configure_steps()
        
        self.logger = logging.getLogger("post_processing")
    
    def _configure_steps(self):
        """Configure which steps are enabled based on the configuration."""
        # Get the post-processing configuration
        post_processing_config = self.config.get("post_processing", {})
        
        # Enable/disable steps based on configuration
        enabled_steps = post_processing_config.get("steps", [])
        
        # Set default enabled status for each step
        for step_name, step in self.steps.items():
            step.enabled = step_name in enabled_steps
        
        # Special case: Sound reapplier must be enabled if camera_movements or transitions are enabled
        if self.steps["camera_movements"].enabled or self.steps["transitions"].enabled:
            self.steps["sound_reapplier"].enabled = True
            logger.info("Automatically enabling sound_reapplier because camera_movements or transitions are enabled")
    
    def _generate_temp_path(self, prefix, suffix=".mp4"):
        """Generate a temporary file path in the cache directory."""
        temp_path = os.path.join(self.cache_dir, f"{prefix}_{int(time.time())}{suffix}")
        self.temp_files.append(temp_path)
        return temp_path
    
    def _start_step(self, step_name):
        """Mark a step as started and log it."""
        step = self.steps[step_name]
        step.start_time = time.time()
        
        # Print a colorful step header
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.CYAN}== STARTING: {Fore.YELLOW}{step.name.upper()} {Fore.CYAN}- {step.description}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")
        
        return step
    
    def _complete_step(self, step_name, output_path):
        """Mark a step as completed and log its execution time."""
        step = self.steps[step_name]
        step.end_time = time.time()
        step.completed = True
        step.output_path = output_path
        
        execution_time = step.execution_time or 0
        
        # Print colorful completion message
        print(f"\n{Fore.GREEN}{'=' * 80}")
        print(f"{Fore.GREEN}== COMPLETED: {Fore.YELLOW}{step.name.upper()} {Fore.GREEN}in {Fore.YELLOW}{execution_time:.2f} seconds")
        print(f"{Fore.GREEN}== Output: {Fore.YELLOW}{output_path}")
        print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}\n")
        
        return step
    
    def _skip_step(self, step_name):
        """Log that a step is being skipped."""
        step = self.steps[step_name]
        
        # Print colorful skip message
        print(f"\n{Fore.BLUE}{'=' * 80}")
        print(f"{Fore.BLUE}== SKIPPING: {Fore.YELLOW}{step.name.upper()} {Fore.BLUE}- {step.description}")
        print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}\n")
        
        return step
    
    def process(
        self,
        input_video_path: str,
        broll_data_path: str,
        output_video_path: Optional[str] = None,
        steps: Optional[List[str]] = None,
        llm_api_key: Optional[str] = None,
        llm_api_url: Optional[str] = None,
        clean_intermediates: bool = True
    ) -> str:
        """
        Run post-processing on the input video.
        
        Args:
            input_video_path: Path to the input video
            broll_data_path: Path to the B-roll cuts JSON file
            output_video_path: Path for the final output video (if None, will use input path with "_post" suffix)
            steps: List of post-processing steps to run (if None, use configuration)
            llm_api_key: API key for LLM (optional, for opportunity detectors)
            llm_api_url: API URL for LLM (optional, for opportunity detectors)
            clean_intermediates: Whether to clean up intermediate files
            
        Returns:
            Path to the processed output video
        """
        # Determine output path
        if output_video_path is None:
            input_path = Path(input_video_path)
            output_video_path = str(input_path.with_stem(f"{input_path.stem}_post"))
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        
        # Override steps if provided
        if steps:
            for step_name in self.steps:
                self.steps[step_name].enabled = step_name in steps
            
            # Ensure sound_reapplier is enabled if needed
            if "camera_movements" in steps or "transitions" in steps:
                self.steps["sound_reapplier"].enabled = True
        
        # Print processing plan
        enabled_steps = [name for name, step in self.steps.items() if step.enabled]
        print(f"\n{Fore.MAGENTA}{'=' * 80}")
        print(f"{Fore.MAGENTA}== POST-PROCESSING PLAN: {Fore.YELLOW}{', '.join(enabled_steps)}")
        print(f"{Fore.MAGENTA}== Input: {Fore.YELLOW}{input_video_path}")
        print(f"{Fore.MAGENTA}== Output: {Fore.YELLOW}{output_video_path}")
        print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")
        
        # Track current video path through the processing steps
        current_video_path = input_video_path
        
        # Keep track of the audio source (for sound reapplier)
        audio_source_path = input_video_path
        
        # Start total timer
        total_start_time = time.time()
        
        try:
            # Create a copy of audio before any processing (will be needed if sound_reapplier is used)
            original_audio_path = self._generate_temp_path("original_audio_backup")
            shutil.copy2(input_video_path, original_audio_path)
            
            # 1. Apply sound effects
            if self.steps["sound_effects"].enabled:
                step = self._start_step("sound_effects")
                
                # Generate sound effect opportunities
                sound_effect_opps_path = os.path.join(
                    self.cache_dir, 
                    f"sound_effect_opportunities_{Path(input_video_path).stem}.json"
                )
                
                # Detect sound effect opportunities
                detector = SoundEffectOpportunityDetector(
                    sound_effects_dir=self.sound_effects_dir,
                    llm_api_url=llm_api_url or self.config.get("llm_api_url"),
                    llm_api_key=llm_api_key or self.config.get("api_keys", {}).get("llm"),
                    llm_model=self.config.get("llm_model", "gpt-4o"),
                    cache_dir=os.path.join(self.cache_dir, "sound_effects"),
                    enable_transitions=self.config.get("sound_effects", {}).get("enable_transitions", True),
                    enable_ambient=self.config.get("sound_effects", {}).get("enable_ambient", True),
                    randomize_selection=self.config.get("sound_effects", {}).get("randomize_selection", True)
                )
                
                detector.detect_opportunities(broll_data_path, sound_effect_opps_path)
                
                # Apply sound effects to video
                sound_effects_output = self._generate_temp_path("sound_effects")
                sound_effect_manager = SoundEffectManager(
                    video_path=current_video_path,
                    output_path=sound_effects_output,
                    opportunities_path=sound_effect_opps_path,
                    global_volume_scale=self.config.get("sound_effects", {}).get("volume", 0.7)
                )
                
                success = sound_effect_manager.process()
                
                if success:
                    current_video_path = sound_effects_output
                    audio_source_path = sound_effects_output
                    self._complete_step("sound_effects", sound_effects_output)
                else:
                    logger.warning("Sound effects processing failed. Using original video.")
                    self._skip_step("sound_effects")
            else:
                self._skip_step("sound_effects")
            
            # 2. Apply background music
            if self.steps["background_music"].enabled:
                step = self._start_step("background_music")
                
                # Generate background music opportunity
                music_opp_path = os.path.join(
                    self.cache_dir, 
                    f"background_music_opportunity_{Path(input_video_path).stem}.json"
                )
                
                # Load broll data for script analysis
                with open(broll_data_path, 'r') as f:
                    broll_data = json.load(f)
                
                # Detect background music opportunity
                music_detector = BackgroundMusicOpportunityDetector(
                    music_dir=self.music_dir,
                    llm_api_url=llm_api_url or self.config.get("llm_api_url"),
                    llm_api_key=llm_api_key or self.config.get("api_keys", {}).get("llm"),
                    llm_model=self.config.get("llm_model", "gpt-4o"),
                    cache_dir=os.path.join(self.cache_dir, "background_music")
                )
                
                # Create dummy script analysis structure from broll data
                script_analysis = {
                    "result": {
                        "transcript": {
                            "full_text": " ".join([cut.get("transcript_segment", "") for cut in broll_data.get("broll_cuts", [])])
                        }
                    }
                }
                
                music_detector.detect_music(script_analysis, music_opp_path)
                
                # Apply background music to video
                background_music_output = self._generate_temp_path("background_music")
                
                # Get background music configuration
                bg_music_config = self.config.get("background_music", {})
                
                music_manager = BackgroundMusicManager(
                    music_dir=self.music_dir,
                    cache_dir=os.path.join(self.cache_dir, "background_music"),
                    base_volume=bg_music_config.get("volume", 0.15),
                    fade_in_duration=bg_music_config.get("fade_in", 2.0),
                    fade_out_duration=bg_music_config.get("fade_out", 3.0)
                )
                
                background_music_output = music_manager.apply_background_music(
                    video_path=current_video_path,
                    broll_data_path=broll_data_path,
                    music_data_path=music_opp_path,
                    output_path=background_music_output,
                    volume_scale=bg_music_config.get("volume", 0.15)
                )
                
                if background_music_output and os.path.exists(background_music_output):
                    current_video_path = background_music_output
                    audio_source_path = background_music_output
                    self._complete_step("background_music", background_music_output)
                else:
                    logger.warning("Background music processing failed. Using previous video.")
                    self._skip_step("background_music")
            else:
                self._skip_step("background_music")
            
            # 3. Apply camera movements (zoom, shake, punch-in)
            if self.steps["camera_movements"].enabled:
                step = self._start_step("camera_movements")
                
                camera_movements_output = self._generate_temp_path("camera_movements")
                
                # Get camera movement configuration
                camera_config = self.config.get("camera_movements", {})
                
                # Apply camera movement effects
                apply_camera_movements(
                    current_video_path, 
                    camera_movements_output,
                    zoom_factor=camera_config.get("zoom_factor", 1.1),
                    shake_intensity=camera_config.get("shake_intensity", 2),
                    punchin_factor=camera_config.get("punchin_factor", 1.08),
                    frame_rate=camera_config.get("frame_rate", 30),
                    zoom_duration=camera_config.get("zoom_duration", 100),
                    shake_duration=camera_config.get("shake_duration", 60),
                    punchin_duration=camera_config.get("punchin_duration", 50),
                    normal_duration=camera_config.get("normal_duration", 40),
                    shake_interval=camera_config.get("shake_interval", 450)
                )
                
                if os.path.exists(camera_movements_output):
                    current_video_path = camera_movements_output
                    self._complete_step("camera_movements", camera_movements_output)
                else:
                    logger.warning("Camera movements processing failed. Using previous video.")
                    self._skip_step("camera_movements")
            else:
                self._skip_step("camera_movements")
            
            # 4. Apply transitions
            if self.steps["transitions"].enabled:
                step = self._start_step("transitions")
                
                transitions_output = self._generate_temp_path("transitions")
                
                # Get transition configuration
                transition_config = self.config.get("transitions", {})
                
                # Initialize the transitions manager
                transition_manager = VideoTransitionEffects(
                    current_video_path,
                    transitions_output,
                    transition_duration=transition_config.get("duration", 20)
                )
                
                # Process with transitions
                transition_manager.process_video_with_motion_transitions(
                    transition_type=transition_config.get("type", "cross_fade"),
                    randomize=transition_config.get("randomize", False),
                    transitions_file=transition_config.get("transitions_file", "./transitions.json")
                )
                
                # Close the transitions manager
                transition_manager.close()
                
                if os.path.exists(transitions_output):
                    current_video_path = transitions_output
                    self._complete_step("transitions", transitions_output)
                else:
                    logger.warning("Transitions processing failed. Using previous video.")
                    self._skip_step("transitions")
            else:
                self._skip_step("transitions")
            
            # 5. Apply sound reapplier (if needed)
            # This should be run if transitions or camera movements were applied
            if self.steps["sound_reapplier"].enabled:
                step = self._start_step("sound_reapplier")
                
                sound_reapplier_output = self._generate_temp_path("sound_reapplier")
                
                # Transfer audio from the audio_source_path to the current_video_path
                success = transfer_audio(
                    current_video_path,  # video source (keep video track)
                    audio_source_path,   # audio source (use audio track from this)
                    sound_reapplier_output
                )
                
                if success:
                    current_video_path = sound_reapplier_output
                    self._complete_step("sound_reapplier", sound_reapplier_output)
                else:
                    logger.warning("Sound reapplication failed. Using previous video.")
                    self._skip_step("sound_reapplier")
            else:
                self._skip_step("sound_reapplier")
            
            # 6. Apply captioning (always the last step)
            if self.steps["captioning"].enabled:
                step = self._start_step("captioning")
                
                captioning_output = self._generate_temp_path("captioning")
                
                # Get captioning configuration
                captioning_config = self.config.get("captioning", {})
                
                # Apply captioning
                upload_and_download(
                    current_video_path,
                    captioning_output,
                    captioning_config.get("api_url", "https://ai-content-pipeline.onrender.com/caption-video")
                )
                
                if os.path.exists(captioning_output) and os.path.getsize(captioning_output) > 0:
                    current_video_path = captioning_output
                    self._complete_step("captioning", captioning_output)
                else:
                    logger.warning("Captioning failed. Using previous video.")
                    self._skip_step("captioning")
            else:
                self._skip_step("captioning")
            
            # Copy the final processed video to the output path
            if current_video_path != output_video_path:
                try:
                    logger.info(f"Copying final output to {output_video_path}")
                    shutil.copy2(current_video_path, output_video_path)
                except Exception as e:
                    logger.error(f"Error copying final output: {e}")
                    output_video_path = current_video_path
            
            # Calculate total processing time
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            # Print summary
            print(f"\n{Fore.GREEN}{'=' * 80}")
            print(f"{Fore.GREEN}== POST-PROCESSING COMPLETE!")
            print(f"{Fore.GREEN}== Total time: {Fore.YELLOW}{total_time:.2f} seconds")
            print(f"{Fore.GREEN}== Final output: {Fore.YELLOW}{output_video_path}")
            print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}\n")
            
            completed_steps = [name for name, step in self.steps.items() if step.completed]
            if completed_steps:
                print(f"{Fore.CYAN}Completed steps: {Fore.YELLOW}{', '.join(completed_steps)}{Style.RESET_ALL}")
            
            # Cleanup intermediate files if requested
            if clean_intermediates:
                self._cleanup_intermediates()
            
            return output_video_path
            
        except Exception as e:
            logger.error(f"Error during post-processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return original video path if processing failed completely
            if current_video_path == input_video_path:
                return input_video_path
            else:
                # Return the latest successfully processed video
                return current_video_path
    
    def _cleanup_intermediates(self):
        """Clean up intermediate files."""
        logger.info("Cleaning up intermediate files...")
        
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
        
        self.temp_files = []
    
    @staticmethod
    def load_config_from_file(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(config_path, 'r') as f:
                    return json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {file_ext}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}


    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Post-processing for automated content pipeline')
        parser.add_argument('--input', required=True, help='Path to the input video')
        parser.add_argument('--broll-data', required=True, help='Path to the B-roll cuts JSON file')
        parser.add_argument('--output', help='Path for the output video (optional)')
        parser.add_argument('--config', default='./config.yaml', help='Path to configuration file')
        parser.add_argument('--cache-dir', default='./cache/post_processing', help='Cache directory')
        parser.add_argument('--sound-effects-dir', default='./assets/sound_effects', 
                            help='Directory containing sound effect files')
        parser.add_argument('--music-dir', default='./assets/background_music', 
                            help='Directory containing background music files')
        
        # LLM API configuration
        parser.add_argument('--llm-api-key', help='API key for LLM (for opportunity detectors)')
        parser.add_argument('--llm-api-url', help='API URL for LLM (for opportunity detectors)')
        
        # Step selection arguments
        parser.add_argument('--steps', nargs='+', 
                    choices=['sound_effects', 'background_music', 'camera_movements', 
                                'transitions', 'sound_reapplier', 'captioning'],
                    help='Specific post-processing steps to run')
        
        # Keeping intermediate files
        parser.add_argument('--keep-intermediates', action='store_true', 
                            help='Keep intermediate files (default: cleanup)')
        
        # Sound effects options
        parser.add_argument('--sound-effects-volume', type=float, default=0.7,
                            help='Global volume scale for sound effects (0.1-1.0)')
        
        # Background music options
        parser.add_argument('--music-volume', type=float, default=0.15,
                            help='Volume level for background music (0.0-1.0)')
        parser.add_argument('--music-fade-in', type=float, default=2.0,
                            help='Duration of music fade-in effect in seconds (default: 2.0)')
        parser.add_argument('--music-fade-out', type=float, default=3.0,
                            help='Duration of music fade-out effect in seconds (default: 3.0)')
        parser.add_argument('--no-smart-ducking', action='store_true',
                            help='Disable smart ducking for background music')
        
        # Camera movement options
        parser.add_argument('--zoom-factor', type=float, default=1.1,
                            help='Zoom factor for camera movements')
        parser.add_argument('--shake-intensity', type=float, default=2.0,
                            help='Intensity of camera shake effect')
        parser.add_argument('--punchin-factor', type=float, default=1.08,
                            help='Factor for punch-in effect')
        parser.add_argument('--camera-frame-rate', type=int, default=30,
                            help='Frame rate for camera movement effects')
        
        # Transition options
        parser.add_argument('--transition-type', default='cross_fade',
                            help='Type of transition to apply')
        parser.add_argument('--transition-duration', type=int, default=20,
                            help='Duration of transitions in frames')
        parser.add_argument('--randomize-transitions', action='store_true',
                            help='Randomize transition types')
        
        # Captions options
        parser.add_argument('--captions-api-url', 
                            default='https://ai-content-pipeline.onrender.com/caption-video',
                            help='API URL for captions service')
        
        return parser.parse_args()

    if __name__ == "__main__":
        args = parse_args()
        
        try:
            # Check if input video exists
            if not os.path.exists(args.input):
                logger.error(f"Input video not found: {args.input}")
                sys.exit(1)
            
            # Check if B-roll data exists
            if not os.path.exists(args.broll_data):
                logger.error(f"B-roll data not found: {args.broll_data}")
                sys.exit(1)
            
            # Load configuration from file
            config = PostProcessingOrchestrator.load_config_from_file(args.config)
            
            # Override configuration with command line arguments
            if not "post_processing" in config:
                config["post_processing"] = {}
            
            # Configure steps
            if args.steps:
                config["post_processing"]["steps"] = args.steps
            
            # Configure sound effects
            if not "sound_effects" in config:
                config["sound_effects"] = {}
            config["sound_effects"]["volume"] = args.sound_effects_volume
            
            # Configure background music
            if not "background_music" in config:
                config["background_music"] = {}
            config["background_music"]["volume"] = args.music_volume
            config["background_music"]["fade_in"] = args.music_fade_in
            config["background_music"]["fade_out"] = args.music_fade_out
            config["background_music"]["smart_ducking"] = not args.no_smart_ducking
            config["background_music"]["music_dir"] = args.music_dir
            
            # Configure camera movements
            if not "camera_movements" in config:
                config["camera_movements"] = {}
            config["camera_movements"]["zoom_factor"] = args.zoom_factor
            config["camera_movements"]["shake_intensity"] = args.shake_intensity
            config["camera_movements"]["punchin_factor"] = args.punchin_factor
            config["camera_movements"]["frame_rate"] = args.camera_frame_rate
            
            # Configure transitions
            if not "transitions" in config:
                config["transitions"] = {}
            config["transitions"]["type"] = args.transition_type
            config["transitions"]["duration"] = args.transition_duration
            config["transitions"]["randomize"] = args.randomize_transitions
            
            # Configure captions
            if not "captioning" in config:
                config["captioning"] = {}
            config["captioning"]["api_url"] = args.captions_api_url
            
            # Initialize orchestrator
            orchestrator = PostProcessingOrchestrator(
                config=config,
                cache_dir=args.cache_dir,
                sound_effects_dir=args.sound_effects_dir,
                music_dir=args.music_dir
            )
            
            # Run post-processing
            output_path = orchestrator.process(
                input_video_path=args.input,
                broll_data_path=args.broll_data,
                output_video_path=args.output,
                steps=args.steps,
                llm_api_key=args.llm_api_key,
                llm_api_url=args.llm_api_url,
                clean_intermediates=not args.keep_intermediates
            )
            
            print(f"Post-processing complete. Output saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)