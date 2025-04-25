# post_processing_orchestrator.py
import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import post-processing components
from sound_effect_manager import SoundEffectManager
from background_music_manager import BackgroundMusicManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostProcessingOrchestrator:
    """
    Coordinates all post-processing steps after the main pipeline completes.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache_dir: str = "./cache/post_processing",
        sound_effects_dir: str = "./assets/sound_effects",
    ):
        """
        Initialize the post-processing orchestrator.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory to store cached files
            sound_effects_dir: Directory containing sound effect files
        """
        self.config = config
        self.cache_dir = cache_dir
        
        # Create cache directory if needed
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize components
        self.sound_effect_manager = SoundEffectManager(
            sound_effects_dir=sound_effects_dir,
            cache_dir=os.path.join(cache_dir, "sound_effects"),
            fade_duration=config.get("sound_effects", {}).get("fade_duration", 0.3),
            volume_adjustment=config.get("sound_effects", {}).get("volume", 0.7),
            max_sounds_per_video=config.get("sound_effects", {}).get("max_sounds_per_video", 5),
            min_gap_between_effects=config.get("sound_effects", {}).get("min_gap", 3.0)
        )
        
        self.background_music_manager = BackgroundMusicManager(
            music_dir=os.path.join(sound_effects_dir, "../background_music"),
            cache_dir=os.path.join(cache_dir, "background_music"),
            base_volume=config.get("background_music", {}).get("volume", 0.15),
            ducking_amount=config.get("background_music", {}).get("ducking", 0.5),
            smart_ducking=config.get("background_music", {}).get("smart_ducking", True)
        )
        
        # Future components will be initialized here:
        # self.transition_effect_manager = TransitionEffectManager(...)
        # self.zoom_effect_manager = ZoomEffectManager(...)
        # self.captioning_service = CaptioningService(...)
        
        self.logger = logging.getLogger(__name__)
    
    def process(
        self,
        input_video_path: str,
        broll_data_path: str,
        output_video_path: Optional[str] = None,
        steps: Optional[List[str]] = None
    ) -> str:
        """
        Run post-processing on the input video.
        
        Args:
            input_video_path: Path to the input video
            broll_data_path: Path to the B-roll cuts JSON file
            output_video_path: Path for the final output video (if None, will use input path with "_post" suffix)
            steps: List of post-processing steps to run (if None, run all steps)
            
        Returns:
            Path to the processed output video
        """
        # Determine output path
        if output_video_path is None:
            input_path = Path(input_video_path)
            output_video_path = str(input_path.with_stem(f"{input_path.stem}_post"))
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        
        # Track current video path through the processing steps
        current_video_path = input_video_path
        
        # Determine steps to run
        all_steps = ["sound_effects", "video_transitions", "zoom_effects", "captioning"]
        steps_to_run = steps or all_steps
        
        # Log the beginning of post-processing
        self.logger.info(f"Starting post-processing on {input_video_path}")
        self.logger.info(f"Steps to run: {', '.join(steps_to_run)}")
        
        try:
            # Apply sound effects
            if "sound_effects" in steps_to_run:
                self.logger.info("Applying sound effects")
                
                # Create intermediate output path
                sound_effects_output = os.path.join(
                    self.cache_dir, 
                    f"{Path(input_video_path).stem}_sound_effects.mp4"
                )
                
                # Apply sound effects
                sound_effects_output = self.sound_effect_manager.apply_sound_effects(
                    video_path=current_video_path,
                    broll_data_path=broll_data_path,
                    output_path=sound_effects_output
                )
                
                # Update current video path
                current_video_path = sound_effects_output
                self.logger.info(f"Sound effects applied, output: {sound_effects_output}")
                
            # Apply background music
            if "background_music" in steps_to_run:
                self.logger.info("Applying background music")
                
                # Create intermediate output path
                background_music_output = os.path.join(
                    self.cache_dir, 
                    f"{Path(input_video_path).stem}_music.mp4"
                )
                
                # Apply background music
                background_music_output = self.background_music_manager.apply_background_music(
                    video_path=current_video_path,
                    broll_data_path=broll_data_path,
                    output_path=background_music_output
                )
                
                # Update current video path
                current_video_path = background_music_output
                self.logger.info(f"Background music applied, output: {background_music_output}")
            
            # Future steps will be added here as they are implemented:
            
            # Apply video transitions
            if "video_transitions" in steps_to_run:
                self.logger.info("Video transitions not yet implemented")
                # Placeholder for future implementation
                # video_transitions_output = self.transition_effect_manager.apply_transitions(...)
                # current_video_path = video_transitions_output
            
            # Apply zoom effects
            if "zoom_effects" in steps_to_run:
                self.logger.info("Zoom effects not yet implemented")
                # Placeholder for future implementation
                # zoom_effects_output = self.zoom_effect_manager.apply_zoom_effects(...)
                # current_video_path = zoom_effects_output
            
            # Apply captioning
            if "captioning" in steps_to_run:
                self.logger.info("Captioning not yet implemented")
                # Placeholder for future implementation
                # captioning_output = self.captioning_service.apply_captions(...)
                # current_video_path = captioning_output
            
            # Copy or rename the final result to the output path
            if current_video_path != output_video_path:
                import shutil
                self.logger.info(f"Copying final output to {output_video_path}")
                shutil.copy2(current_video_path, output_video_path)
            
            return output_video_path
            
        except Exception as e:
            self.logger.error(f"Error during post-processing: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return original video path if processing failed
            return input_video_path
    
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
                import yaml
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
    parser.add_argument('--steps', nargs='+', 
                        choices=['sound_effects', 'video_transitions', 'zoom_effects', 'captioning'],
                        help='Specific post-processing steps to run')
    parser.add_argument('--steps', nargs='+', 
                   choices=['sound_effects', 'background_music', 'video_transitions', 'zoom_effects', 'captioning'],
                   help='Specific post-processing steps to run')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Load configuration from file
        config = PostProcessingOrchestrator.load_config_from_file(args.config) if os.path.exists(args.config) else {}
        
        # Initialize orchestrator
        orchestrator = PostProcessingOrchestrator(
            config=config,
            cache_dir=args.cache_dir,
            sound_effects_dir=args.sound_effects_dir
        )
        
        # Run post-processing
        output_path = orchestrator.process(
            input_video_path=args.input,
            broll_data_path=args.broll_data,
            output_video_path=args.output,
            steps=args.steps
        )
        
        print(f"Post-processing complete. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)