# heygen_client.py
import os
import json
import logging
import argparse
import requests
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from urllib.parse import urlparse
import shutil
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("heygen_client")

class HeyGenClient:
    """
    Client for interacting with the HeyGen API to generate AI avatar videos.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: str = "https://api.heygen.com",
        polling_interval: int = 30,  # Changed from 10 to 30 seconds
        max_retries: int = 30,
        output_dir: str = "./videos"
    ):
        """
        Initialize the HeyGen API client.
        
        Args:
            api_key: HeyGen API key (if None, will use environment variable)
            api_base_url: Base URL for the HeyGen API
            polling_interval: Interval in seconds for polling video status
            max_retries: Maximum number of retry attempts when polling
            output_dir: Directory to save downloaded videos
        """
        self.api_key = api_key or os.environ.get("HEYGEN_API_KEY")
        if not self.api_key:
            raise ValueError("HeyGen API key is required. Set it via constructor or HEYGEN_API_KEY environment variable.")
        
        self.api_base_url = api_base_url
        self.polling_interval = polling_interval
        self.max_retries = max_retries
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_video(
        self,
        script: str,
        avatar_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        talking_photo_id: Optional[str] = None,
        background_url: Optional[str] = None,
        background_color: str = "#f6f6fc",
        dimension: Dict[str, int] = {"width": 1080, "height": 1920},
        caption: bool = False,
        voice_emotion: Optional[str] = None,
        voice_speed: float = 1.0,
        avatar_scale: float = 1.0,
        avatar_offset_x: float = 0.0,
        avatar_offset_y: float = 0.0,
        avatar_style: str = "normal",
        landscape_avatar: bool = False,
        elevenlabs_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a video using the HeyGen API.
        
        Args:
            script: Script text for the video
            avatar_id: ID of the avatar to use (for Avatar mode)
            voice_id: ID of the voice to use
            talking_photo_id: ID of the talking photo to use (for Talking Photo mode)
            background_url: URL of the background image/video
            background_color: Background color for solid color backgrounds
            dimension: Video dimensions (width, height)
            caption: Whether to add captions to the video
            voice_emotion: Voice emotion (Excited, Friendly, Serious, Soothing, Broadcaster)
            voice_speed: Voice speed (0.5-1.5)
            avatar_scale: Scale factor for the avatar (1.0 is default, 2.0 is twice as large)
            avatar_offset_x: Horizontal offset for the avatar (-1.0 to 1.0)
            avatar_offset_y: Vertical offset for the avatar (-1.0 to 1.0)
            avatar_style: Avatar rendering style (normal, closeUp, circle)
            landscape_avatar: Whether the avatar is in landscape format (16:9)
            elevenlabs_settings: Settings for ElevenLabs voice integration
            
        Returns:
            Dictionary containing the response from the API
        """
        logger.info("Creating video with HeyGen API")
        
        # Validate inputs
        if not avatar_id and not talking_photo_id:
            raise ValueError("Either avatar_id or talking_photo_id must be provided")
        
        if avatar_id and talking_photo_id:
            logger.warning("Both avatar_id and talking_photo_id provided. Will use talking_photo_id.")
            avatar_id = None
        
        if not voice_id:
            logger.warning("No voice_id provided. HeyGen will use a default voice.")
        
        # Adjust scale and offset based on landscape_avatar flag
        if landscape_avatar:
            logger.info("Adjusting settings for landscape avatar in portrait video")
            # Override the provided settings with optimized values for landscape avatars
            if avatar_scale < 1.7:  # Only override if not already set to a higher value
                avatar_scale = 1.8  # Larger scale to maximize avatar size
                logger.info(f"Setting avatar_scale to {avatar_scale} for landscape avatar")
            
            # Adjust vertical position if not specified
            if avatar_offset_y == 0.0:
                # Position slightly higher in the frame for better composition
                # Values range from -1.0 (bottom) to 1.0 (top)
                avatar_offset_y = 0.2
                logger.info(f"Setting avatar_offset_y to {avatar_offset_y} for landscape avatar")
        
        # Prepare the API request
        url = f"{self.api_base_url}/v2/video/generate"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        # Build the character settings based on what's provided
        character = {}
        if talking_photo_id:
            character = {
                "type": "talking_photo",
                "talking_photo_id": talking_photo_id,
                "talking_style": "expressive",
                "scale": avatar_scale,
                "offset": {"x": avatar_offset_x, "y": avatar_offset_y}
            }
        elif avatar_id:
            character = {
                "type": "avatar",
                "avatar_id": avatar_id,
                "scale": avatar_scale,
                "offset": {"x": avatar_offset_x, "y": avatar_offset_y},
                "avatar_style": avatar_style
            }
        
        # Build the voice settings
        voice_settings = {
            "type": "text",
            "input_text": script,
            "voice_id": voice_id,
            "speed": voice_speed
        }
        
        # Add optional voice parameters
        if voice_emotion:
            voice_settings["emotion"] = voice_emotion
        
        if elevenlabs_settings:
            voice_settings["elevenlabs_settings"] = elevenlabs_settings
        
        # Build the background settings
        background = {}
        if background_url:
            # Determine if it's a video or image based on the URL
            extension = Path(urlparse(background_url).path).suffix.lower()
            if extension in ['.mp4', '.mov', '.avi', '.webm']:
                background = {
                    "type": "video",
                    "url": background_url,
                    "fit": "cover"
                }
            else:
                background = {
                    "type": "image",
                    "url": background_url,
                    "fit": "cover"
                }
        else:
            background = {
                "type": "color",
                "value": background_color
            }
        
        # Build the full request body
        data = {
            "caption": caption,
            "dimension": dimension,
            "video_inputs": [
                {
                    "character": character,
                    "voice": voice_settings,
                    "background": background
                }
            ]
        }
        
        try:
            # Make the API request
            logger.info(f"Sending create video request to {url}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Failed to create video: {response.status_code} - {response.text}")
                raise Exception(f"Failed to create video: {response.status_code} - {response.text}")
            
            # Parse the response
            result = response.json()
            
            if "data" not in result or "video_id" not in result.get("data", {}):
                logger.error(f"Invalid response format: {result}")
                raise Exception("Invalid response format")
            
            video_id = result["data"]["video_id"]
            logger.info(f"Video creation initiated with ID: {video_id}")
            
            return {
                "video_id": video_id,
                "request": data,
                "response": result
            }
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            raise
    
    def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """
        Get the status of a video.
        
        Args:
            video_id: ID of the video
            
        Returns:
            Dictionary containing the video status information
        """
        try:
            url = f"{self.api_base_url}/v1/video_status.get"
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key
            }
            
            params = {
                "video_id": video_id
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Failed to get video status: {response.status_code} - {response.text}")
                raise Exception(f"Failed to get video status: {response.status_code}")
            
            result = response.json()
            
            if "data" not in result:
                logger.error(f"Invalid response format: {result}")
                raise Exception("Invalid response format")
            
            status_data = result["data"]
            logger.debug(f"Video status: {status_data.get('status')}")
            
            return status_data
            
        except Exception as e:
            logger.error(f"Error getting video status: {e}")
            raise
    
    def wait_for_video_completion(self, video_id: str) -> Dict[str, Any]:
        """
        Poll the API until the video is completed.
        
        Args:
            video_id: ID of the video
            
        Returns:
            Dictionary containing the final video status information
        """
        logger.info(f"Waiting for video {video_id} to complete...")
        
        for attempt in range(self.max_retries):
            try:
                status_data = self.get_video_status(video_id)
                status = status_data.get("status")
                
                # Check if video is completed
                if status == "completed":
                    logger.info(f"Video {video_id} completed successfully!")
                    return status_data
                
                # Check if video failed
                if status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    logger.error(f"Video {video_id} failed: {error_msg}")
                    raise Exception(f"Video generation failed: {error_msg}")
                
                # Video is still processing
                logger.info(f"Video {video_id} is {status}. Checking again in {self.polling_interval} seconds... (Attempt {attempt+1}/{self.max_retries})")
                time.sleep(self.polling_interval)
                
            except Exception as e:
                if "Video generation failed" in str(e):
                    raise  # Re-raise the exception if it's a failure notification
                
                logger.warning(f"Error while polling video status: {e}")
                logger.info(f"Retrying in {self.polling_interval} seconds...")
                time.sleep(self.polling_interval)
        
        # If we get here, we've exceeded the maximum retries
        logger.error(f"Exceeded maximum retries ({self.max_retries}) waiting for video completion")
        raise Exception("Timeout waiting for video completion")
    
    def download_video(self, video_url: str, output_path: Optional[str] = None) -> str:
        """
        Download a video from the given URL.
        
        Args:
            video_url: URL of the video to download
            output_path: Path to save the video (if None, will generate one)
            
        Returns:
            Path to the downloaded video
        """
        if not output_path:
            # Generate a filename based on the current timestamp
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"heygen_video_{timestamp}.mp4")
        
        try:
            logger.info(f"Downloading video from {video_url}")
            
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Download the video
            response = requests.get(video_url, stream=True, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Failed to download video: {response.status_code}")
                raise Exception(f"Failed to download video: {response.status_code}")
            
            # Write the video to file
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            logger.info(f"Video downloaded successfully to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise
    
    def create_and_download_video(
        self,
        script: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create a video and download it once it's completed.
        
        Args:
            script: Script text for the video
            output_path: Path to save the downloaded video
            **kwargs: Additional arguments to pass to create_video()
            
        Returns:
            Path to the downloaded video
        """
        try:
            # Create the video
            create_result = self.create_video(script, **kwargs)
            video_id = create_result["video_id"]
            
            # Wait for the video to complete
            status_data = self.wait_for_video_completion(video_id)
            
            # Get the video URL
            video_url = status_data.get("video_url")
            if not video_url:
                logger.error("No video URL found in status data")
                raise Exception("No video URL found in status data")
            
            # Download the video
            return self.download_video(video_url, output_path)
            
        except Exception as e:
            logger.error(f"Error creating and downloading video: {e}")
            raise

    def list_avatars(self, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available avatars (this is a mock method since the actual API
        endpoint is not provided in the documentation).
        
        Returns:
            List of avatar information dictionaries
        """
        logger.info("This is a mock method. In a real implementation, it would call the HeyGen API to list avatars.")
        logger.info("Please check HeyGen API documentation for the correct endpoint to list avatars.")
        
        # This is a placeholder - in a real implementation, you would call the API
        mock_avatars = []
        
        if verbose:
            for avatar in mock_avatars:
                print(f"Avatar ID: {avatar.get('id')}")
                print(f"Name: {avatar.get('name')}")
                print(f"Type: {avatar.get('type')}")
                print("-" * 30)
        
        return mock_avatars


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a video using HeyGen API')
    parser.add_argument('--script', required=True, help='Path to the script text file')
    parser.add_argument('--output', required=True, help='Path to save the downloaded video')
    parser.add_argument('--api-key', help='HeyGen API key (or set HEYGEN_API_KEY env var)')
    
    # Avatar and Voice options
    avatar_group = parser.add_mutually_exclusive_group(required=True)
    avatar_group.add_argument('--avatar-id', help='ID of the avatar to use')
    avatar_group.add_argument('--talking-photo-id', help='ID of the talking photo to use')
    
    parser.add_argument('--voice-id', help='ID of the voice to use')
    parser.add_argument('--voice-emotion', choices=['Excited', 'Friendly', 'Serious', 'Soothing', 'Broadcaster'],
                        help='Emotion to apply to the voice')
    parser.add_argument('--voice-speed', type=float, default=1.0, help='Speed of the voice (0.5-1.5)')
    
    # Avatar positioning and styling options
    parser.add_argument('--avatar-scale', type=float, default=1.0, 
                      help='Scale factor for the avatar (default: 1.0)')
    parser.add_argument('--avatar-offset-x', type=float, default=0.0, 
                      help='Horizontal offset for the avatar (-1.0 to 1.0)')
    parser.add_argument('--avatar-offset-y', type=float, default=0.0, 
                      help='Vertical offset for the avatar (-1.0 to 1.0)')
    parser.add_argument('--avatar-style', default='normal', 
                      choices=['normal', 'closeUp', 'circle'],
                      help='Avatar rendering style')
    parser.add_argument('--landscape-avatar', action='store_true',
                      help='Optimize settings for a landscape avatar in portrait video')
    
    # Video options
    parser.add_argument('--caption', action='store_true', help='Add captions to the video')
    parser.add_argument('--background-url', help='URL of the background image or video')
    parser.add_argument('--background-color', default='#f6f6fc', help='Background color in hex format')
    
    # ElevenLabs options
    parser.add_argument('--use-elevenlabs', action='store_true', help='Use ElevenLabs voice synthesis')
    parser.add_argument('--elevenlabs-model', default='eleven_turbo_v2', help='ElevenLabs model to use')
    parser.add_argument('--elevenlabs-stability', type=float, default=0.5, help='ElevenLabs stability (0.0-1.0)')
    parser.add_argument('--elevenlabs-similarity', type=float, default=0.75, help='ElevenLabs similarity boost (0.0-1.0)')
    parser.add_argument('--elevenlabs-style', type=float, default=0.0, help='ElevenLabs style (0.0-1.0)')
    
    # Processing options
    parser.add_argument('--polling-interval', type=int, default=30, 
                      help='Interval in seconds for polling video status (default: 30)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Load the script
        with open(args.script, 'r') as f:
            script_text = f.read().strip()
        
        if not script_text:
            logger.error("No script text found in the provided file")
            sys.exit(1)
        
        # Initialize the HeyGen client with custom polling interval
        client = HeyGenClient(
            api_key=args.api_key,
            polling_interval=args.polling_interval
        )
        
        # Prepare elevenlabs settings if needed
        elevenlabs_settings = None
        if args.use_elevenlabs:
            elevenlabs_settings = {
                "model": args.elevenlabs_model,
                "stability": args.elevenlabs_stability,
                "similarity_boost": args.elevenlabs_similarity,
                "style": args.elevenlabs_style
            }
        
        # Create and download the video
        output_path = client.create_and_download_video(
            script=script_text,
            output_path=args.output,
            avatar_id=args.avatar_id,
            voice_id=args.voice_id,
            talking_photo_id=args.talking_photo_id,
            background_url=args.background_url,
            background_color=args.background_color,
            dimension={"width": 1080, "height": 1920},  # 9:16 vertical format
            caption=args.caption,
            voice_emotion=args.voice_emotion,
            voice_speed=args.voice_speed,
            avatar_scale=args.avatar_scale,
            avatar_offset_x=args.avatar_offset_x,
            avatar_offset_y=args.avatar_offset_y,
            avatar_style=args.avatar_style,
            landscape_avatar=args.landscape_avatar,
            elevenlabs_settings=elevenlabs_settings
        )
        
        print(f"\n✅ Video generation successful!")
        print(f"Video saved to: {output_path}")
        print(f"\nRun the pipeline to enhance this video:")
        print(f"python pipeline_orchestrator.py --input {output_path} --output enhanced_video.mp4")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)