# social_media_publisher.py
import os
import sys
import json
import logging
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
import facebook  # Meta/Instagram Graph API SDK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("social_media_publisher")

@dataclass
class PublishingMetadata:
    """Metadata for publishing videos to social media platforms."""
    title: str
    description: str
    hashtags: List[str] = None
    category: str = None
    language: str = "en"
    is_draft: bool = False
    visibility: str = "public"  # public, private, unlisted
    scheduled_time: Optional[int] = None  # Unix timestamp for scheduled posting
    custom_thumbnail: Optional[str] = None  # Path to thumbnail image
    location: Optional[Dict[str, Any]] = None  # Location data if applicable
    
    def get_combined_description(self, include_hashtags: bool = True) -> str:
        """Get the full description including hashtags if requested."""
        full_description = self.description
        
        if include_hashtags and self.hashtags:
            hashtag_text = " ".join([f"#{tag}" for tag in self.hashtags])
            full_description = f"{full_description}\n\n{hashtag_text}"
            
        return full_description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        return {
            "title": self.title,
            "description": self.description,
            "hashtags": self.hashtags,
            "category": self.category,
            "language": self.language,
            "is_draft": self.is_draft,
            "visibility": self.visibility,
            "scheduled_time": self.scheduled_time,
            "custom_thumbnail": self.custom_thumbnail,
            "location": self.location
        }


class SocialMediaPlatform(ABC):
    """Base abstract class for social media platform integrations."""
    
    def __init__(self, auth_config: Dict[str, Any], cache_dir: str = "./cache/publisher"):
        """
        Initialize the platform with authentication configuration.
        
        Args:
            auth_config: Authentication configuration for the platform
            cache_dir: Directory to cache authentication tokens and other data
        """
        self.auth_config = auth_config
        self.cache_dir = cache_dir
        self.platform_name = self.__class__.__name__
        self.is_authenticated = False
        
        # Create cache directory if needed
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(f"social_media_publisher.{self.platform_name}")
    
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the platform.
        
        Returns:
            True if authentication successful, False otherwise
        """
        pass
    
    @abstractmethod
    def publish_video(self, video_path: str, metadata: PublishingMetadata) -> Dict[str, Any]:
        """
        Publish a video to the platform.
        
        Args:
            video_path: Path to the video file
            metadata: Publishing metadata
            
        Returns:
            Dictionary with publishing result information
        """
        pass
    
    @abstractmethod
    def check_video_requirements(self, video_path: str) -> Dict[str, Any]:
        """
        Check if a video meets the platform's requirements.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with validation results
        """
        pass
    
    def save_token_to_cache(self, token_data: Dict[str, Any]) -> None:
        """Save authentication token to cache."""
        try:
            token_path = os.path.join(self.cache_dir, f"{self.platform_name.lower()}_token.json")
            with open(token_path, 'w') as f:
                json.dump(token_data, f)
                
            self.logger.debug(f"Saved token data to {token_path}")
        except Exception as e:
            self.logger.error(f"Error saving token data: {e}")
    
    def load_token_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load authentication token from cache."""
        try:
            token_path = os.path.join(self.cache_dir, f"{self.platform_name.lower()}_token.json")
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    token_data = json.load(f)
                    
                self.logger.debug(f"Loaded token data from {token_path}")
                return token_data
            
            return None
        except Exception as e:
            self.logger.error(f"Error loading token data: {e}")
            return None


class YouTubePublisher(SocialMediaPlatform):
    """Publisher for YouTube Shorts."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/youtube'
    ]
    
    def __init__(self, auth_config: Dict[str, Any], cache_dir: str = "./cache/publisher"):
        """
        Initialize YouTube publisher.
        
        Auth config should contain:
        - client_secrets_file: Path to OAuth client secrets JSON file
        - token_file: Path to save/load OAuth token
        """
        super().__init__(auth_config, cache_dir)
        self.client_secrets_file = auth_config.get("client_secrets_file")
        self.youtube_service = None
    
    def authenticate(self) -> bool:
        """Authenticate with YouTube API."""
        try:
            credentials = None
            
            # Load token from cache if available
            token_data = self.load_token_from_cache()
            if token_data:
                credentials = google.oauth2.credentials.Credentials.from_authorized_user_info(token_data)
                
            # Check if credentials are valid or need refresh
            if credentials and credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                except RefreshError:
                    credentials = None
            
            # If no valid credentials, run the OAuth flow
            if not credentials or not credentials.valid:
                if not self.client_secrets_file or not os.path.exists(self.client_secrets_file):
                    self.logger.error("Client secrets file not found or not specified")
                    return False
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets_file, self.SCOPES)
                credentials = flow.run_local_server(port=0)
                
                # Save the credentials for future use
                token_data = {
                    'token': credentials.token,
                    'refresh_token': credentials.refresh_token,
                    'token_uri': credentials.token_uri,
                    'client_id': credentials.client_id,
                    'client_secret': credentials.client_secret,
                    'scopes': credentials.scopes
                }
                self.save_token_to_cache(token_data)
            
            # Build YouTube service
            self.youtube_service = build('youtube', 'v3', credentials=credentials)
            self.is_authenticated = True
            self.logger.info("Successfully authenticated with YouTube API")
            return True
            
        except Exception as e:
            self.logger.error(f"YouTube authentication error: {e}")
            return False
    
    def check_video_requirements(self, video_path: str) -> Dict[str, Any]:
        """Check if video meets YouTube Shorts requirements."""
        try:
            import ffmpeg
            
            # Probe the video file
            probe = ffmpeg.probe(video_path)
            
            # Get video stream information
            video_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'video'), None)
            
            # Get audio stream information
            audio_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                return {
                    "valid": False,
                    "message": "No video stream found"
                }
            
            # Extract video properties
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe.get('format', {}).get('duration', 0))
            file_size_bytes = int(probe.get('format', {}).get('size', 0))
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # YouTube Shorts requirements as of 2025
            requirements = {
                "aspect_ratio": "Portrait (9:16) recommended for Shorts",
                "optimal_resolution": "1080x1920 recommended",
                "max_duration": "60 seconds",
                "file_size_limit": "10GB",
                "supported_formats": "MP4, MOV",
                "has_audio": "Recommended"
            }
            
            # Validation checks
            is_portrait = height > width
            is_optimal_resolution = (width == 1080 and height == 1920)
            is_duration_valid = duration <= 60
            is_file_size_valid = file_size_mb <= 10 * 1024  # 10GB in MB
            has_audio = audio_stream is not None
            file_extension = os.path.splitext(video_path)[1].lower()
            is_format_valid = file_extension in ['.mp4', '.mov']
            
            # Collect issues
            issues = []
            if not is_portrait:
                issues.append("Video is not in portrait format (recommended for Shorts)")
            if not is_optimal_resolution:
                issues.append(f"Resolution is {width}x{height} (recommended 1080x1920)")
            if not is_duration_valid:
                issues.append(f"Duration is {duration:.1f}s (max 60s for Shorts)")
            if not is_file_size_valid:
                issues.append(f"File size is {file_size_mb:.1f}MB (max 10GB)")
            if not is_format_valid:
                issues.append(f"File format {file_extension} not recommended (use MP4, MOV)")
            if not has_audio:
                issues.append("Video has no audio (audio recommended for better engagement)")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "requirements": requirements,
                "video_info": {
                    "resolution": f"{width}x{height}",
                    "aspect_ratio": f"{width}:{height}",
                    "duration": f"{duration:.1f}s",
                    "file_size": f"{file_size_mb:.1f}MB",
                    "format": os.path.splitext(video_path)[1],
                    "has_audio": has_audio
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking video requirements: {e}")
            return {
                "valid": False,
                "message": f"Error analyzing video: {str(e)}"
            }
    
    def publish_video(self, video_path: str, metadata: PublishingMetadata) -> Dict[str, Any]:
        """Publish a video to YouTube as a Short."""
        if not self.is_authenticated:
            if not self.authenticate():
                return {
                    "success": False,
                    "message": "Authentication failed"
                }
        
        try:
            # Check if video exists
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "message": f"Video file not found: {video_path}"
                }
            
            # Check video requirements
            requirements_check = self.check_video_requirements(video_path)
            if not requirements_check["valid"]:
                return {
                    "success": False,
                    "message": "Video does not meet YouTube Shorts requirements",
                    "details": requirements_check
                }
            
            # Prepare video metadata
            youtube_metadata = {
                "snippet": {
                    "title": metadata.title,
                    "description": metadata.get_combined_description(),
                    "tags": metadata.hashtags or [],
                    "categoryId": metadata.category or "22",  # 22 is "People & Blogs"
                    "defaultLanguage": metadata.language
                },
                "status": {
                    "privacyStatus": metadata.visibility,
                    "selfDeclaredMadeForKids": False
                }
            }
            
            # Add scheduling if specified
            if metadata.scheduled_time:
                publish_time = time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z", 
                    time.gmtime(metadata.scheduled_time)
                )
                youtube_metadata["status"]["publishAt"] = publish_time
            
            # Add custom thumbnail if provided
            if metadata.custom_thumbnail and os.path.exists(metadata.custom_thumbnail):
                has_thumbnail = True
            else:
                has_thumbnail = False
            
            # Create upload request
            media = MediaFileUpload(
                video_path, 
                mimetype='video/mp4', 
                resumable=True
            )
            
            # Execute the request (insert video)
            self.logger.info(f"Uploading video to YouTube: {metadata.title}")
            request = self.youtube_service.videos().insert(
                part=",".join(youtube_metadata.keys()),
                body=youtube_metadata,
                media_body=media,
                notifySubscribers=True
            )
            
            # Handle upload with progress tracking
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    self.logger.info(f"Upload in progress: {int(status.progress() * 100)}%")
            
            video_id = response['id']
            self.logger.info(f"Video uploaded successfully! Video ID: {video_id}")
            
            # Upload custom thumbnail if available
            if has_thumbnail:
                self._upload_thumbnail(video_id, metadata.custom_thumbnail)
            
            # Mark as Short using the hashtag approach (as of 2025, this may have evolved)
            # YouTube currently identifies Shorts based on aspect ratio, duration and #Shorts hashtag
            if "#shorts" not in [h.lower() for h in (metadata.hashtags or [])]:
                # Add #Shorts hashtag if not already present
                updated_description = metadata.get_combined_description()
                if "#Shorts" not in updated_description:
                    updated_description += "\n#Shorts"
                
                # Update video with #Shorts hashtag
                self.youtube_service.videos().update(
                    part="snippet",
                    body={
                        "id": video_id,
                        "snippet": {
                            "title": metadata.title,
                            "description": updated_description,
                            "categoryId": metadata.category or "22"
                        }
                    }
                ).execute()
                
                self.logger.info("Added #Shorts hashtag to ensure short is properly categorized")
            
            return {
                "success": True,
                "platform": "YouTube",
                "video_id": video_id,
                "url": f"https://www.youtube.com/shorts/{video_id}",
                "message": "Video published successfully as a YouTube Short"
            }
            
        except Exception as e:
            self.logger.error(f"Error publishing to YouTube: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error publishing to YouTube: {str(e)}"
            }
    
    def _upload_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Upload a custom thumbnail for a video."""
        try:
            media = MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
            self.youtube_service.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            
            self.logger.info(f"Custom thumbnail uploaded for video {video_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading thumbnail: {e}")
            return False


class TikTokPublisher(SocialMediaPlatform):
    """Publisher for TikTok videos."""
    
    def __init__(self, auth_config: Dict[str, Any], cache_dir: str = "./cache/publisher"):
        """
        Initialize TikTok publisher.
        
        Auth config should contain:
        - client_key: TikTok API client key
        - client_secret: TikTok API client secret
        - access_token: Direct access token if available
        - creator_id: The creator's ID if using the Creator API
        """
        super().__init__(auth_config, cache_dir)
        self.base_url = "https://open.tiktokapis.com/v2"
        self.client_key = auth_config.get("client_key")
        self.client_secret = auth_config.get("client_secret")
        self.access_token = auth_config.get("access_token")
        self.creator_id = auth_config.get("creator_id")
    
    def authenticate(self) -> bool:
        """Authenticate with TikTok API."""
        try:
            # If we already have an access token, check if it's valid
            if self.access_token:
                # Verify token validity by making a test request
                headers = {
                    "Authorization": f"Bearer {self.access_token}"
                }
                response = requests.get(
                    f"{self.base_url}/user/info/",
                    headers=headers
                )
                
                if response.status_code == 200:
                    self.is_authenticated = True
                    self.logger.info("Successfully authenticated with existing token")
                    return True
            
            # If no valid token, try to get a new one using the client credentials flow
            if self.client_key and self.client_secret:
                token_url = f"{self.base_url}/oauth/token/"
                
                response = requests.post(
                    token_url,
                    data={
                        "client_key": self.client_key,
                        "client_secret": self.client_secret,
                        "grant_type": "client_credentials",
                        "scope": "user.info.basic,video.publish"
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data.get("access_token")
                    
                    # Save token to cache
                    self.save_token_to_cache({
                        "access_token": self.access_token,
                        "expires_in": token_data.get("expires_in"),
                        "token_type": token_data.get("token_type"),
                        "scope": token_data.get("scope"),
                        "refresh_token": token_data.get("refresh_token", ""),
                        "timestamp": time.time()
                    })
                    
                    self.is_authenticated = True
                    self.logger.info("Successfully obtained new TikTok access token")
                    return True
                else:
                    self.logger.error(f"Failed to get access token: {response.text}")
            
            self.logger.error("Authentication failed: Missing credentials")
            return False
            
        except Exception as e:
            self.logger.error(f"TikTok authentication error: {e}")
            return False
    
    def check_video_requirements(self, video_path: str) -> Dict[str, Any]:
        """Check if video meets TikTok requirements."""
        try:
            import ffmpeg
            
            # Probe the video file
            probe = ffmpeg.probe(video_path)
            
            # Get video stream information
            video_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'video'), None)
            
            # Get audio stream information
            audio_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                return {
                    "valid": False,
                    "message": "No video stream found"
                }
            
            # Extract video properties
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe.get('format', {}).get('duration', 0))
            file_size_bytes = int(probe.get('format', {}).get('size', 0))
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # TikTok video requirements as of 2025
            requirements = {
                "aspect_ratio": "9:16 preferred (vertical)",
                "resolution": "1080x1920 recommended",
                "max_duration": "10 minutes",
                "file_size_limit": "500MB",
                "supported_formats": "MP4, MOV",
                "has_audio": "Strongly recommended"
            }
            
            # Validation checks
            is_portrait = height > width
            is_optimal_resolution = (width == 1080 and height == 1920)
            is_duration_valid = duration <= 600  # 10 minutes
            is_file_size_valid = file_size_mb <= 500  # 500MB
            has_audio = audio_stream is not None
            file_extension = os.path.splitext(video_path)[1].lower()
            is_format_valid = file_extension in ['.mp4', '.mov']
            
            # Collect issues
            issues = []
            if not is_portrait:
                issues.append("Video is not in portrait format (recommended for TikTok)")
            if not is_optimal_resolution:
                issues.append(f"Resolution is {width}x{height} (recommended 1080x1920)")
            if not is_duration_valid:
                issues.append(f"Duration is {duration:.1f}s (max 10 minutes)")
            if not is_file_size_valid:
                issues.append(f"File size is {file_size_mb:.1f}MB (max 500MB)")
            if not is_format_valid:
                issues.append(f"File format {file_extension} not supported (use MP4, MOV)")
            if not has_audio:
                issues.append("Video has no audio (audio strongly recommended for TikTok)")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "requirements": requirements,
                "video_info": {
                    "resolution": f"{width}x{height}",
                    "aspect_ratio": f"{width}:{height}",
                    "duration": f"{duration:.1f}s",
                    "file_size": f"{file_size_mb:.1f}MB",
                    "format": os.path.splitext(video_path)[1],
                    "has_audio": has_audio
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking video requirements: {e}")
            return {
                "valid": False,
                "message": f"Error analyzing video: {str(e)}"
            }
    
    def publish_video(self, video_path: str, metadata: PublishingMetadata) -> Dict[str, Any]:
        """Publish a video to TikTok."""
        if not self.is_authenticated:
            if not self.authenticate():
                return {
                    "success": False,
                    "message": "Authentication failed"
                }
        
        try:
            # Check if video exists
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "message": f"Video file not found: {video_path}"
                }
            
            # Check video requirements
            requirements_check = self.check_video_requirements(video_path)
            if not requirements_check["valid"]:
                return {
                    "success": False,
                    "message": "Video does not meet TikTok requirements",
                    "details": requirements_check
                }
            
            # TikTok now (as of 2025) uses a two-step process for video upload:
            # 1. Initiate upload and get upload URL
            # 2. Upload the video to the URL
            # 3. Finalize the upload with metadata
            
            # Step 1: Initiate upload
            init_url = f"{self.base_url}/video/upload/"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Get file size
            file_size = os.path.getsize(video_path)
            
            init_response = requests.post(
                init_url,
                headers=headers,
                json={
                    "video_size": file_size,
                    "video_source": "PNS",  # Post & Schedule
                    "creator_id": self.creator_id
                }
            )
            
            if init_response.status_code != 200:
                return {
                    "success": False,
                    "message": f"Failed to initiate upload: {init_response.text}"
                }
                
            init_data = init_response.json()
            upload_url = init_data.get("data", {}).get("upload_url")
            video_id = init_data.get("data", {}).get("video_id")
            
            if not upload_url or not video_id:
                return {
                    "success": False,
                    "message": "Failed to get upload URL or video ID"
                }
            
            # Step 2: Upload the video
            self.logger.info(f"Uploading video to TikTok: {metadata.title}")
            
            with open(video_path, 'rb') as f:
                upload_response = requests.put(
                    upload_url,
                    data=f,
                    headers={
                        "Content-Type": "application/octet-stream"
                    }
                )
                
            if upload_response.status_code not in [200, 201, 204]:
                return {
                    "success": False,
                    "message": f"Failed to upload video: {upload_response.text}"
                }
            
            # Step 3: Finalize upload with metadata
            finalize_url = f"{self.base_url}/video/publish/"
            
            # Prepare hashtags
            hashtags = [{"name": tag} for tag in (metadata.hashtags or [])]
            
            finalize_data = {
                "video_id": video_id,
                "title": metadata.title,
                "description": metadata.description,
                "privacy_level": "PUBLIC" if metadata.visibility == "public" else "SELF_ONLY",
                "hashtags": hashtags,
                "disable_duet": False,
                "disable_stitch": False,
                "disable_comment": False,
                "creator_id": self.creator_id
            }
            
            # Add scheduled time if provided
            if metadata.scheduled_time:
                finalize_data["schedule_time"] = metadata.scheduled_time
            
            finalize_response = requests.post(
                finalize_url,
                headers=headers,
                json=finalize_data
            )
            
            if finalize_response.status_code != 200:
                return {
                    "success": False,
                    "message": f"Failed to publish video: {finalize_response.text}"
                }
                
            finalize_data = finalize_response.json()
            tiktok_post_id = finalize_data.get("data", {}).get("post_id")
            
            # Get the TikTok video URL
            tiktok_url = f"https://www.tiktok.com/@{self.creator_id}/video/{tiktok_post_id}"
            
            return {
                "success": True,
                "platform": "TikTok",
                "video_id": tiktok_post_id,
                "url": tiktok_url,
                "message": "Video published successfully to TikTok"
            }
            
        except Exception as e:
            self.logger.error(f"Error publishing to TikTok: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error publishing to TikTok: {str(e)}"
            }


class InstagramPublisher(SocialMediaPlatform):
    """Publisher for Instagram Reels."""
    
    def __init__(self, auth_config: Dict[str, Any], cache_dir: str = "./cache/publisher"):
        """
        Initialize Instagram publisher.
        
        Auth config should contain:
        - app_id: Meta/Facebook App ID
        - app_secret: Meta/Facebook App Secret
        - access_token: Instagram Graph API access token
        - instagram_account_id: Instagram Account ID for publishing
        """
        super().__init__(auth_config, cache_dir)
        self.app_id = auth_config.get("app_id")
        self.app_secret = auth_config.get("app_secret")
        self.access_token = auth_config.get("access_token")
        self.instagram_account_id = auth_config.get("instagram_account_id")
        self.api_version = "v20.0"  # Updated for 2025
        self.fb_graph_api = None
    
    def authenticate(self) -> bool:
        """Authenticate with Instagram Graph API."""
        try:
            # Try to load token from cache first
            token_data = self.load_token_from_cache()
            if token_data:
                self.access_token = token_data.get("access_token")
            
            # Check if we have a valid access token
            if self.access_token:
                # Verify token validity
                graph_url = f"https://graph.facebook.com/{self.api_version}/me"
                params = {
                    "access_token": self.access_token
                }
                
                response = requests.get(graph_url, params=params)
                
                if response.status_code == 200:
                    # Initialize the Facebook Graph API client
                    self.fb_graph_api = facebook.GraphAPI(self.access_token)
                    self.is_authenticated = True
                    self.logger.info("Successfully authenticated with Instagram API")
                    return True
            
            # If no valid token, we need app_id and app_secret
            if not all([self.app_id, self.app_secret]):
                self.logger.error("Missing credentials for Instagram authentication")
                return False
            
            # In a real application, you would need to implement OAuth workflow
            # This typically involves redirecting user to Meta login page
            # For this module, we assume the access token is provided via auth_config
            
            self.logger.error("No valid access token found. Please provide a valid token.")
            return False
            
        except Exception as e:
            self.logger.error(f"Instagram authentication error: {e}")
            return False
    
    def check_video_requirements(self, video_path: str) -> Dict[str, Any]:
        """Check if video meets Instagram Reels requirements."""
        try:
            import ffmpeg
            
            # Probe the video file
            probe = ffmpeg.probe(video_path)
            
            # Get video stream information
            video_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'video'), None)
            
            # Get audio stream information
            audio_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                return {
                    "valid": False,
                    "message": "No video stream found"
                }
            
            # Extract video properties
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe.get('format', {}).get('duration', 0))
            file_size_bytes = int(probe.get('format', {}).get('size', 0))
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Instagram Reels requirements as of 2025
            requirements = {
                "aspect_ratio": "9:16 preferred (vertical)",
                "resolution": "1080x1920 recommended",
                "max_duration": "90 seconds",
                "file_size_limit": "4GB",
                "supported_formats": "MP4, MOV",
                "has_audio": "Recommended"
            }
            
            # Validation checks
            is_portrait = height > width
            is_optimal_resolution = (width == 1080 and height == 1920)
            is_duration_valid = duration <= 90  # 90 seconds
            is_file_size_valid = file_size_mb <= 4 * 1024  # 4GB
            has_audio = audio_stream is not None
            file_extension = os.path.splitext(video_path)[1].lower()
            is_format_valid = file_extension in ['.mp4', '.mov']
            
            # Collect issues
            issues = []
            if not is_portrait:
                issues.append("Video is not in portrait format (recommended for Reels)")
            if not is_optimal_resolution:
                issues.append(f"Resolution is {width}x{height} (recommended 1080x1920)")
            if not is_duration_valid:
                issues.append(f"Duration is {duration:.1f}s (max 90s for Reels)")
            if not is_file_size_valid:
                issues.append(f"File size is {file_size_mb:.1f}MB (max 4GB)")
            if not is_format_valid:
                issues.append(f"File format {file_extension} not supported (use MP4, MOV)")
            if not has_audio:
                issues.append("Video has no audio (audio recommended for better engagement)")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "requirements": requirements,
                "video_info": {
                    "resolution": f"{width}x{height}",
                    "aspect_ratio": f"{width}:{height}",
                    "duration": f"{duration:.1f}s",
                    "file_size": f"{file_size_mb:.1f}MB",
                    "format": os.path.splitext(video_path)[1],
                    "has_audio": has_audio
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking video requirements: {e}")
            return {
                "valid": False,
                "message": f"Error analyzing video: {str(e)}"
            }
    
    def publish_video(self, video_path: str, metadata: PublishingMetadata) -> Dict[str, Any]:
        """Publish a video to Instagram as a Reel."""
        if not self.is_authenticated:
            if not self.authenticate():
                return {
                    "success": False,
                    "message": "Authentication failed"
                }
        
        try:
            # Check if video exists
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "message": f"Video file not found: {video_path}"
                }
            
            # Check video requirements
            requirements_check = self.check_video_requirements(video_path)
            if not requirements_check["valid"]:
                return {
                    "success": False,
                    "message": "Video does not meet Instagram Reels requirements",
                    "details": requirements_check
                }
            
            # Instagram requires a multi-step process:
            # 1. Create a container for the media
            # 2. Upload the video
            # 3. Publish the reel with metadata
            
            if not self.instagram_account_id:
                return {
                    "success": False,
                    "message": "Instagram account ID not provided"
                }
            
            # Step 1: Create a container
            container_url = f"https://graph.facebook.com/{self.api_version}/{self.instagram_account_id}/media"
            
            # Format caption with hashtags
            caption = metadata.get_combined_description(include_hashtags=True)
            
            container_data = {
                "media_type": "REELS",
                "video_url": "",  # Will be replaced with uploaded video URL
                "caption": caption,
                "access_token": self.access_token
            }
            
            # If scheduling is enabled
            if metadata.scheduled_time:
                container_data["publishing_type"] = "SCHEDULE"
                container_data["publish_time"] = metadata.scheduled_time
            
            # Add location data if provided
            if metadata.location:
                location_id = metadata.location.get("id")
                if location_id:
                    container_data["location_id"] = location_id
            
            # Use Facebook SDK to upload the video
            with open(video_path, 'rb') as video_file:
                video_id = self.fb_graph_api.put_video(
                    video_file, 
                    title=metadata.title,
                    description=caption
                )
            
            if not video_id:
                return {
                    "success": False,
                    "message": "Failed to upload video to Instagram"
                }
            
            # Use the video ID to create the container
            container_data["video_url"] = f"https://graph.facebook.com/{self.api_version}/{video_id}"
            
            container_response = requests.post(container_url, data=container_data)
            
            if container_response.status_code != 200:
                return {
                    "success": False,
                    "message": f"Failed to create media container: {container_response.text}"
                }
            
            container_data = container_response.json()
            creation_id = container_data.get("id")
            
            if not creation_id:
                return {
                    "success": False,
                    "message": "Failed to get creation ID from container response"
                }
            
            # Step 2: Publish the container as a Reel
            publish_url = f"https://graph.facebook.com/{self.api_version}/{self.instagram_account_id}/media_publish"
            
            publish_data = {
                "creation_id": creation_id,
                "access_token": self.access_token
            }
            
            publish_response = requests.post(publish_url, data=publish_data)
            
            if publish_response.status_code != 200:
                return {
                    "success": False,
                    "message": f"Failed to publish Reel: {publish_response.text}"
                }
                
            publish_data = publish_response.json()
            instagram_post_id = publish_data.get("id")
            
            # Construct the URL to the Reel
            # Format will be https://www.instagram.com/reel/{media_id}/
            instagram_url = f"https://www.instagram.com/reel/{instagram_post_id}/"
            
            return {
                "success": True,
                "platform": "Instagram",
                "video_id": instagram_post_id,
                "url": instagram_url,
                "message": "Video published successfully as an Instagram Reel"
            }
            
        except Exception as e:
            self.logger.error(f"Error publishing to Instagram: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error publishing to Instagram: {str(e)}"
            }


class SocialMediaPublisher:
    """
    Main orchestrator for publishing videos to multiple social media platforms.
    """
    
    def __init__(self, config_path: str = None, cache_dir: str = "./cache/publisher"):
        """
        Initialize the social media publisher.
        
        Args:
            config_path: Path to configuration file for social media platforms
            cache_dir: Directory for caching auth tokens and other data
        """
        self.config = self._load_config(config_path)
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Dictionary of platform instances
        self.platforms = {}
        
        # Initialize logger
        self.logger = logging.getLogger("social_media_publisher")
        
        # Initialize enabled platforms
        self._initialize_platforms()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use default empty config.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not config_path or not os.path.exists(config_path):
            return {
                "platforms": {}
            }
            
        try:
            with open(config_path, 'r') as f:
                # Check file extension to determine format
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    return yaml.safe_load(f)
                else:
                    # Default to JSON
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                "platforms": {}
            }
    
    def _initialize_platforms(self):
        """Initialize enabled social media platforms from config."""
        platform_configs = self.config.get("platforms", {})
        
        # Initialize YouTube
        if "youtube" in platform_configs and platform_configs["youtube"].get("enabled", False):
            self.platforms["youtube"] = YouTubePublisher(
                platform_configs["youtube"],
                cache_dir=self.cache_dir
            )
            
        # Initialize TikTok
        if "tiktok" in platform_configs and platform_configs["tiktok"].get("enabled", False):
            self.platforms["tiktok"] = TikTokPublisher(
                platform_configs["tiktok"],
                cache_dir=self.cache_dir
            )
            
        # Initialize Instagram
        if "instagram" in platform_configs and platform_configs["instagram"].get("enabled", False):
            self.platforms["instagram"] = InstagramPublisher(
                platform_configs["instagram"],
                cache_dir=self.cache_dir
            )
        
        # Log enabled platforms
        enabled_platforms = list(self.platforms.keys())
        if enabled_platforms:
            self.logger.info(f"Enabled platforms: {', '.join(enabled_platforms)}")
        else:
            self.logger.warning("No social media platforms enabled")
    
    def add_platform(self, platform_name: str, auth_config: Dict[str, Any]) -> bool:
        """
        Add a new platform instance or update existing one.
        
        Args:
            platform_name: Name of the platform ('youtube', 'tiktok', 'instagram')
            auth_config: Authentication configuration for the platform
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create platform instance based on name
            if platform_name.lower() == "youtube":
                self.platforms["youtube"] = YouTubePublisher(
                    auth_config,
                    cache_dir=self.cache_dir
                )
                return True
            elif platform_name.lower() == "tiktok":
                self.platforms["tiktok"] = TikTokPublisher(
                    auth_config,
                    cache_dir=self.cache_dir
                )
                return True
            elif platform_name.lower() == "instagram":
                self.platforms["instagram"] = InstagramPublisher(
                    auth_config,
                    cache_dir=self.cache_dir
                )
                return True
            else:
                self.logger.error(f"Unsupported platform: {platform_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding platform {platform_name}: {e}")
            return False
    
    def authenticate_platform(self, platform_name: str) -> bool:
        """
        Authenticate with a specific platform.
        
        Args:
            platform_name: Name of the platform to authenticate
            
        Returns:
            True if authentication successful, False otherwise
        """
        if platform_name not in self.platforms:
            self.logger.error(f"Platform not enabled: {platform_name}")
            return False
            
        try:
            return self.platforms[platform_name].authenticate()
        except Exception as e:
            self.logger.error(f"Error authenticating with {platform_name}: {e}")
            return False
    
    def authenticate_all(self) -> Dict[str, bool]:
        """
        Authenticate with all enabled platforms.
        
        Returns:
            Dictionary of platform names mapped to authentication success status
        """
        results = {}
        
        for platform_name, platform in self.platforms.items():
            try:
                success = platform.authenticate()
                results[platform_name] = success
            except Exception as e:
                self.logger.error(f"Error authenticating with {platform_name}: {e}")
                results[platform_name] = False
                
        return results
    
    def check_video_requirements(self, video_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Check if video meets the requirements for all enabled platforms.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary mapping platform names to requirement check results
        """
        results = {}
        
        for platform_name, platform in self.platforms.items():
            try:
                results[platform_name] = platform.check_video_requirements(video_path)
            except Exception as e:
                self.logger.error(f"Error checking requirements for {platform_name}: {e}")
                results[platform_name] = {
                    "valid": False,
                    "message": f"Error: {str(e)}"
                }
                
        return results
    
    def publish_to_platform(
        self, 
        platform_name: str, 
        video_path: str, 
        metadata: Union[PublishingMetadata, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Publish a video to a specific platform.
        
        Args:
            platform_name: Name of the platform to publish to
            video_path: Path to the video file
            metadata: Publishing metadata (can be PublishingMetadata object or dict)
            
        Returns:
            Dictionary with publishing result information
        """
        if platform_name not in self.platforms:
            return {
                "success": False,
                "message": f"Platform not enabled: {platform_name}"
            }
            
        # Convert dict to PublishingMetadata if needed
        if isinstance(metadata, dict):
            metadata = PublishingMetadata(
                title=metadata.get("title", ""),
                description=metadata.get("description", ""),
                hashtags=metadata.get("hashtags"),
                category=metadata.get("category"),
                language=metadata.get("language", "en"),
                is_draft=metadata.get("is_draft", False),
                visibility=metadata.get("visibility", "public"),
                scheduled_time=metadata.get("scheduled_time"),
                custom_thumbnail=metadata.get("custom_thumbnail"),
                location=metadata.get("location")
            )
            
        try:
            platform = self.platforms[platform_name]
            
            # Authenticate if not already authenticated
            if not platform.is_authenticated:
                success = platform.authenticate()
                if not success:
                    return {
                        "success": False,
                        "message": f"Failed to authenticate with {platform_name}"
                    }
            
            # Publish the video
            result = platform.publish_video(video_path, metadata)
            return result
            
        except Exception as e:
            self.logger.error(f"Error publishing to {platform_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error publishing to {platform_name}: {str(e)}"
            }
    
    def publish_to_all(
        self, 
        video_path: str, 
        metadata: Union[PublishingMetadata, Dict[str, Any]],
        platforms: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Publish a video to multiple platforms.
        
        Args:
            video_path: Path to the video file
            metadata: Publishing metadata
            platforms: List of platform names to publish to (None for all enabled)
            
        Returns:
            Dictionary mapping platform names to publishing results
        """
        results = {}
        
        # Determine which platforms to publish to
        target_platforms = platforms or list(self.platforms.keys())
        
        for platform_name in target_platforms:
            if platform_name in self.platforms:
                results[platform_name] = self.publish_to_platform(
                    platform_name, video_path, metadata
                )
                
                # Log result
                if results[platform_name].get("success"):
                    self.logger.info(f"Successfully published to {platform_name}: {results[platform_name].get('url', '')}")
                else:
                    self.logger.error(f"Failed to publish to {platform_name}: {results[platform_name].get('message', '')}")
            else:
                results[platform_name] = {
                    "success": False,
                    "message": f"Platform not enabled: {platform_name}"
                }
                
        return results
    
    def save_config(self, config_path: str) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            # Update config with current platforms
            platforms_config = {}
            for platform_name, platform in self.platforms.items():
                # Don't save sensitive auth info like tokens and secrets
                # Just save enabled status and non-sensitive info
                platforms_config[platform_name] = {
                    "enabled": True
                }
            
            config = {
                "platforms": platforms_config
            }
            
            # Save based on file extension
            if config_path.endswith('.json'):
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif config_path.endswith(('.yaml', '.yml')):
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
            else:
                # Default to JSON
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
            self.logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False


def create_sample_config():
    """
    Create a sample configuration file for social media publishing.
    
    Returns:
        Sample configuration dictionary
    """
    return {
        "platforms": {
            "youtube": {
                "enabled": True,
                "client_secrets_file": "/path/to/client_secrets.json",
                "token_file": "/path/to/youtube_token.json"
            },
            "tiktok": {
                "enabled": True,
                "client_key": "YOUR_TIKTOK_CLIENT_KEY",
                "client_secret": "YOUR_TIKTOK_CLIENT_SECRET",
                "creator_id": "YOUR_TIKTOK_CREATOR_ID"
            },
            "instagram": {
                "enabled": True,
                "app_id": "YOUR_META_APP_ID",
                "app_secret": "YOUR_META_APP_SECRET",
                "instagram_account_id": "YOUR_INSTAGRAM_ACCOUNT_ID"
            }
        }
    }


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Publish videos to social media platforms')
    
    # Main command options
    parser.add_argument('--video', '-v', required=True, help='Path to the video file')
    parser.add_argument('--title', '-t', required=True, help='Title for the video')
    parser.add_argument('--description', '-d', help='Description for the video')
    parser.add_argument('--config', '-c', default='./social_media_config.json', help='Path to config file')
    
    # Platform options
    parser.add_argument('--platforms', '-p', nargs='+', choices=['youtube', 'tiktok', 'instagram'], 
                        help='Platforms to publish to (defaults to all enabled)')
    
    # Metadata options
    parser.add_argument('--hashtags', nargs='+', help='Hashtags for the video')
    parser.add_argument('--visibility', choices=['public', 'private', 'unlisted'], default='public', 
                        help='Video visibility')
    parser.add_argument('--thumbnail', help='Path to custom thumbnail image')
    parser.add_argument('--scheduled-time', type=int, help='Unix timestamp for scheduled posting')
    parser.add_argument('--location', help='Location JSON string for Instagram {"id": "123", "name": "Location Name"}')
    
    # Other options
    parser.add_argument('--check-only', action='store_true', help='Only check requirements, don\'t publish')
    parser.add_argument('--create-config', action='store_true', help='Create a sample config file')
    parser.add_argument('--config-path', default='./social_media_config.json', 
                        help='Path to create sample config file')
    
    return parser.parse_args()


def main():
    """Main entry point for CLI usage."""
    args = parse_args()
    
    # Create sample config if requested
    if args.create_config:
        config = create_sample_config()
        try:
            with open(args.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Sample configuration created at {args.config_path}")
            return 0
        except Exception as e:
            print(f"Error creating sample configuration: {e}")
            return 1
    
    # Initialize publisher
    publisher = SocialMediaPublisher(config_path=args.config)
    
    # If check-only mode, just check requirements and exit
    if args.check_only:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found at {args.video}")
            return 1
            
        results = publisher.check_video_requirements(args.video)
        
        print("\nVideo Requirements Check:")
        print("=" * 50)
        
        all_valid = True
        for platform, result in results.items():
            is_valid = result.get("valid", False)
            all_valid = all_valid and is_valid
            
            print(f"\n{platform.upper()}: {'✅ VALID' if is_valid else '❌ INVALID'}")
            
            if not is_valid and "issues" in result:
                print("Issues:")
                for issue in result["issues"]:
                    print(f"  - {issue}")
            
            if "video_info" in result:
                print("Video Info:")
                for key, value in result["video_info"].items():
                    print(f"  {key}: {value}")
                    
        return 0 if all_valid else 1
    
    # Create publishing metadata
    metadata = PublishingMetadata(
        title=args.title,
        description=args.description or "",
        hashtags=args.hashtags,
        visibility=args.visibility,
        scheduled_time=args.scheduled_time,
        custom_thumbnail=args.thumbnail
    )
    
    # Parse location JSON if provided
    if args.location:
        try:
            metadata.location = json.loads(args.location)
        except:
            print(f"Warning: Invalid location JSON: {args.location}")
    
    # Publish to requested platforms
    results = publisher.publish_to_all(args.video, metadata, args.platforms)
    
    # Display results
    print("\nPublishing Results:")
    print("=" * 50)
    
    all_success = True
    for platform, result in results.items():
        success = result.get("success", False)
        all_success = all_success and success
        
        print(f"\n{platform.upper()}: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Message: {result.get('message', '')}")
        
        if success and "url" in result:
            print(f"URL: {result['url']}")
            
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())