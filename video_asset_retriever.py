# video_asset_retriever.py
import os
import json
import logging
import hashlib
import aiohttp
import asyncio
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import quote_plus
import time
import random
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Store metadata about retrieved video assets."""
    id: str
    width: int
    height: int
    duration: float
    url: str
    preview_image: str
    download_link: str
    local_path: Optional[str] = None
    search_query: Optional[str] = None
    
    def meets_requirements(self, min_duration: float, min_resolution: Tuple[int, int]) -> bool:
        """Check if video meets minimum requirements for duration and resolution."""
        min_width, min_height = min_resolution
        
        # For landscape videos
        landscape_check = self.width >= min_width and self.height >= min_height
        
        # For portrait videos (we also accept these, just flipped)
        portrait_check = self.width >= min_height and self.height >= min_width
        
        # Check both resolution and duration requirements
        return (landscape_check or portrait_check) and self.duration >= min_duration


class VideoAssetRetriever:
    """
    Retrieves video assets from Pexels API based on keywords.
    
    Features:
    - Asynchronous API requests and downloads
    - Local caching to prevent redundant downloads
    - Intelligent query generation and fallback strategies
    - Video filtering based on duration and resolution requirements
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "./cache/videos",
        metadata_cache_path: str = "./cache/metadata",
        min_resolution: Tuple[int, int] = (1080, 1080),  # Minimum of Full HD
        max_results_per_query: int = 5,
        max_concurrent_requests: int = 3,
        max_download_retries: int = 3
    ):
        """
        Initialize the VideoAssetRetriever.
        
        Args:
            api_key: Pexels API key (if None, will use environment variable)
            cache_dir: Directory to store downloaded videos
            metadata_cache_path: Directory to store video metadata
            min_resolution: Minimum resolution (width, height) in pixels
            max_results_per_query: Maximum number of results to return per query
            max_concurrent_requests: Maximum number of concurrent API requests
            max_download_retries: Maximum number of retries for failed downloads
        """
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY")
        if not self.api_key:
            raise ValueError("Pexels API key is required. Set it via constructor or PEXELS_API_KEY environment variable.")
        
        self.cache_dir = cache_dir
        self.metadata_cache_path = metadata_cache_path
        self.min_resolution = min_resolution
        self.max_results_per_query = max_results_per_query
        self.max_concurrent_requests = max_concurrent_requests
        self.max_download_retries = max_download_retries
        
        # Create cache directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.metadata_cache_path, exist_ok=True)
        
        # Base URL for Pexels API
        self.api_base_url = "https://api.pexels.com/videos/search"
        
        # Load metadata cache
        self.metadata_cache = self._load_metadata_cache()
        
        # Dictionary to track rate limiting
        self.rate_limit_info = {
            "remaining": 200,  # Default quota, will be updated with API responses
            "limit": 200,
            "reset_time": None
        }
    
    async def process_broll_cuts(self, broll_cuts_data: Dict) -> Dict:
        """
        Process the B-roll cuts data from the Keyword Extractor.
        
        For each B-roll cut, find and download an appropriate video.
        
        Args:
            broll_cuts_data: Dictionary containing B-roll cuts data
            
        Returns:
            Updated B-roll cuts data with video paths
        """
        try:
            # Extract the B-roll cuts
            broll_cuts = broll_cuts_data.get("broll_cuts", [])
            
            if not broll_cuts:
                logger.warning("No B-roll cuts found in the provided data.")
                return broll_cuts_data
            
            # Process each B-roll cut asynchronously
            tasks = []
            async with aiohttp.ClientSession() as session:
                for cut in broll_cuts:
                    task = self._process_single_cut(session, cut)
                    tasks.append(task)
                
                # Wait for all tasks to complete
                processed_cuts = await asyncio.gather(*tasks)
            
            # Update the B-roll cuts data
            result = {
                "broll_cuts": processed_cuts,
                "metadata": broll_cuts_data.get("metadata", {})
            }
            
            # Save the updated metadata cache
            self._save_metadata_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing B-roll cuts: {e}")
            return broll_cuts_data
    
    async def _process_single_cut(self, session: aiohttp.ClientSession, cut: Dict) -> Dict:
        """
        Process a single B-roll cut by finding and downloading an appropriate video.
        
        Args:
            session: aiohttp client session
            cut: Dictionary containing B-roll cut data
            
        Returns:
            Updated B-roll cut data with video path
        """
        try:
            # Extract information from the cut
            timestamp = cut.get("timestamp", 0)
            duration = cut.get("duration", 0)
            primary_keywords = cut.get("keywords", [])
            alternative_keywords = cut.get("alternative_keywords", [])
            visual_concepts = cut.get("visual_concepts", [])
            
            logger.info(f"Processing B-roll cut at {timestamp}s with duration {duration}s")
            
            # First, try to find a suitable video using primary keywords
            video = await self._find_and_download_video(
                session, 
                primary_keywords, 
                duration
            )
            
            # If no suitable video found, try alternative keyword sets
            if not video and alternative_keywords:
                for alt_keywords in alternative_keywords:
                    video = await self._find_and_download_video(
                        session, 
                        alt_keywords, 
                        duration
                    )
                    if video:
                        break
            
            # If still no video, try visual concepts
            if not video and visual_concepts:
                video = await self._find_and_download_video(
                    session, 
                    visual_concepts, 
                    duration
                )
            
            # Update the cut with the video path
            if video and video.local_path:
                # Deep copy the cut to avoid modifying the original
                updated_cut = cut.copy()
                updated_cut["path"] = video.local_path
                logger.info(f"Found video for B-roll cut at {timestamp}s: {video.local_path}")
                return updated_cut
            else:
                logger.warning(f"Failed to find suitable video for B-roll cut at {timestamp}s")
                # Return the original cut, but with empty path to indicate failure
                updated_cut = cut.copy()
                updated_cut["path"] = ""
                return updated_cut
                
        except Exception as e:
            logger.error(f"Error processing B-roll cut at {cut.get('timestamp', 0)}s: {e}")
            # Return the original cut with empty path
            updated_cut = cut.copy()
            updated_cut["path"] = ""
            return updated_cut
    
    async def _find_and_download_video(
        self, 
        session: aiohttp.ClientSession,
        keywords: List[str], 
        min_duration: float
    ) -> Optional[VideoMetadata]:
        """
        Find and download a video that matches the given keywords and duration.
        
        Args:
            session: aiohttp client session
            keywords: List of keywords to search for
            min_duration: Minimum duration of the video in seconds
            
        Returns:
            VideoMetadata object if successful, None otherwise
        """
        # Construct a search query from the keywords
        query = self._construct_search_query(keywords)
        
        # Check if we already have suitable videos in cache
        cached_videos = self._find_cached_videos(query, min_duration)
        if cached_videos:
            # Return a random cached video to add variety
            return random.choice(cached_videos)
        
        # If not in cache, search for videos
        try:
            logger.info(f"Searching for videos with query: '{query}'")
            
            # Search for videos
            videos = await self._search_videos(session, query)
            
            # Filter videos based on duration and resolution
            suitable_videos = [
                v for v in videos 
                if v.meets_requirements(min_duration, self.min_resolution)
            ]
            
            if not suitable_videos:
                logger.warning(f"No suitable videos found for query: '{query}'")
                return None
            
            # Select a random video from the results (to add variety)
            selected_video = random.choice(suitable_videos)
            
            # Download the video
            downloaded_video = await self._download_video(session, selected_video)
            
            if downloaded_video:
                # Update cache with the downloaded video
                self._add_to_metadata_cache(downloaded_video)
                return downloaded_video
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error finding and downloading video: {e}")
            return None
    
    def _construct_search_query(self, keywords: List[str]) -> str:
        """
        Construct a search query from a list of keywords.
        
        Args:
            keywords: List of keywords
            
        Returns:
            Search query string
        """
        # Filter out empty or too short keywords
        filtered_keywords = [k for k in keywords if k and len(k) > 2]
        
        if not filtered_keywords:
            return "video"  # Fallback generic query
        
        # Limit to top 3 keywords to avoid overly specific queries
        top_keywords = filtered_keywords[:3]
        
        # Join keywords with spaces (Pexels handles this well)
        query = " ".join(top_keywords)
        
        return query
    
    def _find_cached_videos(self, query: str, min_duration: float) -> List[VideoMetadata]:
        """
        Find suitable videos in the cache.
        
        Args:
            query: Search query used to find videos
            min_duration: Minimum duration in seconds
            
        Returns:
            List of suitable VideoMetadata objects from cache
        """
        suitable_videos = []
        
        # Check query-specific cache first
        query_videos = [
            v for v in self.metadata_cache.values()
            if v.search_query == query and 
            v.meets_requirements(min_duration, self.min_resolution) and
            v.local_path and 
            os.path.exists(v.local_path)
        ]
        
        if query_videos:
            return query_videos
        
        # If no query-specific videos, check all videos in cache
        for video in self.metadata_cache.values():
            if (video.meets_requirements(min_duration, self.min_resolution) and
                video.local_path and 
                os.path.exists(video.local_path)):
                suitable_videos.append(video)
        
        return suitable_videos
    
    async def _search_videos(self, session: aiohttp.ClientSession, query: str) -> List[VideoMetadata]:
        """
        Search for videos using the Pexels API.
        
        Args:
            session: aiohttp client session
            query: Search query
            
        Returns:
            List of VideoMetadata objects
        """
        try:
            # Check rate limiting
            await self._respect_rate_limits()
            
            # Prepare request
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }
            
            # URL encode the query
            encoded_query = quote_plus(query)
            
            # Construct full URL with parameters
            url = f"{self.api_base_url}?query={encoded_query}&per_page={self.max_results_per_query}&size=medium"
            
            # Make the request
            logger.info(f"Making API request to: {url}")
            async with session.get(url, headers=headers) as response:
                # Update rate limiting information
                self._update_rate_limits(response)
                
                # Check if the request was successful
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API request failed with status {response.status}: {error_text}")
                    return []
                
                # Parse the response
                data = await response.json()
                
                # Extract video metadata
                videos = []
                for video_data in data.get("videos", []):
                    # Get the best quality video file
                    video_files = video_data.get("video_files", [])
                    
                    if not video_files:
                        continue
                    
                    # Find HD or the highest quality available video file
                    best_file = None
                    for vf in video_files:
                        # Skip non-mp4 files (hls, etc.)
                        if vf.get("file_type") != "video/mp4":
                            continue
                            
                        # Prefer HD quality
                        if vf.get("quality") == "hd":
                            best_file = vf
                            break
                            
                        # Otherwise, pick the file with the highest resolution
                        if (not best_file or 
                            (vf.get("width", 0) * vf.get("height", 0)) > 
                            (best_file.get("width", 0) * best_file.get("height", 0))):
                            best_file = vf
                    
                    if not best_file:
                        continue
                    
                    # Get the first preview image
                    preview_image = ""
                    if video_data.get("video_pictures") and len(video_data["video_pictures"]) > 0:
                        preview_image = video_data["video_pictures"][0].get("picture", "")
                    
                    # Create VideoMetadata object
                    video = VideoMetadata(
                        id=str(video_data.get("id", "")),
                        width=best_file.get("width", 0),
                        height=best_file.get("height", 0),
                        duration=video_data.get("duration", 0),
                        url=video_data.get("url", ""),
                        preview_image=preview_image,
                        download_link=best_file.get("link", ""),
                        search_query=query
                    )
                    
                    videos.append(video)
                
                logger.info(f"Found {len(videos)} videos for query: '{query}'")
                return videos
                
        except Exception as e:
            logger.error(f"Error searching for videos: {e}")
            return []
    
    async def _download_video(self, session: aiohttp.ClientSession, video: VideoMetadata) -> Optional[VideoMetadata]:
        """
        Download a video from its URL.
        
        Args:
            session: aiohttp client session
            video: VideoMetadata object containing download information
            
        Returns:
            Updated VideoMetadata object with local path, or None if download failed
        """
        try:
            # Skip if no download link
            if not video.download_link:
                logger.warning(f"No download link for video {video.id}")
                return None
            
            # Generate a unique filename based on video ID and query
            filename = f"{video.id}_{hashlib.md5(video.search_query.encode()).hexdigest()[:8]}.mp4"
            local_path = os.path.join(self.cache_dir, filename)
            
            # Check if file already exists
            if os.path.exists(local_path):
                logger.info(f"Video already downloaded to {local_path}")
                video.local_path = local_path
                return video
            
            # Download the file
            for attempt in range(self.max_download_retries):
                try:
                    logger.info(f"Downloading video to {local_path} (attempt {attempt+1}/{self.max_download_retries})")
                    
                    # Download using streaming to handle large files
                    async with session.get(video.download_link) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Download failed with status {response.status}: {error_text}")
                            continue
                        
                        # Create a temporary download path to avoid partial downloads
                        temp_path = f"{local_path}.download"
                        
                        # Use synchronous file write with executor to avoid IO blocking
                        content = await response.read()
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._write_file, temp_path, content
                        )
                        
                        # Rename to final path once download is complete
                        os.rename(temp_path, local_path)
                        
                        logger.info(f"Successfully downloaded video to {local_path}")
                        
                        # Update video metadata with local path
                        video.local_path = local_path
                        return video
                        
                except Exception as e:
                    logger.warning(f"Download attempt {attempt+1} failed: {e}")
                    # Wait before retrying
                    await asyncio.sleep(1 * (attempt + 1))
            
            logger.error(f"Failed to download video after {self.max_download_retries} attempts")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def _write_file(self, path: str, content: bytes) -> None:
        """Write content to file (synchronous operation for executor)."""
        with open(path, 'wb') as f:
            f.write(content)
    
    async def _respect_rate_limits(self) -> None:
        """
        Check and respect API rate limits.
        
        Will sleep if necessary to avoid hitting rate limits.
        """
        # Check if we're close to the rate limit
        if self.rate_limit_info["remaining"] <= 3:
            # If we know when the limit resets
            if self.rate_limit_info["reset_time"]:
                current_time = time.time()
                if current_time < self.rate_limit_info["reset_time"]:
                    # Calculate sleep time (add 1 second buffer)
                    sleep_time = self.rate_limit_info["reset_time"] - current_time + 1
                    logger.warning(f"Rate limit almost reached. Sleeping for {sleep_time:.1f}s until reset.")
                    await asyncio.sleep(sleep_time)
            else:
                # If we don't know the reset time, sleep for a conservative amount
                logger.warning(f"Rate limit almost reached. Sleeping for 60s as precaution.")
                await asyncio.sleep(60)
    
    def _update_rate_limits(self, response: aiohttp.ClientResponse) -> None:
        """
        Update rate limit information from API response headers.
        
        Args:
            response: API response object with headers
        """
        try:
            # Update rate limit information from headers if available
            if "X-Ratelimit-Remaining" in response.headers:
                self.rate_limit_info["remaining"] = int(response.headers["X-Ratelimit-Remaining"])
                
            if "X-Ratelimit-Limit" in response.headers:
                self.rate_limit_info["limit"] = int(response.headers["X-Ratelimit-Limit"])
                
            if "X-Ratelimit-Reset" in response.headers:
                # Reset time is typically in seconds since epoch
                self.rate_limit_info["reset_time"] = float(response.headers["X-Ratelimit-Reset"])
                
            logger.debug(f"Rate limits: {self.rate_limit_info['remaining']}/{self.rate_limit_info['limit']}")
            
        except Exception as e:
            logger.warning(f"Failed to update rate limit info: {e}")
    
    def _load_metadata_cache(self) -> Dict[str, VideoMetadata]:
        """
        Load metadata cache from disk.
        
        Returns:
            Dictionary of VideoMetadata objects keyed by video ID
        """
        cache_file = os.path.join(self.metadata_cache_path, "video_metadata.json")
        
        if not os.path.exists(cache_file):
            return {}
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            cache = {}
            for video_id, video_data in data.items():
                cache[video_id] = VideoMetadata(
                    id=video_data.get("id", ""),
                    width=video_data.get("width", 0),
                    height=video_data.get("height", 0),
                    duration=video_data.get("duration", 0),
                    url=video_data.get("url", ""),
                    preview_image=video_data.get("preview_image", ""),
                    download_link=video_data.get("download_link", ""),
                    local_path=video_data.get("local_path"),
                    search_query=video_data.get("search_query", "")
                )
                
            logger.info(f"Loaded {len(cache)} videos from metadata cache")
            return cache
            
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}. Starting with empty cache.")
            return {}
    
    def _save_metadata_cache(self) -> None:
        """Save metadata cache to disk."""
        cache_file = os.path.join(self.metadata_cache_path, "video_metadata.json")
        
        try:
            # Convert VideoMetadata objects to dictionaries
            data = {}
            for video_id, video in self.metadata_cache.items():
                data[video_id] = {
                    "id": video.id,
                    "width": video.width,
                    "height": video.height,
                    "duration": video.duration,
                    "url": video.url,
                    "preview_image": video.preview_image,
                    "download_link": video.download_link,
                    "local_path": video.local_path,
                    "search_query": video.search_query
                }
                
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(data)} videos to metadata cache")
            
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")
    
    def _add_to_metadata_cache(self, video: VideoMetadata) -> None:
        """
        Add a video to the metadata cache.
        
        Args:
            video: VideoMetadata object to add
        """
        self.metadata_cache[video.id] = video
    
    # Synchronous version of process_broll_cuts for direct usage
    def process_broll_cuts_sync(self, broll_cuts_data: Dict) -> Dict:
        """
        Synchronous wrapper for process_broll_cuts.
        
        Args:
            broll_cuts_data: Dictionary containing B-roll cuts data
            
        Returns:
            Updated B-roll cuts data with video paths
        """
        return asyncio.run(self.process_broll_cuts(broll_cuts_data))


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrieve video assets for B-roll based on keywords')
    parser.add_argument('--input', required=True, help='Path to the enhanced keywords JSON file')
    parser.add_argument('--output', required=True, help='Path to save the output JSON file with video paths')
    parser.add_argument('--api-key', help='Pexels API key (or set PEXELS_API_KEY env var)')
    parser.add_argument('--cache-dir', default='./cache/videos', help='Directory to store downloaded videos')
    parser.add_argument('--metadata-cache', default='./cache/metadata', help='Directory to store video metadata')
    parser.add_argument('--min-width', type=int, default=1080, help='Minimum video width')
    parser.add_argument('--min-height', type=int, default=1080, help='Minimum video height')
    parser.add_argument('--max-results', type=int, default=5, help='Maximum results per query')
    parser.add_argument('--force-refresh', action='store_true', help='Ignore cache and force new downloads')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Load input file
        logger.info(f"Loading enhanced keywords from {args.input}")
        with open(args.input, 'r') as f:
            broll_cuts_data = json.load(f)
            
        # Initialize the asset retriever
        retriever = VideoAssetRetriever(
            api_key=args.api_key,
            cache_dir=args.cache_dir,
            metadata_cache_path=args.metadata_cache,
            min_resolution=(args.min_width, args.min_height),
            max_results_per_query=args.max_results
        )
        
        # Process B-roll cuts
        logger.info(f"Processing {len(broll_cuts_data.get('broll_cuts', []))} B-roll cuts")
        result = retriever.process_broll_cuts_sync(broll_cuts_data)
        
        # Save output file
        logger.info(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Print summary
        successful_cuts = sum(1 for cut in result.get("broll_cuts", []) if cut.get("path"))
        total_cuts = len(result.get("broll_cuts", []))
        logger.info(f"Successfully processed {successful_cuts}/{total_cuts} B-roll cuts")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)