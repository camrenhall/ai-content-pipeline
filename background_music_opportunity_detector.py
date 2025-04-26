import os
import json
import logging
import random
import argparse
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import requests
from dataclasses import dataclass, asdict, field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MusicFile:
    """Represents a background music file with its metadata."""
    path: str
    category: str
    tags: List[str]
    filename: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BackgroundMusicSelection:
    """Represents a selected background music track for a video."""
    file_path: str
    category: str
    tags: List[str]
    duration_seconds: float
    reason: str  # Explanation for why this music was selected
    confidence_score: float = 0.0  # 0.0-1.0 score indicating confidence in the selection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BackgroundMusicOpportunityDetector:
    """
    Analyzes video content to determine the most appropriate background 
    music category and selects a suitable music file.
    """
    
    def __init__(
        self,
        music_dir: str = "./assets/background_music",
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o",
        cache_dir: Optional[str] = None,
        override_category: Optional[str] = None
    ):
        """
        Initialize the BackgroundMusicOpportunityDetector.
        
        Args:
            music_dir: Directory containing background music files
            llm_api_url: URL for the LLM API (if None, will use environment variable)
            llm_api_key: API key for the LLM (if None, will use environment variable)
            llm_model: Model name to use for the LLM API
            cache_dir: Directory to cache analysis results
            override_category: Category to use instead of analyzing content
        """
        self.music_dir = Path(music_dir)
        self.llm_api_url = llm_api_url or os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        self.override_category = override_category
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        
        # Initialize music library
        self.music_files: Dict[str, List[MusicFile]] = {}
        self.categories: List[str] = []
        
        # Cache for music files - will be populated when _scan_music_files is called
        self._music_library_cache = None
        
        # Analysis cache - will store previous LLM analyses
        self._analysis_cache = {}
        
        # Load analysis cache if it exists
        self._load_analysis_cache()
        
        # Initialize music library
        self._scan_music_files()
    
    def _scan_music_files(self, force_refresh: bool = False) -> Dict[str, List[MusicFile]]:
        """
        Scan the music directory to build a library of available music files.
        
        Args:
            force_refresh: Whether to force a fresh scan instead of using cache
            
        Returns:
            Dictionary of music files by category
        """
        # Check if we have a cached library and we're not forcing a refresh
        if self._music_library_cache is not None and not force_refresh:
            return self._music_library_cache
            
        self.logger.info(f"Scanning music directory: {self.music_dir}")
        
        # Check if music directory exists
        if not self.music_dir.exists():
            self.logger.warning(f"Music directory not found: {self.music_dir}")
            self.music_files = {}
            self.categories = []
            return {}
        
        # Initialize music library
        self.music_files = {}
        
        # Scan all subdirectories (categories)
        for category_dir in self.music_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            self.music_files[category] = []
            
            # Scan music files in this category
            for file_path in category_dir.glob("*.mp3"):
                # Parse tags from filename (format: primary_tag1_tag2.mp3)
                filename = file_path.name
                tags = filename.split(".")[0].split("_")
                
                music_file = MusicFile(
                    path=str(file_path),
                    category=category,
                    tags=tags,
                    filename=filename
                )
                
                self.music_files[category].append(music_file)
                
        # Store the list of categories
        self.categories = list(self.music_files.keys())
        
        # Store the library in cache
        self._music_library_cache = self.music_files
        
        # Log the results
        total_files = sum(len(files) for files in self.music_files.values())
        self.logger.info(f"Found {total_files} music files in {len(self.categories)} categories")
        for category, files in self.music_files.items():
            self.logger.info(f"  {category}: {len(files)} files")
            
        return self.music_files
        
    def detect_music(self, script_analysis: Dict[str, Any], output_path: Optional[str] = None, force_refresh: bool = False) -> BackgroundMusicSelection:
        """
        Analyze script content to determine appropriate background music.
        
        Args:
            script_analysis: Script analysis data from script_analyzer.py
            output_path: Path to save the analysis results
            force_refresh: Whether to force a fresh analysis instead of using cache
            
        Returns:
            BackgroundMusicSelection object with selected music information
        """
        self.logger.info("Detecting appropriate background music")
        
        # Handle both direct and pipeline-generated script analysis formats
        if "result" in script_analysis:
            # Pipeline format: extract from result.transcript
            transcript_data = script_analysis.get("result", {}).get("transcript", {})
            full_text = transcript_data.get("full_text", "")
            duration_seconds = script_analysis.get("result", {}).get("duration_seconds", 0)
            if not duration_seconds:
                duration_seconds = transcript_data.get("metadata", {}).get("duration_seconds", 0)
        else:
            # Direct format from script analyzer CLI
            full_text = script_analysis.get("full_text", "")
            duration_seconds = script_analysis.get("metadata", {}).get("duration_seconds", 0)
        
        if not full_text:
            self.logger.warning("No text content found in script analysis")
            return self._select_fallback_music(duration_seconds)
        
        # Check if override category is specified
        if self.override_category:
            self.logger.info(f"Using override category: {self.override_category}")
            return self._select_music_from_category(self.override_category, full_text, duration_seconds)
        
        # Generate a unique key for the analysis cache
        cache_key = self._generate_cache_key(full_text)
        
        # Check if we have a cached analysis
        if not force_refresh and cache_key in self._analysis_cache:
            cached_result = self._analysis_cache[cache_key]
            self.logger.info(f"Using cached analysis for music selection (key: {cache_key[:8]})")
            
            category = cached_result["category"]
            confidence = cached_result["confidence"]
            reason = cached_result["reason"]
            
            # Select music using the cached analysis
            selection = self._select_music_from_category(category, full_text, duration_seconds, reason, confidence)
        else:
            # Use LLM to determine appropriate music category
            category, confidence, reason = self._analyze_content_for_music(full_text)
            
            # Cache the analysis
            self._analysis_cache[cache_key] = {
                "category": category,
                "confidence": confidence,
                "reason": reason,
                "timestamp": time.time()
            }
            self._save_analysis_cache()
            
            # Select music from the determined category
            selection = self._select_music_from_category(category, full_text, duration_seconds, reason, confidence)
        
        # Save to output file if provided
        if output_path:
            self._save_to_json(selection, output_path)
            
        return selection
    
    def _analyze_content_for_music(self, text: str) -> Tuple[str, float, str]:
        """
        Use LLM to analyze content and determine appropriate music category.
        
        Args:
            text: Script content to analyze
            
        Returns:
            Tuple of (category, confidence, reason)
        """
        # Check if LLM API is available
        if not self.llm_api_key:
            self.logger.warning("No LLM API key available, using random music category")
            if self.categories:
                category = random.choice(self.categories)
                return category, 0.5, "Selected randomly due to no LLM API access"
            else:
                return "ambient", 0.5, "No music categories available, defaulting to ambient"
        
        try:
            # Ensure we have scanned the music library
            self._scan_music_files()
            
            # List available categories for LLM
            categories_str = ", ".join(self.categories) if self.categories else "ambient, calm, dramatic, energetic, inspirational"
            
            # Prepare the prompt for the LLM
            prompt = f"""
            You are an expert music supervisor for videos. Analyze the following script content and determine the most appropriate background music category. 
            
            SCRIPT CONTENT:
            {text}
            
            AVAILABLE MUSIC CATEGORIES:
            {categories_str}
            
            Based solely on the content and tone of the script:
            1. Select the MOST appropriate single music category from the available categories
            2. Provide a brief explanation for why this category fits the content
            3. Assign a confidence score (0.0-1.0) for your selection
            
            Respond in JSON format:
            {{
              "category": "selected_category",
              "confidence": 0.7,
              "reason": "Brief explanation of why this category fits"
            }}
            """
            
            # Call the LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Parse the response
            if isinstance(llm_response, dict) and "category" in llm_response:
                category = llm_response.get("category", "")
                confidence = float(llm_response.get("confidence", 0.5))
                reason = llm_response.get("reason", "")
                
                # Validate category
                if category not in self.categories:
                    self.logger.warning(f"LLM selected invalid category: {category}")
                    if self.categories:
                        category = random.choice(self.categories)
                        reason = f"Falling back to random category (original suggestion '{category}' not available)"
                        confidence = 0.5
                    else:
                        category = "ambient"
                        reason = "No music categories available, defaulting to ambient"
                        confidence = 0.5
                
                self.logger.info(f"LLM selected music category: {category} (confidence: {confidence:.2f})")
                return category, confidence, reason
            else:
                self.logger.warning("Invalid LLM response format, using random music category")
                if self.categories:
                    category = random.choice(self.categories)
                    return category, 0.5, "Selected randomly due to invalid LLM response"
                else:
                    return "ambient", 0.5, "No music categories available, defaulting to ambient"
                
        except Exception as e:
            self.logger.error(f"Error analyzing content for music: {e}")
            if self.categories:
                category = random.choice(self.categories)
                return category, 0.5, f"Selected randomly due to error: {str(e)}"
            else:
                return "ambient", 0.5, "No music categories available, defaulting to ambient"
    
    def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM API with the prompt."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            data = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,  # Lower temperature for more consistent results
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                self.llm_api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return {}
                
            result = response.json()
            
            # Extract the content from the response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                self.logger.error("No content in LLM response")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            return {}
    
    def _select_music_from_category(
        self, 
        category: str, 
        content: str, 
        duration_seconds: float,
        reason: str = "",
        confidence: float = 0.5
    ) -> BackgroundMusicSelection:
        """
        Select an appropriate music file from the given category.
        
        Args:
            category: Music category
            content: Content to help select appropriate music
            duration_seconds: Duration of the video in seconds
            reason: Reason for category selection
            confidence: Confidence score for the category selection
            
        Returns:
            BackgroundMusicSelection object
        """
        # Ensure we have scanned the music library
        self._scan_music_files()
        
        # Check if we have music files for this category
        if category not in self.music_files or not self.music_files[category]:
            self.logger.warning(f"No music files found for category: {category}")
            return self._select_fallback_music(duration_seconds)
        
        # Use weighted random selection based on tag relevance to content
        if content:
            weighted_music_files = self._calculate_weights(category, content)
            if weighted_music_files:
                # Select based on weights
                weights = [weight for _, weight in weighted_music_files]
                files = [music_file for music_file, _ in weighted_music_files]
                
                selected_file = random.choices(files, weights=weights, k=1)[0]
                self.logger.info(f"Selected music file with weighted random: {selected_file.filename}")
            else:
                # Fall back to pure random selection
                selected_file = random.choice(self.music_files[category])
                self.logger.info(f"Selected music file randomly: {selected_file.filename}")
        else:
            # If no content for context, use pure random selection
            selected_file = random.choice(self.music_files[category])
            self.logger.info(f"Selected music file randomly (no content): {selected_file.filename}")
        
        # Create selection object
        selection = BackgroundMusicSelection(
            file_path=selected_file.path,
            category=category,
            tags=selected_file.tags,
            duration_seconds=duration_seconds,
            reason=reason or f"Selected {category} music based on content analysis",
            confidence_score=confidence
        )
        
        return selection
    
    def _calculate_weights(self, category: str, content: str) -> List[Tuple[MusicFile, float]]:
        """
        Calculate relevance weights for music files in a category based on content.
        
        Args:
            category: Music category
            content: Content to calculate relevance against
            
        Returns:
            List of (MusicFile, weight) tuples
        """
        # Ensure we have music files for this category
        if category not in self.music_files or not self.music_files[category]:
            return []
            
        # Prepare content for comparison
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        # Calculate weights for each music file
        weighted_files = []
        
        for music_file in self.music_files[category]:
            # Default base weight
            weight = 1.0
            
            # Check for tag matches in content
            for tag in music_file.tags:
                tag_lower = tag.lower()
                
                # Direct tag match in content
                if tag_lower in content_lower:
                    weight += 0.5
                
                # Tag is a word in content
                if tag_lower in content_words:
                    weight += 0.3
                
                # Content contains tag as substring
                if tag_lower in content_lower:
                    weight += 0.2
            
            weighted_files.append((music_file, weight))
        
        return weighted_files
    
    def _select_fallback_music(self, duration_seconds: float) -> BackgroundMusicSelection:
        """
        Select fallback music when primary selection fails.
        
        Args:
            duration_seconds: Duration of the video in seconds
            
        Returns:
            BackgroundMusicSelection object
        """
        # Ensure we have scanned the music library
        self._scan_music_files()
        
        # If we have any music files at all, select one randomly
        all_files = []
        for category, files in self.music_files.items():
            all_files.extend(files)
            
        if all_files:
            selected_file = random.choice(all_files)
            self.logger.info(f"Selected fallback music file: {selected_file.filename}")
            
            selection = BackgroundMusicSelection(
                file_path=selected_file.path,
                category=selected_file.category,
                tags=selected_file.tags,
                duration_seconds=duration_seconds,
                reason="Fallback selection due to unavailable primary category",
                confidence_score=0.2
            )
            
            return selection
        
        # If no music files at all, return a dummy selection
        self.logger.warning("No music files found, returning dummy selection")
        return BackgroundMusicSelection(
            file_path="",
            category="none",
            tags=[],
            duration_seconds=duration_seconds,
            reason="No music files available",
            confidence_score=0.0
        )
    
    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: Text to generate a cache key for
            
        Returns:
            Cache key string
        """
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_analysis_cache(self) -> None:
        """Load analysis cache from disk."""
        if not self.cache_dir:
            return
            
        cache_file = os.path.join(self.cache_dir, "music_analysis_cache.json")
        
        if not os.path.exists(cache_file):
            self._analysis_cache = {}
            return
            
        try:
            with open(cache_file, 'r') as f:
                self._analysis_cache = json.load(f)
                
            self.logger.info(f"Loaded {len(self._analysis_cache)} entries from analysis cache")
        except Exception as e:
            self.logger.warning(f"Failed to load analysis cache: {e}. Starting with empty cache.")
            self._analysis_cache = {}
    
    def _save_analysis_cache(self) -> None:
        """Save analysis cache to disk."""
        if not self.cache_dir:
            return
            
        cache_file = os.path.join(self.cache_dir, "music_analysis_cache.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._analysis_cache, f, indent=2)
                
            self.logger.info(f"Saved {len(self._analysis_cache)} entries to analysis cache")
        except Exception as e:
            self.logger.error(f"Failed to save analysis cache: {e}")
    
    def _prune_old_cache_entries(self, max_age_days: int = 30) -> None:
        """
        Remove old entries from the analysis cache.
        
        Args:
            max_age_days: Maximum age of cache entries in days
        """
        if not self._analysis_cache:
            return
            
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        keys_to_remove = []
        for key, entry in self._analysis_cache.items():
            # Skip entries without timestamp
            if "timestamp" not in entry:
                continue
                
            age = current_time - entry["timestamp"]
            if age > max_age_seconds:
                keys_to_remove.append(key)
        
        # Remove old entries
        for key in keys_to_remove:
            del self._analysis_cache[key]
            
        if keys_to_remove:
            self.logger.info(f"Pruned {len(keys_to_remove)} old entries from analysis cache")
            self._save_analysis_cache()# background_music_opportunity_detector.py
    
    def _save_to_json(self, selection: BackgroundMusicSelection, output_path: str) -> None:
        """
        Save music selection to a JSON file.
        
        Args:
            selection: BackgroundMusicSelection object
            output_path: Path to save the JSON file
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Format for the downstream components - ensure compatibility with BackgroundMusicManager
            output_data = {
                "file_path": selection.file_path,
                "category": selection.category,
                "tags": selection.tags,
                "duration_seconds": selection.duration_seconds,
                "reason": selection.reason
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            self.logger.info(f"Saved music selection to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save music selection to JSON: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect appropriate background music for a video')
    parser.add_argument('--script', required=True, help='Path to the script analysis JSON file')
    parser.add_argument('--output', required=True, help='Path to save the music selection JSON file')
    parser.add_argument('--music-dir', default='./assets/background_music', 
                       help='Directory containing background music files')
    parser.add_argument('--llm-api-url', help='URL for the LLM API')
    parser.add_argument('--llm-api-key', help='API key for the LLM')
    parser.add_argument('--llm-model', default='gpt-4o', help='Model name to use for the LLM API')
    parser.add_argument('--cache-dir', default='./cache', help='Directory to cache analysis results')
    parser.add_argument('--category', help='Override category selection (skip content analysis)')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh analysis instead of using cache')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Load script analysis
        logger.info(f"Loading script analysis from {args.script}")
        with open(args.script, 'r') as f:
            script_analysis = json.load(f)
        
        # Initialize detector
        detector = BackgroundMusicOpportunityDetector(
            music_dir=args.music_dir,
            llm_api_url=args.llm_api_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            cache_dir=args.cache_dir,
            override_category=args.category
        )
        
        # Detect music
        logger.info("Detecting appropriate background music")
        selection = detector.detect_music(
            script_analysis, 
            args.output,
            force_refresh=args.force_refresh
        )
        
        # Print results
        logger.info(f"Selected music: {selection.file_path}")
        logger.info(f"Category: {selection.category}")
        logger.info(f"Tags: {', '.join(selection.tags)}")
        logger.info(f"Confidence: {selection.confidence_score:.2f}")
        logger.info(f"Reason: {selection.reason}")
        logger.info(f"Output saved to: {args.output}")
        
        # Exit with success
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()