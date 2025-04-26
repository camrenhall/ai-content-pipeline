# sound_effect_opportunity_detector.py
import json
import logging
import os
import random
import argparse
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SoundEffectOpportunity:
    """Represents an opportunity to add a sound effect to a video."""
    timestamp: float          # Start time in seconds
    duration: float           # Maximum duration to play the sound effect
    type: str                 # Type of sound effect (e.g., "transition_in", "transition_out", "ambient")
    sound_category: str       # Category of sound effect (e.g., "whoosh", "impact", "foley")
    sound_tags: List[str]     # Tags describing the sound effect
    sound_file_path: str      # Path to the sound effect file
    reason: str               # Justification for adding this sound effect
    context: str              # Context from the transcript or B-roll description
    broll_id: Optional[int] = None  # ID of the associated B-roll cut
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SoundEffectOpportunityDetector:
    """
    Detects opportunities to add sound effects to a video, based on B-roll cuts
    and transcript context. Intelligently selects appropriate sound effects from
    the available library.
    """
    
    def __init__(
        self, 
        sound_effects_dir: str = "./assets/sound_effects",
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o",
        cache_dir: Optional[str] = None,
        enable_transitions: bool = True,
        enable_ambient: bool = True,
        randomize_selection: bool = True,
        randomize_weight: float = 0.7  # Higher values mean more randomization
    ):
        """
        Initialize the Sound Effect Opportunity Detector.
        
        Args:
            sound_effects_dir: Directory containing sound effect files
            llm_api_url: URL for the LLM API (if None, will use environment variable)
            llm_api_key: API key for the LLM (if None, will use environment variable)
            llm_model: Model name to use for the LLM API
            cache_dir: Directory to cache analysis results
            enable_transitions: Whether to add transition sound effects
            enable_ambient: Whether to add ambient sound effects
            randomize_selection: Whether to randomize sound effect selection
            randomize_weight: Weight for randomization (0-1, higher = more random)
        """
        self.sound_effects_dir = sound_effects_dir
        self.llm_api_url = llm_api_url or os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.enable_transitions = enable_transitions
        self.enable_ambient = enable_ambient
        self.randomize_selection = randomize_selection
        self.randomize_weight = max(0.0, min(1.0, randomize_weight))  # Ensure value is between 0-1
        
        # Initialize sound effect library
        self.sound_effects = {}
        self.sound_effects_by_category = {}
        self.sound_effect_categories = []
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load sound effect library
        self._initialize_sound_effect_library()
    
    def _initialize_sound_effect_library(self):
        """Initialize the sound effect library from files on disk."""
        self.logger.info(f"Initializing sound effect library from {self.sound_effects_dir}")
        
        # Check if cache directory exists
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "sound_effects_cache.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                        self.sound_effects = cache_data["sound_effects"]
                        self.sound_effects_by_category = cache_data["sound_effects_by_category"]
                        self.sound_effect_categories = cache_data["sound_effect_categories"]
                        
                    # Validate cache by checking if all files still exist
                    cache_valid = True
                    for se_id, se_data in self.sound_effects.items():
                        if not os.path.exists(se_data["file_path"]):
                            cache_valid = False
                            break
                            
                    if cache_valid:
                        self.logger.info(f"Loaded {len(self.sound_effects)} sound effects from cache")
                        return
                    else:
                        self.logger.info("Cache invalid (missing files), rescanning sound effects")
                except Exception as e:
                    self.logger.warning(f"Failed to load sound effects cache: {e}, rescanning")
        
        # Scan for sound effects
        self.sound_effects = {}
        self.sound_effects_by_category = {}
        
        sound_effects_dir = Path(self.sound_effects_dir)
        if not sound_effects_dir.exists():
            self.logger.error(f"Sound effects directory not found: {self.sound_effects_dir}")
            raise FileNotFoundError(f"Sound effects directory not found: {self.sound_effects_dir}")
        
        # Scan all subdirectories
        for category_dir in sound_effects_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            self.sound_effects_by_category[category] = []
            
            for file_path in category_dir.glob("**/*.mp3"):
                se_id = f"{category}_{file_path.stem}"
                file_name = file_path.name
                
                # Parse tags from filename (format: primary_tag1_tag2.mp3)
                tags = file_name.split(".")[0].split("_")
                
                # First tag is typically the main descriptor (e.g., "impact", "knife", "sting")
                primary_tag = tags[0] if tags else ""
                
                # Remaining tags are additional descriptors
                descriptor_tags = tags[1:] if len(tags) > 1 else []
                
                self.sound_effects[se_id] = {
                    "id": se_id,
                    "file_path": str(file_path),
                    "file_name": file_name,
                    "category": category,
                    "primary_tag": primary_tag,
                    "descriptor_tags": descriptor_tags,
                    "all_tags": tags
                }
                
                self.sound_effects_by_category[category].append(se_id)
        
        # Store the list of categories
        self.sound_effect_categories = list(self.sound_effects_by_category.keys())
        
        self.logger.info(f"Found {len(self.sound_effects)} sound effects in {len(self.sound_effect_categories)} categories")
        
        # Save to cache if cache directory exists
        if cache_file:
            try:
                cache_data = {
                    "sound_effects": self.sound_effects,
                    "sound_effects_by_category": self.sound_effects_by_category,
                    "sound_effect_categories": self.sound_effect_categories
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                    
                self.logger.info(f"Saved sound effects library to cache: {cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save sound effects cache: {e}")
    
    def detect_opportunities(self, broll_cuts_file: str, output_file: Optional[str] = None) -> List[SoundEffectOpportunity]:
        """
        Detect opportunities to add sound effects based on B-roll cuts.
        
        Args:
            broll_cuts_file: Path to the JSON file with B-roll cuts
            output_file: Path to save the sound effect opportunities
            
        Returns:
            List of SoundEffectOpportunity objects
        """
        self.logger.info(f"Detecting sound effect opportunities from {broll_cuts_file}")
        
        # Load B-roll cuts data
        with open(broll_cuts_file, 'r') as f:
            broll_data = json.load(f)
        
        broll_cuts = broll_data.get("broll_cuts", [])
        if not broll_cuts:
            self.logger.warning("No B-roll cuts found in the provided file")
            return []
        
        # Get video duration from the last B-roll cut (approximate)
        video_duration = 0
        for cut in broll_cuts:
            end_time = cut.get("timestamp", 0) + cut.get("duration", 0)
            video_duration = max(video_duration, end_time)
        
        # Add extra duration to account for content after the last B-roll
        video_duration += 10.0  # Assuming at least 10 more seconds after the last B-roll
        
        # Detect transition sound effect opportunities
        transition_opportunities = []
        if self.enable_transitions:
            transition_opportunities = self._detect_transition_opportunities(broll_cuts, video_duration)
            self.logger.info(f"Detected {len(transition_opportunities)} transition sound effect opportunities")
        
        # Detect ambient sound effect opportunities for B-roll segments
        ambient_opportunities = []
        if self.enable_ambient:
            ambient_opportunities = self._detect_ambient_opportunities(broll_cuts, video_duration)
            self.logger.info(f"Detected {len(ambient_opportunities)} ambient sound effect opportunities")
        
        # Combine all opportunities
        opportunities = transition_opportunities + ambient_opportunities
        
        # Sort by timestamp
        opportunities.sort(key=lambda x: x.timestamp)
        
        # Save to output file if provided
        if output_file:
            self._save_to_json(opportunities, output_file)
        
        return opportunities
    
    def _detect_transition_opportunities(self, broll_cuts: List[Dict], video_duration: float) -> List[SoundEffectOpportunity]:
        """
        Detect opportunities for transition sound effects.
        
        Args:
            broll_cuts: List of B-roll cut dictionaries
            video_duration: Estimated total video duration in seconds
            
        Returns:
            List of SoundEffectOpportunity objects for transitions
        """
        opportunities = []
        
        for i, cut in enumerate(broll_cuts):
            # Extract B-roll information
            timestamp = cut.get("timestamp", 0)
            duration = cut.get("duration", 0)
            reason = cut.get("reason", "")
            transcript_segment = cut.get("transcript_segment", "")
            
            # Ensure B-roll doesn't exceed video duration
            if timestamp >= video_duration:
                self.logger.warning(f"B-roll cut at {timestamp}s exceeds video duration, skipping")
                continue
            
            # Add transition IN sound effect (whoosh)
            in_sound = self._select_sound_effect("whooshes", context=reason or transcript_segment)
            if in_sound:
                opportunities.append(SoundEffectOpportunity(
                    timestamp=max(0, timestamp - 0.2),  # Start slightly before the B-roll
                    duration=min(1.0, duration / 3),   # Use at most 1/3 of the B-roll duration
                    type="transition_in",
                    sound_category="whooshes",
                    sound_tags=self.sound_effects[in_sound]["all_tags"],
                    sound_file_path=self.sound_effects[in_sound]["file_path"],
                    reason=f"Transition into B-roll at {timestamp}s",
                    context=transcript_segment,
                    broll_id=i
                ))

            # Add transition OUT sound effect (impact)
            out_timestamp = timestamp + duration
            # Ensure we don't exceed video duration
            if out_timestamp < video_duration:
                out_sound = self._select_sound_effect("impacts", context=reason or transcript_segment)
                if out_sound:
                    opportunities.append(SoundEffectOpportunity(
                        timestamp=out_timestamp,
                        duration=1.0,  # Typical impact duration
                        type="transition_out",
                        sound_category="impacts",
                        sound_tags=self.sound_effects[out_sound]["all_tags"],
                        sound_file_path=self.sound_effects[out_sound]["file_path"],
                        reason=f"Transition out of B-roll at {out_timestamp}s",
                        context=transcript_segment,
                        broll_id=i
                    ))
        
        return opportunities
    
    def _detect_ambient_opportunities(self, broll_cuts: List[Dict], video_duration: float) -> List[SoundEffectOpportunity]:
        """
        Detect opportunities for ambient sound effects during B-roll segments.
        
        Args:
            broll_cuts: List of B-roll cut dictionaries
            video_duration: Estimated total video duration in seconds
            
        Returns:
            List of SoundEffectOpportunity objects for ambient sounds
        """
        opportunities = []
        
        for i, cut in enumerate(broll_cuts):
            # Extract B-roll information
            timestamp = cut.get("timestamp", 0)
            duration = cut.get("duration", 0)
            reason = cut.get("reason", "")
            transcript_segment = cut.get("transcript_segment", "")
            keywords = cut.get("keywords", [])
            
            # Ensure B-roll doesn't exceed video duration
            if timestamp >= video_duration:
                continue
                
            # Skip very short B-roll cuts (less than 1.5 seconds)
            if duration < 1.5:
                continue
            
            # Build context for LLM analysis
            context = f"B-roll content: {reason}\n"
            context += f"Transcript: {transcript_segment}\n"
            if keywords:
                context += f"Keywords: {', '.join(keywords)}\n"
            
            # Use LLM to analyze if we need ambient sound effects and what type
            ambient_analysis = self._analyze_ambient_sound_needs(context)
            
            if ambient_analysis["add_sound"] and ambient_analysis["category"]:
                category = ambient_analysis["category"]
                
                # Ensure category exists in our library
                if category not in self.sound_effects_by_category:
                    # Try to find the closest matching category
                    category = self._find_closest_category(category)
                
                if category:
                    # Select sound effect
                    sound_id = self._select_sound_effect(
                        category,
                        context=context,
                        preferred_tags=ambient_analysis.get("tags", [])
                    )
                    
                    if sound_id:
                        # Start ambient sound at the beginning of B-roll with a small offset
                        ambient_start = timestamp + 0.3  # Small offset after the transition in
                        
                        # End ambient sound before the transition out
                        ambient_duration = duration - 0.5  # Leave room for transition out
                        
                        # Ensure we have a positive duration
                        if ambient_duration <= 0:
                            continue
                        
                        opportunities.append(SoundEffectOpportunity(
                            timestamp=ambient_start,
                            duration=ambient_duration,
                            type="ambient",
                            sound_category=category,
                            sound_tags=self.sound_effects[sound_id]["all_tags"],
                            sound_file_path=self.sound_effects[sound_id]["file_path"],
                            reason=ambient_analysis["reason"],
                            context=context,
                            broll_id=i
                        ))
        
        return opportunities
    
    def _analyze_ambient_sound_needs(self, context: str) -> Dict[str, Any]:
        """
        Use LLM to analyze if ambient sound effects are needed and what type.
        
        Args:
            context: Context about the B-roll content
            
        Returns:
            Dictionary with analysis results
        """
        # Default response if LLM analysis fails
        default_response = {
            "add_sound": False,
            "category": None,
            "tags": [],
            "reason": "No ambient sound needed"
        }
        
        # Check if LLM API key is available
        if not self.llm_api_key:
            self.logger.warning("No LLM API key available, skipping ambient sound analysis")
            return default_response
        
        try:
            prompt = f"""
            You are a sound designer analyzing a video segment to determine if ambient sound effects would enhance the viewing experience. 
            
            Here's information about the video segment:
            {context}
            
            The available sound effect categories are:
            {', '.join(self.sound_effect_categories)}
            
            Determine if an ambient sound effect would enhance this B-roll segment. Be selective - not every B-roll needs sound effects.
            Only recommend sound effects if there's a clear match between the context and available sound categories.
            
            Respond in JSON format:
            {{
              "add_sound": true/false,
              "category": "category_name", (must be one from the list above, or null if add_sound is false)
              "tags": ["tag1", "tag2"], (descriptive tags to help select the right sound)
              "reason": "Explanation of why this sound effect fits this segment"
            }}
            """
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            # Validate response
            if not isinstance(response, dict) or "add_sound" not in response:
                self.logger.warning("Invalid LLM response format, using default response")
                return default_response
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error in ambient sound analysis: {e}")
            return default_response
    
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
    
    def _select_sound_effect(
        self, 
        category: str, 
        context: str = "", 
        preferred_tags: List[str] = []
    ) -> Optional[str]:
        """
        Select an appropriate sound effect from the given category.
        
        Args:
            category: Sound effect category
            context: Context to help select appropriate sound
            preferred_tags: Preferred tags for sound selection
            
        Returns:
            Sound effect ID if found, None otherwise
        """
        if category not in self.sound_effects_by_category:
            self.logger.warning(f"Sound effect category not found: {category}")
            return None
        
        available_sounds = self.sound_effects_by_category[category]
        if not available_sounds:
            return None
            
        # If no preferred tags or no context, select randomly
        if (not preferred_tags and not context) or not self.llm_api_key or random.random() < 0.3:
            return random.choice(available_sounds)
        
        # If we have preferred tags, try to match them
        if preferred_tags:
            # Score each sound based on tag matches
            scored_sounds = []
            for sound_id in available_sounds:
                sound = self.sound_effects[sound_id]
                
                # Calculate score based on tag matches
                score = 0
                for tag in preferred_tags:
                    if tag.lower() in [t.lower() for t in sound["all_tags"]]:
                        score += 1
                
                scored_sounds.append((sound_id, score))
                
            # If we have any matches, select from the best matches
            if any(score > 0 for _, score in scored_sounds):
                # Get the maximum score
                max_score = max(score for _, score in scored_sounds)
                
                # Filter to only the sounds with the highest score
                best_sounds = [sound_id for sound_id, score in scored_sounds if score == max_score]
                
                # If randomizing, select randomly from the best matches
                if self.randomize_selection:
                    return random.choice(best_sounds)
                else:
                    # Otherwise, just take the first one
                    return best_sounds[0]
        
        # If we reach here, just select randomly from the category
        return random.choice(available_sounds)
    
    def _find_closest_category(self, desired_category: str) -> Optional[str]:
        """
        Find the closest matching category in our library.
        
        Args:
            desired_category: The desired category name
            
        Returns:
            Closest matching category name if found, None otherwise
        """
        desired_lower = desired_category.lower()
        
        # Check for partial matches
        for category in self.sound_effect_categories:
            if desired_lower in category.lower() or category.lower() in desired_lower:
                return category
        
        # Define mappings for common alternatives
        category_mappings = {
            "ambient": "foley",
            "background": "foley",
            "music": "stingers",
            "fx": "foley",
            "effect": "foley",
            "transition": "whooshes",
            "hit": "impacts",
            "swoosh": "whooshes",
            "interface": "ui",
            "button": "ui",
            "notification": "ui",
            "alert": "ui",
            "environmental": "foley",
            "nature": "foley",
            "sound": "foley"
        }
        
        # Check for mappings
        for key, value in category_mappings.items():
            if key in desired_lower and value in self.sound_effect_categories:
                return value
        
        # If no match found, return None
        return None
    
    def _save_to_json(self, opportunities: List[SoundEffectOpportunity], output_path: str) -> None:
        """
        Save sound effect opportunities to a JSON file.
        
        Args:
            opportunities: List of SoundEffectOpportunity objects
            output_path: Path to save the JSON file
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Format for the downstream components
            sound_effect_opportunities = []
            for opp in opportunities:
                sound_effect_opportunities.append(opp.to_dict())
            
            # Create output structure
            output_data = {
                "sound_effect_opportunities": sound_effect_opportunities,
                "metadata": {
                    "count": len(opportunities),
                    "version": "1.0",
                    "categories": {category: len([o for o in opportunities if o.sound_category == category]) for category in set(o.sound_category for o in opportunities)}
                }
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            self.logger.info(f"Saved {len(opportunities)} sound effect opportunities to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save sound effect opportunities to JSON: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect sound effect opportunities in a video based on B-roll cuts')
    parser.add_argument('--broll', required=True, help='Path to the B-roll cuts JSON file')
    parser.add_argument('--output', required=True, help='Path to save the sound effect opportunities JSON file')
    parser.add_argument('--sound-effects-dir', default='./assets/sound_effects', help='Directory containing sound effect files')
    parser.add_argument('--llm-api-url', help='URL for the LLM API')
    parser.add_argument('--llm-api-key', help='API key for the LLM')
    parser.add_argument('--llm-model', default='gpt-4o', help='Model name to use for the LLM API')
    parser.add_argument('--cache-dir', default='./cache', help='Directory to cache analysis results')
    parser.add_argument('--disable-transitions', action='store_true', help='Disable transition sound effects')
    parser.add_argument('--disable-ambient', action='store_true', help='Disable ambient sound effects')
    parser.add_argument('--randomize-selection', action='store_true', help='Randomize sound effect selection')
    parser.add_argument('--randomize-weight', type=float, default=0.7, help='Weight for randomization (0-1)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Initialize the detector
        detector = SoundEffectOpportunityDetector(
            sound_effects_dir=args.sound_effects_dir,
            llm_api_url=args.llm_api_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            cache_dir=args.cache_dir,
            enable_transitions=not args.disable_transitions,
            enable_ambient=not args.disable_ambient,
            randomize_selection=args.randomize_selection,
            randomize_weight=args.randomize_weight
        )
        
        # Detect opportunities
        logger.info(f"Detecting sound effect opportunities for {args.broll}")
        opportunities = detector.detect_opportunities(
            args.broll,
            args.output
        )
        
        # Print summary
        logger.info(f"Detected {len(opportunities)} sound effect opportunities")
        
        # Count by type
        type_counts = {}
        for opp in opportunities:
            type_counts[opp.type] = type_counts.get(opp.type, 0) + 1
        
        for t, count in type_counts.items():
            logger.info(f"  {t}: {count}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)