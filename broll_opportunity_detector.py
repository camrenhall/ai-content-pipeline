# broll_opportunity_detector.py
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import requests
from script_analyzer import Transcript, TranscriptSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BRollOpportunity:
    """Represents a detected opportunity for B-roll insertion."""
    timestamp: float  # Start time in seconds
    duration: float   # Duration in seconds
    score: float      # Confidence/relevance score (0-1)
    keywords: List[str]  # Keywords for B-roll search
    reason: str       # Justification for this opportunity
    transcript_segment: str  # The transcript text this relates to
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BRollOpportunityDetector:
    """
    Analyzes video transcripts to identify ideal moments for B-roll insertion.
    
    This detector uses various strategies to find natural pauses, topic transitions, 
    emphasis points, and other indicators of good B-roll placement opportunities.
    """
    
    def __init__(
        self, 
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o",  # Add this parameter
        min_separation: float = 4.0,
        max_opportunities: int = 5,
        min_opportunity_duration: float = 1.5,
        max_opportunity_duration: float = 4.0,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the B-Roll opportunity detector.
        
        Args:
            llm_api_url: URL for the LLM API (if None, will use environment variable)
            llm_api_key: API key for the LLM (if None, will use environment variable)
            llm_model: Model name to use for the LLM API
            min_separation: Minimum separation between opportunities (seconds)
            max_opportunities: Maximum number of opportunities to return
            min_opportunity_duration: Minimum duration for an opportunity
            max_opportunity_duration: Maximum duration for an opportunity
            cache_dir: Directory to cache analysis results
        """
        self.llm_api_url = llm_api_url or os.environ.get("LLM_API_URL")
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.llm_model = llm_model  # Store the model name
        
        self.min_separation = min_separation
        self.max_opportunities = max_opportunities
        self.min_opportunity_duration = min_opportunity_duration
        self.max_opportunity_duration = max_opportunity_duration
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
    
    def detect_opportunities(
        self, 
        transcript: Transcript, 
        strategy: str = "hybrid",
        force_refresh: bool = False
    ) -> List[BRollOpportunity]:
        """
        Detect B-roll opportunities in the provided transcript.
        
        Args:
            transcript: The transcript to analyze
            strategy: Detection strategy ('rule_based', 'llm_based', or 'hybrid')
            force_refresh: Whether to force a fresh analysis even if cached results exist
            
        Returns:
            List of BRollOpportunity objects
        """
        # Check for cached results
        cache_path = None
        if self.cache_dir:
            # Create a unique identifier based on the transcript content
            transcript_hash = self._hash_transcript(transcript)
            cache_path = os.path.join(self.cache_dir, f"{transcript_hash}.broll_opportunities.json")
            
            if not force_refresh and os.path.exists(cache_path):
                self.logger.info(f"Loading cached B-roll opportunities from {cache_path}")
                return self._load_cached_opportunities(cache_path)
        
        # Detect opportunities using the specified strategy
        if strategy == "rule_based":
            opportunities = self._detect_opportunities_rule_based(transcript)
        elif strategy == "llm_based":
            opportunities = self._detect_opportunities_llm(transcript)
        elif strategy == "hybrid":
            opportunities = self._detect_opportunities_hybrid(transcript)
        else:
            raise ValueError(f"Unknown detection strategy: {strategy}")
        
        # Apply post-processing to ensure opportunities meet requirements
        opportunities = self._post_process_opportunities(opportunities, transcript)
        
        # Ensure all opportunities have complete information
        opportunities = self._ensure_complete_opportunities(opportunities, transcript)
        
        # Cache the results if needed
        if cache_path:
            self._save_opportunities_to_cache(opportunities, cache_path)
            
        return opportunities
    
    def _hash_transcript(self, transcript: Transcript) -> str:
        """Create a unique hash for the transcript for caching purposes."""
        import hashlib
        transcript_content = transcript.full_text + str(transcript.duration_seconds)
        return hashlib.md5(transcript_content.encode()).hexdigest()
    
    def _save_opportunities_to_cache(self, opportunities: List[BRollOpportunity], cache_path: str) -> None:
        """Save detected opportunities to cache."""
        try:
            data = {
                "opportunities": [opp.to_dict() for opp in opportunities],
                "metadata": {
                    "count": len(opportunities),
                    "detection_version": "1.0"
                }
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved {len(opportunities)} B-roll opportunities to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save opportunities to cache: {e}")
    
    def _load_cached_opportunities(self, cache_path: str) -> List[BRollOpportunity]:
        """Load opportunities from cache file."""
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            opportunities = []
            for opp_data in data["opportunities"]:
                opportunities.append(BRollOpportunity(
                    timestamp=opp_data["timestamp"],
                    duration=opp_data["duration"],
                    score=opp_data["score"],
                    keywords=opp_data["keywords"],
                    reason=opp_data["reason"],
                    transcript_segment=opp_data["transcript_segment"]
                ))
                
            self.logger.info(f"Loaded {len(opportunities)} B-roll opportunities from cache")
            return opportunities
        except Exception as e:
            self.logger.warning(f"Failed to load opportunities from cache: {e}")
            return []
    
    def _detect_opportunities_rule_based(self, transcript: Transcript) -> List[BRollOpportunity]:
        """
        Rule-based approach to detect B-roll opportunities.
        
        This approach uses heuristics about natural pauses, topic transitions,
        and other patterns to identify good B-roll insertion points.
        """
        opportunities = []
        
        # Get all segments and their timing info
        segments = transcript.segments
        if not segments:
            return []
            
        # Build a list of potential transition points
        transition_points = []
        
        # Strategy 1: Look for transitional phrases and natural pauses
        transition_phrases = [
            "for example", "such as", "consider", "let's see", "for instance",
            "importantly", "additionally", "moreover", "however", "but", 
            "on the other hand", "first", "second", "third", "finally",
            "in conclusion", "to summarize", "basically", "essentially",
            "now", "next", "then", "after", "before", "during", "while",
            "imagine", "picture", "visualize", "think about", "look at",
            "what if", "when", "where", "why", "how", "who", "which"
        ]
        
        for i, segment in enumerate(segments):
            segment_text = segment.text.lower()
            segment_start_sec = segment.start / 1000
            segment_end_sec = segment.end / 1000
            segment_duration = segment_end_sec - segment_start_sec
            
            # Check for transitional phrases
            score = 0.0
            matching_phrases = []
            for phrase in transition_phrases:
                if phrase in segment_text:
                    score += 0.2
                    matching_phrases.append(phrase)
            
            # Natural pause detection - check for pause after this segment
            pause_duration = 0
            if i < len(segments) - 1:
                next_start = segments[i+1].start
                pause_duration = (next_start - segment.end) / 1000
                if pause_duration > 0.5:  # Significant pause (500ms)
                    score += min(pause_duration * 0.3, 0.3)  # Cap at 0.3
            
            # Check for segment that introduces a new concept
            if i > 0:
                if len(segment_text.split()) >= 3:  # Reasonable length segment
                    score += 0.1
                    
                # Prefer statements over questions for B-roll
                if not segment_text.endswith('?'):
                    score += 0.1
            
            # Give higher scores to segments occurring at natural transition points
            # (e.g., 1/4, 1/3, 1/2, 2/3, 3/4 through the video)
            video_duration = transcript.duration_seconds
            relative_position = segment_start_sec / video_duration
            
            # Check if segment is near a natural transition point
            transition_points = [0.25, 0.33, 0.5, 0.66, 0.75]
            for point in transition_points:
                if abs(relative_position - point) < 0.05:  # Within 5% of transition point
                    score += 0.2
                    break
            
            # Ensure we have a reasonable duration
            if segment_duration < self.min_opportunity_duration:
                score *= 0.5  # Penalize short segments
            elif segment_duration > self.max_opportunity_duration:
                # Still usable but we'll trim it later
                score *= 0.8  # Small penalty
            
            # If score is significant enough, add as an opportunity
            if score >= 0.3:
                # Extract keywords for this segment
                keywords = self._extract_keywords(segment.text)
                
                # Create a reason for this opportunity
                reason_components = []
                if matching_phrases:
                    reason_components.append(f"Contains transitional phrase(s): {', '.join(matching_phrases)}")
                if pause_duration > 0.5:
                    reason_components.append(f"Natural pause of {pause_duration:.2f} seconds after this segment")
                for point in transition_points:
                    if abs(relative_position - point) < 0.05:
                        reason_components.append(f"Located at natural transition point ({relative_position:.0%} through video)")
                        break
                
                reason = " | ".join(reason_components) if reason_components else "Good general opportunity"
                
                # Create the opportunity
                duration = min(max(segment_duration, self.min_opportunity_duration), self.max_opportunity_duration)
                
                opportunities.append(BRollOpportunity(
                    timestamp=segment_start_sec,
                    duration=duration,
                    score=score,
                    keywords=keywords,
                    reason=reason,
                    transcript_segment=segment.text
                ))
        
        # Sort by score in descending order
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        return opportunities
    
    def _detect_opportunities_llm(self, transcript: Transcript) -> List[BRollOpportunity]:
        """
        Use LLM to detect B-roll opportunities.
        
        This approach asks an LLM to analyze the transcript and identify good
        B-roll insertion points based on context and meaning.
        """
        if not self.llm_api_url or not self.llm_api_key:
            self.logger.warning("LLM API URL or key not provided. Falling back to rule-based detection.")
            return self._detect_opportunities_rule_based(transcript)
        
        opportunities = []
        
        try:
            # Format the transcript for the LLM
            formatted_transcript = self._format_transcript_for_llm(transcript)
            
            # Create the prompt for the LLM
            prompt = self._create_llm_prompt(formatted_transcript, transcript.duration_seconds)
            
            # Call the LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Parse LLM response into opportunities
            opportunities = self._parse_llm_response(llm_response, transcript)
            
        except Exception as e:
            self.logger.error(f"LLM-based detection failed: {e}")
            self.logger.info("Falling back to rule-based detection")
            opportunities = self._detect_opportunities_rule_based(transcript)
        
        return opportunities
    
    def _detect_opportunities_hybrid(self, transcript: Transcript) -> List[BRollOpportunity]:
        """
        Hybrid approach combining rule-based and LLM-based methods.
        
        This approach uses both rule-based methods and LLM analysis, then
        combines and ranks the results.
        """
        # Get opportunities from both approaches
        rule_based_opps = self._detect_opportunities_rule_based(transcript)
        
        # Try to get LLM opportunities, fall back to just rule-based if LLM fails
        try:
            llm_opps = self._detect_opportunities_llm(transcript)
        except Exception as e:
            self.logger.warning(f"LLM detection failed in hybrid approach: {e}")
            llm_opps = []
        
        # If either method failed completely, return the other's results
        if not rule_based_opps:
            return llm_opps
        if not llm_opps:
            return rule_based_opps
        
        # Combine and deduplicate opportunities
        all_opps = rule_based_opps.copy()
        
        # Check for overlapping opportunities from LLM and merge if needed
        for llm_opp in llm_opps:
            # Check if this opportunity overlaps with any existing one
            overlapping = False
            for i, existing_opp in enumerate(all_opps):
                # If opportunities are within 1 second of each other, consider them overlapping
                if abs(existing_opp.timestamp - llm_opp.timestamp) < 1.0:
                    # Combine them by taking the higher score and merging keywords and reasons
                    overlapping = True
                    if llm_opp.score > existing_opp.score:
                        # Create a hybrid reason WITHOUT the prefixes
                        hybrid_reason = llm_opp.reason
                        
                        # Update the existing opportunity with the better data
                        all_opps[i] = BRollOpportunity(
                            timestamp=llm_opp.timestamp,  # Use LLM's timestamp if score is higher
                            duration=max(existing_opp.duration, llm_opp.duration),  # Take the longer duration
                            score=max(existing_opp.score, llm_opp.score) + 0.1,  # Boost score slightly for hybrid matches
                            keywords=list(set(existing_opp.keywords + llm_opp.keywords)),  # Combine keywords
                            reason=hybrid_reason,  # Clean reason from LLM
                            transcript_segment=llm_opp.transcript_segment  # Prefer LLM's segment
                        )
                    break
            
            # If no overlap, add the LLM opportunity directly
            if not overlapping:
                all_opps.append(llm_opp)
        
        # Re-sort by score
        all_opps.sort(key=lambda x: x.score, reverse=True)
        
        return all_opps
    
    def _post_process_opportunities(self, opportunities: List[BRollOpportunity], transcript: Transcript) -> List[BRollOpportunity]:
        """
        Apply post-processing to refine the opportunities.
        
        This includes:
        1. Ensuring minimum separation between opportunities
        2. Limiting to max_opportunities
        3. Adjusting durations to be within bounds
        4. Ensuring no temporal conflicts
        5. Ensuring B-roll doesn't extend beyond video duration
        """
        if not opportunities:
            return []
        
        # Sort by score (highest first)
        sorted_opps = sorted(opportunities, key=lambda x: x.score, reverse=True)
        
        # Get total video duration
        video_duration = transcript.duration_seconds
        
        # Take the top opportunities while ensuring minimum separation
        filtered_opps = []
        for opp in sorted_opps:
            # Check if this opportunity conflicts with any selected ones
            conflicts = False
            for selected_opp in filtered_opps:
                # Check for temporal conflict (too close to an existing opportunity)
                if abs(selected_opp.timestamp - opp.timestamp) < self.min_separation:
                    conflicts = True
                    break
            
            # If no conflicts, add this opportunity
            if not conflicts:
                # Adjust duration to be within bounds
                duration = max(min(opp.duration, self.max_opportunity_duration), self.min_opportunity_duration)
                
                # Ensure B-roll doesn't extend beyond video duration
                timestamp = opp.timestamp
                if timestamp + duration > video_duration:
                    # Adjust duration to end exactly at video end
                    duration = max(video_duration - timestamp, self.min_opportunity_duration)
                    
                    # If the adjusted duration is too short, don't use this opportunity
                    if duration < self.min_opportunity_duration:
                        continue
                
                adjusted_opp = BRollOpportunity(
                    timestamp=timestamp,
                    duration=duration,
                    score=opp.score,
                    keywords=opp.keywords,
                    reason=opp.reason,
                    transcript_segment=opp.transcript_segment
                )
                
                filtered_opps.append(adjusted_opp)
                
                # Stop if we have enough opportunities
                if len(filtered_opps) >= self.max_opportunities:
                    break
        
        # Sort the final list by timestamp
        return sorted(filtered_opps, key=lambda x: x.timestamp)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for B-roll search.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Remove stopwords
        stopwords = [
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
            "be", "been", "being", "in", "on", "at", "to", "for", "with", "by", 
            "about", "against", "between", "into", "through", "during", "before", 
            "after", "above", "below", "from", "up", "down", "of", "off", "over", 
            "under", "again", "further", "then", "once", "here", "there", "when", 
            "where", "why", "how", "all", "any", "both", "each", "few", "more", 
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
            "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
            "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", 
            "that", "this", "these", "those", "am", "have", "has", "had", "do", 
            "does", "did", "doing", "it", "its", "it's", "itself", "they", "them", 
            "their", "theirs", "themselves", "what", "which", "who", "whom", "whose", 
            "as", "im", "youre", "hes", "shes", "its", "were", "theyre"
        ]
        
        # Simple keyword extraction
        words = text.lower().split()
        filtered_words = [word.strip('.,?!;:()[]{}""\'') for word in words if word.lower().strip('.,?!;:()[]{}""\'') not in stopwords]
        
        # Get unique words with length > 3
        unique_words = list(set([word for word in filtered_words if len(word) > 3]))
        
        # Filter out very common words that might not be in stopwords
        common_words = ["going", "think", "know", "like", "just", "really", "things", "thing"]
        unique_words = [word for word in unique_words if word not in common_words]
        
        # Take up to 5 most meaningful keywords
        return unique_words[:5]
    
    def _format_transcript_for_llm(self, transcript: Transcript) -> str:
        """Format the transcript for the LLM prompt."""
        formatted_lines = []
        for i, segment in enumerate(transcript.segments):
            start_time = segment.start / 1000  # Convert to seconds
            end_time = segment.end / 1000
            formatted_lines.append(f"[{start_time:.2f}s - {end_time:.2f}s] {segment.text}")
        
        return "\n".join(formatted_lines)
    
    def _create_llm_prompt(self, formatted_transcript: str, video_duration: float) -> str:
        """Create the prompt for the LLM."""
        return f"""
            You are an expert video editor who understands effective B-roll placement. You are analyzing a video transcript to identify the best 3-5 places to insert B-roll footage. B-roll is supplementary footage that enriches the main video by providing visual context, emphasizing points, or breaking monotony.

            The transcript below shows text segments with their timestamps (in seconds).

            TRANSCRIPT:
            {formatted_transcript}

            TOTAL VIDEO DURATION: {video_duration:.2f} seconds

            Identify 3-5 specific points in this video where B-roll would be most effective, considering:
            1. Natural pauses or transitions between topics
            2. Points where visual illustration would enhance understanding
            3. Statements that would benefit from visual reinforcement
            4. Places where breaking the "talking head" format would maintain viewer engagement
            5. Key concepts or examples that could be visualized
            6. Ensure at least {self.min_separation} seconds between B-roll placement points
            7. IMPORTANT: Make sure that a B-roll insertion point plus its duration does not exceed the total video duration of {video_duration:.2f} seconds

            For each recommended B-roll opportunity, provide:
            1. Timestamp (in seconds) to insert the B-roll
            2. Suggested duration (between {self.min_opportunity_duration} and {self.max_opportunity_duration} seconds, and ensuring timestamp + duration <= {video_duration:.2f})
            3. Keywords to search for appropriate B-roll content (be specific and descriptive)
            4. A detailed justification for why B-roll would be effective at this point

            Your "reason" field must be extremely clear and descriptive. Write a complete sentence that explicitly describes what kind of visuals would work best here and why they enhance the content. This reason will be used by another AI to search for and select appropriate B-roll footage, so make it as helpful and specific as possible.

            Format your response as valid JSON with this structure:
            {{
            "opportunities": [
                {{
                "timestamp": 12.4,
                "duration": 2.5,
                "keywords": ["specific keyword 1", "specific keyword 2", "specific keyword 3"],
                "reason": "A detailed, specific explanation of what kind of visuals would enhance this moment and why."
                }}
            ]
            }}

            Focus only on the most impactful opportunities for B-roll insertion.
            """
    
    def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM API with the prompt."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}"
            }
            
            data = {
                "model": self.llm_model,  # Use the stored model name
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
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")
                
            result = response.json()
            
            # Extract the content from the response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                raise Exception("No content in LLM response")
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            raise
    
    def _parse_llm_response(self, llm_response: Dict[str, Any], transcript: Transcript) -> List[BRollOpportunity]:
        """Parse the LLM response into BRollOpportunity objects."""
        opportunities = []
        
        try:
            # Get the video duration from the transcript
            video_duration = transcript.duration_seconds
            
            # Extract opportunities from the response
            if "opportunities" in llm_response:
                for opp_data in llm_response["opportunities"]:
                    timestamp = float(opp_data.get("timestamp", 0))
                    duration = float(opp_data.get("duration", 2.0))
                    
                    # Validate timestamp and duration are within video bounds
                    if timestamp < 0 or timestamp >= video_duration:
                        self.logger.warning(f"Discarding opportunity with invalid timestamp: {timestamp}")
                        continue
                        
                    # If duration extends beyond video end, adjust it
                    if timestamp + duration > video_duration:
                        duration = video_duration - timestamp
                        self.logger.info(f"Adjusted duration to {duration} to prevent exceeding video length")
                        
                        # If adjusted duration is too short, skip this opportunity
                        if duration < self.min_opportunity_duration:
                            self.logger.warning(f"Discarding opportunity that cannot fit minimum duration")
                            continue
                    
                    keywords = opp_data.get("keywords", [])
                    reason = opp_data.get("reason", "")
                    
                    # Find the corresponding transcript segment
                    segment = transcript.get_segment_at_time(timestamp)
                    segment_text = segment.text if segment else ""
                    
                    # If we couldn't find a segment at the exact timestamp, look nearby
                    if not segment_text:
                        # Look for the nearest segment
                        nearest_segment = None
                        nearest_distance = float('inf')
                        
                        for seg in transcript.segments:
                            seg_mid_time = (seg.start + seg.end) / 2000.0  # Convert to seconds
                            distance = abs(seg_mid_time - timestamp)
                            
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest_segment = seg
                        
                        if nearest_segment and nearest_distance < 3.0:  # Within 3 seconds
                            segment_text = nearest_segment.text
                            
                    # If still no segment text, use the full text (last resort)
                    if not segment_text:
                        # Take a short excerpt from the full text around this point
                        relative_pos = timestamp / video_duration
                        words = transcript.full_text.split()
                        if words:
                            approx_word_index = int(relative_pos * len(words))
                            start_index = max(0, approx_word_index - 5)
                            end_index = min(len(words), approx_word_index + 5)
                            segment_text = " ".join(words[start_index:end_index])
                    
                    # If no keywords provided, extract them
                    if not keywords and segment_text:
                        keywords = self._extract_keywords(segment_text)
                    
                    # Create the opportunity
                    opportunities.append(BRollOpportunity(
                        timestamp=timestamp,
                        duration=duration,
                        score=0.8,  # Give LLM opportunities a high base score
                        keywords=keywords,
                        reason=reason,  # Clean reason
                        transcript_segment=segment_text  # Ensure there's always segment text
                    ))
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            
        return opportunities
    
    def _ensure_complete_opportunities(self, opportunities: List[BRollOpportunity], transcript: Transcript) -> List[BRollOpportunity]:
        """Ensure that all opportunities have complete information."""
        complete_opps = []
        
        for opp in opportunities:
            # Copy the opportunity to avoid modifying the original
            new_opp = BRollOpportunity(
                timestamp=opp.timestamp,
                duration=opp.duration,
                score=opp.score,
                keywords=opp.keywords.copy() if opp.keywords else [],
                reason=opp.reason,
                transcript_segment=opp.transcript_segment
            )
            
            # Ensure transcript segment is not empty
            if not new_opp.transcript_segment:
                segment = transcript.get_segment_at_time(new_opp.timestamp)
                if segment:
                    new_opp.transcript_segment = segment.text
                else:
                    # Find nearest segment if exact match not found
                    nearest_segment = None
                    nearest_distance = float('inf')
                    
                    for seg in transcript.segments:
                        seg_mid_time = (seg.start + seg.end) / 2000.0  # Convert to seconds
                        distance = abs(seg_mid_time - new_opp.timestamp)
                        
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_segment = seg
                    
                    if nearest_segment:
                        new_opp.transcript_segment = nearest_segment.text
                    else:
                        # Last resort: take a short excerpt from the full transcript
                        relative_pos = new_opp.timestamp / transcript.duration_seconds
                        words = transcript.full_text.split()
                        if words:
                            approx_word_index = int(relative_pos * len(words))
                            start_index = max(0, approx_word_index - 5)
                            end_index = min(len(words), approx_word_index + 5)
                            new_opp.transcript_segment = " ".join(words[start_index:end_index])
            
            # Ensure keywords are not empty
            if not new_opp.keywords and new_opp.transcript_segment:
                new_opp.keywords = self._extract_keywords(new_opp.transcript_segment)
            
            # Remove any "LLM:" or "Rule:" prefixes from the reason
            if new_opp.reason:
                # Remove prefixes like "LLM: " or "Rule: "
                new_opp.reason = new_opp.reason.replace("LLM: ", "").replace("Rule: ", "")
                
                # Remove the pattern "| Rule: something"
                if " | Rule: " in new_opp.reason:
                    new_opp.reason = new_opp.reason.split(" | Rule: ")[0]
                    
                # Ensure the reason is detailed enough
                if len(new_opp.reason.split()) < 10:
                    # If the reason is too short, expand it based on the transcript segment
                    if new_opp.transcript_segment:
                        new_opp.reason += f" This moment discusses '{new_opp.transcript_segment}' which can be visually enhanced with B-roll footage that illustrates these concepts."
            
            complete_opps.append(new_opp)
        
        return complete_opps
    
    def export_to_json(self, opportunities: List[BRollOpportunity], output_path: str) -> None:
        """
        Export opportunities to a JSON file in the format expected by the B-roll inserter.
        
        Args:
            opportunities: List of BRollOpportunity objects
            output_path: Path to save the JSON file
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Format for the B-roll inserter
            broll_cuts = []
            for opp in opportunities:
                # Note: The 'path' for the B-roll video will be populated later by the
                # Video Asset Retriever component
                broll_cuts.append({
                    "timestamp": opp.timestamp,
                    "duration": opp.duration,
                    "keywords": opp.keywords,
                    "reason": opp.reason,
                    "transcript_segment": opp.transcript_segment,
                    # Placeholder for the actual B-roll path
                    "path": ""
                })
            
            # Create output structure
            output_data = {
                "broll_cuts": broll_cuts,
                "metadata": {
                    "count": len(opportunities),
                    "detection_version": "1.0"
                }
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                    
            self.logger.info(f"Exported {len(opportunities)} B-roll opportunities to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export opportunities to JSON: {e}")
            return False


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect B-roll opportunities in a video transcript')
    parser.add_argument('--transcript', required=True, help='Path to the transcript JSON file')
    parser.add_argument('--output', required=True, help='Path to save the opportunities JSON file')
    parser.add_argument('--strategy', default='hybrid', choices=['rule_based', 'llm_based', 'hybrid'], 
                        help='Detection strategy to use')
    parser.add_argument('--llm-api-url', help='URL for the LLM API')
    parser.add_argument('--llm-api-key', help='API key for the LLM')
    parser.add_argument('--llm-model', default='gpt-4o', help='Model name to use for the LLM API')  # Add this
    parser.add_argument('--min-separation', type=float, default=4.0, 
                        help='Minimum seconds between B-roll insertions')
    parser.add_argument('--max-opportunities', type=int, default=5, 
                        help='Maximum number of opportunities to return')
    parser.add_argument('--min-duration', type=float, default=1.5, 
                        help='Minimum duration for B-roll (seconds)')
    parser.add_argument('--max-duration', type=float, default=4.0, 
                        help='Maximum duration for B-roll (seconds)')
    parser.add_argument('--cache-dir', default='./cache', help='Directory to cache analysis results')
    parser.add_argument('--force-refresh', action='store_true', help='Force refreshing the analysis')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Load the transcript
        logger.info(f"Loading transcript from {args.transcript}")
        transcript = Transcript.load(args.transcript)
        
        # Initialize the detector
        detector = BRollOpportunityDetector(
            llm_api_url=args.llm_api_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,  # Pass the model name
            min_separation=args.min_separation,
            max_opportunities=args.max_opportunities,
            min_opportunity_duration=args.min_duration,
            max_opportunity_duration=args.max_duration,
            cache_dir=args.cache_dir
)
        
        # Detect opportunities
        logger.info(f"Detecting B-roll opportunities using {args.strategy} strategy")
        opportunities = detector.detect_opportunities(
            transcript,
            strategy=args.strategy,
            force_refresh=args.force_refresh
        )
        
        # Print summary
        logger.info(f"Found {len(opportunities)} B-roll opportunities")
        for i, opp in enumerate(opportunities):
            logger.info(f"Opportunity {i+1}: {opp.timestamp:.2f}s - {opp.duration:.2f}s, Score: {opp.score:.2f}")
            logger.info(f"  Reason: {opp.reason}")
            logger.info(f"  Keywords: {', '.join(opp.keywords)}")
            logger.info(f"  Text: \"{opp.transcript_segment}\"")
        
        # Export to JSON
        detector.export_to_json(opportunities, args.output)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)