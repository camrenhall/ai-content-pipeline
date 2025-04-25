# keyword_extractor.py
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class KeywordSet:
    """Represents a set of keywords for a specific B-roll opportunity."""
    primary: List[str]  # Main keywords to try first
    alternatives: List[List[str]]  # Alternative sets to try if primary fails
    visual_concepts: List[str]  # Specific visual elements to look for
    abstract_concepts: List[str]  # Abstract themes that might be harder to visualize
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EnhancedBRollOpportunity:
    """B-roll opportunity enhanced with optimized keywords."""
    timestamp: float  # Start time in seconds
    duration: float   # Duration in seconds
    score: float      # Confidence/relevance score (0-1)
    original_keywords: List[str]  # Original keywords from detector
    enhanced_keywords: KeywordSet  # Enhanced keyword sets
    reason: str       # Justification for this opportunity
    transcript_segment: str  # The transcript text this relates to
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "score": self.score,
            "keywords": self.enhanced_keywords.primary,  # Use primary keywords as main set
            "alternative_keywords": [alt for alt in self.enhanced_keywords.alternatives],
            "visual_concepts": self.enhanced_keywords.visual_concepts,
            "abstract_concepts": self.enhanced_keywords.abstract_concepts,
            "original_keywords": self.original_keywords,
            "reason": self.reason,
            "transcript_segment": self.transcript_segment
        }


class KeywordExtractor:
    """
    Analyzes B-roll opportunities and generates optimized search terms for video retrieval.
    
    This extractor uses NLP techniques and/or LLM to generate high-quality keywords
    that will lead to relevant and visually appropriate B-roll footage.
    """
    
    def __init__(
        self, 
        llm_api_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o",
        use_llm: bool = True,
        cache_dir: Optional[str] = None,
        max_workers: int = 4
    ):
        """
        Initialize the Keyword Extractor.
        
        Args:
            llm_api_url: URL for the LLM API (if None, will use environment variable)
            llm_api_key: API key for the LLM (if None, will use environment variable)
            llm_model: Model name to use for the LLM API
            use_llm: Whether to use LLM for keyword enhancement
            cache_dir: Directory to cache analysis results
            max_workers: Maximum number of concurrent workers for parallel processing
        """
        self.llm_api_url = llm_api_url or os.environ.get("LLM_API_URL")
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.llm_model = llm_model
        self.use_llm = use_llm
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        
        # Visual concept prioritization - words that are highly visual
        self.visual_priority_terms = [
            "showing", "display", "interface", "screen", "hands", "person", "people",
            "animation", "motion", "movement", "visual", "chart", "graph", "diagram",
            "illustration", "demo", "example", "process", "workflow", "step-by-step",
            "closeup", "view", "perspective", "scene", "setting", "environment",
            "background", "foreground", "silhouette", "shadow", "light", "color",
            "texture", "pattern", "shape", "form", "design", "layout", "composition",
            "zoom", "pan", "track", "dolly", "tilt", "aerial", "drone", "timelapse",
            "slowmotion", "fastmotion", "transition", "fade", "dissolve", "cut",
            "wipe", "montage", "sequence", "series", "collage", "mosaic", "grid"
        ]
        
        # Weak keywords that are too general or abstract alone
        self.weak_keywords = [
            "the", "and", "a", "an", "in", "on", "at", "by", "to", "for", "with", 
            "of", "from", "about", "as", "like", "good", "bad", "best", "worst",
            "great", "terrible", "awesome", "amazing", "interesting", "boring",
            "exciting", "dull", "fun", "sad", "happy", "angry", "frustrated",
            "content", "upset", "pleased", "proud", "ashamed", "guilty", "innocent",
            "responsible", "irresponsible", "smart", "intelligent", "dumb", "stupid",
            "clever", "wise", "foolish", "right", "wrong", "correct", "incorrect",
            "true", "false", "real", "fake", "authentic", "inauthentic", "genuine",
            "ingenuine", "honest", "dishonest", "trustworthy", "untrustworthy",
            "reliable", "unreliable", "consistent", "inconsistent", "stable",
            "unstable", "steady", "unsteady", "sure", "unsure", "certain",
            "uncertain", "definite", "indefinite", "clear", "unclear", "ambiguous",
            "unambiguous", "precise", "imprecise", "accurate", "inaccurate",
            "exact", "inexact", "specific", "general", "detailed", "vague",
            "concrete", "abstract", "wild", "future", "isn't", "what's", "just"
        ]
    
    def process_opportunities(
        self, 
        opportunities_file: str,
        output_file: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[EnhancedBRollOpportunity]:
        """
        Process B-roll opportunities from a JSON file and enhance keywords.
        
        Args:
            opportunities_file: Path to the JSON file with B-roll opportunities
            output_file: Path to save the enhanced opportunities (if None, will not save)
            force_refresh: Whether to force keyword enhancement even if cached results exist
            
        Returns:
            List of EnhancedBRollOpportunity objects
        """
        # Check for cached results
        cache_path = None
        if self.cache_dir and output_file:
            # Create a cache filename based on the opportunities file
            base_name = os.path.splitext(os.path.basename(opportunities_file))[0]
            cache_path = os.path.join(self.cache_dir, f"{base_name}.enhanced_keywords.json")
            
            if not force_refresh and os.path.exists(cache_path):
                self.logger.info(f"Loading cached enhanced keywords from {cache_path}")
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                enhanced_opportunities = self._load_from_json(data)
                return enhanced_opportunities
        
        # Load original opportunities
        with open(opportunities_file, 'r') as f:
            data = json.load(f)
        
        original_opportunities = data.get("broll_cuts", data.get("opportunities", []))
        
        if not original_opportunities:
            self.logger.warning(f"No B-roll opportunities found in {opportunities_file}")
            return []
        
        # Process opportunities in parallel for efficiency
        enhanced_opportunities = self._process_opportunities_parallel(original_opportunities)
        
        # Save results if output file provided
        if output_file:
            self._save_to_json(enhanced_opportunities, output_file)
            
            # Cache the results if needed
            if cache_path:
                self.logger.info(f"Caching enhanced keywords to {cache_path}")
                self._save_to_json(enhanced_opportunities, cache_path)
        
        return enhanced_opportunities
    
    def _process_opportunities_parallel(self, opportunities: List[Dict]) -> List[EnhancedBRollOpportunity]:
        """Process multiple opportunities in parallel using ThreadPoolExecutor."""
        enhanced_opportunities = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_opportunity = {
                executor.submit(self._enhance_keywords_for_opportunity, opp): opp 
                for opp in opportunities
            }
            
            # Process as they complete
            for future in as_completed(future_to_opportunity):
                opportunity = future_to_opportunity[future]
                try:
                    enhanced_opp = future.result()
                    enhanced_opportunities.append(enhanced_opp)
                except Exception as e:
                    self.logger.error(f"Error processing opportunity at {opportunity.get('timestamp')}: {e}")
                    # Create a basic enhanced opportunity with original keywords
                    enhanced_opp = EnhancedBRollOpportunity(
                        timestamp=opportunity.get("timestamp", 0),
                        duration=opportunity.get("duration", 0),
                        score=opportunity.get("score", 0.5),
                        original_keywords=opportunity.get("keywords", []),
                        enhanced_keywords=KeywordSet(
                            primary=opportunity.get("keywords", []),
                            alternatives=[],
                            visual_concepts=[],
                            abstract_concepts=[]
                        ),
                        reason=opportunity.get("reason", ""),
                        transcript_segment=opportunity.get("transcript_segment", "")
                    )
                    enhanced_opportunities.append(enhanced_opp)
        
        # Sort by timestamp
        enhanced_opportunities.sort(key=lambda x: x.timestamp)
        
        return enhanced_opportunities
    
    def _enhance_keywords_for_opportunity(self, opportunity: Dict) -> EnhancedBRollOpportunity:
        """
        Enhance keywords for a single B-roll opportunity.
        
        Args:
            opportunity: Dictionary containing opportunity data
            
        Returns:
            EnhancedBRollOpportunity object with optimized keywords
        """
        timestamp = opportunity.get("timestamp", 0)
        duration = opportunity.get("duration", 0)
        score = opportunity.get("score", 0.5)
        original_keywords = opportunity.get("keywords", [])
        reason = opportunity.get("reason", "")
        transcript_segment = opportunity.get("transcript_segment", "")
        
        self.logger.info(f"Enhancing keywords for opportunity at {timestamp}s")
        
        # Use LLM if available, otherwise fall back to rule-based approach
        if self.use_llm and self.llm_api_url and self.llm_api_key:
            try:
                enhanced_keywords = self._enhance_keywords_llm(
                    original_keywords, 
                    reason, 
                    transcript_segment
                )
            except Exception as e:
                self.logger.warning(f"LLM enhancement failed: {e}. Falling back to rule-based.")
                enhanced_keywords = self._enhance_keywords_rule_based(
                    original_keywords, 
                    reason, 
                    transcript_segment
                )
        else:
            enhanced_keywords = self._enhance_keywords_rule_based(
                original_keywords, 
                reason, 
                transcript_segment
            )
        
        # Create enhanced opportunity
        enhanced_opp = EnhancedBRollOpportunity(
            timestamp=timestamp,
            duration=duration,
            score=score,
            original_keywords=original_keywords,
            enhanced_keywords=enhanced_keywords,
            reason=reason,
            transcript_segment=transcript_segment
        )
        
        return enhanced_opp
    
    def _enhance_keywords_rule_based(
        self, 
        original_keywords: List[str],
        reason: str,
        transcript_segment: str
    ) -> KeywordSet:
        """
        Rule-based approach to enhance keywords.
        
        Args:
            original_keywords: Original keywords from detector
            reason: Justification for this opportunity
            transcript_segment: The transcript text this relates to
            
        Returns:
            KeywordSet object with enhanced keywords
        """
        # Start with original keywords, filter out weak ones
        filtered_keywords = [
            kw for kw in original_keywords 
            if kw.lower() not in self.weak_keywords and len(kw) > 3
        ]
        
        # If we have too few keywords, extract more from reason and transcript
        if len(filtered_keywords) < 3:
            # Extract keywords from reason
            reason_words = reason.lower().replace('.', ' ').replace(',', ' ').split()
            reason_keywords = [
                word for word in reason_words 
                if word not in self.weak_keywords and len(word) > 3
            ]
            
            # Extract keywords from transcript segment
            transcript_words = transcript_segment.lower().replace('.', ' ').replace(',', ' ').split()
            transcript_keywords = [
                word for word in transcript_words 
                if word not in self.weak_keywords and len(word) > 3
            ]
            
            # Combine all sources, removing duplicates
            all_keywords = list(set(filtered_keywords + reason_keywords + transcript_keywords))
            
            # Sort by length (longer words tend to be more specific)
            all_keywords.sort(key=len, reverse=True)
            
            # Take top keywords
            filtered_keywords = all_keywords[:5]
        
        # Identify visual concepts from the reason and transcript
        visual_concepts = []
        for visual_term in self.visual_priority_terms:
            if visual_term in reason.lower():
                # Extract phrases with visual terms (take 3 words before and after)
                words = reason.lower().split()
                for i, word in enumerate(words):
                    if visual_term in word:
                        start = max(0, i-3)
                        end = min(len(words), i+4)
                        phrase = " ".join(words[start:end])
                        if phrase not in visual_concepts:
                            visual_concepts.append(phrase)
        
        # If we don't have enough visual concepts, add some general ones based on keywords
        if len(visual_concepts) < 2:
            for keyword in filtered_keywords:
                visual_concepts.append(f"{keyword} visual")
                visual_concepts.append(f"{keyword} example")
        
        # Generate alternative keyword sets
        alternatives = []
        
        # Alt set 1: Focus on nouns and adjectives
        noun_adj_keywords = filtered_keywords.copy()
        if len(noun_adj_keywords) > 2:
            alternatives.append(noun_adj_keywords)
        
        # Alt set 2: More abstract concepts
        abstract_keywords = []
        if "technology" in reason.lower() or "tech" in reason.lower():
            abstract_keywords.extend(["innovation", "digital", "modern", "advanced"])
        if "future" in reason.lower():
            abstract_keywords.extend(["tomorrow", "beyond", "next generation", "cutting edge"])
        if "AI" in reason or "artificial intelligence" in reason.lower():
            abstract_keywords.extend(["machine learning", "neural network", "algorithm", "data processing"])
        
        if abstract_keywords:
            alternatives.append(abstract_keywords)
        
        # Alt set 3: Combine with "footage" or "video" keywords
        footage_keywords = [f"{kw} footage" for kw in filtered_keywords[:3]]
        alternatives.append(footage_keywords)
        
        # Create final keyword set
        keyword_set = KeywordSet(
            primary=filtered_keywords,
            alternatives=alternatives,
            visual_concepts=visual_concepts[:5],  # Limit to top 5
            abstract_concepts=abstract_keywords
        )
        
        return keyword_set
    
    def _enhance_keywords_llm(
        self, 
        original_keywords: List[str],
        reason: str,
        transcript_segment: str
    ) -> KeywordSet:
        """
        Use LLM to enhance keywords.
        
        Args:
            original_keywords: Original keywords from detector
            reason: Justification for this opportunity
            transcript_segment: The transcript text this relates to
            
        Returns:
            KeywordSet object with enhanced keywords
        """
        try:
            # Format the context for the LLM
            prompt = self._create_llm_prompt(original_keywords, reason, transcript_segment)
            
            # Call the LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Parse LLM response into KeywordSet
            keyword_set = self._parse_llm_response(llm_response)
            
            # If parsing failed, fall back to rule-based approach
            if keyword_set is None:
                self.logger.warning("Failed to parse LLM response. Falling back to rule-based.")
                return self._enhance_keywords_rule_based(original_keywords, reason, transcript_segment)
            
            return keyword_set
            
        except Exception as e:
            self.logger.error(f"LLM-based enhancement failed: {e}")
            raise
    
    def _create_llm_prompt(
        self, 
        original_keywords: List[str],
        reason: str,
        transcript_segment: str
    ) -> str:
        """Create the prompt for the LLM."""
        return f"""
            You are an expert video producer specialized in selecting perfect B-roll footage. Your task is to generate optimal search terms for finding the best B-roll footage on stock video sites like Pexels.

            For context, the original video has this segment: "{transcript_segment}"

            The reason B-roll is needed here is: "{reason}"

            The original keywords detected are: {", ".join(original_keywords)}

            Please generate enhanced keywords that will lead to visually compelling and relevant B-roll footage. The keywords should be specific enough to return good results but not so specific that they return no results.

            Consider the following:
            1. Prioritize visual concepts that can be represented in video
            2. Include concrete nouns and actions rather than abstract concepts
            3. Consider both literal and metaphorical visual representations
            4. Provide multiple alternative keyword sets to try if the primary set doesn't yield good results
            5. Separate purely visual concepts (what the footage literally shows) from abstract concepts (what the footage represents)

            Format your response as valid JSON with this structure:
            {{
              "primary_keywords": ["keyword1", "keyword2", "keyword3"],
              "alternative_keyword_sets": [
                ["alt_set1_kw1", "alt_set1_kw2"],
                ["alt_set2_kw1", "alt_set2_kw2"]
              ],
              "visual_concepts": ["specific visual element 1", "specific visual element 2"],
              "abstract_concepts": ["abstract theme 1", "abstract theme 2"]
            }}

            The primary_keywords should be 3-5 highly specific and effective search terms.
            The alternative_keyword_sets should contain 2-3 alternative approaches if the primary doesn't work.
            The visual_concepts should describe what the footage literally shows.
            The abstract_concepts should express themes or ideas the footage represents.

            All keywords should be optimized for video search on stock footage sites.
            """
    
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
            
            # Add exponential backoff for API rate limiting
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.llm_api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 429:  # Too Many Requests
                        retry_delay *= 2  # Exponential backoff
                        self.logger.warning(f"Rate limited. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                        
                    if response.status_code != 200:
                        raise Exception(f"LLM API error: {response.status_code} - {response.text}")
                        
                    result = response.json()
                    
                    # Extract the content from the response
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return json.loads(content)
                    else:
                        raise Exception("No content in LLM response")
                        
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse LLM response as JSON: {response.text}")
                    raise
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_delay *= 2
                        self.logger.warning(f"API call failed: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise
            
            raise Exception("Max retries exceeded")
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            raise
    
    def _parse_llm_response(self, llm_response: Dict[str, Any]) -> Optional[KeywordSet]:
        """Parse the LLM response into a KeywordSet object."""
        try:
            primary_keywords = llm_response.get("primary_keywords", [])
            alternative_keyword_sets = llm_response.get("alternative_keyword_sets", [])
            visual_concepts = llm_response.get("visual_concepts", [])
            abstract_concepts = llm_response.get("abstract_concepts", [])
            
            # Validate that we have at least primary keywords
            if not primary_keywords:
                self.logger.warning("No primary keywords found in LLM response")
                return None
            
            # Create and return KeywordSet
            return KeywordSet(
                primary=primary_keywords,
                alternatives=alternative_keyword_sets,
                visual_concepts=visual_concepts,
                abstract_concepts=abstract_concepts
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _save_to_json(self, enhanced_opportunities: List[EnhancedBRollOpportunity], output_path: str) -> None:
        """Save enhanced opportunities to a JSON file."""
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Format for the downstream components
            enhanced_broll_cuts = [opp.to_dict() for opp in enhanced_opportunities]
            
            # Create output structure
            output_data = {
                "broll_cuts": enhanced_broll_cuts,
                "metadata": {
                    "count": len(enhanced_opportunities),
                    "enhancement_version": "1.0"
                }
            }
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            self.logger.info(f"Saved {len(enhanced_opportunities)} enhanced opportunities to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save enhanced opportunities to JSON: {e}")
    
    def _load_from_json(self, data: Dict) -> List[EnhancedBRollOpportunity]:
        """Load enhanced opportunities from JSON data."""
        opportunities = []
        
        for opp_data in data.get("broll_cuts", []):
            # Extract keywords
            primary_keywords = opp_data.get("keywords", [])
            alternative_keywords = opp_data.get("alternative_keywords", [])
            visual_concepts = opp_data.get("visual_concepts", [])
            abstract_concepts = opp_data.get("abstract_concepts", [])
            original_keywords = opp_data.get("original_keywords", primary_keywords)
            
            # Create KeywordSet
            keyword_set = KeywordSet(
                primary=primary_keywords,
                alternatives=alternative_keywords,
                visual_concepts=visual_concepts,
                abstract_concepts=abstract_concepts
            )
            
            # Create EnhancedBRollOpportunity
            opportunity = EnhancedBRollOpportunity(
                timestamp=opp_data.get("timestamp", 0),
                duration=opp_data.get("duration", 0),
                score=opp_data.get("score", 0.5),
                original_keywords=original_keywords,
                enhanced_keywords=keyword_set,
                reason=opp_data.get("reason", ""),
                transcript_segment=opp_data.get("transcript_segment", "")
            )
            
            opportunities.append(opportunity)
        
        return opportunities


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance keywords for B-roll opportunities')
    parser.add_argument('--input', required=True, help='Path to the B-roll opportunities JSON file')
    parser.add_argument('--output', required=True, help='Path to save the enhanced opportunities JSON file')
    parser.add_argument('--llm-api-url', help='URL for the LLM API')
    parser.add_argument('--llm-api-key', help='API key for the LLM')
    parser.add_argument('--llm-model', default='gpt-4o', help='Model name to use for the LLM API')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM enhancement and use only rule-based approach')
    parser.add_argument('--cache-dir', default='./cache', help='Directory to cache analysis results')
    parser.add_argument('--force-refresh', action='store_true', help='Force refreshing the enhancement')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of concurrent workers')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Initialize the extractor
        extractor = KeywordExtractor(
            llm_api_url=args.llm_api_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            use_llm=not args.no_llm,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers
        )
        
        # Process opportunities
        logger.info(f"Enhancing keywords for B-roll opportunities in {args.input}")
        enhanced_opportunities = extractor.process_opportunities(
            args.input,
            args.output,
            force_refresh=args.force_refresh
        )
        
        # Print summary
        logger.info(f"Enhanced keywords for {len(enhanced_opportunities)} B-roll opportunities")
        for i, opp in enumerate(enhanced_opportunities):
            logger.info(f"Opportunity {i+1}: {opp.timestamp:.2f}s - {opp.duration:.2f}s")
            logger.info(f"  Original keywords: {', '.join(opp.original_keywords)}")
            logger.info(f"  Enhanced primary keywords: {', '.join(opp.enhanced_keywords.primary)}")
            if opp.enhanced_keywords.alternatives:
                logger.info(f"  Alternative sets: {len(opp.enhanced_keywords.alternatives)}")
            logger.info(f"  Visual concepts: {', '.join(opp.enhanced_keywords.visual_concepts)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)