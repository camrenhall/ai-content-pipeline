# broll_opportunity_detector.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Import our transcript structures from script_analyzer
from script_analyzer import Transcript, TranscriptSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BRollOpportunity:
    """Represents a detected opportunity for B-roll insertion."""
    timestamp: float
    duration: float
    keywords: List[str]
    description: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "keywords": self.keywords,
            "description": self.description,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BRollOpportunity':
        return cls(
            timestamp=data["timestamp"],
            duration=data["duration"],
            keywords=data["keywords"],
            description=data["description"],
            confidence=data["confidence"]
        )


class BRollOpportunityDetector:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via constructor or OPENAI_API_KEY environment variable.")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    def detect_opportunities(
        self,
        transcript: Transcript,
        min_count: int = 3,
        max_count: int = 5,
        min_duration: float = 2.0,
        max_duration: float = 5.0,
        min_separation: float = 10.0,
        cache_dir: Optional[str] = None,
        force_refresh: bool = False,
        video_path: Optional[str] = None
    ) -> List[BRollOpportunity]:
        opportunities_cache_path = None
        if cache_dir and video_path:
            os.makedirs(cache_dir, exist_ok=True)
            import hashlib
            video_hash = hashlib.md5(video_path.encode()).hexdigest()
            opportunities_cache_path = os.path.join(cache_dir, f"{video_hash}.opportunities.json")

            if not force_refresh and os.path.exists(opportunities_cache_path):
                self.logger.info(f"Loading cached B-roll opportunities from {opportunities_cache_path}")
                try:
                    with open(opportunities_cache_path, 'r') as f:
                        data = json.load(f)
                        return [BRollOpportunity.from_dict(item) for item in data]
                except Exception as e:
                    self.logger.warning(f"Failed to load cached opportunities: {e}")

        transcript_text = transcript.to_llm_format()
        video_duration = transcript.segments[-1].end_seconds if transcript.segments else 0

        prompt = self._generate_prompt(
            transcript_text=transcript_text,
            video_duration=video_duration,
            min_count=min_count,
            max_count=max_count,
            min_duration=min_duration,
            max_duration=max_duration,
            min_separation=min_separation
        )

        self.logger.info("Querying LLM to identify B-roll opportunities")
        opportunities = self._query_llm(prompt)

        if opportunities_cache_path:
            self.logger.info(f"Caching B-roll opportunities to {opportunities_cache_path}")
            with open(opportunities_cache_path, 'w') as f:
                json.dump([o.to_dict() for o in opportunities], f, indent=2)

        return opportunities

    def _generate_prompt(
        self,
        transcript_text: str,
        video_duration: float,
        min_count: int,
        max_count: int,
        min_duration: float,
        max_duration: float,
        min_separation: float
    ) -> str:
        return f"""
You are an expert video editor tasked with identifying the best moments to insert B-roll footage in a video based on its transcript.

VIDEO TRANSCRIPT:
{transcript_text}

Your task is to identify {min_count}-{max_count} optimal moments in this video where B-roll footage would enhance the viewer's understanding, maintain engagement, or illustrate key concepts.

GUIDELINES:
1. Choose moments where the speaker is discussing concrete visual concepts
2. Look for natural pauses, topic transitions, and emphasis points
3. Consider viewer engagement - where might attention be waning?
4. Ensure B-roll opportunities are distributed throughout the video, not clustered
5. Each B-roll should last between {min_duration} and {max_duration} seconds
6. Maintain at least {min_separation} seconds between B-roll segments
7. Avoid cutting during critical explanations where seeing the speaker is important
8. The video is approximately {video_duration:.1f} seconds in total

FOR EACH OPPORTUNITY, PROVIDE:
- TIMESTAMP: The exact timestamp (in seconds) where the B-roll should begin
- DURATION: How long the B-roll should last (in seconds)
- KEYWORDS: 3-5 specific search terms that would return appropriate B-roll footage
- DESCRIPTION: Brief justification explaining why this moment is ideal for B-roll

RESPONSE FORMAT: JSON array of objects with timestamp, duration, keywords, description
"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _query_llm(self, prompt: str) -> List[BRollOpportunity]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional video editor with expertise in B-roll placement."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                n=1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            json_str = json_match.group(1) if json_match else content.strip()

            if json_str.startswith('```') and json_str.endswith('```'):
                json_str = json_str[3:-3].strip()

            data = json.loads(json_str)

            if isinstance(data, dict):
                opportunities_data = data.get("opportunities") or data.get("broll_opportunities") or next(
                    (v for v in data.values() if isinstance(v, list)), None)
                if opportunities_data is None:
                    raise ValueError("Could not find opportunities in LLM response")
            else:
                opportunities_data = data

            return [
                BRollOpportunity.from_dict({**item, "confidence": item.get("confidence", 0.9)})
                for item in opportunities_data
            ]

        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            raise

    def validate_opportunities(
        self,
        opportunities: List[BRollOpportunity],
        video_duration: float,
        min_duration: float = 2.0,
        max_duration: float = 5.0,
        min_separation: float = 10.0
    ) -> List[BRollOpportunity]:
        opportunities.sort(key=lambda o: o.timestamp)
        validated = []
        last_end_time = 0

        for opp in opportunities:
            if opp.timestamp < 0 or opp.timestamp >= video_duration - 5:
                self.logger.warning(f"Skipping opportunity at {opp.timestamp}s - invalid timestamp")
                continue

            if opp.timestamp < last_end_time + min_separation:
                self.logger.warning(f"Skipping opportunity at {opp.timestamp}s - too close to previous B-roll")
                continue

            duration = max(min(opp.duration, max_duration), min_duration)
            end_time = min(opp.timestamp + duration, video_duration - 2)
            adjusted_duration = end_time - opp.timestamp

            if adjusted_duration < min_duration:
                self.logger.warning(f"Skipping opportunity at {opp.timestamp}s - too short")
                continue

            validated.append(BRollOpportunity(
                timestamp=opp.timestamp,
                duration=adjusted_duration,
                keywords=opp.keywords,
                description=opp.description,
                confidence=opp.confidence
            ))

            last_end_time = opp.timestamp + adjusted_duration

        return validated


if __name__ == "__main__":
    import argparse
    from script_analyzer import ScriptAnalyzer, Transcript

    parser = argparse.ArgumentParser(description='Detect B-roll opportunities in a video transcript')
    parser.add_argument('--transcript')
    parser.add_argument('--video')
    parser.add_argument('--openai-key')
    parser.add_argument('--assemblyai-key')
    parser.add_argument('--cache-dir', default='./cache')
    parser.add_argument('--force-refresh', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--min-count', type=int, default=3)
    parser.add_argument('--max-count', type=int, default=5)
    parser.add_argument('--min-duration', type=float, default=2.0)
    parser.add_argument('--max-duration', type=float, default=5.0)
    parser.add_argument('--min-separation', type=float, default=10.0)

    args = parser.parse_args()

    if args.transcript:
        transcript = Transcript.load(args.transcript)
    elif args.video:
        if not args.assemblyai_key:
            parser.error("--assemblyai-key is required when transcribing a video")
        analyzer = ScriptAnalyzer(api_key=args.assemblyai_key)
        transcript = analyzer.analyze(
            args.video,
            cache_dir=args.cache_dir,
            force_refresh=args.force_refresh
        )
    else:
        parser.error("Either --transcript or --video must be provided")

    detector = BRollOpportunityDetector(api_key=args.openai_key)

    opportunities = detector.detect_opportunities(
        transcript=transcript,
        min_count=args.min_count,
        max_count=args.max_count,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_separation=args.min_separation,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
        video_path=args.video
    )

    video_duration = transcript.segments[-1].end_seconds if transcript.segments else 0
    validated_opportunities = detector.validate_opportunities(
        opportunities=opportunities,
        video_duration=video_duration,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_separation=args.min_separation
    )

    print(f"Detected {len(validated_opportunities)} valid B-roll opportunities:")
    for i, opp in enumerate(validated_opportunities):
        print(f"\nOpportunity {i+1}:")
        print(f"  Timestamp: {opp.timestamp:.2f}s")
        print(f"  Duration: {opp.duration:.2f}s")
        print(f"  Keywords: {', '.join(opp.keywords)}")
        print(f"  Description: {opp.description}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump([o.to_dict() for o in validated_opportunities], f, indent=2)
        print(f"\nOpportunities saved to {args.output}")
