# improved_script_analyzer.py
import os
import assemblyai as aai
from typing import Dict, List, Optional, Union, Tuple
import json
import logging
from dataclasses import dataclass, asdict
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    """Represents a segment of the transcript with timing information."""
    text: str
    start: float  # Start time in milliseconds
    end: float    # End time in milliseconds
    speaker: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert segment to dictionary for JSON serialization."""
        return asdict(self)


class Transcript:
    """A structured representation of a video transcript with utility methods."""
    
    def __init__(self, segments=None, full_text=None):
        self.segments = segments or []
        self._full_text = full_text
        
    @property
    def full_text(self) -> str:
        """Get the full text of the transcript."""
        if self._full_text is None:
            self._full_text = " ".join(segment.text for segment in self.segments)
        return self._full_text
    
    @property
    def duration_seconds(self) -> float:
        """Get total duration of the transcript in seconds."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end / 1000.0
    
    def get_segment_at_time(self, timestamp_seconds: float) -> Optional[TranscriptSegment]:
        """Find the segment that contains the given timestamp (in seconds)."""
        timestamp_ms = timestamp_seconds * 1000.0
        for segment in self.segments:
            if segment.start <= timestamp_ms <= segment.end:
                return segment
        return None
    
    def to_dict(self) -> Dict:
        """Convert transcript to dictionary for JSON serialization."""
        return {
            "full_text": self.full_text,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": {
                "segment_count": len(self.segments),
                "duration_seconds": self.duration_seconds
            }
        }
    
    def to_json(self) -> str:
        """Serialize transcript to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Transcript':
        """Create a Transcript from JSON."""
        data = json.loads(json_str)
        segments = [
            TranscriptSegment(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                speaker=s.get("speaker"),
                confidence=s.get("confidence", 1.0)
            ) for s in data["segments"]
        ]
        transcript = cls()
        transcript.segments = segments
        transcript._full_text = data["full_text"]
        return transcript
    
    def save(self, file_path: str) -> None:
        """Save transcript to a file as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: str) -> 'Transcript':
        """Load transcript from a JSON file."""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())


class ScriptAnalyzer:
    """
    Analyzes video content to extract and structure the script with timing information.
    Uses AssemblyAI for transcription.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ScriptAnalyzer.
        
        Args:
            api_key: AssemblyAI API key (optional, can be set via environment variable)
        """
        # Set API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("AssemblyAI API key is required. Set it via constructor or ASSEMBLYAI_API_KEY environment variable.")
        
        # Initialize AssemblyAI settings
        aai.settings.api_key = self.api_key
        
        self.logger = logging.getLogger(__name__)
    
    def get_cache_path(self, video_path: str, cache_dir: str) -> str:
        """Generate a consistent cache file path for a given video."""
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a filename based on the video path (MD5 hash for uniqueness)
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        
        return os.path.join(cache_dir, f"{video_hash}.transcript.json")
    
    def analyze(self, video_path: str, 
                cache_dir: Optional[str] = None,
                force_refresh: bool = False,
                **kwargs) -> Transcript:
        """
        Analyze a video to extract its script with timing information.
        
        Args:
            video_path: Path to the video file or URL
            cache_dir: Directory to store/load cached transcripts
            force_refresh: Force re-transcription even if cached result exists
            **kwargs: Additional transcription parameters
            
        Returns:
            Transcript object containing the structured script
        """
        # Check for cached transcript
        transcript_cache_path = None
        if cache_dir:
            transcript_cache_path = self.get_cache_path(video_path, cache_dir)
            
            # Check if we have a cached result
            if not force_refresh and os.path.exists(transcript_cache_path):
                self.logger.info(f"Loading cached transcript from {transcript_cache_path}")
                try:
                    return Transcript.load(transcript_cache_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load cached transcript: {e}")
        
        # Configure transcription
        config = self._create_config(**kwargs)
        
        # Start transcription
        self.logger.info(f"Starting transcription of {video_path}")
        try:
            transcript = aai.Transcriber(config=config).transcribe(video_path)
            
            if hasattr(transcript, 'status') and transcript.status == "error":
                raise RuntimeError(f"Transcription failed: {transcript.error}")
                
            # Process the transcript result
            structured_transcript = self._process_transcript_result(transcript)
            
            # Cache the result if needed
            if transcript_cache_path:
                self.logger.info(f"Caching transcript to {transcript_cache_path}")
                structured_transcript.save(transcript_cache_path)
            
            return structured_transcript
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe video: {e}")
            raise
    
    def _create_config(self, **kwargs) -> aai.TranscriptionConfig:
        """Create transcription configuration with sensible defaults."""
        config_params = {
            'speech_model': aai.SpeechModel.best,
            'punctuate': True,
            'format_text': True,
            'dual_channel': False,
            'language_detection': True,
            'entity_detection': True,
            'speaker_labels': kwargs.get('multiple_speakers', False),
        }
        
        # Override defaults with any provided kwargs
        config_params.update({k: v for k, v in kwargs.items() if k in [
            'speech_model', 'punctuate', 'format_text', 'dual_channel', 
            'language_detection', 'entity_detection', 'speaker_labels'
        ]})
        
        return aai.TranscriptionConfig(**config_params)
    
    def _process_transcript_result(self, transcript_result) -> Transcript:
        """
        Process the AssemblyAI transcript result into our structured Transcript object
        with more granular segmentation.
        
        Args:
            transcript_result: AssemblyAI transcript result
            
        Returns:
            Structured Transcript object
        """
        segments = []
        
        # Extract words with timing information
        if hasattr(transcript_result, 'words') and transcript_result.words:
            # Process using word-level timestamps for more accuracy
            current_segment_text = []
            current_segment_start = None
            current_segment_end = None
            current_speaker = None
            
            # Configurable parameters for segmentation
            max_segment_duration = 5000  # Maximum segment duration in milliseconds (5 seconds)
            max_segment_words = 15      # Maximum words per segment
            
            for word in transcript_result.words:
                # Check for segmentation conditions:
                # 1. Natural sentence boundaries
                is_sentence_end = False
                if current_segment_text and current_segment_text[-1].strip().endswith(('.', '!', '?')):
                    is_sentence_end = True
                    
                # 2. Phrase boundaries (commas, semicolons, etc.)
                is_phrase_boundary = False
                if current_segment_text and current_segment_text[-1].strip().endswith((',', ';', ':')):
                    is_phrase_boundary = True
                    
                # 3. Duration thresholds
                exceeds_duration = False
                if (current_segment_start is not None and 
                    word.end - current_segment_start > max_segment_duration):
                    exceeds_duration = True
                    
                # 4. Word count thresholds
                exceeds_word_count = False
                if len(current_segment_text) >= max_segment_words:
                    exceeds_word_count = True
                    
                # 5. Speaker change (if diarization enabled)
                speaker_changed = False
                if hasattr(word, 'speaker') and current_speaker is not None and word.speaker != current_speaker:
                    speaker_changed = True
                    
                # 6. Significant pauses (if present in the data)
                has_significant_pause = False
                if (current_segment_end is not None and 
                    word.start - current_segment_end > 800):  # 800ms pause threshold
                    has_significant_pause = True
                
                # Decide whether to start a new segment
                should_segment = (
                    current_segment_start is None or  # First word
                    is_sentence_end or                # End of sentence
                    speaker_changed or                # Different speaker
                    has_significant_pause or          # Long pause
                    exceeds_duration or               # Segment too long (time)
                    exceeds_word_count or             # Segment too long (words)
                    (is_phrase_boundary and 
                    (len(current_segment_text) > 5 or word.start - current_segment_start > 2000))  # Phrase boundary with sufficient content
                )
                
                if should_segment and current_segment_text:
                    # Save current segment
                    segments.append(TranscriptSegment(
                        text=' '.join(current_segment_text),
                        start=current_segment_start,
                        end=current_segment_end,
                        speaker=current_speaker,
                        confidence=1.0  # AssemblyAI doesn't provide confidence per segment
                    ))
                    
                    # Start a new segment
                    current_segment_text = []
                    current_segment_start = word.start
                    current_speaker = getattr(word, 'speaker', None)
                elif current_segment_start is None:
                    # Initialize first segment
                    current_segment_start = word.start
                    current_speaker = getattr(word, 'speaker', None)
                
                # Add word to current segment
                current_segment_text.append(word.text)
                current_segment_end = word.end
            
            # Add the last segment
            if current_segment_text:
                segments.append(TranscriptSegment(
                    text=' '.join(current_segment_text),
                    start=current_segment_start,
                    end=current_segment_end,
                    speaker=current_speaker,
                    confidence=1.0
                ))
        else:
            # Fallback - just use the full text with estimated timing
            full_text = getattr(transcript_result, 'text', '')
            audio_duration = getattr(transcript_result, 'audio_duration', 0)
            segments.append(TranscriptSegment(
                text=full_text,
                start=0,
                end=audio_duration * 1000 if audio_duration else 0,
                speaker=None,
                confidence=1.0
            ))
        
        transcript = Transcript()
        transcript.segments = segments
        transcript._full_text = getattr(transcript_result, 'text', ' '.join([s.text for s in segments]))
        return transcript


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe a video and output transcript JSON')
    parser.add_argument('video_path', help='Path or URL to the video file')
    parser.add_argument('--api-key', help='AssemblyAI API key (or set ASSEMBLYAI_API_KEY env var)')
    parser.add_argument('--cache-dir', default='./cache', help='Directory to store cached transcripts')
    parser.add_argument('--force-refresh', action='store_true', help='Force re-transcription even if cached')
    parser.add_argument('--output-json', help='Path to save the transcript JSON (defaults to cache dir)')
    parser.add_argument('--multiple-speakers', action='store_true', help='Enable speaker diarization')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize and run analyzer
    analyzer = ScriptAnalyzer(api_key=args.api_key)
    transcript = analyzer.analyze(
        args.video_path,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
        multiple_speakers=args.multiple_speakers
    )
    
    # Print summary
    print(f"Transcript contains {len(transcript.segments)} segments")
    print(f"Total duration: {transcript.duration_seconds:.2f} seconds")
    print(f"Full text length: {len(transcript.full_text)} characters")
    
    # Determine output path (use hash-based cache path by default)
    if args.output_json:
        # User specified output path
        output_path = args.output_json
    else:
        # Use the cache path only
        output_path = analyzer.get_cache_path(args.video_path, args.cache_dir)
    
    # Write transcript to file
    transcript.save(output_path)
    print(f"Transcript JSON saved to {output_path}")