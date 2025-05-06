# pipeline_orchestrator.py
import argparse
import concurrent.futures
import logging
import os
import sys
import time
import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import hashlib
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("pipeline_orchestrator")

# Import pipeline components
try:
    from script_analyzer import ScriptAnalyzer, Transcript
    from broll_opportunity_detector import BRollOpportunityDetector
    from keyword_extractor import KeywordExtractor
    from video_asset_retriever import VideoAssetRetriever
    from video_portrait_transformer import crop_to_portrait
    from enhanced_broll_inserter import insert_multiple_brolls, save_edited_video
    from post_processing_orchestrator import PostProcessingOrchestrator
    from moviepy import VideoFileClip
except ImportError as e:
    logger.error(f"Failed to import pipeline components: {e}")
    logger.error("Please ensure all pipeline components are installed and in the Python path.")
    sys.exit(1)


class PipelineStepStatus(Enum):
    """Status of a pipeline step execution."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class PipelineStep:
    """Represents a step in the pipeline with metadata."""
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    status: PipelineStepStatus = PipelineStepStatus.NOT_STARTED
    result: Any = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[Exception] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


class PipelineState(Enum):
    """Overall state of the pipeline."""
    INITIALIZING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()


@dataclass
class PipelineContext:
    """Context object passed between pipeline steps."""
    config: Dict[str, Any]
    input_video_path: str
    output_video_path: str
    cache_dir: str
    intermediate_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_cache_path(self, step_name: str, extension: str = "json") -> str:
        """Generate a cache path for a specific step."""
        filename = f"{hashlib.md5(self.input_video_path.encode()).hexdigest()}_{step_name}.{extension}"
        return os.path.join(self.cache_dir, filename)


class ConfigManager:
    """Manages loading and validation of configuration."""
    
    def __init__(self, config_path: str = "config.yaml", profile: str = "default"):
        self.config_path = config_path
        self.profile = profile
        self.config = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with profile selection."""
        try:
            with open(self.config_path, 'r') as f:
                all_configs = yaml.safe_load(f)
                
            # Load default config first
            config = all_configs.get("default", {})
            
            # Override with profile-specific config if available and requested
            if self.profile != "default" and self.profile in all_configs:
                self._deep_update(config, all_configs[self.profile])
            
            # Load secrets from secrets file
            self._load_secrets(config)
                
            # Validate config
            self._validate_config(config)
            
            # Apply environment variable overrides
            self._apply_env_vars(config)
            
            self.config = config
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if (key in base_dict and isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def _load_secrets(self, config: Dict[str, Any]) -> None:
        """Load secrets from properties file and add to config."""
        # Get secrets file path from config or use default
        secrets_file = config.get("secrets_file", "secrets.properties")
        
        # Initialize api_keys dictionary if not present
        if "api_keys" not in config:
            config["api_keys"] = {}
            
        # Try to load secrets
        try:
            # Check if secrets file exists
            if not os.path.exists(secrets_file):
                logger.warning(f"Secrets file not found at {secrets_file}")
                return
                
            # Load secrets from properties file
            secrets = {}
            with open(secrets_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key-value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        secrets[key.strip()] = value.strip()
            
            # Map secrets to API keys
            api_key_mapping = {
                "ASSEMBLYAI_API_KEY": "assemblyai",
                "LLM_API_KEY": "llm",
                "PEXELS_API_KEY": "pexels"
            }
            
            # Add API keys from secrets
            for secret_key, config_key in api_key_mapping.items():
                if secret_key in secrets:
                    config["api_keys"][config_key] = secrets[secret_key]
            
            # Add other secrets/endpoints as needed
            if "LLM_API_ENDPOINT" in secrets:
                config["llm_api_url"] = secrets["LLM_API_ENDPOINT"]
                
            logger.info(f"Loaded secrets from {secrets_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load secrets from {secrets_file}: {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration has required fields."""
        required_fields = [
            "cache_dir",
            "api_keys"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Check API keys
        required_api_keys = ["assemblyai", "llm", "pexels"]
        for key in required_api_keys:
            if key not in config["api_keys"] or not config["api_keys"][key]:
                logger.warning(f"Missing API key for {key}. Some pipeline steps may fail.")
    
    def _apply_env_vars(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides to config."""
        # Override API keys from environment variables
        for key in config.get("api_keys", {}):
            env_var_name = f"{key.upper()}_API_KEY"
            if env_var_name in os.environ:
                config["api_keys"][key] = os.environ[env_var_name]
                
        # Override LLM API URL if specified in environment
        if "LLM_API_ENDPOINT" in os.environ:
            config["llm_api_url"] = os.environ["LLM_API_ENDPOINT"]


class PipelineOrchestrator:
    """
    Main orchestrator class that manages the content pipeline workflow.
    """
    
    def __init__(
        self, 
        input_video_path: str,
        output_video_path: str,
        config_path: str = "config.yaml",
        profile: str = "default",
        start_step: Optional[str] = None,
        end_step: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_refresh: bool = False,
        parallel_execution: bool = False
    ):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.start_step = start_step
        self.end_step = end_step
        self.force_refresh = force_refresh
        self.parallel_execution = parallel_execution
        
        # Initialize config manager and load configuration
        self.config_manager = ConfigManager(config_path, profile)
        self.config = self.config_manager.load_config()
        
        # Set up cache directory
        self.cache_dir = cache_dir or self.config.get("cache_dir", "./cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up pipeline context
        self.context = PipelineContext(
            config=self.config,
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            cache_dir=self.cache_dir
        )
        
        # Initialize pipeline state
        self.state = PipelineState.INITIALIZING
        self.start_time = None
        self.end_time = None
        
        # Define pipeline steps
        self.steps = {}
        self._define_pipeline_steps()
        
        # Setup logging to file
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Set up logging to file."""
        logs_dir = os.path.join(self.cache_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"pipeline_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    def _define_pipeline_steps(self):
        """Define all steps in the pipeline with their dependencies."""
        self.steps = {
            "analyze_script": PipelineStep(
                name="analyze_script",
                function=self._analyze_script,
                dependencies=[]
            ),
            "detect_broll_opportunities": PipelineStep(
                name="detect_broll_opportunities",
                function=self._detect_broll_opportunities,
                dependencies=["analyze_script"]
            ),
            "extract_keywords": PipelineStep(
                name="extract_keywords",
                function=self._extract_keywords,
                dependencies=["detect_broll_opportunities"]
            ),
            "retrieve_videos": PipelineStep(
                name="retrieve_videos",
                function=self._retrieve_videos,
                dependencies=["extract_keywords"]
            ),
            "transform_videos": PipelineStep(
                name="transform_videos",
                function=self._transform_videos,
                dependencies=["retrieve_videos"]
            ),
            "assemble_video": PipelineStep(
                name="assemble_video",
                function=self._assemble_video,
                dependencies=["transform_videos"]
            ),
            "post_process": PipelineStep(
                name="post_process",
                function=self._apply_post_processing,
                dependencies=["assemble_video"]
            )
        }
    
    def _get_execution_order(self) -> List[str]:
        """
        Determine the execution order of pipeline steps based on dependencies
        and start/end step constraints.
        """
        # Topological sort for dependency order
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected at step: {step_name}")
            
            if step_name in visited:
                return
                
            temp_visited.add(step_name)
            
            for dep in self.steps[step_name].dependencies:
                visit(dep)
                
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        # Filter based on start and end steps
        if self.start_step or self.end_step:
            filtered_order = []
            include = False if self.start_step else True
            
            for step_name in order:
                if step_name == self.start_step:
                    include = True
                
                if include:
                    filtered_order.append(step_name)
                
                if step_name == self.end_step:
                    break
                    
            return filtered_order
        
        return order
    
    def _can_execute_step(self, step_name: str) -> bool:
        """Check if a step can be executed based on its dependencies."""
        for dep in self.steps[step_name].dependencies:
            if self.steps[dep].status != PipelineStepStatus.COMPLETED:
                return False
        return True
    
    def _execute_step(self, step_name: str) -> PipelineStepStatus:
        """Execute a single pipeline step with retry logic."""
        step = self.steps[step_name]
        step.status = PipelineStepStatus.IN_PROGRESS
        step.start_time = time.time()
        
        logger.info(f"Starting pipeline step: {step_name}")
        
        # Check cache for step result if not forcing refresh
        cache_hit = False
        if not self.force_refresh:
            try:
                cached_result = self._load_from_cache(step_name)
                if cached_result is not None:
                    step.result = cached_result
                    step.status = PipelineStepStatus.COMPLETED
                    step.end_time = time.time()
                    logger.info(f"Using cached result for step: {step_name}")
                    cache_hit = True
            except Exception as e:
                logger.warning(f"Failed to load cached result for {step_name}: {e}")
        
        # Execute step if not using cached result
        if not cache_hit:
            for attempt in range(step.max_retries):
                if attempt > 0:
                    logger.warning(f"Retrying step {step_name} (attempt {attempt+1}/{step.max_retries})")
                    
                try:
                    result = step.function(self.context)
                    step.result = result
                    step.status = PipelineStepStatus.COMPLETED
                    
                    # Cache the result
                    self._save_to_cache(step_name, result)
                    
                    break
                except Exception as e:
                    step.error = e
                    step.retry_count += 1
                    logger.error(f"Error in step {step_name}: {e}")
                    
                    # Last attempt failed
                    if attempt == step.max_retries - 1:
                        logger.error(f"Step {step_name} failed after {step.max_retries} attempts")
                        step.status = PipelineStepStatus.FAILED
                    
                    # Add backoff between retries
                    if attempt < step.max_retries - 1:
                        backoff_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Waiting {backoff_time}s before retry...")
                        time.sleep(backoff_time)
        
        step.end_time = time.time()
        execution_time = step.execution_time or 0
        
        if step.status == PipelineStepStatus.COMPLETED:
            logger.info(f"Step {step_name} completed in {execution_time:.2f}s")
            # Store result in context
            self.context.intermediate_outputs[step_name] = step.result
        elif step.status == PipelineStepStatus.FAILED:
            logger.error(f"Step {step_name} failed after {execution_time:.2f}s")
        
        return step.status
    
    def _load_from_cache(self, step_name: str) -> Any:
        """Load step result from cache."""
        cache_path = self.context.get_cache_path(step_name)
        
        # Skip if cache file doesn't exist
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Check if cache is valid (input video unchanged)
            if data.get("input_video") != self.input_video_path:
                logger.info(f"Cache for {step_name} is for different input video, ignoring")
                return None
                
            return data.get("result")
        except Exception as e:
            logger.warning(f"Failed to load cache for {step_name}: {e}")
            return None
    
    def _save_to_cache(self, step_name: str, result: Any) -> None:
        """Save step result to cache."""
        cache_path = self.context.get_cache_path(step_name)
        
        try:
            # Create cache data with metadata
            cache_data = {
                "step": step_name,
                "input_video": self.input_video_path,
                "timestamp": time.time(),
                "result": result
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.debug(f"Saved cache for {step_name} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {step_name}: {e}")
    
    def run(self) -> bool:
        """Run the pipeline from start to finish."""
        self.state = PipelineState.RUNNING
        self.start_time = time.time()
        
        logger.info(f"Starting pipeline for video: {self.input_video_path}")
        
        try:
            # Get execution order based on dependencies
            execution_order = self._get_execution_order()
            logger.info(f"Pipeline execution order: {execution_order}")
            
            # Execute steps in order
            if self.parallel_execution:
                self._run_parallel(execution_order)
            else:
                self._run_sequential(execution_order)
            
            # Check if pipeline completed successfully
            success = all(
                self.steps[step_name].status == PipelineStepStatus.COMPLETED
                for step_name in execution_order
            )
            
            if success:
                self.state = PipelineState.COMPLETED
                logger.info("Pipeline completed successfully")
            else:
                self.state = PipelineState.FAILED
                failed_steps = [
                    step_name for step_name in execution_order
                    if self.steps[step_name].status == PipelineStepStatus.FAILED
                ]
                logger.error(f"Pipeline failed. Failed steps: {failed_steps}")
            
            self.end_time = time.time()
            total_time = self.end_time - self.start_time
            logger.info(f"Total pipeline execution time: {total_time:.2f}s")
            
            # Generate execution report
            self._generate_report()
            
            return success
            
        except Exception as e:
            self.state = PipelineState.FAILED
            self.end_time = time.time()
            logger.error(f"Pipeline failed with exception: {e}")
            return False
    
    def _run_sequential(self, execution_order: List[str]) -> None:
        """Run pipeline steps sequentially."""
        for step_name in execution_order:
            status = self._execute_step(step_name)
            
            # Stop execution if a step fails
            if status == PipelineStepStatus.FAILED:
                logger.error(f"Step {step_name} failed, stopping pipeline execution")
                break
    
    def _run_parallel(self, execution_order: List[str]) -> None:
        """Run pipeline steps with parallelism when possible."""
        pending_steps = set(execution_order)
        completed_steps = set()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            
            while pending_steps:
                # Find steps that can be executed
                executable_steps = [
                    step_name for step_name in pending_steps
                    if all(dep in completed_steps for dep in self.steps[step_name].dependencies)
                ]
                
                if not executable_steps:
                    # Check if we're waiting for futures to complete
                    if futures:
                        # Wait for at least one future to complete
                        done, _ = concurrent.futures.wait(
                            futures.values(), 
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        # Process completed futures
                        for future in done:
                            step_name = next(k for k, v in futures.items() if v == future)
                            try:
                                status = future.result()
                                if status == PipelineStepStatus.FAILED:
                                    logger.error(f"Step {step_name} failed in parallel execution")
                            except Exception as e:
                                logger.error(f"Exception in parallel step {step_name}: {e}")
                                self.steps[step_name].status = PipelineStepStatus.FAILED
                            
                            # Remove from futures and pending, add to completed
                            del futures[step_name]
                            pending_steps.remove(step_name)
                            completed_steps.add(step_name)
                    else:
                        # No executable steps and no pending futures - we're stuck
                        logger.error("Parallel execution deadlock detected, cannot proceed")
                        break
                else:
                    # Submit executable steps to the executor
                    for step_name in executable_steps:
                        logger.info(f"Submitting step for parallel execution: {step_name}")
                        futures[step_name] = executor.submit(self._execute_step, step_name)
            
            # Wait for any remaining futures
            if futures:
                concurrent.futures.wait(futures.values())
                
                # Process remaining results
                for step_name, future in futures.items():
                    try:
                        status = future.result()
                    except Exception as e:
                        logger.error(f"Exception in parallel step {step_name}: {e}")
                        self.steps[step_name].status = PipelineStepStatus.FAILED
    
    def _generate_report(self) -> None:
        """Generate an execution report with metrics."""
        report = {
            "pipeline_id": hashlib.md5(f"{self.input_video_path}_{time.time()}".encode()).hexdigest()[:8],
            "input_video": self.input_video_path,
            "output_video": self.output_video_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.end_time - self.start_time if self.end_time and self.start_time else None,
            "status": self.state.name,
            "steps": {}
        }
        
        for step_name, step in self.steps.items():
            report["steps"][step_name] = {
                "status": step.status.name,
                "start_time": step.start_time,
                "end_time": step.end_time,
                "execution_time": step.execution_time,
                "retry_count": step.retry_count,
                "artifacts": step.artifacts
            }
        
        # Save report to file
        report_path = os.path.join(
            self.cache_dir, 
            f"pipeline_report_{report['pipeline_id']}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Pipeline execution report saved to {report_path}")
    
    # Pipeline step implementations
    
    def _analyze_script(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 1: Analyze script and extract transcript."""
        logger.info("Analyzing script from video")
        
        # Get API key from config
        api_key = context.config["api_keys"].get("assemblyai")
        
        # Initialize ScriptAnalyzer
        analyzer = ScriptAnalyzer(api_key=api_key)
        
        # Create transcript cache directory
        transcript_cache_dir = os.path.join(context.cache_dir, "transcripts")
        os.makedirs(transcript_cache_dir, exist_ok=True)
        
        # Analyze video to get transcript
        transcript = analyzer.analyze(
            context.input_video_path,
            cache_dir=transcript_cache_dir,
            force_refresh=self.force_refresh
        )
        
        # Save transcript to a specific file for reference
        transcript_path = os.path.join(
            transcript_cache_dir,
            f"{hashlib.md5(context.input_video_path.encode()).hexdigest()}.transcript.json"
        )
        transcript.save(transcript_path)
        
        # Store artifact path
        self.steps["analyze_script"].artifacts["transcript_path"] = transcript_path
        
        # Return results
        return {
            "transcript": transcript.to_dict(),
            "transcript_path": transcript_path,
            "duration_seconds": transcript.duration_seconds,
            "segment_count": len(transcript.segments)
        }
    
    def _detect_broll_opportunities(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 2: Detect B-roll opportunities in the transcript."""
        logger.info("Detecting B-roll opportunities")
        
        # Get result from previous step
        script_result = context.intermediate_outputs.get("analyze_script")
        if not script_result:
            raise ValueError("Script analysis result not found in context")
        
        # Load transcript from file
        transcript_path = script_result["transcript_path"]
        transcript = Transcript.load(transcript_path)
        
        # Create B-roll opportunities cache directory
        opportunities_cache_dir = os.path.join(context.cache_dir, "opportunities")
        os.makedirs(opportunities_cache_dir, exist_ok=True)
        
        # Get API key from config
        llm_api_key = context.config["api_keys"].get("llm")
        llm_api_url = context.config.get("llm_api_url", "https://api.openai.com/v1/chat/completions")
        
        # Detection strategy from config
        strategy = context.config.get("broll_detection_strategy", "hybrid")
        
        # Initialize detector
        detector = BRollOpportunityDetector(
            llm_api_key=llm_api_key,
            llm_api_url=llm_api_url,
            llm_model=context.config.get("llm_model", "gpt-4o"),
            min_separation=context.config.get("min_broll_separation", 4.0),
            max_opportunities=context.config.get("max_broll_opportunities", 5),
            min_opportunity_duration=context.config.get("min_broll_duration", 1.5),
            max_opportunity_duration=context.config.get("max_broll_duration", 4.0),
            cache_dir=opportunities_cache_dir
        )
        
        # Detect opportunities
        opportunities = detector.detect_opportunities(
            transcript,
            strategy=strategy,
            force_refresh=self.force_refresh
        )
        
        # Save opportunities to file
        opportunities_path = os.path.join(
            opportunities_cache_dir,
            f"{hashlib.md5(context.input_video_path.encode()).hexdigest()}.opportunities.json"
        )
        detector.export_to_json(opportunities, opportunities_path)
        
        # Store artifact path
        self.steps["detect_broll_opportunities"].artifacts["opportunities_path"] = opportunities_path
        
        # Return results
        return {
            "opportunities": [opp.__dict__ for opp in opportunities],
            "opportunities_path": opportunities_path,
            "count": len(opportunities)
        }
    
    def _extract_keywords(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 3: Extract and enhance keywords for B-roll search."""
        logger.info("Extracting and enhancing keywords")
        
        # Get result from previous step
        opportunities_result = context.intermediate_outputs.get("detect_broll_opportunities")
        if not opportunities_result:
            raise ValueError("B-roll opportunities result not found in context")
        
        # Get opportunities file path
        opportunities_path = opportunities_result["opportunities_path"]
        
        # Create keywords cache directory
        keywords_cache_dir = os.path.join(context.cache_dir, "keywords")
        os.makedirs(keywords_cache_dir, exist_ok=True)
        
        # Output path for enhanced keywords
        enhanced_keywords_path = os.path.join(
            keywords_cache_dir,
            f"{hashlib.md5(context.input_video_path.encode()).hexdigest()}.enhanced_keywords.json"
        )
        
        # Get API key from config
        llm_api_key = context.config["api_keys"].get("llm")
        llm_api_url = context.config.get("llm_api_url", "https://api.openai.com/v1/chat/completions")
        
        # Initialize keyword extractor
        extractor = KeywordExtractor(
            llm_api_key=llm_api_key,
            llm_api_url=llm_api_url,
            llm_model=context.config.get("llm_model", "gpt-4o"),
            use_llm=context.config.get("use_llm_for_keywords", True),
            cache_dir=keywords_cache_dir,
            max_workers=context.config.get("keyword_extraction_workers", 4)
        )
        
        # Process opportunities to enhance keywords
        enhanced_opportunities = extractor.process_opportunities(
            opportunities_path,
            enhanced_keywords_path,
            force_refresh=self.force_refresh
        )
        
        # Store artifact path
        self.steps["extract_keywords"].artifacts["enhanced_keywords_path"] = enhanced_keywords_path
        
        # Return results
        return {
            "enhanced_keywords_path": enhanced_keywords_path,
            "count": len(enhanced_opportunities),
            "enhanced_opportunities": [opp.to_dict() for opp in enhanced_opportunities]
        }
    
    def _retrieve_videos(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 4: Retrieve videos for B-roll based on keywords."""
        logger.info("Retrieving videos for B-roll")
        
        # Get result from previous step
        keywords_result = context.intermediate_outputs.get("extract_keywords")
        if not keywords_result:
            raise ValueError("Enhanced keywords result not found in context")
        
        # Get enhanced keywords file path
        enhanced_keywords_path = keywords_result["enhanced_keywords_path"]
        
        # Load enhanced keywords file
        with open(enhanced_keywords_path, 'r') as f:
            enhanced_keywords_data = json.load(f)
        
        # Create videos cache directory
        videos_cache_dir = os.path.join(context.cache_dir, "videos")
        metadata_cache_dir = os.path.join(context.cache_dir, "metadata")
        os.makedirs(videos_cache_dir, exist_ok=True)
        os.makedirs(metadata_cache_dir, exist_ok=True)
        
        # Output path for retrieved videos
        retrieved_videos_path = os.path.join(
            videos_cache_dir,
            f"{hashlib.md5(context.input_video_path.encode()).hexdigest()}.videos.json"
        )
        
        # Get API key from config
        pexels_api_key = context.config["api_keys"].get("pexels")
        
        # Initialize video asset retriever
        retriever = VideoAssetRetriever(
            api_key=pexels_api_key,
            cache_dir=videos_cache_dir,
            metadata_cache_path=metadata_cache_dir,
            min_resolution=context.config.get("min_video_resolution", (1080, 1080)),
            max_results_per_query=context.config.get("max_results_per_query", 5),
            max_concurrent_requests=context.config.get("max_concurrent_requests", 3),
            max_download_retries=context.config.get("max_download_retries", 3)
        )
        
        # Process opportunities to retrieve videos
        retrieved_videos_data = retriever.process_broll_cuts_sync(enhanced_keywords_data)
        
        # Save results to file
        with open(retrieved_videos_path, 'w') as f:
            json.dump(retrieved_videos_data, f, indent=2)
        
        # Store artifact path
        self.steps["retrieve_videos"].artifacts["retrieved_videos_path"] = retrieved_videos_path
        
        # Count successful retrievals
        successful_retrievals = sum(1 for cut in retrieved_videos_data.get("broll_cuts", []) if cut.get("path"))
        
        # Return results
        return {
            "retrieved_videos_path": retrieved_videos_path,
            "total_opportunities": len(retrieved_videos_data.get("broll_cuts", [])),
            "successful_retrievals": successful_retrievals,
            "success_rate": successful_retrievals / len(retrieved_videos_data.get("broll_cuts", [])) if retrieved_videos_data.get("broll_cuts") else 0
        }
    
    def _transform_videos(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 5: Transform retrieved videos to portrait format."""
        logger.info("Transforming videos to portrait format")
        
        # Get result from previous step
        retrieval_result = context.intermediate_outputs.get("retrieve_videos")
        if not retrieval_result:
            raise ValueError("Video retrieval result not found in context")
        
        # Get retrieved videos file path
        retrieved_videos_path = retrieval_result["retrieved_videos_path"]
        
        # Load retrieved videos data
        with open(retrieved_videos_path, 'r') as f:
            retrieved_videos_data = json.load(f)
        
        # Create transformed videos cache directory
        transformed_videos_dir = os.path.join(context.cache_dir, "transformed_videos")
        os.makedirs(transformed_videos_dir, exist_ok=True)
        
        # Output path for transformed videos data
        transformed_videos_path = os.path.join(
            transformed_videos_dir,
            f"{hashlib.md5(context.input_video_path.encode()).hexdigest()}.transformed_videos.json"
        )
        
        # Get transformation parameters from config
        target_width = context.config.get("target_video_width", 1080)
        target_height = context.config.get("target_video_height", 1920)
        
        # Transform all videos
        transformed_broll_cuts = []
        
        for cut in retrieved_videos_data.get("broll_cuts", []):
            # Skip if no video path
            if not cut.get("path") or not os.path.exists(cut.get("path")):
                logger.warning(f"Skipping transformation for cut at {cut.get('timestamp')}s: No video path")
                transformed_broll_cuts.append(cut)
                continue
            
            # Define output path for transformed video
            input_path = cut["path"]
            filename = os.path.basename(input_path)
            base_name, ext = os.path.splitext(filename)
            transformed_path = os.path.join(transformed_videos_dir, f"{base_name}_portrait{ext}")
            
            # Skip transformation if already exists, unless force_refresh is True
            if os.path.exists(transformed_path) and not self.force_refresh:
                logger.info(f"Using existing transformed video: {transformed_path}")
                
                # Update the cut with the transformed path
                transformed_cut = cut.copy()
                transformed_cut["path"] = transformed_path
                transformed_broll_cuts.append(transformed_cut)
                continue
            
            # Transform the video
            logger.info(f"Transforming video: {input_path} -> {transformed_path}")
            try:
                result_path = crop_to_portrait(
                    input_path, 
                    transformed_path, 
                    width=target_width,
                    height=target_height
                )
                
                if result_path:
                    # Update the cut with the transformed path
                    transformed_cut = cut.copy()
                    transformed_cut["path"] = result_path
                    transformed_broll_cuts.append(transformed_cut)
                    logger.info(f"Successfully transformed video to {result_path}")
                else:
                    logger.error(f"Failed to transform video: {input_path}")
                    transformed_broll_cuts.append(cut)  # Keep original
            except Exception as e:
                logger.error(f"Error transforming video {input_path}: {e}")
                transformed_broll_cuts.append(cut)  # Keep original
        
        # Create result data
        transformed_videos_data = {
            "broll_cuts": transformed_broll_cuts,
            "metadata": retrieved_videos_data.get("metadata", {})
        }
        
        # Save to file
        with open(transformed_videos_path, 'w') as f:
            json.dump(transformed_videos_data, f, indent=2)
        
        # Store artifact path
        self.steps["transform_videos"].artifacts["transformed_videos_path"] = transformed_videos_path
        
        # Count successful transformations
        successful_transformations = sum(
            1 for cut in transformed_broll_cuts 
            if cut.get("path") and os.path.exists(cut.get("path")) and "_portrait" in cut.get("path")
        )
        
        # Return results
        return {
            "transformed_videos_path": transformed_videos_path,
            "total_videos": len(transformed_broll_cuts),
            "successful_transformations": successful_transformations,
            "success_rate": successful_transformations / len(transformed_broll_cuts) if transformed_broll_cuts else 0
        }
    
    def _assemble_video(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 6: Assemble final video with B-roll insertions."""
        logger.info("Assembling final video with B-roll insertions")
        
        # Get result from previous step
        transform_result = context.intermediate_outputs.get("transform_videos")
        if not transform_result:
            raise ValueError("Video transformation result not found in context")
        
        # Get transformed videos file path
        transformed_videos_path = transform_result["transformed_videos_path"]
        
        # Load transformed videos data
        with open(transformed_videos_path, 'r') as f:
            transformed_videos_data = json.load(f)
        
        # Get input and output video paths
        input_video_path = context.input_video_path
        output_video_path = context.output_video_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        
        # Extract B-roll cuts with non-empty paths
        broll_cuts = []
        for cut in transformed_videos_data.get("broll_cuts", []):
            if cut.get("path") and os.path.exists(cut.get("path")):
                broll_cuts.append({
                    "path": cut["path"],
                    "timestamp": cut["timestamp"],
                    "duration": cut["duration"]
                })
        
        if not broll_cuts:
            logger.warning("No valid B-roll cuts found for assembly")
            
            # If no B-roll cuts, simply copy the input video to output
            logger.info(f"No B-roll to insert, copying input video to output")
            shutil.copy2(input_video_path, output_video_path)
            
            return {
                "input_video_path": input_video_path,
                "output_video_path": output_video_path,
                "broll_cuts_inserted": 0,
                "success": True
            }
        
        # Insert B-roll clips into main video
        logger.info(f"Inserting {len(broll_cuts)} B-roll clips into main video")
        try:
            # Load main video
            main_video = VideoFileClip(input_video_path)
            
            # Insert B-roll clips
            final_clip = insert_multiple_brolls(input_video_path, broll_cuts)
            
            # Save the final video
            if final_clip:
                save_edited_video(final_clip, output_video_path)
                logger.info(f"Successfully saved assembled video to {output_video_path}")
                
                # Close clips to free resources
                if hasattr(final_clip, 'close'):
                    final_clip.close()
                if hasattr(main_video, 'close'):
                    main_video.close()
                
                return {
                    "input_video_path": input_video_path,
                    "output_video_path": output_video_path,
                    "broll_cuts_inserted": len(broll_cuts),
                    "success": True
                }
            else:
                logger.error("Failed to create the final video")
                return {
                    "input_video_path": input_video_path,
                    "output_video_path": None,
                    "broll_cuts_inserted": 0,
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error assembling video: {e}")
            return {
                "input_video_path": input_video_path,
                "output_video_path": None,
                "broll_cuts_inserted": 0,
                "success": False,
                "error": str(e)
            }
    
    def _apply_post_processing(self, context: PipelineContext) -> Dict[str, Any]:
        """Step 7: Apply post-processing effects to the assembled video."""
        logger.info("Applying post-processing effects")
        
        # Get result from previous step
        assembly_result = context.intermediate_outputs.get("assemble_video")
        if not assembly_result:
            raise ValueError("Video assembly result not found in context")
        
        # Get paths
        input_video_path = assembly_result["output_video_path"]
        if not os.path.exists(input_video_path):
            raise ValueError(f"Assembled video not found at {input_video_path}")
        
        # Get B-roll data path from previous steps
        broll_data_path = context.intermediate_outputs.get("retrieve_videos", {}).get("retrieved_videos_path")
        if not broll_data_path:
            # Try to get from transform_videos step if not in retrieve_videos
            broll_data_path = context.intermediate_outputs.get("transform_videos", {}).get("transformed_videos_path")
        
        if not broll_data_path:
            raise ValueError("B-roll data not found in context")
        
        # Determine output path
        output_video_path = os.path.join(
            os.path.dirname(context.output_video_path),
            f"{Path(context.output_video_path).stem}_post{Path(context.output_video_path).suffix}"
        )
        
        # Create post-processing cache directory
        post_processing_cache_dir = os.path.join(context.cache_dir, "post_processing")
        os.makedirs(post_processing_cache_dir, exist_ok=True)
        
        # Get post-processing configuration
        post_processing_config = context.config.get("post_processing", {})
        
        # Get background music configuration
        bg_music_config = context.config.get("background_music", {})
        
        # Initialize orchestrator with updated configuration
        orchestrator = PostProcessingOrchestrator(
            config=context.config,
            cache_dir=post_processing_cache_dir,
            sound_effects_dir=post_processing_config.get("sound_effects", {}).get(
                "sound_effects_dir", "./assets/sound_effects"
            ),
            music_dir=bg_music_config.get("music_dir", "./assets/background_music")
        )
        
        # Get enabled steps
        enabled_steps = post_processing_config.get("steps", [])
        
        # Get LLM API information from context
        llm_api_key = context.config.get("api_keys", {}).get("llm")
        llm_api_url = context.config.get("llm_api_url")
        
        # Run post-processing
        final_output_path = orchestrator.process(
            input_video_path=input_video_path,
            broll_data_path=broll_data_path,
            output_video_path=output_video_path,
            steps=enabled_steps,
            llm_api_key=llm_api_key,
            llm_api_url=llm_api_url,
            clean_intermediates=True  # Clean up intermediate files after processing
        )
        
        # Store artifact path
        self.steps["post_process"].artifacts["post_processed_video_path"] = final_output_path
        
        # Update the final output video path in context
        context.output_video_path = final_output_path
        
        # Return results
        return {
            "input_video_path": input_video_path,
            "output_video_path": final_output_path,
            "broll_data_path": broll_data_path,
            "applied_steps": enabled_steps,
            "success": os.path.exists(final_output_path)
        }


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Content Pipeline Orchestrator')
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path for output video')
    
    # Optional arguments
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--profile', default='default', help='Config profile to use')
    parser.add_argument('--cache-dir', help='Directory for caching (overrides config)')
    parser.add_argument('--start-step', choices=[
        'analyze_script', 'detect_broll_opportunities', 'extract_keywords',
        'retrieve_videos', 'transform_videos', 'assemble_video'
    ], help='Start pipeline from this step')
    parser.add_argument('--end-step', choices=[
        'analyze_script', 'detect_broll_opportunities', 'extract_keywords',
        'retrieve_videos', 'transform_videos', 'assemble_video'
    ], help='End pipeline at this step')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh and ignore cache')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution where possible')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if input video exists
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)
    
    # Run the pipeline
    try:
        orchestrator = PipelineOrchestrator(
            input_video_path=args.input,
            output_video_path=args.output,
            config_path=args.config,
            profile=args.profile,
            start_step=args.start_step,
            end_step=args.end_step,
            cache_dir=args.cache_dir,
            force_refresh=args.force_refresh,
            parallel_execution=args.parallel
        )
        
        success = orchestrator.run()
        
        if success:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline orchestration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()