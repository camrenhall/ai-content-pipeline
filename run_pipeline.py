#!/usr/bin/env python3
"""
Helper script for running the content pipeline with common configurations.
This script provides a simpler interface to the pipeline orchestrator.
"""

import os
import sys
import argparse
import logging
from typing import Optional, List
import time
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("run_pipeline")


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    required_modules = [
        "script_analyzer", 
        "broll_opportunity_detector",
        "keyword_extractor", 
        "video_asset_retriever", 
        "video_portrait_transformer",
        "enhanced_broll_inserter",
        "moviepy"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        return False
    
    return True


def check_api_keys() -> bool:
    """Check if API keys are set in environment variables."""
    required_keys = [
        "ASSEMBLYAI_API_KEY",
        "LLM_API_KEY",
        "PEXELS_API_KEY"
    ]
    
    missing_keys = []
    
    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
    
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        logger.warning("Set these environment variables or configure them in config.yaml")
        return False
    
    return True


def run_pipeline(
    input_path: str,
    output_path: str,
    profile: str = "default",
    cache_dir: Optional[str] = None,
    start_step: Optional[str] = None,
    end_step: Optional[str] = None,
    force_refresh: bool = False,
    parallel: bool = False,
    verbose: bool = False
) -> int:
    """
    Run the content pipeline with the specified parameters.
    
    Args:
        input_path: Path to the input video file
        output_path: Path where the output video will be saved
        profile: Configuration profile to use
        cache_dir: Custom cache directory (overrides profile setting)
        start_step: Start pipeline from this step
        end_step: End pipeline at this step
        force_refresh: Force refresh and ignore cache
        parallel: Enable parallel execution where possible
        verbose: Enable verbose logging
        
    Returns:
        Return code (0 for success, non-zero for failure)
    """
    # Build command
    cmd = [
        "python", "pipeline_orchestrator.py",
        "--input", input_path,
        "--output", output_path,
        "--profile", profile
    ]
    
    # Add optional arguments
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    if start_step:
        cmd.extend(["--start-step", start_step])
    if end_step:
        cmd.extend(["--end-step", end_step])
    if force_refresh:
        cmd.append("--force-refresh")
    if parallel:
        cmd.append("--parallel")
    if verbose:
        cmd.append("--verbose")
    
    # Print command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Start timer
    start_time = time.time()
    
    # Run the pipeline
    result = subprocess.run(cmd)
    
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Log result
    if result.returncode == 0:
        logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
    else:
        logger.error(f"Pipeline failed with return code {result.returncode}")
    
    return result.returncode


def setup_environment() -> None:
    """Set up additional environment settings if needed."""
    # You can add environment variable setup here if needed
    pass


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Run Content Pipeline with common configurations')
    
    # Input/output arguments
    parser.add_argument('input', help='Path to input video')
    parser.add_argument('output', help='Path for output video')
    
    # Pipeline configuration
    parser.add_argument('--profile', default='default', 
                       choices=['default', 'development', 'production', 'high_quality', 'fast'],
                       help='Configuration profile to use')
    parser.add_argument('--cache-dir', help='Custom cache directory')
    
    # Pipeline execution
    parser.add_argument('--start-step', 
                       choices=['analyze_script', 'detect_broll_opportunities', 
                                'extract_keywords', 'retrieve_videos', 
                                'transform_videos', 'assemble_video'],
                       help='Start pipeline from this step')
    parser.add_argument('--end-step', 
                       choices=['analyze_script', 'detect_broll_opportunities', 
                                'extract_keywords', 'retrieve_videos', 
                                'transform_videos', 'assemble_video'],
                       help='End pipeline at this step')
    
    # Execution options
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh and ignore cache')
    parser.add_argument('--parallel', action='store_true', 
                       help='Enable parallel execution where possible')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    # Check dependencies
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check dependencies if requested
    if args.check_deps:
        deps_ok = check_dependencies()
        keys_ok = check_api_keys()
        
        if deps_ok and keys_ok:
            logger.info("All dependencies and API keys are properly configured")
            return 0
        else:
            return 1
    
    # Check if input video exists
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Run the pipeline
    return run_pipeline(
        input_path=args.input,
        output_path=args.output,
        profile=args.profile,
        cache_dir=args.cache_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        force_refresh=args.force_refresh,
        parallel=args.parallel,
        verbose=args.verbose
    )


if __name__ == "__main__":
    result = main()
    sys.exit(result)