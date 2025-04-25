# Automated Content Pipeline

An end-to-end pipeline for automatically enhancing videos with context-appropriate B-roll footage.

## Overview

This pipeline automates the process of analyzing a video, identifying opportunities for B-roll insertion, finding relevant B-roll footage, and assembling a final enhanced video. The system is composed of modular components orchestrated by a central pipeline manager.

### Pipeline Components

1. **Script Analyzer** - Extracts and processes the script from the video
2. **B-Roll Opportunity Detector** - Identifies optimal points for B-roll insertion
3. **Keyword Extractor** - Generates relevant search terms for B-roll content
4. **Video Asset Retriever** - Interfaces with Pexels API to find and download videos
5. **Video Transformer** - Converts videos to required format (portrait mode)
6. **Video Assembler** - Combines main video with B-roll clips
7. **Pipeline Orchestrator** - Manages the overall workflow and error handling

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg installed and available in your PATH
- API keys for:
  - AssemblyAI (for transcription)
  - OpenAI or another LLM provider (for analysis)
  - Pexels (for video retrieval)

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/content-pipeline.git
   cd content-pipeline
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys in `secrets.properties`
   ```properties
   # secrets.properties
   ASSEMBLYAI_API_KEY=your_assemblyai_key_here
   LLM_API_KEY=your_openai_key_here
   PEXELS_API_KEY=your_pexels_api_key_here
   ```
   
   Alternatively, you can set these as environment variables:
   ```bash
   export ASSEMBLYAI_API_KEY="your_assemblyai_key"
   export LLM_API_KEY="your_openai_key"
   export PEXELS_API_KEY="your_pexels_key"
   ```
   
   ⚠️ **Important:** Add `secrets.properties` to your `.gitignore` file to prevent committing sensitive information to your repository.

## Usage

### Basic Usage

The simplest way to run the pipeline is using the `run_pipeline.py` script:

```bash
python run_pipeline.py input_video.mp4 output_video.mp4
```

### Advanced Usage

For more control, you can use the orchestrator directly:

```bash
python pipeline_orchestrator.py --input input_video.mp4 --output output_video.mp4 --profile high_quality
```

### Configuration Profiles

The pipeline supports different configuration profiles:

- `default` - Standard settings for most videos
- `development` - Optimized for faster testing
- `production` - More reliable settings for production use
- `high_quality` - Maximum quality settings
- `fast` - Optimized for speed over quality

Select a profile using the `--profile` argument:

```bash
python run_pipeline.py input_video.mp4 output_video.mp4 --profile high_quality
```

### Partial Pipeline Execution

You can run only specific parts of the pipeline:

```bash
# Run from the beginning up to keyword extraction
python run_pipeline.py input_video.mp4 output_video.mp4 --end-step extract_keywords

# Start from video retrieval to the end
python run_pipeline.py input_video.mp4 output_video.mp4 --start-step retrieve_videos

# Run just a specific section
python run_pipeline.py input_video.mp4 output_video.mp4 --start-step transform_videos --end-step assemble_video
```

### Additional Options

- `--force-refresh`: Ignore cache and reprocess all steps
- `--parallel`: Enable parallel execution where possible
- `--verbose`: Show detailed logging information
- `--cache-dir`: Specify a custom cache directory

## Configuration

### Main Configuration

The pipeline is configured through `config.yaml`. You can modify this file to:

- Configure LLM models and endpoints
- Adjust B-roll detection parameters
- Set video quality settings
- Control retry behavior
- Define custom profiles

See the comments in `config.yaml` for details on each setting.

### Secrets Management

Sensitive information like API keys is stored separately in `secrets.properties` to avoid accidentally committing credentials to version control.

**Secrets File Format:**
```properties
# API Keys
ASSEMBLYAI_API_KEY=your_assemblyai_key_here
LLM_API_KEY=your_openai_key_here
PEXELS_API_KEY=your_pexels_api_key_here

# Optional service endpoints
LLM_API_ENDPOINT=https://api.openai.com/v1/chat/completions
```

**Security Best Practices:**
1. Never commit `secrets.properties` to version control
2. Add `secrets.properties` to your `.gitignore` file
3. Consider using environment variables in production environments
4. For team environments, use a secure method to share secrets (like a password manager)

## Pipeline Workflow

1. **Script Analysis**
   - Transcribes the video using AssemblyAI
   - Segments the transcript with timing information

2. **B-Roll Opportunity Detection**
   - Analyzes the transcript to find ideal moments for B-roll
   - Uses a hybrid approach combining rule-based heuristics and LLM analysis

3. **Keyword Extraction**
   - Enhances the detected opportunities with optimized search terms
   - Generates multiple keyword sets for each opportunity

4. **Video Asset Retrieval**
   - Searches Pexels for relevant B-roll footage
   - Downloads and caches videos for future use

5. **Video Transformation**
   - Converts landscape videos to portrait format
   - Optimizes for target resolution

6. **Video Assembly**
   - Inserts B-roll clips at the right moments
   - Preserves original audio throughout

## Monitoring and Logging

The pipeline creates detailed logs and reports:

- Runtime logs in the `logs` directory under your cache
- Execution reports with timing information for each step
- Cached intermediate outputs for debugging

## Troubleshooting

If you encounter issues:

1. Check the logs in `cache/logs/`
2. Ensure all API keys are correctly set
3. Verify FFmpeg is installed and available
4. Try running with `--verbose` for more detailed logs
5. Run with `--force-refresh` if cache corruption is suspected

## License

[MIT License](LICENSE)