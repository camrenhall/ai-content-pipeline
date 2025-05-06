# script_generator.py
import os
import json
import logging
import argparse
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("script_generator")

class ScriptGenerator:
    """
    Generates engaging scripts for short-form videos from natural language prompts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-4o",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize the ScriptGenerator.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            api_url: API URL for the LLM
            model: Model to use (default is GPT-4o)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via constructor or OPENAI_API_KEY environment variable.")
        
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_script(self, prompt: str, duration: int = 20) -> str:
        """
        Generate an engaging script from a natural language prompt.
        
        Args:
            prompt: Natural language prompt describing the video topic
            duration: Target duration in seconds (10, 15, 20, 25, 30, 35, or 40)
            
        Returns:
            Script text
        """
        logger.info(f"Generating {duration}s script for prompt: '{prompt}'")
        
        # Ensure valid duration
        valid_durations = [10, 15, 20, 25, 30, 35, 40]
        if duration not in valid_durations:
            closest_duration = min(valid_durations, key=lambda x: abs(x - duration))
            logger.warning(f"Invalid duration {duration}s, using closest valid duration: {closest_duration}s")
            duration = closest_duration
        
        # Construct the system prompt for the LLM
        system_prompt = self._create_system_prompt(duration)
        
        # Call the LLM API
        response = self._call_llm_api(system_prompt, prompt)
        
        # Return the script text directly
        return response
    
    def _create_system_prompt(self, duration: int) -> str:
        """Create the system prompt for the LLM based on the desired duration."""
        
        # Adjust content strategy based on duration
        content_strategy = ""
        if duration <= 15:
            content_strategy = """
            For ULTRA-SHORT videos (10-15 seconds):
            - Focus on one powerful hook or surprising fact
            - Create immediate curiosity in the first 1-2 seconds
            - End with a clear, simple call-to-action
            - Make just ONE point extremely well
            """
        elif duration <= 25:
            content_strategy = """
            For SHORT videos (20-25 seconds):
            - Start with a powerful hook (first 2-3 seconds)
            - Present 1-2 key points that deliver on the hook's promise
            - Use simple, direct language with strong emotional appeal
            - End with a clear call-to-action that creates engagement
            """
        else:
            content_strategy = """
            For STANDARD videos (30-40 seconds):
            - Open with an irresistible hook (first 3 seconds)
            - Follow a structure: hook → problem → solution → call-to-action
            - Include 2-3 memorable points with supporting evidence
            - Create urgency in the call-to-action
            """
        
        system_prompt = f"""You are an expert short-form video script writer. Your task is to create an engaging script based on the user's prompt.

TARGET DETAILS:
- Duration: EXACTLY {duration} seconds
- Format: Vertical short-form video

{content_strategy}

ELEMENTS TO INCLUDE:
1. A strong opening hook that grabs attention immediately
2. Clear, simple language that works well when spoken aloud
3. A conversational, authentic tone
4. A clear call-to-action at the end

IMPORTANT: Generate ONLY the script text that would be spoken in the video. Do not include any headers, formatting, metadata, or instructions. The response should be PURELY the script, ready to be read aloud.

The script should sound natural and conversational, exactly as it would be spoken by a real person.
"""
        return system_prompt
    
    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM API with the system and user prompts.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            LLM response text
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                raise Exception(f"LLM API error: {response.status_code}")
                
            result = response.json()
            
            # Extract the content from the response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content.strip()
            else:
                logger.error("No content in LLM response")
                raise Exception("No content in LLM response")
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise

    def save_script_to_file(self, script: str, output_path: str) -> None:
        """
        Save the script to a file.
        
        Args:
            script: Script text
            output_path: Path to save the script
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write the script to file
            with open(output_path, 'w') as f:
                f.write(script)
                
            logger.info(f"Saved script to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save script to file: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a script for a short-form video')
    parser.add_argument('prompt', help='Natural language prompt describing the video topic')
    parser.add_argument('--output', required=True, help='Path to save the script file')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--duration', type=int, default=20, choices=[10, 15, 20, 25, 30, 35, 40], 
                        help='Target duration in seconds (default: 20)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Initialize the script generator
        generator = ScriptGenerator(api_key=args.api_key)
        
        # Generate script
        script = generator.generate_script(
            prompt=args.prompt,
            duration=args.duration
        )
        
        # Save script to file
        generator.save_script_to_file(script, args.output)
        
        # Print brief summary
        print(f"Script generated successfully!")
        print(f"Script saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        import sys
        sys.exit(1)