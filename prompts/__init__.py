"""
Prompts Module

Provides utility functions to load prompts from markdown files.
"""

from pathlib import Path


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from a markdown file.
    
    Args:
        prompt_name: Name of the prompt file (without .md extension)
        
    Returns:
        The prompt content as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / f"{prompt_name}.md"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    return prompt_file.read_text(encoding='utf-8').strip()
