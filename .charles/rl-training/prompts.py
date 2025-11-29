"""
Prompt Configuration Loader for Agent0.

Loads prompt presets from prompts.toml for curriculum and executor agents.

Paper Reference:
- Table 6: Executor Agent prompt template
- Table 7: Curriculum Agent prompt template  
- Table 8: Hyperparameters including prompt configurations

Usage:
    from prompts import load_prompts, get_curriculum_prompts, get_executor_prompts
    
    # Load specific preset
    prompts = load_prompts("data_scientist")
    
    # Or use environment variable
    prompts = load_prompts()  # Uses PROMPT_PRESET from .env
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python

from loguru import logger


@dataclass
class PromptConfig:
    """
    Configuration for a prompt preset.
    
    Attributes:
        name: Human-readable name of the preset
        description: Description of what this preset trains for
        curriculum_system: System prompt for curriculum agent
        curriculum_user: User prompt template for curriculum agent
        executor_system: System prompt for executor agent
        executor_user: User prompt template for executor agent (contains {problem})
    """
    name: str
    description: str
    curriculum_system: str
    curriculum_user: str
    executor_system: str
    executor_user: str


# Default prompts file location
PROMPTS_FILE = Path(__file__).parent / "prompts.toml"


def load_prompts_toml(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the prompts.toml configuration file.
    
    Args:
        path: Path to prompts.toml (default: same directory as this file)
    
    Returns:
        Parsed TOML as dictionary
    
    Raises:
        FileNotFoundError: If prompts.toml doesn't exist
    """
    path = path or PROMPTS_FILE
    
    if not path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {path}\n"
            f"Please create prompts.toml with prompt configurations."
        )
    
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_available_presets(path: Optional[Path] = None) -> list[str]:
    """
    Get list of available prompt presets.
    
    Returns:
        List of preset names (e.g., ['math', 'data_scientist', 'coding'])
    """
    config = load_prompts_toml(path)
    return list(config.keys())


def load_prompts(
    preset: Optional[str] = None,
    path: Optional[Path] = None,
) -> PromptConfig:
    """
    Load a prompt preset configuration.
    
    Args:
        preset: Name of preset to load (e.g., 'math', 'data_scientist')
                If None, uses PROMPT_PRESET environment variable
        path: Path to prompts.toml
    
    Returns:
        PromptConfig with all prompts for the preset
    
    Raises:
        ValueError: If preset doesn't exist
    
    Example:
        >>> prompts = load_prompts("data_scientist")
        >>> print(prompts.curriculum_system[:50])
        'You are an expert problem setter for training AI...'
    """
    # Get preset from env if not specified
    if preset is None:
        preset = os.getenv("PROMPT_PRESET", "math")
    
    config = load_prompts_toml(path)
    
    if preset not in config:
        available = list(config.keys())
        raise ValueError(
            f"Unknown prompt preset: '{preset}'\n"
            f"Available presets: {available}"
        )
    
    preset_config = config[preset]
    
    prompt_config = PromptConfig(
        name=preset_config.get("name", preset),
        description=preset_config.get("description", ""),
        curriculum_system=preset_config["curriculum"]["system"],
        curriculum_user=preset_config["curriculum"]["user"],
        executor_system=preset_config["executor"]["system"],
        executor_user=preset_config["executor"]["user"],
    )
    
    logger.info(f"📝 Loaded prompt preset: {prompt_config.name}")
    logger.debug(f"   Description: {prompt_config.description}")
    
    return prompt_config


def get_curriculum_prompts(
    preset: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Get curriculum agent prompts (system, user).
    
    Convenience function for curriculum training.
    
    Args:
        preset: Prompt preset name (or uses PROMPT_PRESET env var)
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    config = load_prompts(preset)
    return config.curriculum_system, config.curriculum_user


def get_executor_prompts(
    preset: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Get executor agent prompts (system, user).
    
    The user prompt contains {problem} placeholder.
    
    Args:
        preset: Prompt preset name (or uses PROMPT_PRESET env var)
    
    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    config = load_prompts(preset)
    return config.executor_system, config.executor_user


def main():
    """Test prompt loading."""
    print("=" * 60)
    print("Agent0 Prompt Configuration Test")
    print("=" * 60)
    
    # List available presets
    presets = get_available_presets()
    print(f"\n📋 Available presets: {presets}")
    
    # Load each preset and show preview
    for preset in presets:
        print(f"\n{'='*60}")
        print(f"🎯 Preset: {preset}")
        print("=" * 60)
        
        config = load_prompts(preset)
        print(f"   Name: {config.name}")
        print(f"   Description: {config.description}")
        print(f"\n   Curriculum System (first 200 chars):")
        print(f"   {config.curriculum_system[:200]}...")
        print(f"\n   Executor System (first 200 chars):")
        print(f"   {config.executor_system[:200]}...")
    
    print("\n✅ Prompt loading test complete!")


if __name__ == "__main__":
    main()
