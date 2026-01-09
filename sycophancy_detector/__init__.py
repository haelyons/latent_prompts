"""
Sycophancy Direction Detector for Llama 3.1 8B

This package provides tools for detecting sycophancy in LLM outputs by extracting
behavioral directions using difference-in-means from contrastive pairs, then
measuring how new prompts and generations project onto those directions.

Based on the "Sycophancy Is Not One Thing" paper methodology.
"""

from .detector import SycophancyDetector
from .direction_extractor import DirectionExtractor
from .prompt_encoder import PromptActivationEncoder
from .hooks import ActivationCapture
from .utils import DEFAULT_CONFIG, load_directions, save_directions

__version__ = "0.1.0"

__all__ = [
    "SycophancyDetector",
    "DirectionExtractor", 
    "PromptActivationEncoder",
    "ActivationCapture",
    "DEFAULT_CONFIG",
    "load_directions",
    "save_directions",
]
