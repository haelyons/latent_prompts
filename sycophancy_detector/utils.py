"""
Utility functions for sycophancy detection.

Provides configuration, direction caching, and loading helpers.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


# Default configuration
DEFAULT_CONFIG = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "layers": [12, 14, 16, 18, 20],
    "canonical_layer": 16,  # Middle of optimal range for Llama 3.1 8B
    "behaviors": ["syc", "ga", "pr"],
    "behavior_names": {
        "syc": "sycophantic_agreement",
        "ga": "genuine_agreement", 
        "pr": "sycophantic_praise",
    },
    "pool_strategy": "last",  # EOS token position
    "factorial_datasets": [
        "math_factorial.json",
        "math_factorial_small.json",
        "claims_factorial.json",
        "cities_pos_factorial.json",
        "cities_neg_factorial.json",
        "larger_than_factorial.json",
        "smaller_than_factorial.json",
        "counterfactual_factorial.json",
        "companies_factorial.json",
    ],
    "direction_cache_dir": "out/directions/",
    "batch_size": 8,
    "max_length": 512,
}


@dataclass
class DirectionMetadata:
    """Metadata for a saved direction vector."""
    behavior: str
    layer: int
    pos_count: int
    neg_count: int
    pos_mean_norm: float
    neg_mean_norm: float
    direction_norm: float
    datasets_used: List[str]
    model_id: str
    

def get_direction_path(
    cache_dir: str,
    behavior: str,
    layer: int,
) -> str:
    """
    Get the standard path for a direction file.
    
    Following the paper's convention: wDiffMean_raw_{behavior}_L{layer}.npy
    """
    return os.path.join(cache_dir, f"wDiffMean_raw_{behavior}_L{layer}.npy")


def get_metadata_path(
    cache_dir: str,
    behavior: str,
    layer: int,
) -> str:
    """Get path for direction metadata JSON."""
    return os.path.join(cache_dir, f"wDiffMean_raw_{behavior}_L{layer}_meta.json")


def save_direction(
    direction: np.ndarray,
    cache_dir: str,
    behavior: str,
    layer: int,
    metadata: Optional[DirectionMetadata] = None,
) -> str:
    """
    Save a direction vector to disk.
    
    Args:
        direction: Direction vector (D,)
        cache_dir: Directory to save to
        behavior: Behavior name
        layer: Layer index
        metadata: Optional metadata to save
        
    Returns:
        Path to saved file
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save direction
    path = get_direction_path(cache_dir, behavior, layer)
    np.save(path, direction)
    
    # Save metadata if provided
    if metadata is not None:
        meta_path = get_metadata_path(cache_dir, behavior, layer)
        with open(meta_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
    
    return path


def load_direction(
    cache_dir: str,
    behavior: str,
    layer: int,
    return_metadata: bool = False,
) -> np.ndarray:
    """
    Load a direction vector from disk.
    
    Args:
        cache_dir: Directory containing directions
        behavior: Behavior name
        layer: Layer index
        return_metadata: Whether to also return metadata
        
    Returns:
        Direction vector (D,), or tuple (direction, metadata) if return_metadata
    """
    path = get_direction_path(cache_dir, behavior, layer)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Direction not found: {path}")
    
    direction = np.load(path)
    
    if return_metadata:
        meta_path = get_metadata_path(cache_dir, behavior, layer)
        metadata = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = DirectionMetadata(**json.load(f))
        return direction, metadata
    
    return direction


def load_directions(
    cache_dir: str,
    behaviors: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load multiple direction vectors.
    
    Args:
        cache_dir: Directory containing directions
        behaviors: List of behaviors to load (default: from config)
        layers: List of layers to load (default: from config)
        
    Returns:
        Nested dict: behavior -> layer -> direction
    """
    behaviors = behaviors or DEFAULT_CONFIG["behaviors"]
    layers = layers or DEFAULT_CONFIG["layers"]
    
    result = {}
    
    for behavior in behaviors:
        result[behavior] = {}
        for layer in layers:
            try:
                direction = load_direction(cache_dir, behavior, layer)
                result[behavior][layer] = direction
            except FileNotFoundError:
                pass  # Skip missing directions
    
    return result


def save_directions(
    directions: Dict[str, Dict[int, np.ndarray]],
    cache_dir: str,
) -> List[str]:
    """
    Save multiple direction vectors.
    
    Args:
        directions: Nested dict behavior -> layer -> direction
        cache_dir: Directory to save to
        
    Returns:
        List of saved file paths
    """
    saved = []
    
    for behavior, layer_dirs in directions.items():
        for layer, direction in layer_dirs.items():
            path = save_direction(direction, cache_dir, behavior, layer)
            saved.append(path)
    
    return saved


def directions_exist(
    cache_dir: str,
    behaviors: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
) -> bool:
    """
    Check if all required direction files exist.
    
    Args:
        cache_dir: Directory to check
        behaviors: Behaviors to check for
        layers: Layers to check for
        
    Returns:
        True if all directions exist
    """
    behaviors = behaviors or DEFAULT_CONFIG["behaviors"]
    layers = layers or DEFAULT_CONFIG["layers"]
    
    for behavior in behaviors:
        for layer in layers:
            path = get_direction_path(cache_dir, behavior, layer)
            if not os.path.exists(path):
                return False
    
    return True


def list_available_directions(cache_dir: str) -> List[Dict[str, Any]]:
    """
    List all available direction files in a directory.
    
    Returns:
        List of dicts with behavior, layer, and path info
    """
    if not os.path.exists(cache_dir):
        return []
    
    available = []
    
    for filename in os.listdir(cache_dir):
        if filename.startswith("wDiffMean_raw_") and filename.endswith(".npy"):
            # Parse: wDiffMean_raw_{behavior}_L{layer}.npy
            parts = filename[len("wDiffMean_raw_"):-len(".npy")]
            # Find the _L{number} suffix
            if "_L" in parts:
                idx = parts.rfind("_L")
                behavior = parts[:idx]
                try:
                    layer = int(parts[idx+2:])
                    available.append({
                        "behavior": behavior,
                        "layer": layer,
                        "path": os.path.join(cache_dir, filename),
                    })
                except ValueError:
                    pass
    
    return sorted(available, key=lambda x: (x["behavior"], x["layer"]))


def get_factorial_data_dir() -> str:
    """
    Get the path to the factorial data directory.
    
    Looks for the disentangle-sycophancy submodule.
    """
    # Try relative to this file
    this_dir = Path(__file__).parent.parent
    
    candidates = [
        this_dir / "disentangle-sycophancy" / "data" / "factorial",
        this_dir / ".." / "disentangle-sycophancy" / "data" / "factorial",
        Path("disentangle-sycophancy") / "data" / "factorial",
    ]
    
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    
    raise FileNotFoundError(
        "Could not find factorial data directory. "
        "Ensure disentangle-sycophancy submodule is checked out."
    )


def get_factorial_dataset_paths(
    data_dir: Optional[str] = None,
    datasets: Optional[List[str]] = None,
) -> List[str]:
    """
    Get full paths to factorial dataset files.
    
    Args:
        data_dir: Base directory (default: auto-detect)
        datasets: List of dataset filenames (default: from config)
        
    Returns:
        List of existing dataset paths
    """
    if data_dir is None:
        data_dir = get_factorial_data_dir()
    
    datasets = datasets or DEFAULT_CONFIG["factorial_datasets"]
    
    paths = []
    for ds in datasets:
        path = os.path.join(data_dir, ds)
        if os.path.exists(path):
            paths.append(path)
    
    return paths


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    
    return float(np.dot(a, b) / (a_norm * b_norm))


def cosine_similarity_batch(
    vectors: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between batch of vectors and a direction.
    
    Args:
        vectors: (N, D) array
        direction: (D,) array
        
    Returns:
        (N,) array of similarities
    """
    # Normalize vectors
    vec_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vec_norms = np.clip(vec_norms, 1e-8, None)
    vectors_normed = vectors / vec_norms
    
    # Normalize direction
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-8:
        return np.zeros(vectors.shape[0])
    direction_normed = direction / dir_norm
    
    # Dot product
    return vectors_normed @ direction_normed


def format_scores(scores: Dict[str, float], precision: int = 4) -> str:
    """Format score dict for display."""
    parts = []
    for k, v in sorted(scores.items()):
        parts.append(f"{k}: {v:.{precision}f}")
    return ", ".join(parts)


# Llama 3.1 8B specific constants
LLAMA_31_8B_CONFIG = {
    "num_layers": 32,
    "hidden_size": 4096,
    "optimal_layers": list(range(12, 21)),  # 12-20 inclusive
    "canonical_layer": 16,
    "eot_token_id": 128009,  # <|eot_id|>
}
