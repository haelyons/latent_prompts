"""
Direction extraction from factorial datasets using difference-in-means.

Computes behavioral directions (sycophancy, genuine agreement, praise) from
the paper's contrastive pair datasets.
"""

import json
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

from .hooks import ActivationCapture, _get_decoder_layers


@dataclass
class ContrastivePair:
    """A contrastive pair with positive (exhibits behavior) and negative (doesn't) examples."""
    positive: str
    negative: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExtractionResult:
    """Result of direction extraction."""
    direction: np.ndarray  # (D,) normalized direction vector
    layer: int
    behavior: str
    pos_count: int
    neg_count: int
    pos_mean_norm: float
    neg_mean_norm: float
    

class DirectionExtractor:
    """
    Extract behavioral directions from factorial datasets using difference-in-means.
    
    Follows the methodology from "Sycophancy Is Not One Thing":
    - Load contrastive pairs from factorial datasets
    - Compute class means μ+ and μ- via forward passes
    - Direction = (μ+ - μ-) / ||μ+ - μ-||
    
    Usage:
        extractor = DirectionExtractor(model, tokenizer)
        direction = extractor.compute_direction_from_dataset(
            "data/factorial/math_factorial.json",
            behavior_label="syc",
            layer=16
        )
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        """
        Args:
            model: HuggingFace transformer model
            tokenizer: Associated tokenizer
            device: Device for computation (inferred from model if None)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Infer device from model
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device
        
        # Get model hidden size
        self.hidden_size = model.config.hidden_size
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    def load_factorial_dataset(
        self,
        dataset_path: str,
        behavior_label: str = "syc",
        max_samples: Optional[int] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Load factorial dataset and split by behavior label.
        
        Args:
            dataset_path: Path to factorial JSON file
            behavior_label: Label field to split on (e.g., "syc", "ga", "pr")
            max_samples: Maximum samples per class (for faster testing)
            
        Returns:
            Tuple of (positive_samples, negative_samples)
        """
        with open(dataset_path, "r") as f:
            data = json.load(f)
        
        positive = []
        negative = []
        
        for item in data:
            label = item.get(behavior_label, 0)
            if label == 1:
                positive.append(item)
            else:
                negative.append(item)
        
        if max_samples is not None:
            positive = positive[:max_samples]
            negative = negative[:max_samples]
        
        return positive, negative
    
    def _build_text(self, sample: Dict) -> str:
        """Build full text from prompt and response."""
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        return prompt + response
    
    def _get_last_token_activation(
        self,
        texts: List[str],
        layer: int,
    ) -> torch.Tensor:
        """
        Get activation at last token position for a batch of texts.
        
        Args:
            texts: List of text strings
            layer: Layer index to extract from
            
        Returns:
            Tensor of shape (B, D) with activations
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        
        # Get sequence lengths for finding last real token
        lengths = attention_mask.sum(dim=1)  # (B,)
        
        # Use activation capture
        capture = ActivationCapture(self.model, [layer], capture_all_tokens=True)
        capture.register()
        
        try:
            with torch.no_grad():
                self.model(input_ids, attention_mask=attention_mask)
            
            # Get captured activations
            if layer not in capture.captured or not capture.captured[layer]:
                raise RuntimeError(f"Failed to capture activations for layer {layer}")
            
            hidden = capture.captured[layer][-1]  # (B, S, D)
            
            # Extract activation at last real token for each sequence
            batch_size = hidden.size(0)
            activations = []
            for i in range(batch_size):
                last_idx = lengths[i].item() - 1
                activations.append(hidden[i, last_idx, :])
            
            return torch.stack(activations)  # (B, D)
            
        finally:
            capture.remove()
    
    def compute_class_mean(
        self,
        samples: List[Dict],
        layer: int,
        desc: str = "Computing mean",
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute mean activation for a class of samples.
        
        Args:
            samples: List of sample dictionaries
            layer: Layer to extract from
            desc: Description for progress bar
            
        Returns:
            Tuple of (mean_activation, count)
        """
        if not samples:
            return torch.zeros(self.hidden_size, device="cpu"), 0
        
        # Accumulate on CPU to save GPU memory
        running_sum = torch.zeros(self.hidden_size, device="cpu", dtype=torch.float32)
        count = 0
        
        # Process in batches
        texts = [self._build_text(s) for s in samples]
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc, leave=False):
            batch_texts = texts[i : i + self.batch_size]
            
            activations = self._get_last_token_activation(batch_texts, layer)
            
            # Accumulate on CPU
            running_sum += activations.sum(dim=0).to("cpu", dtype=torch.float32)
            count += activations.size(0)
            
            # Free GPU memory
            del activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        mean = running_sum / max(1, count)
        return mean, count
    
    def compute_direction(
        self,
        positive_samples: List[Dict],
        negative_samples: List[Dict],
        layer: int,
        behavior: str = "unknown",
        normalize: bool = True,
    ) -> ExtractionResult:
        """
        Compute direction using difference-in-means.
        
        Args:
            positive_samples: Samples exhibiting the behavior
            negative_samples: Samples not exhibiting the behavior
            layer: Layer to extract from
            behavior: Name of the behavior (for logging)
            normalize: Whether to normalize to unit vector
            
        Returns:
            ExtractionResult with direction vector and metadata
        """
        # Compute class means
        pos_mean, pos_count = self.compute_class_mean(
            positive_samples, layer, f"Positive ({behavior})"
        )
        neg_mean, neg_count = self.compute_class_mean(
            negative_samples, layer, f"Negative ({behavior})"
        )
        
        # Compute direction
        direction = pos_mean - neg_mean
        
        # Normalize
        direction_norm = torch.norm(direction).item()
        if normalize and direction_norm > 1e-8:
            direction = direction / direction_norm
        
        return ExtractionResult(
            direction=direction.numpy(),
            layer=layer,
            behavior=behavior,
            pos_count=pos_count,
            neg_count=neg_count,
            pos_mean_norm=torch.norm(pos_mean).item(),
            neg_mean_norm=torch.norm(neg_mean).item(),
        )
    
    def compute_direction_from_dataset(
        self,
        dataset_path: str,
        behavior_label: str,
        layer: int,
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> ExtractionResult:
        """
        Compute direction from a factorial dataset file.
        
        Args:
            dataset_path: Path to factorial JSON file
            behavior_label: Label field to use (e.g., "syc", "ga", "pr")
            layer: Layer to extract from
            max_samples: Max samples per class (None for all)
            normalize: Whether to normalize direction
            
        Returns:
            ExtractionResult with direction and metadata
        """
        # Load and split dataset
        positive, negative = self.load_factorial_dataset(
            dataset_path, behavior_label, max_samples
        )
        
        print(f"Loaded {len(positive)} positive, {len(negative)} negative samples for '{behavior_label}'")
        
        return self.compute_direction(
            positive, negative, layer, behavior_label, normalize
        )
    
    def compute_directions_multi_layer(
        self,
        dataset_path: str,
        behavior_label: str,
        layers: List[int],
        max_samples: Optional[int] = None,
    ) -> Dict[int, ExtractionResult]:
        """
        Compute directions for multiple layers.
        
        Args:
            dataset_path: Path to factorial JSON file
            behavior_label: Label field to use
            layers: List of layers to extract from
            max_samples: Max samples per class
            
        Returns:
            Dict mapping layer index to ExtractionResult
        """
        # Load dataset once
        positive, negative = self.load_factorial_dataset(
            dataset_path, behavior_label, max_samples
        )
        
        print(f"Computing directions for layers {layers} on '{behavior_label}'")
        print(f"  Positive: {len(positive)}, Negative: {len(negative)}")
        
        results = {}
        for layer in tqdm(layers, desc="Layers"):
            results[layer] = self.compute_direction(
                positive, negative, layer, behavior_label, normalize=True
            )
        
        return results
    
    def compute_all_behaviors(
        self,
        dataset_paths: List[str],
        behaviors: List[str],
        layers: List[int],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Dict[int, ExtractionResult]]:
        """
        Compute directions for multiple behaviors and layers.
        
        Args:
            dataset_paths: List of factorial dataset paths
            behaviors: List of behavior labels to extract
            layers: List of layers to extract from
            max_samples: Max samples per class
            
        Returns:
            Nested dict: behavior -> layer -> ExtractionResult
        """
        all_results = {}
        
        for behavior in behaviors:
            print(f"\n=== Extracting '{behavior}' direction ===")
            
            # Aggregate samples from all datasets
            all_positive = []
            all_negative = []
            
            for path in dataset_paths:
                if os.path.exists(path):
                    pos, neg = self.load_factorial_dataset(path, behavior, None)
                    all_positive.extend(pos)
                    all_negative.extend(neg)
            
            if max_samples is not None:
                all_positive = all_positive[:max_samples]
                all_negative = all_negative[:max_samples]
            
            print(f"Total: {len(all_positive)} positive, {len(all_negative)} negative")
            
            # Compute for each layer
            layer_results = {}
            for layer in tqdm(layers, desc=f"{behavior} layers"):
                layer_results[layer] = self.compute_direction(
                    all_positive, all_negative, layer, behavior, normalize=True
                )
            
            all_results[behavior] = layer_results
        
        return all_results


def aggregate_factorial_datasets(
    data_dir: str,
    behavior_label: str,
    dataset_patterns: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Aggregate samples from multiple factorial datasets.
    
    Args:
        data_dir: Directory containing factorial JSON files
        behavior_label: Label to split on
        dataset_patterns: List of dataset filenames (or None for all)
        
    Returns:
        Tuple of (all_positive, all_negative)
    """
    if dataset_patterns is None:
        # Default: use all factorial files
        dataset_patterns = [
            "math_factorial.json",
            "claims_factorial.json",
            "cities_pos_factorial.json",
            "cities_neg_factorial.json",
            "larger_than_factorial.json",
            "smaller_than_factorial.json",
        ]
    
    all_positive = []
    all_negative = []
    
    for pattern in dataset_patterns:
        path = os.path.join(data_dir, pattern)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            
            for item in data:
                label = item.get(behavior_label, 0)
                if label == 1:
                    all_positive.append(item)
                else:
                    all_negative.append(item)
    
    return all_positive, all_negative
