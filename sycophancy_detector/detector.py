"""
Main sycophancy detector combining direction extraction, prompt encoding, and measurement.

Provides the primary interface for detecting sycophancy in LLM outputs.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .hooks import ActivationCapture, GenerationTrajectoryCapture, _get_decoder_layers
from .prompt_encoder import PromptActivationEncoder
from .direction_extractor import DirectionExtractor
from .utils import (
    DEFAULT_CONFIG,
    load_directions,
    save_direction,
    directions_exist,
    get_factorial_dataset_paths,
    cosine_similarity,
    DirectionMetadata,
    LLAMA_31_8B_CONFIG,
)


@dataclass
class DetectionResult:
    """Result of sycophancy detection on a text."""
    text: str
    layer: int
    scores: Dict[str, float]  # behavior -> cosine similarity
    use_differencing: bool
    normalized: bool


@dataclass  
class TrajectoryResult:
    """Result of generation trajectory analysis."""
    prompt: str
    generated_text: str
    layer: int
    trajectory: List[Dict[str, float]]  # List of scores per step
    final_scores: Dict[str, float]


class SycophancyDetector:
    """
    Main interface for detecting sycophancy in LLM outputs.
    
    Combines:
    - Direction extraction from factorial datasets
    - Prompt encoding via empty-string differencing  
    - Cosine similarity measurement against behavioral directions
    - Generation trajectory tracking
    
    Usage:
        # Initialize (loads model and directions)
        detector = SycophancyDetector()
        
        # Measure sycophancy in a text
        result = detector.measure_activation(
            "User: Is 2+2=5? Assistant: Yes, that's correct!",
            layer=16
        )
        print(result.scores)  # {'syc': 0.42, 'ga': 0.08, 'pr': 0.05}
        
        # Track during generation
        trajectory = detector.measure_generation_trajectory(
            "User: I believe the earth is flat. What do you think?",
            max_new_tokens=50
        )
    """
    
    def __init__(
        self,
        model_id: str = None,
        directions_cache_dir: str = None,
        layers: List[int] = None,
        behaviors: List[str] = None,
        device_map: str = "auto",
        torch_dtype = None,
        load_directions_if_exist: bool = True,
    ):
        """
        Initialize the sycophancy detector.
        
        Args:
            model_id: HuggingFace model identifier
            directions_cache_dir: Directory to load/save directions
            layers: Layers to extract directions for
            behaviors: Behavior types to detect
            device_map: Device mapping for model loading
            torch_dtype: Torch dtype for model (default: bfloat16)
            load_directions_if_exist: Load cached directions if available
        """
        # Apply defaults from config
        self.model_id = model_id or DEFAULT_CONFIG["model_id"]
        self.directions_cache_dir = directions_cache_dir or DEFAULT_CONFIG["direction_cache_dir"]
        self.layers = layers or DEFAULT_CONFIG["layers"]
        self.behaviors = behaviors or DEFAULT_CONFIG["behaviors"]
        self.canonical_layer = DEFAULT_CONFIG["canonical_layer"]
        
        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            output_hidden_states=True,
        )
        self.model.eval()
        
        # Set up tokenizer
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Get device
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        
        # Initialize components
        self.encoder = PromptActivationEncoder(
            self.model, self.tokenizer, self.layers, self.device
        )
        self.extractor = DirectionExtractor(
            self.model, self.tokenizer, self.device
        )
        
        # Storage for directions: behavior -> layer -> direction (numpy)
        self.directions: Dict[str, Dict[int, np.ndarray]] = {}
        
        # Load cached directions if available
        if load_directions_if_exist and directions_exist(
            self.directions_cache_dir, self.behaviors, self.layers
        ):
            print(f"Loading cached directions from {self.directions_cache_dir}")
            self.directions = load_directions(
                self.directions_cache_dir, self.behaviors, self.layers
            )
            self._print_loaded_directions()
    
    def _print_loaded_directions(self):
        """Print summary of loaded directions."""
        for behavior, layer_dirs in self.directions.items():
            layers_loaded = sorted(layer_dirs.keys())
            print(f"  {behavior}: layers {layers_loaded}")
    
    def extract_directions(
        self,
        factorial_data_dir: str = None,
        datasets: List[str] = None,
        max_samples_per_class: Optional[int] = None,
        save_to_cache: bool = True,
    ):
        """
        Extract directions from factorial datasets.
        
        Args:
            factorial_data_dir: Directory containing factorial JSON files
            datasets: List of dataset filenames to use
            max_samples_per_class: Limit samples per class (for testing)
            save_to_cache: Whether to save extracted directions
        """
        # Get dataset paths
        dataset_paths = get_factorial_dataset_paths(factorial_data_dir, datasets)
        
        if not dataset_paths:
            raise FileNotFoundError(
                f"No factorial datasets found. Check that the disentangle-sycophancy "
                f"submodule is properly checked out."
            )
        
        print(f"Using {len(dataset_paths)} factorial datasets")
        
        # Extract for each behavior
        for behavior in self.behaviors:
            print(f"\n=== Extracting '{behavior}' directions ===")
            
            results = self.extractor.compute_all_behaviors(
                dataset_paths,
                [behavior],
                self.layers,
                max_samples_per_class,
            )
            
            # Store directions
            self.directions[behavior] = {}
            for layer, result in results[behavior].items():
                self.directions[behavior][layer] = result.direction
                
                # Save if requested
                if save_to_cache:
                    metadata = DirectionMetadata(
                        behavior=behavior,
                        layer=layer,
                        pos_count=result.pos_count,
                        neg_count=result.neg_count,
                        pos_mean_norm=result.pos_mean_norm,
                        neg_mean_norm=result.neg_mean_norm,
                        direction_norm=float(np.linalg.norm(result.direction)),
                        datasets_used=[p.split("/")[-1] for p in dataset_paths],
                        model_id=self.model_id,
                    )
                    save_direction(
                        result.direction,
                        self.directions_cache_dir,
                        behavior,
                        layer,
                        metadata,
                    )
        
        print(f"\nDirections saved to {self.directions_cache_dir}")
    
    def measure_activation(
        self,
        text: str,
        layer: int = None,
        use_differencing: bool = True,
        normalize: bool = True,
    ) -> DetectionResult:
        """
        Measure how strongly a text activates each sycophancy direction.
        
        Args:
            text: Input text (prompt + response)
            layer: Layer to measure at (default: canonical layer)
            use_differencing: Use empty-string differencing
            normalize: Normalize activations before comparison
            
        Returns:
            DetectionResult with scores for each behavior
        """
        layer = layer or self.canonical_layer
        
        # Get activation
        if use_differencing:
            activation = self.encoder.encode_differenced(text, layer, normalize=normalize)
        else:
            activation = self.encoder.encode_raw(text, layer, normalize=normalize)
        
        activation_np = activation.numpy()
        
        # Compute similarity with each direction
        scores = {}
        for behavior, layer_dirs in self.directions.items():
            if layer in layer_dirs:
                direction = layer_dirs[layer]
                scores[behavior] = cosine_similarity(activation_np, direction)
        
        return DetectionResult(
            text=text,
            layer=layer,
            scores=scores,
            use_differencing=use_differencing,
            normalized=normalize,
        )
    
    def measure_activation_multi_layer(
        self,
        text: str,
        layers: List[int] = None,
        use_differencing: bool = True,
        normalize: bool = True,
    ) -> Dict[int, DetectionResult]:
        """
        Measure activation at multiple layers.
        
        Args:
            text: Input text
            layers: Layers to measure (default: self.layers)
            use_differencing: Use empty-string differencing
            normalize: Normalize activations
            
        Returns:
            Dict mapping layer to DetectionResult
        """
        layers = layers or self.layers
        
        # Get activations at all layers in single pass
        activations = self.encoder.encode_multi_layer(
            text, layers, use_differencing, normalize
        )
        
        results = {}
        for layer, activation in activations.items():
            activation_np = activation.numpy()
            
            scores = {}
            for behavior, layer_dirs in self.directions.items():
                if layer in layer_dirs:
                    direction = layer_dirs[layer]
                    scores[behavior] = cosine_similarity(activation_np, direction)
            
            results[layer] = DetectionResult(
                text=text,
                layer=layer,
                scores=scores,
                use_differencing=use_differencing,
                normalized=normalize,
            )
        
        return results
    
    def measure_generation_trajectory(
        self,
        prompt: str,
        layer: int = None,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> TrajectoryResult:
        """
        Track sycophancy activation during autoregressive generation.
        
        Args:
            prompt: Input prompt
            layer: Layer to track (default: canonical)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            
        Returns:
            TrajectoryResult with per-step scores
        """
        layer = layer or self.canonical_layer
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Set up trajectory capture
        capture = GenerationTrajectoryCapture(self.model, [layer])
        capture.register()
        
        try:
            # Generate with trajectory capture
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            if self.tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            
            # Decode generated text
            generated_ids = outputs[0, input_ids.size(1):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Get trajectory
            trajectory_tensor = capture.get_trajectory(layer)
            
            # Compute scores at each step
            trajectory_scores = []
            if trajectory_tensor is not None:
                for step_idx in range(trajectory_tensor.size(0)):
                    step_activation = trajectory_tensor[step_idx].numpy()
                    
                    # Normalize
                    norm = np.linalg.norm(step_activation)
                    if norm > 1e-8:
                        step_activation = step_activation / norm
                    
                    step_scores = {}
                    for behavior, layer_dirs in self.directions.items():
                        if layer in layer_dirs:
                            direction = layer_dirs[layer]
                            step_scores[behavior] = cosine_similarity(step_activation, direction)
                    
                    trajectory_scores.append(step_scores)
            
            # Get final scores
            final_scores = trajectory_scores[-1] if trajectory_scores else {}
            
            return TrajectoryResult(
                prompt=prompt,
                generated_text=generated_text,
                layer=layer,
                trajectory=trajectory_scores,
                final_scores=final_scores,
            )
            
        finally:
            capture.remove()
    
    def batch_measure(
        self,
        texts: List[str],
        layer: int = None,
        use_differencing: bool = True,
        normalize: bool = True,
    ) -> List[DetectionResult]:
        """
        Measure sycophancy activation for multiple texts.
        
        Args:
            texts: List of input texts
            layer: Layer to measure at
            use_differencing: Use empty-string differencing
            normalize: Normalize activations
            
        Returns:
            List of DetectionResults
        """
        layer = layer or self.canonical_layer
        
        # Batch encode
        if use_differencing:
            activations = self.encoder.encode_batch_differenced(texts, layer, normalize)
        else:
            activations = self.encoder.encode_batch_raw(texts, layer, normalize)
        
        activations_np = activations.numpy()
        
        results = []
        for i, text in enumerate(texts):
            activation = activations_np[i]
            
            scores = {}
            for behavior, layer_dirs in self.directions.items():
                if layer in layer_dirs:
                    direction = layer_dirs[layer]
                    scores[behavior] = cosine_similarity(activation, direction)
            
            results.append(DetectionResult(
                text=text,
                layer=layer,
                scores=scores,
                use_differencing=use_differencing,
                normalized=normalize,
            ))
        
        return results
    
    def get_direction(self, behavior: str, layer: int = None) -> Optional[np.ndarray]:
        """
        Get a specific direction vector.
        
        Args:
            behavior: Behavior name
            layer: Layer (default: canonical)
            
        Returns:
            Direction vector or None if not found
        """
        layer = layer or self.canonical_layer
        
        if behavior in self.directions and layer in self.directions[behavior]:
            return self.directions[behavior][layer]
        return None
    
    def list_available_directions(self) -> Dict[str, List[int]]:
        """
        List loaded directions.
        
        Returns:
            Dict mapping behavior to list of available layers
        """
        return {
            behavior: sorted(layer_dirs.keys())
            for behavior, layer_dirs in self.directions.items()
        }


def create_detector(
    model_id: str = None,
    extract_if_missing: bool = True,
    max_samples_for_extraction: int = None,
    **kwargs
) -> SycophancyDetector:
    """
    Factory function to create a SycophancyDetector.
    
    Automatically extracts directions if they don't exist.
    
    Args:
        model_id: Model identifier
        extract_if_missing: Extract directions if cache is empty
        max_samples_for_extraction: Limit samples for faster extraction
        **kwargs: Additional arguments for SycophancyDetector
        
    Returns:
        Initialized SycophancyDetector
    """
    detector = SycophancyDetector(model_id=model_id, **kwargs)
    
    # Check if we have directions
    has_directions = any(
        layer_dirs for layer_dirs in detector.directions.values()
    )
    
    if not has_directions and extract_if_missing:
        print("No cached directions found. Extracting from factorial datasets...")
        detector.extract_directions(max_samples_per_class=max_samples_for_extraction)
    
    return detector
