"""
PyTorch forward hooks for activation extraction.

Provides clean interfaces for capturing hidden states from transformer layers
during forward passes and generation.
"""

import torch
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


def _get_decoder_layers(model):
    """
    Locate decoder layers for different model architectures.
    
    Supports: Llama, GPT-2/GPT-Neo, Gemma variants.
    """
    # Llama-style: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2/Neo style: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # Gemma decoder variant
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    # Top-level layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not locate decoder layers for this model architecture.")


def _hidden_to_block_index(hidden_idx: int, model) -> int:
    """
    Convert a hidden_states index (0..num_layers) to decoder block index (0..num_layers-1).
    
    hidden_states[0] = embeddings, hidden_states[1..N] = layer outputs
    decoder blocks are indexed 0..N-1
    """
    if hidden_idx is None:
        raise ValueError("Hidden index cannot be None")
    
    layers = _get_decoder_layers(model)
    num_blocks = len(layers)
    
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(model.config, "n_layer", num_blocks)
    
    # hidden_idx 0 is embeddings, can't hook there
    if hidden_idx == 0:
        raise ValueError("Cannot extract from embeddings (hidden index 0). Use layer >= 1.")
    
    # hidden_idx == num_layers maps to last block
    if hidden_idx == num_layers:
        return num_blocks - 1
    
    # Map hidden_idx to block: hidden 1 -> block 0, hidden 2 -> block 1, etc.
    if 1 <= hidden_idx <= num_layers:
        return hidden_idx - 1
    
    # Fallback: clamp to valid range
    return max(0, min(num_blocks - 1, hidden_idx - 1))


class ActivationCapture:
    """
    Capture hidden state activations from transformer layers using forward hooks.
    
    Usage:
        capture = ActivationCapture(model, layers=[12, 16, 20])
        capture.register()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        activations = capture.get_last_token_activations()
        capture.clear()
        capture.remove()
    """
    
    def __init__(
        self, 
        model, 
        layers: List[int],
        capture_all_tokens: bool = False
    ):
        """
        Args:
            model: HuggingFace transformer model
            layers: List of layer indices to capture (1-indexed hidden states)
            capture_all_tokens: If True, capture all tokens; else only last token
        """
        self.model = model
        self.layers = layers
        self.capture_all_tokens = capture_all_tokens
        self._decoder_layers = _get_decoder_layers(model)
        
        # Storage for captured activations
        self.captured: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.handles: List[Any] = []
        
    def _make_hook(self, layer_idx: int):
        """Create a forward hook for the specified layer."""
        def hook(module, input, output):
            # Handle different output formats
            if hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                hidden = output[0] if torch.is_tensor(output[0]) else None
            elif torch.is_tensor(output):
                hidden = output
            else:
                return
            
            if hidden is None:
                return
                
            # Capture activations
            if self.capture_all_tokens:
                # Store full sequence: (B, S, D)
                self.captured[layer_idx].append(hidden.detach().clone())
            else:
                # Store only last token: (B, D)
                self.captured[layer_idx].append(hidden[:, -1, :].detach().clone())
        
        return hook
    
    def register(self):
        """Register forward hooks on target layers."""
        self.remove()  # Clear any existing hooks
        
        for layer_idx in self.layers:
            block_idx = _hidden_to_block_index(layer_idx, self.model)
            if block_idx < len(self._decoder_layers):
                handle = self._decoder_layers[block_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear(self):
        """Clear captured activations."""
        self.captured = {l: [] for l in self.layers}
    
    def get_last_token_activations(self) -> Dict[int, torch.Tensor]:
        """
        Get the most recently captured activations for each layer.
        
        Returns:
            Dict mapping layer index to activation tensor (D,)
        """
        result = {}
        for layer_idx, activations in self.captured.items():
            if activations:
                last_act = activations[-1]
                # If we captured all tokens, extract last token
                if self.capture_all_tokens and last_act.dim() == 3:
                    result[layer_idx] = last_act[:, -1, :].squeeze(0)
                else:
                    result[layer_idx] = last_act.squeeze(0)
        return result
    
    def get_all_activations(self) -> Dict[int, List[torch.Tensor]]:
        """Get all captured activations for each layer."""
        return self.captured
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False


class GenerationTrajectoryCapture(ActivationCapture):
    """
    Specialized capture for tracking activations during autoregressive generation.
    
    Captures activation at each generation step for trajectory analysis.
    """
    
    def __init__(self, model, layers: List[int]):
        super().__init__(model, layers, capture_all_tokens=False)
        self.trajectory: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    
    def _make_hook(self, layer_idx: int):
        """Create hook that appends to trajectory."""
        def hook(module, input, output):
            if hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                hidden = output[0] if torch.is_tensor(output[0]) else None
            elif torch.is_tensor(output):
                hidden = output
            else:
                return
            
            if hidden is not None:
                # Append last token activation to trajectory
                self.trajectory[layer_idx].append(
                    hidden[:, -1, :].detach().cpu()
                )
        
        return hook
    
    def clear(self):
        """Clear trajectory and captured activations."""
        super().clear()
        self.trajectory = {l: [] for l in self.layers}
    
    def get_trajectory(self, layer: int) -> Optional[torch.Tensor]:
        """
        Get the activation trajectory for a layer.
        
        Returns:
            Tensor of shape (num_steps, D) or None if no trajectory
        """
        if layer in self.trajectory and self.trajectory[layer]:
            return torch.stack([t.squeeze(0) for t in self.trajectory[layer]])
        return None
    
    def get_all_trajectories(self) -> Dict[int, torch.Tensor]:
        """Get trajectories for all captured layers."""
        return {
            layer: self.get_trajectory(layer)
            for layer in self.layers
            if self.get_trajectory(layer) is not None
        }


@contextmanager
def capture_activations(model, layers: List[int], capture_all_tokens: bool = False):
    """
    Context manager for capturing activations.
    
    Usage:
        with capture_activations(model, [12, 16, 20]) as capture:
            outputs = model(input_ids)
            activations = capture.get_last_token_activations()
    """
    capture = ActivationCapture(model, layers, capture_all_tokens)
    capture.register()
    try:
        yield capture
    finally:
        capture.remove()
