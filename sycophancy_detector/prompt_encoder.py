"""
Prompt activation encoder using empty-string differencing.

Converts prompts into activation space by computing:
    activation(prompt) - activation("")
    
This captures what the prompt "adds" to the model's representation,
following the CAA/STA methodology.
"""

import torch
from typing import Dict, List, Optional
from .hooks import ActivationCapture


class PromptActivationEncoder:
    """
    Encode prompts into activation space via empty-string differencing.
    
    The key insight is that activation(prompt) - activation("") captures
    what the prompt contributes to the model's internal representation,
    removing baseline activation patterns.
    
    Usage:
        encoder = PromptActivationEncoder(model, tokenizer, layers=[12, 16, 20])
        
        # Get differenced encoding
        activation = encoder.encode_differenced("User: What is 2+2?", layer=16)
        
        # Or raw encoding
        activation = encoder.encode_raw("User: What is 2+2?", layer=16)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        layers: List[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: HuggingFace transformer model
            tokenizer: Associated tokenizer
            layers: Layers to support (default: [12, 14, 16, 18, 20])
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers or [12, 14, 16, 18, 20]
        
        # Infer device
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device
        
        # Hidden size for validation
        self.hidden_size = model.config.hidden_size
        
        # Cache for empty string activations (computed once per layer)
        self._empty_cache: Dict[int, torch.Tensor] = {}
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def _get_activation(
        self,
        text: str,
        layer: int,
    ) -> torch.Tensor:
        """
        Get activation at last token position for a text.
        
        Args:
            text: Input text
            layer: Layer index to extract from
            
        Returns:
            Activation tensor of shape (D,)
        """
        # Tokenize
        if not text:
            # Empty string: use just BOS/padding behavior
            inputs = self.tokenizer("", return_tensors="pt")
        else:
            inputs = self.tokenizer(text, return_tensors="pt")
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None
        
        # Capture activation
        capture = ActivationCapture(self.model, [layer], capture_all_tokens=True)
        capture.register()
        
        try:
            with torch.no_grad():
                if attention_mask is not None:
                    self.model(input_ids, attention_mask=attention_mask)
                else:
                    self.model(input_ids)
            
            # Get captured hidden states
            if layer not in capture.captured or not capture.captured[layer]:
                raise RuntimeError(f"Failed to capture activation for layer {layer}")
            
            hidden = capture.captured[layer][-1]  # (1, S, D)
            
            # Return last token activation
            return hidden[0, -1, :].cpu()  # (D,)
            
        finally:
            capture.remove()
    
    def _get_empty_activation(self, layer: int) -> torch.Tensor:
        """
        Get cached activation for empty string.
        
        Computes once and caches for efficiency.
        """
        if layer not in self._empty_cache:
            self._empty_cache[layer] = self._get_activation("", layer)
        return self._empty_cache[layer]
    
    def clear_cache(self):
        """Clear the empty string activation cache."""
        self._empty_cache = {}
    
    def encode_raw(
        self,
        text: str,
        layer: int,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Get raw activation embedding (no differencing).
        
        Args:
            text: Input text
            layer: Layer to extract from
            normalize: Whether to normalize to unit vector
            
        Returns:
            Activation tensor of shape (D,)
        """
        activation = self._get_activation(text, layer)
        
        if normalize:
            norm = torch.norm(activation)
            if norm > 1e-8:
                activation = activation / norm
        
        return activation
    
    def encode_differenced(
        self,
        text: str,
        layer: int,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Encode text via empty-string differencing.
        
        Computes: activation(text) - activation("")
        
        This captures what the text "adds" to the model's representation.
        
        Args:
            text: Input text
            layer: Layer to extract from
            normalize: Whether to normalize to unit vector
            
        Returns:
            Differenced activation tensor of shape (D,)
        """
        text_activation = self._get_activation(text, layer)
        empty_activation = self._get_empty_activation(layer)
        
        differenced = text_activation - empty_activation
        
        if normalize:
            norm = torch.norm(differenced)
            if norm > 1e-8:
                differenced = differenced / norm
        
        return differenced
    
    def encode_batch_raw(
        self,
        texts: List[str],
        layer: int,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Batch encode multiple texts (raw activation).
        
        Args:
            texts: List of input texts
            layer: Layer to extract from
            normalize: Whether to normalize each to unit vector
            
        Returns:
            Tensor of shape (N, D)
        """
        if not texts:
            return torch.zeros(0, self.hidden_size)
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Get sequence lengths
        lengths = attention_mask.sum(dim=1)
        
        # Capture activations
        capture = ActivationCapture(self.model, [layer], capture_all_tokens=True)
        capture.register()
        
        try:
            with torch.no_grad():
                self.model(input_ids, attention_mask=attention_mask)
            
            hidden = capture.captured[layer][-1]  # (B, S, D)
            
            # Extract last real token for each sequence
            batch_size = hidden.size(0)
            activations = []
            for i in range(batch_size):
                last_idx = lengths[i].item() - 1
                activations.append(hidden[i, last_idx, :])
            
            result = torch.stack(activations).cpu()  # (B, D)
            
            if normalize:
                norms = torch.norm(result, dim=1, keepdim=True)
                norms = torch.clamp(norms, min=1e-8)
                result = result / norms
            
            return result
            
        finally:
            capture.remove()
    
    def encode_batch_differenced(
        self,
        texts: List[str],
        layer: int,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Batch encode multiple texts with empty-string differencing.
        
        Args:
            texts: List of input texts
            layer: Layer to extract from
            normalize: Whether to normalize each to unit vector
            
        Returns:
            Tensor of shape (N, D)
        """
        raw = self.encode_batch_raw(texts, layer, normalize=False)
        empty = self._get_empty_activation(layer)
        
        differenced = raw - empty.unsqueeze(0)
        
        if normalize:
            norms = torch.norm(differenced, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            differenced = differenced / norms
        
        return differenced
    
    def encode_multi_layer(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        use_differencing: bool = True,
        normalize: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """
        Encode text at multiple layers in a single forward pass.
        
        Args:
            text: Input text
            layers: Layers to extract (default: self.layers)
            use_differencing: Whether to use empty-string differencing
            normalize: Whether to normalize
            
        Returns:
            Dict mapping layer index to activation tensor
        """
        layers = layers or self.layers
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        # Capture all requested layers
        capture = ActivationCapture(self.model, layers, capture_all_tokens=True)
        capture.register()
        
        try:
            with torch.no_grad():
                self.model(input_ids)
            
            result = {}
            for layer in layers:
                if layer in capture.captured and capture.captured[layer]:
                    hidden = capture.captured[layer][-1]  # (1, S, D)
                    activation = hidden[0, -1, :].cpu()  # Last token
                    
                    if use_differencing:
                        empty = self._get_empty_activation(layer)
                        activation = activation - empty
                    
                    if normalize:
                        norm = torch.norm(activation)
                        if norm > 1e-8:
                            activation = activation / norm
                    
                    result[layer] = activation
            
            return result
            
        finally:
            capture.remove()
