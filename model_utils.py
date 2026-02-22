"""
Provides hook-based activation extraction that works at any model scale,
including multi-GPU setups where output_hidden_states=True would OOM.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DIRECTIONS_DIR

POOL_STRATEGY = "pre_eos"

def load_model_and_tokenizer(profile: dict):
    """
    Load model and tokenizer from a model profile dict.

    For 70B on 2× H100, profile["device_map"] = "auto" handles layer sharding
    via accelerate. For 8B, profile["device_map"] = "cuda" pins to one GPU.
    """
    model_id = profile["model_id"]
    print(f"Loading model: {model_id} (device_map={profile['device_map']})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=profile["device_map"],
    )
    model.eval()

    device_info = {p.device for p in model.parameters()}
    print(f"  Model devices: {device_info}")
    return model, tokenizer


def build_special_token_set(tokenizer) -> Set[int]:
    """Build set of special token IDs to skip during pooling."""
    special_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    return special_ids


def find_pool_position(
    input_ids: torch.Tensor,
    special_ids: Set[int],
    eos_id: int,
    strategy: str = POOL_STRATEGY,
) -> int:
    """
    Find the token position to extract the activation from.

    'pre_eos': last token before the final EOS, skipping special tokens.
    'last':    last non-special token.

    Matches the logic in 01_extract_directions.get_pooled_activation and
    03_measure_per_token_dual.find_measurement_position.
    """
    ids = input_ids[0] if input_ids.dim() > 1 else input_ids
    seq_len = len(ids)

    if strategy == "pre_eos":
        eos_positions = (ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            idx = eos_positions[-1].item() - 1
        else:
            idx = seq_len - 1
    else:
        idx = seq_len - 1

    while idx > 0 and ids[idx].item() in special_ids:
        idx -= 1
    return max(0, idx)


class ActivationExtractor:
    """
    Extracts activations at specified layers and token positions using forward
    hooks.  Only the requested layers are captured — no output_hidden_states,
    so memory stays flat regardless of total layer count.

    Handles multi-GPU: each hook detaches and moves to CPU immediately, so
    it doesn't matter which device a given layer lives on.

    Usage:
        extractor = ActivationExtractor(model, layers=[55, 60, 65])
        acts = extractor.extract(input_ids, position=42)
        # acts = {55: Tensor(hidden_dim,), 60: ..., 65: ...}
    """

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = sorted(layers)
        self._captured: Dict[int, Optional[torch.Tensor]] = {}

    def extract(
        self,
        input_ids: torch.Tensor,
        position: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Run a single forward pass and return {layer: activation} at `position`.

        Args:
            input_ids: (1, seq_len) token IDs, already on the correct device(s).
            position:  token index to extract from.

        Returns:
            Dict mapping layer index to a float32 CPU tensor of shape (hidden_dim,).
        """
        self._captured = {}
        handles = []

        for layer_idx in self.layers:
            handle = self.model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, position)
            )
            handles.append(handle)

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return dict(self._captured)

    def _make_hook(self, layer_idx: int, position: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            act = hidden[0, position, :].detach().float().cpu()
            self._captured[layer_idx] = act
        return hook


# replace 01 get_pooled_activation
def get_pooled_activations(
    model,
    tokenizer,
    text: str,
    layers: List[int],
    special_ids: Set[int],
    pool_strategy: str = POOL_STRATEGY,
) -> Dict[int, torch.Tensor]:
    """
    Tokenize text, find pool position, extract activations at all requested
    layers in a single forward pass.

    Drop-in multi-layer replacement for 01_extract_directions.get_pooled_activation.

    Returns:
        {layer: float32 CPU tensor of shape (hidden_dim,)}
    """
    inputs = tokenizer(text, return_tensors="pt")
    # Move to first model device (accelerate handles cross-device routing)
    first_device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(first_device)

    position = find_pool_position(
        input_ids, special_ids, tokenizer.eos_token_id, strategy=pool_strategy
    )

    extractor = ActivationExtractor(model, layers)
    return extractor.extract(input_ids, position)


BEHAVIOR_NAMES = ["sya", "ga", "sypr"]

def direction_filename(behavior: str, layer: int, n_examples: int, model_tag: str = "") -> str:
    """
    Build a direction filename.
    model_tag is e.g. '70b' — omitted for 8B to preserve backward compat.
    """
    prefix = f"{behavior}_{model_tag}_" if model_tag else f"{behavior}_"

    if n_examples >= 1000:
        thousands = n_examples // 1000
        hundreds = (n_examples % 1000) // 100
        count_str = f"{thousands}k{hundreds}" if hundreds else f"{thousands}k"
    else:
        count_str = str(n_examples)

    return f"{prefix}layer{layer}_n{count_str}_svd.pt"


def find_direction_file(
    behavior: str,
    layer: int,
    directions_dir: Path = DIRECTIONS_DIR,
    model_tag: str = "",
) -> Path:
    """Find the most recent direction file matching behavior, layer, and optional model tag."""
    if model_tag:
        pattern = f"{behavior}_{model_tag}_layer{layer}_*.pt"
    else:
        pattern = f"{behavior}_layer{layer}_*.pt"

    matches = list(directions_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No direction file for {behavior} layer {layer} "
            f"(model_tag={model_tag!r}) in {directions_dir}"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_directions(
    layers: List[int],
    directions_dir: Path = DIRECTIONS_DIR,
    model_tag: str = "",
) -> Dict[str, Dict[int, dict]]:
    """
    Load direction files for all behaviors and layers.

    Returns:
        {behavior: {layer: {"direction": Tensor, "auroc": float, ...}}}
    """
    directions: Dict[str, Dict[int, dict]] = {}
    for behavior in BEHAVIOR_NAMES:
        directions[behavior] = {}
        for layer in layers:
            filepath = find_direction_file(behavior, layer, directions_dir, model_tag)
            saved = torch.load(str(filepath), weights_only=False)
            directions[behavior][layer] = saved
            auroc = saved.get("auroc", 0)
            print(f"  {behavior.upper()} layer {layer}: {filepath.name} (AUROC={auroc:.4f})")
    return directions
