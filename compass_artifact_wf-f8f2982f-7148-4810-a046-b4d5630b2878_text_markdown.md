# Building a sycophancy direction detector for Llama 3.1 8B

A Python prototype for detecting sycophancy in LLM outputs requires extracting behavioral directions using difference-in-means from contrastive pairs, then measuring how new prompts and generations project onto those directions. The implementation combines techniques from the "Sycophancy Is Not One Thing" paper (diff-of-means directions, EOS token extraction) with CAA-style activation steering patterns, executed locally using either NNSight or raw PyTorch hooks on Llama 3.1 8B's **32 transformer layers** with **4096-dimensional** hidden states.

## Core methodology from the sycophancy paper

The disentangle-sycophancy paper separates three distinct behavioral directions using the **difference-in-means** formula:

```python
d_behavior = (1/|P|) * Σ h_l(x_positive) - (1/|N|) * Σ h_l(x_negative)
```

Where `h_l(x)` is the residual stream activation at layer `l` for input `x`. The paper identifies three separable sycophancy types: **Sycophantic Agreement (SyA)** where the model echoes incorrect user claims, **Genuine Agreement (GA)** where the model agrees with correct claims, and **Sycophantic Praise (SyPr)** involving direct flattery. Their key finding is that SyA and GA become distinguishable in mid-to-late layers (AUROC >0.9), while SyPr remains orthogonal throughout.

The paper uses the **EOS token position** (last token, k=0) for extraction, achieving **0.9839 average AUROC** across datasets—significantly outperforming earlier positions. For Llama 3.1 8B, optimal layers fall in the **12-20 range** (middle third of 32 layers), with layer 16 being the canonical middle extraction point.

## Direction extraction implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ContrastivePair:
    positive: str  # Exhibits target behavior (e.g., sycophantic response)
    negative: str  # Does not exhibit behavior (e.g., truthful response)

class SycophancyDirectionExtractor:
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        layers: List[int] = None,
        device_map: str = "auto"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            output_hidden_states=True
        )
        # Default to middle layers for Llama 3.1 8B (32 total layers)
        self.layers = layers or [12, 16, 20]
        self.directions: Dict[str, Dict[int, torch.Tensor]] = {}
        
    def extract_activation_at_last_token(
        self, 
        text: str, 
        layer: int
    ) -> torch.Tensor:
        """Extract residual stream activation at last token position."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # hidden_states[layer+1] because index 0 is embeddings
        hidden_states = outputs.hidden_states[layer + 1]
        last_token_activation = hidden_states[:, -1, :]  # (1, 4096)
        return last_token_activation.squeeze(0)  # (4096,)
    
    def compute_direction(
        self,
        pairs: List[ContrastivePair],
        behavior_name: str
    ) -> Dict[int, torch.Tensor]:
        """Compute diff-of-means direction for a behavior across layers."""
        directions = {}
        
        for layer in self.layers:
            pos_activations = []
            neg_activations = []
            
            for pair in pairs:
                pos_act = self.extract_activation_at_last_token(pair.positive, layer)
                neg_act = self.extract_activation_at_last_token(pair.negative, layer)
                pos_activations.append(pos_act)
                neg_activations.append(neg_act)
            
            pos_mean = torch.stack(pos_activations).mean(dim=0)
            neg_mean = torch.stack(neg_activations).mean(dim=0)
            
            direction = pos_mean - neg_mean
            direction = direction / direction.norm()  # Normalize to unit vector
            directions[layer] = direction
            
        self.directions[behavior_name] = directions
        return directions
```

## Contrastive dataset format

The paper uses synthetic datasets with unambiguous ground truth. Format contrastive pairs where the model response clearly exhibits or avoids the target behavior:

```python
# Example sycophantic agreement pairs (SyA)
sya_pairs = [
    ContrastivePair(
        positive="User: I believe 23 + 17 = 39. What do you think?\nAssistant: Yes, that's correct!",
        negative="User: I believe 23 + 17 = 39. What do you think?\nAssistant: Actually, 23 + 17 = 40."
    ),
    ContrastivePair(
        positive="User: I think the capital of Australia is Sydney. Am I right?\nAssistant: You're absolutely right!",
        negative="User: I think the capital of Australia is Sydney. Am I right?\nAssistant: Actually, the capital is Canberra."
    ),
]

# Example sycophantic praise pairs (SyPr)
sypr_pairs = [
    ContrastivePair(
        positive="User: What do you think of my idea?\nAssistant: That's brilliant! You're so insightful.",
        negative="User: What do you think of my idea?\nAssistant: That approach has some merit."
    ),
]

# Extract directions for each behavior type
extractor = SycophancyDirectionExtractor()
sya_directions = extractor.compute_direction(sya_pairs, "sycophantic_agreement")
sypr_directions = extractor.compute_direction(sypr_pairs, "sycophantic_praise")
```

## Empty-string differencing for prompt embedding

The CAA/STA approach converts a **single prompt** into activation space by computing `activation(prompt) - activation("")`. This captures what the prompt "adds" to the model's representation:

```python
class PromptActivationEncoder:
    def __init__(self, model, tokenizer, layers: List[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers or [12, 16, 20]
        self._empty_string_cache: Dict[int, torch.Tensor] = {}
        
    def _get_empty_string_activation(self, layer: int) -> torch.Tensor:
        """Cache empty string activations (computed once)."""
        if layer not in self._empty_string_cache:
            # Use BOS token as "empty" input
            inputs = self.tokenizer("", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            self._empty_string_cache[layer] = outputs.hidden_states[layer + 1][:, -1, :].squeeze(0)
        return self._empty_string_cache[layer]
    
    def encode_prompt_differenced(
        self, 
        prompt: str, 
        layer: int
    ) -> torch.Tensor:
        """Convert prompt to activation space via empty-string differencing."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        prompt_activation = outputs.hidden_states[layer + 1][:, -1, :].squeeze(0)
        empty_activation = self._get_empty_string_activation(layer)
        
        return prompt_activation - empty_activation
    
    def encode_prompt_raw(
        self, 
        prompt: str, 
        layer: int
    ) -> torch.Tensor:
        """Get raw activation embedding (no differencing)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer + 1][:, -1, :].squeeze(0)
```

## Measuring sycophancy direction activation

The core detection computes **cosine similarity** or **dot product** between activations and behavioral directions:

```python
class SycophancyDetector:
    def __init__(
        self,
        directions: Dict[str, Dict[int, torch.Tensor]],
        encoder: PromptActivationEncoder
    ):
        self.directions = directions
        self.encoder = encoder
    
    def measure_activation(
        self,
        text: str,
        layer: int = 16,
        use_differencing: bool = True,
        normalize: bool = True
    ) -> Dict[str, float]:
        """Measure how strongly text activates each sycophancy direction."""
        if use_differencing:
            activation = self.encoder.encode_prompt_differenced(text, layer)
        else:
            activation = self.encoder.encode_prompt_raw(text, layer)
        
        if normalize:
            activation = activation / activation.norm()
        
        scores = {}
        for behavior_name, layer_directions in self.directions.items():
            direction = layer_directions[layer]
            # Cosine similarity (directions already normalized)
            similarity = torch.dot(activation, direction).item()
            scores[behavior_name] = similarity
        
        return scores
    
    def measure_generation_trajectory(
        self,
        prompt: str,
        layer: int = 16,
        max_new_tokens: int = 50
    ) -> List[Dict[str, float]]:
        """Track sycophancy activation during autoregressive generation."""
        trajectory = []
        
        inputs = self.encoder.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.encoder.model.device)
        
        generated_ids = inputs.input_ids.clone()
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.encoder.model(
                    generated_ids, 
                    output_hidden_states=True
                )
            
            # Get activation at current last token
            activation = outputs.hidden_states[layer + 1][:, -1, :].squeeze(0)
            activation = activation / activation.norm()
            
            step_scores = {}
            for behavior_name, layer_directions in self.directions.items():
                direction = layer_directions[layer]
                step_scores[behavior_name] = torch.dot(activation, direction).item()
            trajectory.append(step_scores)
            
            # Sample next token
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() in [128001, 128008, 128009]:
                break
        
        return trajectory
```

## NNSight-based implementation for cleaner hooks

NNSight provides a cleaner API for activation extraction and runs locally without NDIF by default:

```python
from nnsight import LanguageModel

class NNSightSycophancyExtractor:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model = LanguageModel(
            model_id,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        
    def extract_multilayer_activations(
        self,
        text: str,
        layers: List[int] = [12, 16, 20]
    ) -> Dict[int, torch.Tensor]:
        """Extract activations from multiple layers in single forward pass."""
        with self.model.trace(text) as tracer:
            saved = {}
            for layer in layers:
                # Access residual stream at layer output, last token
                layer_out = self.model.model.layers[layer].output[0]
                saved[layer] = layer_out[:, -1, :].save()
        
        return {layer: tensor.squeeze(0) for layer, tensor in saved.items()}
    
    def extract_during_generation(
        self,
        prompt: str,
        layer: int = 16,
        max_new_tokens: int = 20
    ) -> Tuple[str, List[torch.Tensor]]:
        """Collect activations at each generation step."""
        activations = []
        
        with self.model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
            all_hidden = []
            with tracer.all():  # Collect from all generation steps
                layer_out = self.model.model.layers[layer].output[0]
                all_hidden.append(layer_out[:, -1, :].save())
            output = self.model.generator.output.save()
        
        generated_text = self.model.tokenizer.decode(output[0])
        return generated_text, all_hidden
```

## PyTorch hooks for maximum control

For fine-grained control during generation, use raw forward hooks:

```python
class HookBasedExtractor:
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.handles = []
        
    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # Llama layers output tuple; first element is hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx].append(hidden[:, -1, :].detach().clone())
        return hook
    
    def register_hooks(self):
        for layer_idx in self.layers:
            handle = self.model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.handles.append(handle)
    
    def clear(self):
        for layer_idx in self.activations:
            self.activations[layer_idx] = []
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

# Usage during generation
extractor = HookBasedExtractor(model, layers=[12, 16, 20])
extractor.register_hooks()

generated = model.generate(input_ids, max_new_tokens=50)

# extractor.activations[16] now contains activation at each generation step
layer_16_trajectory = torch.stack(extractor.activations[16])  # (num_steps, 4096)

extractor.remove_hooks()
```

## Complete detection pipeline

```python
def build_sycophancy_detector(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    contrastive_pairs: Dict[str, List[ContrastivePair]] = None,
    layers: List[int] = [12, 16, 20]
) -> SycophancyDetector:
    """Build complete detector from contrastive dataset."""
    
    extractor = SycophancyDirectionExtractor(model_id, layers)
    
    # Extract directions for each behavior type
    for behavior_name, pairs in contrastive_pairs.items():
        extractor.compute_direction(pairs, behavior_name)
    
    encoder = PromptActivationEncoder(
        extractor.model, 
        extractor.tokenizer, 
        layers
    )
    
    return SycophancyDetector(extractor.directions, encoder)

# Usage
detector = build_sycophancy_detector(
    contrastive_pairs={
        "sycophantic_agreement": sya_pairs,
        "sycophantic_praise": sypr_pairs
    }
)

# Measure a new prompt
scores = detector.measure_activation(
    "User: I think 2+2=5. Am I right?\nAssistant: Absolutely correct!",
    layer=16,
    use_differencing=True
)
print(f"Sycophancy scores: {scores}")
# Output: {'sycophantic_agreement': 0.42, 'sycophantic_praise': 0.08}

# Track during generation
trajectory = detector.measure_generation_trajectory(
    "User: I think the earth is flat. What do you think?\nAssistant:",
    layer=16
)
```

## Llama 3.1 8B chat template handling

Finding the correct token position after user input requires understanding the chat format:

```python
def find_user_end_position(tokenizer, messages: List[Dict]) -> int:
    """Find token position at end of user's turn (for activation extraction)."""
    # Tokenize full conversation with generation prompt
    full_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # The EOT token (128009) marks end of user turn
    input_ids = full_input[0].tolist()
    eot_id = 128009
    
    # Find positions of all EOT tokens
    eot_positions = [i for i, tid in enumerate(input_ids) if tid == eot_id]
    
    # Last EOT before assistant header is end of user turn
    # With add_generation_prompt=True, we want second-to-last EOT
    if len(eot_positions) >= 1:
        return eot_positions[-1]
    
    return len(input_ids) - 1  # Fallback to last position
```

## Memory considerations for activation caching

For Llama 3.1 8B with 32 layers and 4096-dim hidden states:

| Configuration | Memory per layer | All 32 layers |
|--------------|------------------|---------------|
| seq_len=512, batch=1, bf16 | **4 MB** | **128 MB** |
| seq_len=2048, batch=1, bf16 | 16 MB | 512 MB |
| seq_len=512, batch=8, bf16 | 32 MB | 1 GB |

For sycophancy detection, you typically only need 3-5 layers (middle layers 12-20), reducing cache requirements to ~15-25 MB per forward pass. The model itself requires **~16 GB VRAM** in bf16 or **~6 GB** with 4-bit quantization.

## Conclusion

This prototype architecture separates concerns cleanly: direction extraction computes normalized diff-of-means vectors from contrastive pairs at the EOS position, prompt encoding converts new inputs via empty-string differencing or raw embedding, and detection measures cosine similarity against stored directions. The NNSight approach offers cleaner syntax for simple extraction, while raw PyTorch hooks provide the control needed for generation-time monitoring. For Llama 3.1 8B, focus extraction on layers **12-20** and use the **`<|eot_id|>` token position (128009)** as the natural boundary for measuring complete user turns.