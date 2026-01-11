#!/usr/bin/env python3
"""
Extract disentangled sycophancy directions using the paper's DiffMean approach.

Datasets used:
- math_factorial, claims_factorial, companies_factorial
- cities_pos_factorial, cities_neg_factorial
- larger_than_factorial, smaller_than_factorial
- sp_en_trans_factorial, counterfactual_factorial

Behaviors extracted:
- SyA (Sycophantic Agreement): syc=1 vs syc=0, where user_claim_is_correct=False
- GA (Genuine Agreement): ga=1 vs ga=0, where user_claim_is_correct=True  
- SyPr (Sycophantic Praise): pr=1 vs pr=0 (uses pr field directly per paper)

Method: Direct label-based DiffMean (paper's approach)
- Extracts activations from ALL examples
- Groups by label, computes mean difference
- Aggregates across datasets via SVD

To run: python 01_extract_directions.py
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_ID,
    EXTRACTION_LAYER,
    N_PAIRS_PER_DATASET,
    FACTORIAL_DATASETS,
    FACTORIAL_DATA_DIR,
    DISENTANGLE_DATA_PATH,
    DIRECTIONS_DIR,
    VALIDATION_SPLIT,
    MIN_AUROC_THRESHOLD,
    SEED,
    BEHAVIORS,
)

# Add disentangle-sycophancy to path for utilities
DISENTANGLE_PATH = DISENTANGLE_DATA_PATH.parent
sys.path.insert(0, str(DISENTANGLE_PATH / "src"))

try:
    from utils.file_io import load_json
except ImportError:
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

LAYERS = [EXTRACTION_LAYER]

DATA_DIR = FACTORIAL_DATA_DIR
OUTPUT_DIR = str(DIRECTIONS_DIR)

POOL_STRATEGY = "pre_eos"

def load_all_factorial_datasets() -> Dict[str, List[Dict]]:
    """Load all 9 factorial datasets."""
    datasets = {}
    print(f"Loading {len(FACTORIAL_DATASETS)} factorial datasets...")
    
    for filename in FACTORIAL_DATASETS:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  WARNING: {filename} not found, skipping")
            continue
        
        data = load_json(str(path))
        name = filename.replace(".json", "")
        datasets[name] = data
        print(f"  {name}: {len(data):,} examples")
    
    total = sum(len(d) for d in datasets.values())
    print(f"  Total: {total:,} examples across {len(datasets)} datasets")
    return datasets


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def build_special_token_set(tokenizer) -> Set[int]:
    """Build set of special token IDs to skip during pooling."""
    special_ids = set()
    if hasattr(tokenizer, 'all_special_ids'):
        special_ids.update(tokenizer.all_special_ids)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    return special_ids


def get_pooled_activation(
    model, 
    tokenizer, 
    text: str, 
    layer: int,
    special_ids: Set[int],
    pool_strategy: str = "pre_eos"
) -> torch.Tensor:
    """
    Extract hidden state at the appropriate token position.
    
    Paper's approach (Appendix F): "extract h at the end of sentence token 
    following the response at the post-layernorm residual stream"
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text (prompt + response)
        layer: Layer index to extract from
        special_ids: Set of special token IDs to skip
        pool_strategy: 'pre_eos' (default) or 'last'
    
    Returns:
        Hidden state tensor at the selected position
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    seq_len = len(input_ids)
    
    # Find the appropriate token position
    if pool_strategy == "pre_eos":
        # Find EOS position
        eos_id = tokenizer.eos_token_id
        eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)[0]
        
        if len(eos_positions) > 0:
            # Use LAST EOS (handles chat templates with multiple turns)
            idx = eos_positions[-1].item() - 1
        else:
            # No EOS found, start from last token
            idx = seq_len - 1
        
        # Skip special tokens going backwards
        while idx > 0 and input_ids[idx].item() in special_ids:
            idx -= 1
            
    else:  # 'last' strategy
        # Start from last token
        idx = seq_len - 1
        # Skip special tokens going backwards
        while idx > 0 and input_ids[idx].item() in special_ids:
            idx -= 1
    
    # Ensure valid index
    idx = max(0, idx)
    
    # Extract hidden state
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states[0] = embeddings, hidden_states[layer+1] = after layer `layer`
    hidden_states = outputs.hidden_states[layer + 1]
    activation = hidden_states[0, idx, :].float().cpu()
    
    return activation


def filter_data_for_behavior(data: List[Dict], behavior: str) -> List[Dict]:
    """
    Filter dataset for behavior-specific examples.
    
    Per paper:
    - SyA: Only where user_claim_is_correct=False (user is wrong)
    - GA: Only where user_claim_is_correct=True (user is correct)
    - SyPr: All examples (pr field already encodes the label correctly)
    """
    if behavior == "sya":
        # SyA: user must be wrong, exclude praise to isolate agreement
        return [d for d in data 
                if not d.get("user_claim_is_correct", True)
                and not d.get("praise_present", False)]
    elif behavior == "ga":
        # GA: user must be correct, exclude praise to isolate agreement
        return [d for d in data 
                if d.get("user_claim_is_correct", False)
                and not d.get("praise_present", False)]
    elif behavior == "sypr":
        # SyPr: use ALL examples - pr field correctly encodes:
        #   pr=1: sycophantic praise (praise_present=True, not negated, positive adjective)
        #   pr=0: no praise OR negated praise OR neutral phrases
        return data
    else:
        return data


def compute_diffmean_direction_label_based(
    model,
    tokenizer,
    data: List[Dict],
    label_key: str,
    layer: int,
    special_ids: Set[int],
    max_examples: Optional[int] = None,
    desc: str = ""
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute DiffMean direction using the paper's label-based approach.
    
    Instead of building explicit pairs, accumulates means across ALL examples
    grouped by label value.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data: List of examples with 'prompt', 'response', and label_key fields
        label_key: Field name for the label (e.g., 'syc', 'ga', 'pr')
        layer: Layer index to extract from
        special_ids: Set of special token IDs
        max_examples: Optional limit on examples per class
        desc: Description for progress bar
    
    Returns:
        (direction, pos_count, neg_count)
    """
    # Separate by label
    pos_examples = [d for d in data if d.get(label_key, 0) == 1]
    neg_examples = [d for d in data if d.get(label_key, 0) == 0]
    
    # Optionally limit examples
    if max_examples is not None:
        random.shuffle(pos_examples)
        random.shuffle(neg_examples)
        pos_examples = pos_examples[:max_examples]
        neg_examples = neg_examples[:max_examples]
    
    if not pos_examples or not neg_examples:
        return None, 0, 0
    
    # Get hidden dimension from first example
    sample_text = pos_examples[0]["prompt"] + pos_examples[0]["response"]
    sample_act = get_pooled_activation(model, tokenizer, sample_text, layer, special_ids, POOL_STRATEGY)
    hidden_dim = sample_act.shape[0]
    
    # Accumulate sums
    pos_sum = torch.zeros(hidden_dim)
    neg_sum = torch.zeros(hidden_dim)
    
    # Process positive examples
    for ex in tqdm(pos_examples, desc=f"{desc} pos", leave=False):
        text = ex["prompt"] + ex["response"]
        act = get_pooled_activation(model, tokenizer, text, layer, special_ids, POOL_STRATEGY)
        pos_sum += act
    
    # Process negative examples
    for ex in tqdm(neg_examples, desc=f"{desc} neg", leave=False):
        text = ex["prompt"] + ex["response"]
        act = get_pooled_activation(model, tokenizer, text, layer, special_ids, POOL_STRATEGY)
        neg_sum += act
    
    # Compute means
    pos_mean = pos_sum / len(pos_examples)
    neg_mean = neg_sum / len(neg_examples)
    
    # Direction = mean(pos) - mean(neg), normalized
    direction = pos_mean - neg_mean
    direction = direction / direction.norm()
    
    return direction, len(pos_examples), len(neg_examples)


def aggregate_directions_svd(directions: List[torch.Tensor]) -> torch.Tensor:
    """
    Aggregate multiple direction vectors via SVD.
    
    Per paper: "normalized and stacked into a matrix M, from which we compute
    an orthonormal basis U via SVD" - then take top principal component.
    """
    valid_directions = [d for d in directions if d is not None]
    
    if not valid_directions:
        raise ValueError("No valid directions to aggregate")
    
    # Stack into matrix (hidden_dim x n_datasets)
    M = torch.stack(valid_directions, dim=1)
    
    # SVD: M = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    
    # Top principal component (first column of U)
    principal_direction = U[:, 0]
    
    # Report variance explained
    total_var = (S ** 2).sum()
    first_var = S[0] ** 2
    var_explained = (first_var / total_var).item()
    
    print(f"    SVD: {len(valid_directions)} directions → PC1 explains {var_explained:.1%} variance")
    
    return principal_direction


def compute_auroc(
    model,
    tokenizer,
    direction: torch.Tensor,
    data: List[Dict],
    label_key: str,
    layer: int,
    special_ids: Set[int],
    n_samples: int = 500
) -> float:
    """Compute AUROC to validate direction quality."""
    from sklearn.metrics import roc_auc_score
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    scores = []
    labels = []
    
    for sample in tqdm(samples, desc=f"AUROC {label_key}", leave=False):
        text = sample["prompt"] + sample["response"]
        activation = get_pooled_activation(model, tokenizer, text, layer, special_ids, POOL_STRATEGY)
        activation_norm = activation / activation.norm()
        
        score = torch.dot(activation_norm, direction).item()
        scores.append(score)
        labels.append(sample.get(label_key, 0))
    
    if len(set(labels)) < 2:
        return 0.5
    
    try:
        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = 1.0 - auroc
        return auroc
    except Exception:
        return 0.5


def check_orthogonality(directions: Dict, layer: int) -> Dict[str, float]:
    """Check pairwise cosine similarity between directions."""
    sya = directions["sya"][layer]
    ga = directions["ga"][layer]
    sypr = directions["sypr"][layer]
    
    return {
        "sya_ga": torch.dot(sya, ga).item(),
        "sya_sypr": torch.dot(sya, sypr).item(),
        "ga_sypr": torch.dot(ga, sypr).item(),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"Model: {MODEL_ID}")
    print(f"Layers: {LAYERS}")
    print(f"Max examples per class per dataset: {N_PAIRS_PER_DATASET}")
    print(f"Datasets: {len(FACTORIAL_DATASETS)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Pooling strategy: {POOL_STRATEGY}")
    
    # Load all datasets
    all_datasets = load_all_factorial_datasets()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Build special token set for pooling
    special_ids = build_special_token_set(tokenizer)
    print(f"Special token IDs to skip: {special_ids}")
    
    # Behavior configurations
    # label_key: field in data that contains the binary label
    behavior_configs = {
        "sya": {"label_key": "syc", "description": "Sycophantic Agreement - echoing incorrect user claims"},
        "ga": {"label_key": "ga", "description": "Genuine Agreement - echoing correct user claims"},
        "sypr": {"label_key": "pr", "description": "Sycophantic Praise - excessive flattery"},
    }
    
    directions = {}
    per_dataset_directions = {}
    per_dataset_counts = {}
    
    for layer in LAYERS:
        print(f"\nLayer {layer}")
        
        for behavior in ["sya", "ga", "sypr"]:
            config = behavior_configs[behavior]
            label_key = config["label_key"]
            
            print(f"\n  Extracting {behavior.upper()} directions from each dataset...")
            print(f"    Label field: {label_key}")
            
            dataset_directions = []
            per_dataset_directions.setdefault(behavior, {}).setdefault(layer, {})
            per_dataset_counts.setdefault(behavior, {}).setdefault(layer, {})
            
            for dataset_name, data in all_datasets.items():
                # Filter data for this behavior
                filtered_data = filter_data_for_behavior(data, behavior)
                
                if len(filtered_data) < 20:
                    print(f"    {dataset_name}: skipped (only {len(filtered_data)} examples after filtering)")
                    continue
                
                # Compute direction using label-based DiffMean
                direction, pos_count, neg_count = compute_diffmean_direction_label_based(
                    model, tokenizer,
                    filtered_data,
                    label_key,
                    layer,
                    special_ids,
                    max_examples=N_PAIRS_PER_DATASET,
                    desc=f"    {dataset_name}"
                )
                
                if direction is not None:
                    dataset_directions.append(direction)
                    per_dataset_directions[behavior][layer][dataset_name] = direction
                    per_dataset_counts[behavior][layer][dataset_name] = (pos_count, neg_count)
                    print(f"    {dataset_name}: pos={pos_count}, neg={neg_count} ✓")
                else:
                    print(f"    {dataset_name}: skipped (missing pos or neg examples)")
            
            # Aggregate via SVD
            print(f"  Aggregating {behavior.upper()} via SVD...")
            aggregated = aggregate_directions_svd(dataset_directions)
            
            directions.setdefault(behavior, {})[layer] = aggregated
    
    # Combine all datasets for validation
    all_data = []
    for data in all_datasets.values():
        all_data.extend(data)
    random.shuffle(all_data)
    val_data = all_data[int(len(all_data) * (1 - VALIDATION_SPLIT)):]
    
    print(f"\nComputing AUROC on {len(val_data):,} held-out examples...")
    
    auroc_results = {}
    
    for behavior in ["sya", "ga", "sypr"]:
        label_key = behavior_configs[behavior]["label_key"]
        auroc_results[behavior] = {}
        
        for layer in LAYERS:
            # Filter validation data appropriately
            if behavior == "sya":
                val_subset = [d for d in val_data if not d.get("user_claim_is_correct", True)]
            elif behavior == "ga":
                val_subset = [d for d in val_data if d.get("user_claim_is_correct", False)]
            else:  # sypr
                val_subset = val_data
            
            auroc = compute_auroc(
                model, tokenizer,
                directions[behavior][layer],
                val_subset, label_key, layer,
                special_ids,
                n_samples=min(500, len(val_subset))
            )
            auroc_results[behavior][layer] = auroc
            
            status = "✓" if auroc >= MIN_AUROC_THRESHOLD else "⚠"
            print(f"  {behavior.upper()} layer {layer}: AUROC = {auroc:.4f} {status}")
    
    print(f"\nOrthogonality (layer {LAYERS[0]}):")
    ortho = check_orthogonality(directions, LAYERS[0])
    print(f"  SyA-GA cosine:   {ortho['sya_ga']:+.4f}")
    print(f"  SyA-SyPr cosine: {ortho['sya_sypr']:+.4f}")
    print(f"  GA-SyPr cosine:  {ortho['ga_sypr']:+.4f}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nSaving directions...")
    
    def format_sample_count(n: int) -> str:
        if n >= 1000:
            thousands = n // 1000
            hundreds = (n % 1000) // 100
            if hundreds > 0:
                return f"{thousands}k{hundreds}"
            return f"{thousands}k"
        return str(n)
    
    for behavior in ["sya", "ga", "sypr"]:
        for layer in LAYERS:
            # Calculate total examples used
            counts = per_dataset_counts[behavior][layer]
            total_pos = sum(c[0] for c in counts.values())
            total_neg = sum(c[1] for c in counts.values())
            total_examples = total_pos + total_neg
            
            count_str = format_sample_count(total_examples)
            filename = f"{behavior}_layer{layer}_n{count_str}_svd.pt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            direction_data = {
                "direction": directions[behavior][layer],
                "behavior": behavior,
                "description": behavior_configs[behavior]["description"],
                "layer": layer,
                "n_examples": total_examples,
                "n_pos": total_pos,
                "n_neg": total_neg,
                "n_datasets": len(per_dataset_directions[behavior][layer]),
                "datasets_used": list(per_dataset_directions[behavior][layer].keys()),
                "model_id": MODEL_ID,
                "auroc": auroc_results[behavior][layer],
                "method": "label_based_diffmean_svd",
                "pool_strategy": POOL_STRATEGY,
            }
            torch.save(direction_data, filepath)
            print(f"  Saved: {filename}")
    
    print(f"  Datasets aggregated: {len(all_datasets)}")
    print(f"  Pool strategy: {POOL_STRATEGY}")
    for behavior in ["sya", "ga", "sypr"]:
        counts = per_dataset_counts[behavior][LAYERS[0]]
        total = sum(c[0] + c[1] for c in counts.values())
        print(f"  {behavior.upper()}: {total} examples, AUROC={auroc_results[behavior][LAYERS[0]]:.4f}")


if __name__ == "__main__":
    main()
