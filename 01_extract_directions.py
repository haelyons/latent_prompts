#!/usr/bin/env python3
"""
Datasets used:
- math_factorial, claims_factorial, companies_factorial
- cities_pos_factorial, cities_neg_factorial
- larger_than_factorial, smaller_than_factorial
- sp_en_trans_factorial, counterfactual_factorial

Behaviors extracted:
- SyA (Sycophantic Agreement): syc=1 vs syc=0, where user_claim_is_correct=False
- GA (Genuine Agreement): ga=1 vs ga=0, where user_claim_is_correct=True  
- SyPr (Sycophantic Praise): pr=1 (positive praise) vs pr=0 (neutral/negated)

To run: python 01_extract_directions.py
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

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

# Layer for extraction (from config)
LAYERS = [EXTRACTION_LAYER]

DATA_DIR = FACTORIAL_DATA_DIR
OUTPUT_DIR = str(DIRECTIONS_DIR)


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


def build_sya_pairs_from_dataset(data: List[Dict], n_pairs: int) -> List[Tuple[str, str]]:
    """Build SyA contrastive pairs from a single dataset."""
    user_wrong = [d for d in data if not d.get("user_claim_is_correct", True)]
    user_wrong_no_praise = [d for d in user_wrong if not d.get("praise_present", False)]
    
    pos = [d for d in user_wrong_no_praise if d.get("syc", 0) == 1]
    neg = [d for d in user_wrong_no_praise if d.get("syc", 0) == 0]
    
    if not pos or not neg:
        return []
    
    pairs = []
    prompt_to_neg = {}
    for n in neg:
        prompt_to_neg.setdefault(n["prompt"], []).append(n)
    
    random.shuffle(pos)
    for p in pos:
        if p["prompt"] in prompt_to_neg and prompt_to_neg[p["prompt"]]:
            n = prompt_to_neg[p["prompt"]].pop(0)
            pairs.append((p["prompt"] + p["response"], n["prompt"] + n["response"]))
            if len(pairs) >= n_pairs:
                break
    
    return pairs


def build_ga_pairs_from_dataset(data: List[Dict], n_pairs: int) -> List[Tuple[str, str]]:
    """Build GA contrastive pairs from a single dataset."""
    user_correct = [d for d in data if d.get("user_claim_is_correct", False)]
    user_correct_no_praise = [d for d in user_correct if not d.get("praise_present", False)]
    
    pos = [d for d in user_correct_no_praise if d.get("ga", 0) == 1]
    neg = [d for d in user_correct_no_praise if d.get("ga", 0) == 0]
    
    if not pos or not neg:
        return []
    
    pairs = []
    prompt_to_neg = {}
    for n in neg:
        prompt_to_neg.setdefault(n["prompt"], []).append(n)
    
    random.shuffle(pos)
    for p in pos:
        if p["prompt"] in prompt_to_neg and prompt_to_neg[p["prompt"]]:
            n = prompt_to_neg[p["prompt"]].pop(0)
            pairs.append((p["prompt"] + p["response"], n["prompt"] + n["response"]))
            if len(pairs) >= n_pairs:
                break
    
    return pairs


def build_sypr_pairs_from_dataset(data: List[Dict], n_pairs: int) -> List[Tuple[str, str]]:
    """Build SyPr contrastive pairs from a single dataset."""
    pos_praise = [d for d in data if d.get("praise_present", False) and not d.get("praise_negated", False)]
    no_praise = [d for d in data if not d.get("praise_present", False)]
    
    if not pos_praise or not no_praise:
        return []
    
    pairs = []
    neutral_lookup = {}
    for n in no_praise:
        key = (n["prompt"], n.get("response_value"))
        neutral_lookup.setdefault(key, []).append(n)
    
    random.shuffle(pos_praise)
    for p in pos_praise:
        key = (p["prompt"], p.get("response_value"))
        if key in neutral_lookup and neutral_lookup[key]:
            n = neutral_lookup[key].pop(0)
            pairs.append((p["prompt"] + p["response"], n["prompt"] + n["response"]))
            if len(pairs) >= n_pairs:
                break
    
    return pairs


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


def get_last_token_activation(model, tokenizer, text: str, layer: int) -> torch.Tensor:
    """Extract the hidden state at the last token position for a given layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer + 1]
    last_token_act = hidden_states[0, -1, :].float().cpu()
    
    return last_token_act


def compute_diffmean_direction(
    model, tokenizer, pairs: List[Tuple[str, str]], layer: int, desc: str = ""
) -> torch.Tensor:
    """Compute normalized difference-in-means direction."""
    if not pairs:
        return None
    
    pos_activations = []
    neg_activations = []
    
    for pos_text, neg_text in tqdm(pairs, desc=desc, leave=False):
        pos_act = get_last_token_activation(model, tokenizer, pos_text, layer)
        neg_act = get_last_token_activation(model, tokenizer, neg_text, layer)
        pos_activations.append(pos_act)
        neg_activations.append(neg_act)
    
    pos_mean = torch.stack(pos_activations).mean(dim=0)
    neg_mean = torch.stack(neg_activations).mean(dim=0)
    
    direction = pos_mean - neg_mean
    direction = direction / direction.norm()
    
    return direction


def aggregate_directions_svd(directions: List[torch.Tensor]) -> torch.Tensor:
    """
    Aggregate multiple direction vectors via SVD.
    
    Per paper: "normalized and stacked into a matrix M, from which we compute
    an orthonormal basis U via SVD" - then take top principal component.
    """
    # Filter out None directions
    valid_directions = [d for d in directions if d is not None]
    
    if not valid_directions:
        raise ValueError("No valid directions to aggregate")
    
    # Stack into matrix (hidden_dim x n_datasets)
    M = torch.stack(valid_directions, dim=1)  # Shape: (D, N)
    
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


def compute_auroc(model, tokenizer, direction: torch.Tensor, data: List[Dict], 
                  label_key: str, layer: int, n_samples: int = 500) -> float:
    """Compute AUROC to validate direction quality."""
    from sklearn.metrics import roc_auc_score
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    scores = []
    labels = []
    
    for sample in tqdm(samples, desc=f"AUROC {label_key}", leave=False):
        text = sample["prompt"] + sample["response"]
        activation = get_last_token_activation(model, tokenizer, text, layer)
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
    print(f"Pairs per dataset: {N_PAIRS_PER_DATASET}")
    print(f"Datasets: {len(FACTORIAL_DATASETS)}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    # Load all datasets
    all_datasets = load_all_factorial_datasets()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Build pair functions for each behavior
    pair_builders = {
        "sya": build_sya_pairs_from_dataset,
        "ga": build_ga_pairs_from_dataset,
        "sypr": build_sypr_pairs_from_dataset,
    }
    
    directions = {}
    per_dataset_directions = {}
    
    for layer in LAYERS:
        print(f"\nl:{layer}")
        
        for behavior in ["sya", "ga", "sypr"]:
            print(f"\n  Extracting {behavior.upper()} directions from each dataset...")
            
            dataset_directions = []
            per_dataset_directions.setdefault(behavior, {}).setdefault(layer, {})
            
            for dataset_name, data in all_datasets.items():
                # Build pairs for this dataset
                pairs = pair_builders[behavior](data, N_PAIRS_PER_DATASET)
                
                if len(pairs) < 10:
                    print(f"    {dataset_name}: skipped (only {len(pairs)} pairs)")
                    continue
                
                # Compute direction for this dataset
                direction = compute_diffmean_direction(
                    model, tokenizer, pairs, layer,
                    desc=f"    {dataset_name}"
                )
                
                if direction is not None:
                    dataset_directions.append(direction)
                    per_dataset_directions[behavior][layer][dataset_name] = direction
                    print(f"    {dataset_name}: {len(pairs)} pairs ✓")
            
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
    
    label_map = {"sya": "syc", "ga": "ga", "sypr": "pr"}
    auroc_results = {}
    
    for behavior in ["sya", "ga", "sypr"]:
        label_key = label_map[behavior]
        auroc_results[behavior] = {}
        
        for layer in LAYERS:
            if behavior == "sya":
                val_subset = [d for d in val_data if not d.get("user_claim_is_correct", True)]
            elif behavior == "ga":
                val_subset = [d for d in val_data if d.get("user_claim_is_correct", False)]
            else:
                val_subset = val_data
            
            auroc = compute_auroc(
                model, tokenizer,
                directions[behavior][layer],
                val_subset, label_key, layer,
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
    
    behavior_descriptions = {
        "sya": "Sycophantic Agreement - echoing incorrect user claims",
        "ga": "Genuine Agreement - echoing correct user claims", 
        "sypr": "Sycophantic Praise - excessive flattery",
    }
    
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
            n_datasets = len([d for d in per_dataset_directions[behavior][layer].values() if d is not None])
            total_pairs = n_datasets * N_PAIRS_PER_DATASET
            count_str = format_sample_count(total_pairs)
            filename = f"{behavior}_layer{layer}_n{count_str}_svd.pt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            direction_data = {
                "direction": directions[behavior][layer],
                "behavior": behavior,
                "description": behavior_descriptions[behavior],
                "layer": layer,
                "n_pairs": total_pairs,
                "n_datasets": n_datasets,
                "datasets_used": list(per_dataset_directions[behavior][layer].keys()),
                "model_id": MODEL_ID,
                "auroc": auroc_results[behavior][layer],
                "method": "svd_aggregation",
            }
            torch.save(direction_data, filepath)
            print(f"  Saved: {filename}")
    
    print(f"Datasets aggregated: {len(all_datasets)}")
    print(f"Total pairs processed: ~{len(all_datasets) * N_PAIRS_PER_DATASET * 3}")
    for behavior in ["sya", "ga", "sypr"]:
        print(f"{behavior.upper()}: AUROC={auroc_results[behavior][LAYERS[0]]:.4f}")
    print(f"\nSyA-GA separation: {ortho['sya_ga']:+.4f} (lower = better disentanglement)")


if __name__ == "__main__":
    main()
