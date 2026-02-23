#!/usr/bin/env python3
"""
Extract disentangled sycophancy directions for large models (70B default).

Same methodology as 01_extract_directions.py (DiffMean + SVD on 9 factorial
datasets) but uses hook-based activation extraction (model_utils) instead of
output_hidden_states=True, which would OOM at 70B.  Extracts at multiple
candidate layers in a single forward pass per example.

Behaviors extracted:
  SyA  — Sycophantic Agreement (syc=1 vs syc=0, user_claim_is_correct=False)
  GA   — Genuine Agreement     (ga=1  vs ga=0,  user_claim_is_correct=True)
  SyPr — Sycophantic Praise    (pr=1  vs pr=0,  all examples)

Usage:
    python 01b_extract_directions_70b.py                  # 70B @ layers [55,60,65]
    python 01b_extract_directions_70b.py --model 8b       # 8B  @ layer  [24]
    python 01b_extract_directions_70b.py --layers 47,55,60,65
    python 01b_extract_directions_70b.py --dry-run        # data loading only
"""

import sys
import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import torch
import numpy as np
from tqdm import tqdm

from config import (
    get_profile,
    N_PAIRS_PER_DATASET,
    FACTORIAL_DATASETS,
    FACTORIAL_DATA_DIR,
    DISENTANGLE_DATA_PATH,
    DIRECTIONS_DIR,
    VALIDATION_SPLIT,
    MIN_AUROC_THRESHOLD,
    VALIDATION_SAMPLES,
    SEED,
)
from model_utils import (
    load_model_and_tokenizer,
    build_special_token_set,
    get_pooled_activations,
    direction_filename,
    BEHAVIOR_NAMES,
)

DISENTANGLE_PATH = DISENTANGLE_DATA_PATH.parent
sys.path.insert(0, str(DISENTANGLE_PATH / "src"))

try:
    from utils.file_io import load_json
except ImportError:
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


BEHAVIOR_CONFIGS = {
    "sya": {"label_key": "syc", "description": "Sycophantic Agreement - echoing incorrect user claims"},
    "ga":  {"label_key": "ga",  "description": "Genuine Agreement - echoing correct user claims"},
    "sypr": {"label_key": "pr", "description": "Sycophantic Praise - excessive flattery"},
}


def load_all_factorial_datasets() -> Dict[str, List[Dict]]:
    datasets = {}
    print(f"Loading {len(FACTORIAL_DATASETS)} factorial datasets...")
    for filename in FACTORIAL_DATASETS:
        path = FACTORIAL_DATA_DIR / filename
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


def filter_data_for_behavior(data: List[Dict], behavior: str) -> List[Dict]:
    """Filter dataset for behavior-specific examples (identical to 01)."""
    if behavior == "sya":
        return [d for d in data
                if not d.get("user_claim_is_correct", True)
                and not d.get("praise_present", False)]
    elif behavior == "ga":
        return [d for d in data
                if d.get("user_claim_is_correct", False)
                and not d.get("praise_present", False)]
    elif behavior == "sypr":
        return data
    else:
        return data


# multi-layer diff-mean
def compute_diffmean_multilayer(
    model,
    tokenizer,
    data: List[Dict],
    label_key: str,
    layers: List[int],
    special_ids: Set[int],
    max_examples: Optional[int] = None,
    desc: str = "",
    pool_strategy: str = "eos",
) -> Tuple[Dict[int, Optional[torch.Tensor]], int, int]:
    """
    Label-based DiffMean at all candidate layers in one forward pass per
    example.  Returns ({layer: normalised_direction}, pos_count, neg_count).
    """
    pos_examples = [d for d in data if d.get(label_key, 0) == 1]
    neg_examples = [d for d in data if d.get(label_key, 0) == 0]

    if max_examples is not None:
        random.shuffle(pos_examples)
        random.shuffle(neg_examples)
        pos_examples = pos_examples[:max_examples]
        neg_examples = neg_examples[:max_examples]

    if not pos_examples or not neg_examples:
        return {layer: None for layer in layers}, 0, 0

    # Probe hidden dimension
    sample_text = pos_examples[0]["prompt"] + pos_examples[0]["response"]
    sample_acts = get_pooled_activations(model, tokenizer, sample_text, layers, special_ids, pool_strategy)
    hidden_dim = next(iter(sample_acts.values())).shape[0]

    pos_sums = {l: torch.zeros(hidden_dim) for l in layers}
    neg_sums = {l: torch.zeros(hidden_dim) for l in layers}

    for ex in tqdm(pos_examples, desc=f"{desc} pos", leave=False):
        text = ex["prompt"] + ex["response"]
        acts = get_pooled_activations(model, tokenizer, text, layers, special_ids, pool_strategy)
        for l in layers:
            pos_sums[l] += acts[l]

    for ex in tqdm(neg_examples, desc=f"{desc} neg", leave=False):
        text = ex["prompt"] + ex["response"]
        acts = get_pooled_activations(model, tokenizer, text, layers, special_ids, pool_strategy)
        for l in layers:
            neg_sums[l] += acts[l]

    directions = {}
    for l in layers:
        d = pos_sums[l] / len(pos_examples) - neg_sums[l] / len(neg_examples)
        d = d / d.norm()
        directions[l] = d

    return directions, len(pos_examples), len(neg_examples)

# previous version + variance explained
def aggregate_directions_svd(
    directions: List[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    valid = [d for d in directions if d is not None]
    if not valid:
        raise ValueError("No valid directions to aggregate")
    M = torch.stack(valid, dim=1)
    U, S, _Vt = torch.linalg.svd(M, full_matrices=False)
    principal = U[:, 0]
    var_explained = (S[0] ** 2 / (S ** 2).sum()).item()
    print(f"    SVD: {len(valid)} directions → PC1 explains {var_explained:.1%} variance")
    return principal, var_explained


# AUROC validation — multi-layer
def compute_auroc_all_layers(
    model,
    tokenizer,
    directions_by_layer: Dict[int, torch.Tensor],
    data: List[Dict],
    label_key: str,
    layers: List[int],
    special_ids: Set[int],
    n_samples: int = 500,
    pool_strategy: str = "eos",
) -> Tuple[Dict[int, float], Dict[int, torch.Tensor]]:
    """
    AUROC for one behavior across all candidate layers, sharing forward
    passes.  Also aligns direction sign: if raw AUROC < 0.5 the direction
    is flipped so that positive projection ≈ sycophantic.

    Returns (auroc_by_layer, aligned_directions_by_layer).
    """
    from sklearn.metrics import roc_auc_score

    samples = random.sample(data, min(n_samples, len(data)))
    scores_by_layer: Dict[int, list] = {l: [] for l in layers}
    labels: List[int] = []

    for sample in tqdm(samples, desc=f"AUROC {label_key}", leave=False):
        text = sample["prompt"] + sample["response"]
        acts = get_pooled_activations(model, tokenizer, text, layers, special_ids, pool_strategy)
        for l in layers:
            act_norm = acts[l] / acts[l].norm()
            scores_by_layer[l].append(torch.dot(act_norm, directions_by_layer[l]).item())
        labels.append(sample.get(label_key, 0))

    if len(set(labels)) < 2:
        return {l: 0.5 for l in layers}, dict(directions_by_layer)

    aurocs = {}
    aligned = {}
    for l in layers:
        try:
            raw = roc_auc_score(labels, scores_by_layer[l])
        except Exception:
            raw = 0.5
        if raw < 0.5:
            aurocs[l] = 1.0 - raw
            aligned[l] = -directions_by_layer[l]
        else:
            aurocs[l] = raw
            aligned[l] = directions_by_layer[l]
    return aurocs, aligned


def check_orthogonality(
    directions: Dict[str, Dict[int, torch.Tensor]], layer: int,
) -> Dict[str, float]:
    sya = directions["sya"][layer]
    ga = directions["ga"][layer]
    sypr = directions["sypr"][layer]
    return {
        "sya_ga":   torch.dot(sya, ga).item(),
        "sya_sypr": torch.dot(sya, sypr).item(),
        "ga_sypr":  torch.dot(ga, sypr).item(),
    }


def parse_args():
    from config import MODEL_PROFILES
    parser = argparse.ArgumentParser(
        description="Extract sycophancy directions (multi-scale, hook-based)",
    )
    parser.add_argument(
        "--model", default="70b", choices=list(MODEL_PROFILES.keys()),
        help="Model profile (default: 70b)",
    )
    parser.add_argument(
        "--layers", default=None, type=str,
        help="Override candidate layers, comma-separated (e.g. 47,55,60,65)",
    )
    parser.add_argument(
        "--pool-strategy", default="eos",
        choices=["eos", "pre_eos", "last"],
        help="Token position for activation extraction (default: eos)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and filter data only — no model, no GPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    profile = get_profile(args.model)
    model_tag = args.model if args.model != "8b" else ""
    pool_strategy = args.pool_strategy

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else profile["candidate_layers"]
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Model:     {profile['model_id']}")
    print(f"Layers:    {layers}")
    print(f"N layers:  {profile['n_layers']}")
    print(f"Pool:      {pool_strategy}")
    print(f"Model tag: {model_tag or '(none — 8B compat)'}")
    print(f"Max examples per class per dataset: {N_PAIRS_PER_DATASET}")
    print(f"Datasets:  {len(FACTORIAL_DATASETS)}")
    print()

    all_datasets = load_all_factorial_datasets()

    if args.dry_run:
        print("\n--- dry-run: data summary ---")
        for behavior in BEHAVIOR_NAMES:
            label_key = BEHAVIOR_CONFIGS[behavior]["label_key"]
            for ds_name, ds_data in all_datasets.items():
                filtered = filter_data_for_behavior(ds_data, behavior)
                pos = sum(1 for d in filtered if d.get(label_key, 0) == 1)
                neg = sum(1 for d in filtered if d.get(label_key, 0) == 0)
                print(f"  {behavior.upper():4s} {ds_name:30s}  pos={pos:4d}  neg={neg:4d}")
        return

    model, tokenizer = load_model_and_tokenizer(profile)
    special_ids = build_special_token_set(tokenizer)
    print(f"Special token IDs to skip: {special_ids}")

    # Structure: directions[behavior][layer] = aggregated direction tensor
    directions: Dict[str, Dict[int, torch.Tensor]] = {}
    per_dataset_directions: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    per_dataset_counts: Dict[str, Dict[int, Dict[str, Tuple[int, int]]]] = {}
    svd_variance: Dict[str, Dict[int, float]] = {}

    for behavior in BEHAVIOR_NAMES:
        label_key = BEHAVIOR_CONFIGS[behavior]["label_key"]
        print(f"Extracting {behavior.upper()} (label: {label_key})")

        per_dataset_directions[behavior] = {l: {} for l in layers}
        per_dataset_counts[behavior] = {l: {} for l in layers}
        svd_variance[behavior] = {}

        # Per-dataset DiffMean (all layers in one pass per example)
        dataset_dirs_by_layer: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

        for ds_name, ds_data in all_datasets.items():
            filtered = filter_data_for_behavior(ds_data, behavior)
            if len(filtered) < 20:
                print(f"  {ds_name}: skipped ({len(filtered)} examples after filtering)")
                continue

            dir_by_layer, n_pos, n_neg = compute_diffmean_multilayer(
                model, tokenizer, filtered, label_key, layers, special_ids,
                max_examples=N_PAIRS_PER_DATASET,
                desc=f"  {ds_name}",
                pool_strategy=pool_strategy,
            )

            for l in layers:
                if dir_by_layer[l] is not None:
                    dataset_dirs_by_layer[l].append(dir_by_layer[l])
                    per_dataset_directions[behavior][l][ds_name] = dir_by_layer[l]
                    per_dataset_counts[behavior][l][ds_name] = (n_pos, n_neg)

            print(f"  {ds_name}: pos={n_pos}, neg={n_neg} ✓")

        # SVD aggregation per layer
        directions[behavior] = {}
        for l in layers:
            print(f"\n  Aggregating {behavior.upper()} layer {l}...")
            agg, var_exp = aggregate_directions_svd(dataset_dirs_by_layer[l])
            directions[behavior][l] = agg
            svd_variance[behavior][l] = var_exp

    all_data = []
    for ds_data in all_datasets.values():
        all_data.extend(ds_data)
    random.shuffle(all_data)
    val_data = all_data[int(len(all_data) * (1 - VALIDATION_SPLIT)):]
    print(f"\nComputing AUROC on {len(val_data):,} held-out examples...")

    auroc_results: Dict[str, Dict[int, float]] = {}

    for behavior in BEHAVIOR_NAMES:
        label_key = BEHAVIOR_CONFIGS[behavior]["label_key"]

        if behavior == "sya":
            val_subset = [d for d in val_data if not d.get("user_claim_is_correct", True)]
        elif behavior == "ga":
            val_subset = [d for d in val_data if d.get("user_claim_is_correct", False)]
        else:
            val_subset = val_data

        aurocs, aligned = compute_auroc_all_layers(
            model, tokenizer, directions[behavior],
            val_subset, label_key, layers, special_ids,
            n_samples=min(VALIDATION_SAMPLES, len(val_subset)),
            pool_strategy=pool_strategy,
        )
        auroc_results[behavior] = aurocs
        directions[behavior] = aligned  # sign-aligned

        for l in layers:
            status = "✓" if aurocs[l] >= MIN_AUROC_THRESHOLD else "⚠"
            print(f"  {behavior.upper()} layer {l}: AUROC = {aurocs[l]:.4f} {status}")

    for l in layers:
        print(f"\nOrthogonality (layer {l}):")
        ortho = check_orthogonality(directions, l)
        print(f"  SyA-GA cosine:   {ortho['sya_ga']:+.4f}")
        print(f"  SyA-SyPr cosine: {ortho['sya_sypr']:+.4f}")
        print(f"  GA-SyPr cosine:  {ortho['ga_sypr']:+.4f}")

    save_dir = get_directions_dir(model_tag)
    os.makedirs(str(save_dir), exist_ok=True)
    print(f"\nSaving directions to {save_dir}/")

    for behavior in BEHAVIOR_NAMES:
        for l in layers:
            counts = per_dataset_counts[behavior][l]
            total_pos = sum(c[0] for c in counts.values())
            total_neg = sum(c[1] for c in counts.values())
            total_examples = total_pos + total_neg

            fname = direction_filename(behavior, l, total_examples)
            filepath = save_dir / fname

            direction_data = {
                "direction": directions[behavior][l],
                "behavior": behavior,
                "description": BEHAVIOR_CONFIGS[behavior]["description"],
                "layer": l,
                "n_examples": total_examples,
                "n_pos": total_pos,
                "n_neg": total_neg,
                "n_datasets": len(per_dataset_directions[behavior][l]),
                "datasets_used": list(per_dataset_directions[behavior][l].keys()),
                "model_id": profile["model_id"],
                "model_tag": model_tag,
                "auroc": auroc_results[behavior][l],
                "svd_variance_explained": svd_variance[behavior][l],
                "method": "label_based_diffmean_svd",
                "pool_strategy": pool_strategy,
            }
            torch.save(direction_data, str(filepath))
            print(f"  Saved: {fname}")

    print("SUMMARY")
    print(f"Model: {profile['model_id']}")
    print(f"Pool strategy: {pool_strategy}")
    print(f"Datasets aggregated: {len(all_datasets)}")

    header = f"{'Behavior':<8}"
    for l in layers:
        header += f"  {'L' + str(l):>10}"
    print(f"\n{header}")

    print(f"{'':8}" + "  ".join(f"{'AUROC':>10}" for _ in layers))
    for behavior in BEHAVIOR_NAMES:
        row = f"{behavior.upper():<8}"
        for l in layers:
            row += f"  {auroc_results[behavior][l]:>10.4f}"
        print(row)

    print(f"\n{'':8}" + "  ".join(f"{'SVD var%':>10}" for _ in layers))
    for behavior in BEHAVIOR_NAMES:
        row = f"{behavior.upper():<8}"
        for l in layers:
            row += f"  {svd_variance[behavior][l]:>9.1%}"
        print(row)

    print(f"\n{'':8}" + "  ".join(f"{'ortho':>10}" for _ in layers))
    for l in layers:
        ortho = check_orthogonality(directions, l)
        print(f"  Layer {l}:  SyA-GA {ortho['sya_ga']:+.3f}  "
              f"SyA-SyPr {ortho['sya_sypr']:+.3f}  "
              f"GA-SyPr {ortho['ga_sypr']:+.3f}")

    # Cross-layer direction stability (if multiple layers)
    if len(layers) > 1:
        print(f"\nCross-layer direction cosine similarity:")
        for behavior in BEHAVIOR_NAMES:
            pairs = []
            for i, l1 in enumerate(layers):
                for l2 in layers[i + 1:]:
                    d1 = directions[behavior][l1]
                    d2 = directions[behavior][l2]
                    cos = torch.dot(d1, d2).item()
                    pairs.append(f"L{l1}-L{l2}={cos:+.3f}")
            print(f"  {behavior.upper():4s}: {', '.join(pairs)}")


if __name__ == "__main__":
    main()
