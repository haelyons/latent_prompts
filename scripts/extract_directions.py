#!/usr/bin/env python3
"""
CLI script to pre-compute sycophancy direction vectors from factorial datasets.

Usage:
    python scripts/extract_directions.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/extract_directions.py --model meta-llama/Llama-3.1-8B-Instruct --max-samples 100
    python scripts/extract_directions.py --layers 12 14 16 18 20 --behaviors syc ga pr
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sycophancy_detector.direction_extractor import DirectionExtractor
from sycophancy_detector.utils import (
    DEFAULT_CONFIG,
    get_factorial_dataset_paths,
    get_factorial_data_dir,
    save_direction,
    DirectionMetadata,
    list_available_directions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract sycophancy direction vectors from factorial datasets."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model_id"],
        help=f"HuggingFace model identifier (default: {DEFAULT_CONFIG['model_id']})",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_CONFIG["direction_cache_dir"],
        help=f"Directory to save directions (default: {DEFAULT_CONFIG['direction_cache_dir']})",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing factorial datasets (default: auto-detect)",
    )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIG["layers"],
        help=f"Layers to extract directions for (default: {DEFAULT_CONFIG['layers']})",
    )
    
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=DEFAULT_CONFIG["behaviors"],
        help=f"Behavior labels to extract (default: {DEFAULT_CONFIG['behaviors']})",
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific dataset files to use (default: all available)",
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per class (for faster testing)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size for processing (default: {DEFAULT_CONFIG['batch_size']})",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', 'cuda:0', 'cpu' (default: auto)",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype (default: auto -> bfloat16 if CUDA available)",
    )
    
    parser.add_argument(
        "--list-existing",
        action="store_true",
        help="List existing directions in output directory and exit",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # List existing and exit if requested
    if args.list_existing:
        print(f"Directions in {args.output_dir}:")
        existing = list_available_directions(args.output_dir)
        if existing:
            for info in existing:
                print(f"  {info['behavior']} L{info['layer']}: {info['path']}")
        else:
            print("  (none found)")
        return 0
    
    # Determine dtype
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Determine device
    if args.device == "auto":
        device_map = "auto"
    else:
        device_map = args.device
    
    print(f"=== Sycophancy Direction Extraction ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Layers: {args.layers}")
    print(f"Behaviors: {args.behaviors}")
    print(f"Dtype: {torch_dtype}")
    print()
    
    # Find data directory
    if args.data_dir is None:
        try:
            args.data_dir = get_factorial_data_dir()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify --data-dir or ensure disentangle-sycophancy is available")
            return 1
    
    # Get dataset paths
    dataset_paths = get_factorial_dataset_paths(args.data_dir, args.datasets)
    
    if not dataset_paths:
        print(f"Error: No factorial datasets found in {args.data_dir}")
        return 1
    
    print(f"Found {len(dataset_paths)} datasets:")
    for path in dataset_paths:
        print(f"  - {os.path.basename(path)}")
    print()
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        output_hidden_states=True,
    )
    model.eval()
    
    # Set up tokenizer
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    
    print(f"Model loaded on {device}")
    print()
    
    # Create extractor
    extractor = DirectionExtractor(
        model, tokenizer, device, batch_size=args.batch_size
    )
    
    # Extract directions for each behavior
    for behavior in args.behaviors:
        print(f"\n{'='*60}")
        print(f"Extracting '{behavior}' directions")
        print(f"{'='*60}")
        
        # Compute directions for all layers
        results = extractor.compute_all_behaviors(
            dataset_paths,
            [behavior],
            args.layers,
            args.max_samples,
        )
        
        # Save each direction
        for layer, result in results[behavior].items():
            metadata = DirectionMetadata(
                behavior=behavior,
                layer=layer,
                pos_count=result.pos_count,
                neg_count=result.neg_count,
                pos_mean_norm=result.pos_mean_norm,
                neg_mean_norm=result.neg_mean_norm,
                direction_norm=float(result.direction @ result.direction)**0.5,
                datasets_used=[os.path.basename(p) for p in dataset_paths],
                model_id=args.model,
            )
            
            path = save_direction(
                result.direction,
                args.output_dir,
                behavior,
                layer,
                metadata,
            )
            
            if args.verbose:
                print(f"  Saved: {path}")
                print(f"    pos_count={result.pos_count}, neg_count={result.neg_count}")
                print(f"    direction_norm={metadata.direction_norm:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Directions saved to: {args.output_dir}")
    print(f"{'='*60}")
    
    # Summary
    print("\nSummary:")
    existing = list_available_directions(args.output_dir)
    for info in existing:
        print(f"  {info['behavior']} L{info['layer']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
