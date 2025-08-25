#!/usr/bin/env python3
"""Prepare SWE-bench Lite dataset for hermetic evaluation.

This script downloads and caches the SWE-bench Lite dataset from Hugging Face
for use in hermetic evaluation runs where network access is blocked.
"""

import argparse
import os
import sys
from pathlib import Path
from datasets import load_dataset
import json


def main():
    parser = argparse.ArgumentParser(description="Prepare SWE-bench Lite dataset cache")
    parser.add_argument(
        "--dataset",
        default="SWE-bench/SWE-bench_Lite",
        help="Dataset name on Hugging Face"
    )
    parser.add_argument(
        "--cache_dir",
        default="~/.cache/huggingface/swebench_lite",
        help="Cache directory for the dataset"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset integrity after download"
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir).expanduser().absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing SWE-bench Lite dataset cache...")
    print(f"Dataset: {args.dataset}")
    print(f"Cache directory: {cache_dir}")
    
    # Set HF cache environment
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    
    # Load both splits to cache them
    print("\nLoading dev split...")
    dev_dataset = load_dataset(
        args.dataset,
        split="dev",
        cache_dir=str(cache_dir / "datasets")
    )
    dev_count = len(dev_dataset)
    print(f"✓ Dev split loaded: {dev_count} instances")
    
    print("\nLoading test split...")
    test_dataset = load_dataset(
        args.dataset,
        split="test",
        cache_dir=str(cache_dir / "datasets")
    )
    test_count = len(test_dataset)
    print(f"✓ Test split loaded: {test_count} instances")
    
    # Verify expected counts
    EXPECTED_DEV = 23
    EXPECTED_TEST = 300
    
    if dev_count != EXPECTED_DEV:
        print(f"WARNING: Expected {EXPECTED_DEV} dev instances, got {dev_count}")
    if test_count != EXPECTED_TEST:
        print(f"WARNING: Expected {EXPECTED_TEST} test instances, got {test_count}")
    
    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "dev_count": dev_count,
        "test_count": test_count,
        "cache_dir": str(cache_dir),
        "columns": list(dev_dataset.column_names) if dev_dataset else [],
    }
    
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved to {metadata_path}")
    
    if args.verify:
        print("\nVerifying dataset integrity...")
        
        # Check required columns
        required_columns = [
            "instance_id", "repo", "base_commit", "patch",
            "test_patch", "problem_statement", "hints_text",
            "version", "environment_setup_commit",
            "FAIL_TO_PASS", "PASS_TO_PASS"
        ]
        
        missing_columns = [col for col in required_columns if col not in dev_dataset.column_names]
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            sys.exit(1)
        
        print("✓ All required columns present")
        
        # Sample first instance from each split
        print("\nSample dev instance:")
        print(f"  instance_id: {dev_dataset[0]['instance_id']}")
        print(f"  repo: {dev_dataset[0]['repo']}")
        
        print("\nSample test instance:")
        print(f"  instance_id: {test_dataset[0]['instance_id']}")
        print(f"  repo: {test_dataset[0]['repo']}")
    
    print(f"\n✅ Dataset cache prepared successfully at {cache_dir}")
    print("\nFor hermetic runs, set:")
    print(f"  export HF_DATASETS_OFFLINE=1")
    print(f"  export HF_HOME={cache_dir}")
    print(f"  export HF_DATASETS_CACHE={cache_dir}/datasets")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())