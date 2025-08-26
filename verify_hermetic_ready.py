#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify hermetic infrastructure is ready for T1.2 acceptance testing."""

import os
import sys
from pathlib import Path


def check_hermetic_ready():
    """Check if all components are ready for hermetic evaluation."""
    
    print("=== Hermetic Readiness Check for T1.2 ===\n")
    
    ready = True
    
    # 1. Check environment variables
    print("1. Environment variables:")
    if os.environ.get("HERMES_HERMETIC") == "1":
        print("  ✓ HERMES_HERMETIC=1")
    else:
        print("  ✗ HERMES_HERMETIC not set to 1")
        print("    Run: export HERMES_HERMETIC=1")
        ready = False
    
    if os.environ.get("HF_DATASETS_OFFLINE") == "1":
        print("  ✓ HF_DATASETS_OFFLINE=1")
    else:
        print("  ✗ HF_DATASETS_OFFLINE not set to 1")
        print("    Run: export HF_DATASETS_OFFLINE=1")
        ready = False
    
    # 2. Check dataset cache
    print("\n2. Dataset cache:")
    cache_dir = Path.home() / ".cache/huggingface/datasets/SWE-bench___swe-bench_lite"
    if cache_dir.exists():
        print(f"  ✓ SWE-bench Lite cached at {cache_dir}")
    else:
        print(f"  ✗ Dataset not cached")
        print(f"    Run: python scripts/prepare_swebench_data.py")
        ready = False
    
    # 3. Check repos manifest
    print("\n3. Repo mirrors:")
    manifest_path = Path("data/repos_manifest.json")
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        num_repos = len(manifest.get("repos", []))
        print(f"  ✓ Repos manifest exists with {num_repos} repos")
        
        # Check if any mirrors actually exist
        mirrors_dir = Path("data/mirrors")
        if mirrors_dir.exists():
            mirrors = list(mirrors_dir.glob("*.git"))
            print(f"  ✓ Found {len(mirrors)} local mirrors")
        else:
            print(f"  ✗ No mirrors directory found")
            ready = False
    else:
        print(f"  ✗ Repos manifest not found")
        print(f"    Run: python scripts/prepare_swebench_repos.py --instances_file configs/swebench_lite_slice20.txt")
        ready = False
    
    # 4. Check slice20 config
    print("\n4. Slice20 configuration:")
    slice20_path = Path("configs/swebench_lite_slice20.txt")
    if slice20_path.exists():
        with open(slice20_path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"  ✓ Slice20 config exists with {len(lines)} instances")
    else:
        print(f"  ✗ Slice20 config not found")
        ready = False
    
    # 5. Check hermetic infrastructure
    print("\n5. Hermetic infrastructure:")
    try:
        from env.hermetic_repos import HermeticRepoManager
        print("  ✓ HermeticRepoManager imports successfully")
    except ImportError as e:
        print(f"  ✗ HermeticRepoManager import failed: {e}")
        ready = False
    
    try:
        from agents.real_tester import RealTester
        print("  ✓ RealTester imports successfully")
    except ImportError as e:
        print(f"  ✗ RealTester import failed: {e}")
        ready = False
    
    # 6. Check protobuf
    print("\n6. Protobuf schema:")
    try:
        from proto import baseline_pb2
        req = baseline_pb2.TestRequest()
        req.repo = "test/repo"
        req.base_commit = "abc123"
        req.test_patch = "test patch"
        print("  ✓ TestRequest has required fields")
    except Exception as e:
        print(f"  ✗ Proto error: {e}")
        ready = False
    
    # 7. Check MCP configuration
    print("\n7. MCP configuration:")
    config_path = Path("configs/generation.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        threshold = config.get("mcp", {}).get("inline_max_bytes", 32768)
        print(f"  ✓ MCP inline threshold: {threshold} bytes")
        if threshold == 1024:
            print("    ✓ Set to 1KB for aggressive anchoring")
        else:
            print(f"    ! Consider setting to 1024 for T1.2 acceptance")
    
    # Summary
    print("\n" + "=" * 50)
    if ready:
        print("✅ READY for hermetic evaluation")
        print("\nRun commands:")
        print("  # C arm")
        print("  python3 -m eval.run_arms --arm C --seed 123 \\")
        print("    --dataset swebench_lite --split test \\")
        print("    --instances_file configs/swebench_lite_slice20.txt \\")
        print("    --gen_cfg configs/generation.yaml")
        print("\n  # PM arm")
        print("  python3 -m eval.run_arms --arm PM --seed 123 \\")
        print("    --dataset swebench_lite --split test \\")
        print("    --instances_file configs/swebench_lite_slice20.txt \\")
        print("    --gen_cfg configs/generation.yaml")
    else:
        print("❌ NOT READY - fix issues above first")
        print("\nRequired setup commands:")
        print("  1. export HERMES_HERMETIC=1")
        print("  2. export HF_DATASETS_OFFLINE=1")
        print("  3. python scripts/prepare_swebench_data.py")
        print("  4. python scripts/prepare_swebench_repos.py \\")
        print("       --instances_file configs/swebench_lite_slice20.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(check_hermetic_ready())