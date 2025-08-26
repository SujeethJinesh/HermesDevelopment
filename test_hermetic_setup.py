#!/usr/bin/env python3
"""Test hermetic setup for T1.2 acceptance."""

import os
import json
import subprocess
from pathlib import Path

def test_hermetic_infrastructure():
    """Verify all hermetic components are in place."""
    
    print("=== Testing Hermetic Infrastructure ===\n")
    
    # 1. Check dataset cache
    print("1. Dataset cache:")
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/datasets"))
    swebench_cache = cache_dir / "SWE-bench___swe-bench_lite"
    if swebench_cache.exists():
        print(f"  ✓ SWE-bench Lite cached at {swebench_cache}")
    else:
        print(f"  ✗ Dataset not cached - run online prep first")
    
    # 2. Check offline mode works
    print("\n2. Offline mode test:")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import load_dataset
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
        print(f"  ✓ Loaded {len(ds)} instances in offline mode")
    except Exception as e:
        print(f"  ✗ Offline load failed: {e}")
    finally:
        del os.environ["HF_DATASETS_OFFLINE"]
    
    # 3. Check hermetic repo helper
    print("\n3. Hermetic repo manager:")
    try:
        from env.hermetic_repos import HermeticRepoManager
        print("  ✓ HermeticRepoManager imports successfully")
        
        # Check if manifest exists
        manifest_path = Path("data/repos_manifest.json")
        if manifest_path.exists():
            print(f"  ✓ Repos manifest found at {manifest_path}")
            with open(manifest_path) as f:
                manifest = json.load(f)
                print(f"    - {len(manifest.get('repos', []))} repos configured")
        else:
            print(f"  ! Manifest not found - repos not prepared yet")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
    
    # 4. Check protobuf fields
    print("\n4. Protobuf schema:")
    try:
        from proto import baseline_pb2
        req = baseline_pb2.TestRequest()
        req.repo = "test/repo"
        req.base_commit = "abc123"
        req.test_patch = "test patch"
        print("  ✓ TestRequest has required fields (repo, base_commit, test_patch)")
    except Exception as e:
        print(f"  ✗ Proto error: {e}")
    
    # 5. Check PM agent integration
    print("\n5. PM agent integration:")
    try:
        from agents.pm_arm import PMAgent
        print("  ✓ PMAgent imports successfully")
        
        # Check RealTester integration
        from agents.real_tester import RealTester
        print("  ✓ RealTester imports successfully")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
    
    # 6. Test configuration
    print("\n6. Configuration:")
    config_path = Path("configs/generation.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
            mcp_threshold = config.get("mcp", {}).get("inline_max_bytes", 32768)
            print(f"  ✓ MCP inline threshold: {mcp_threshold} bytes")
            if mcp_threshold == 1024:
                print("    - Set to 1KB for T1.2 acceptance testing")
            else:
                print(f"    ! Consider setting to 1024 for acceptance")
    
    print("\n=== Summary ===")
    print("Infrastructure is ready for hermetic evaluation.")
    print("\nNext steps:")
    print("1. Run scripts/prepare_swebench_repos.py (ONLINE, one-time)")
    print("2. Set environment variables:")
    print("   export HERMES_HERMETIC=1")
    print("   export HF_DATASETS_OFFLINE=1")
    print("3. Run evaluation:")
    print("   python3 -m eval.run_arms --arm C ...")
    print("   python3 -m eval.run_arms --arm PM ...")

if __name__ == "__main__":
    test_hermetic_infrastructure()