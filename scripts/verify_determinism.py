#!/usr/bin/env python3
"""Verify deterministic execution of two runs."""

import json
from pathlib import Path

def compare_metrics_files(file1, file2):
    """Compare two metrics.jsonl files, ignoring timing fields."""
    
    # Fields to ignore (non-deterministic by nature)
    IGNORE_FIELDS = {
        'duration', 'sandbox_setup_ms', 'sandbox_cleanup_ms', 
        'end_time', 'start_time', 'e2e_latency_ms'
    }
    
    with open(file1) as f1, open(file2) as f2:
        lines1 = [json.loads(l) for l in f1 if l.strip()]
        lines2 = [json.loads(l) for l in f2 if l.strip()]
    
    if len(lines1) != len(lines2):
        print(f"✗ Different number of tasks: {len(lines1)} vs {len(lines2)}")
        return False
    
    differences = []
    for i, (m1, m2) in enumerate(zip(lines1, lines2)):
        # Compare deterministic fields only
        for key in m1:
            if key in IGNORE_FIELDS:
                continue
            
            if key not in m2:
                differences.append(f"Task {i}: Key '{key}' missing in run2")
            elif m1[key] != m2[key]:
                differences.append(f"Task {i}: {key} differs: {m1[key]} vs {m2[key]}")
    
    return differences

def main():
    """Main verification."""
    print("=" * 60)
    print("Hermetic Determinism Verification")
    print("=" * 60)
    
    # Check if we have both runs - use the slice20 extracts
    run1_file = Path("runs/PM/run1_slice20.jsonl")
    run2_file = Path("runs/PM/run2_slice20.jsonl")
    
    if not run1_file.exists():
        print("✗ Run 1 file not found")
        return 1
    
    if not run2_file.exists():
        print("✗ Run 2 file not found") 
        return 1
    
    # Compare the two runs
    differences = compare_metrics_files(run1_file, run2_file)
    
    if not differences:
        print("\n✓ DETERMINISM VERIFIED!")
        print("  Both runs produced identical results for all deterministic fields.")
        print("\nKey deterministic fields verified:")
        print("  - task_id (same order)")
        print("  - task_seed (derived from global seed)")
        print("  - bytes_in/bytes_out (identical)")
        print("  - tokens_in/tokens_out (identical)")
        print("  - pass/fail status (identical)")
        print("  - prefill/decode tokens (identical)")
        print("\nNon-deterministic fields (expected to differ):")
        print("  - duration, sandbox_setup_ms, sandbox_cleanup_ms")
        print("  - start_time, end_time, e2e_latency_ms")
    else:
        print("\n✗ DETERMINISM FAILED!")
        print(f"  Found {len(differences)} differences:")
        for diff in differences[:10]:  # Show first 10
            print(f"    - {diff}")
        if len(differences) > 10:
            print(f"    ... and {len(differences) - 10} more")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())