#!/usr/bin/env python3
"""Minimal acceptance check matching exact spec requirements."""

import json
import sys
import statistics
import glob


def load_bytes_pass(run_dir):
    """Load bytes_per_solve and pass_at_1 metrics from run directory."""
    bytes_vals, passes = [], []
    metrics_file = f"{run_dir}/metrics.jsonl"
    
    try:
        with open(metrics_file) as f:
            for line in f:
                rec = json.loads(line)
                if "bytes_per_solve" in rec:
                    bytes_vals.append(rec["bytes_per_solve"])
                if "pass_at_1" in rec:
                    passes.append(rec["pass_at_1"])
    except FileNotFoundError:
        print(f"ERROR: {metrics_file} not found")
        return None, None
    
    if not bytes_vals or not passes:
        print(f"ERROR: No metrics found in {metrics_file}")
        return None, None
    
    return statistics.mean(bytes_vals), statistics.mean(passes)


def main(c_dir, pm_dir):
    """Check T1.2 acceptance criteria."""
    c_bytes, c_pass = load_bytes_pass(c_dir)
    pm_bytes, pm_pass = load_bytes_pass(pm_dir)
    
    if c_bytes is None or pm_bytes is None:
        print("ERROR: Missing metrics. Run hermetic evaluation first.")
        sys.exit(1)
    
    print(f"C:  bytes/solve={c_bytes:.1f}  pass@1={c_pass:.3f}")
    print(f"PM: bytes/solve={pm_bytes:.1f}  pass@1={pm_pass:.3f}")
    
    ok_bytes = pm_bytes < c_bytes
    ok_pass = abs(pm_pass - c_pass) <= 0.02
    
    print(f"ACCEPT_BYTES={ok_bytes}  ACCEPT_PASS={ok_pass}")
    
    if ok_bytes and ok_pass:
        print("✅ T1.2 ACCEPTED")
    else:
        print("❌ T1.2 NOT ACCEPTED")
    
    sys.exit(0 if (ok_bytes and ok_pass) else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: check_acceptance_minimal.py <C_dir> <PM_dir>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])