#!/usr/bin/env python3
"""Calculate p95 metrics from transport RTT logs."""

import json
import numpy as np
from pathlib import Path

def calculate_p95(values):
    """Calculate 95th percentile."""
    if not values:
        return 0
    return np.percentile(values, 95)

def main():
    """Calculate message_path_ms_p95 from RTT logs."""
    
    # Process PM arm RTTs
    pm_rtts = []
    pm_file = Path("runs/PM/transport_rtts.jsonl")
    if pm_file.exists():
        with open(pm_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pm_rtts.append(data["rtt_ms"])
    
    # Process C arm RTTs  
    c_rtts = []
    c_file = Path("runs/C/transport_rtts.jsonl")
    if c_file.exists():
        with open(c_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    c_rtts.append(data["rtt_ms"])
    
    print("=" * 60)
    print("Message Path RTT Analysis")
    print("=" * 60)
    
    if pm_rtts:
        print(f"\nPM Arm RTTs (n={len(pm_rtts)}):")
        print(f"  Min:    {min(pm_rtts):.3f} ms")
        print(f"  Median: {np.median(pm_rtts):.3f} ms")
        print(f"  Mean:   {np.mean(pm_rtts):.3f} ms")
        print(f"  P95:    {calculate_p95(pm_rtts):.3f} ms")
        print(f"  Max:    {max(pm_rtts):.3f} ms")
        print(f"  Raw values: {pm_rtts}")
    
    if c_rtts:
        print(f"\nC Arm RTTs (n={len(c_rtts)}):")
        print(f"  Min:    {min(c_rtts):.3f} ms")
        print(f"  Median: {np.median(c_rtts):.3f} ms")
        print(f"  Mean:   {np.mean(c_rtts):.3f} ms")
        print(f"  P95:    {calculate_p95(c_rtts):.3f} ms")
        print(f"  Max:    {max(c_rtts):.3f} ms")
        print(f"  Raw values: {c_rtts}")
    
    # Combined for overall message_path_ms_p95
    all_rtts = pm_rtts + c_rtts
    if all_rtts:
        print(f"\nCombined RTTs (n={len(all_rtts)}):")
        print(f"  message_path_ms_p95: {calculate_p95(all_rtts):.3f} ms")
        
        # Acceptance check
        p95 = calculate_p95(all_rtts)
        print(f"\nAcceptance Criteria:")
        print(f"  Target: < 10 ms (goal), < 20 ms (acceptable)")
        print(f"  Actual: {p95:.3f} ms")
        if p95 < 10:
            print("  ✓ GOAL MET (<10ms)")
        elif p95 < 20:
            print("  ✓ ACCEPTABLE (<20ms)")
        else:
            print("  ✗ NOT MET (>=20ms)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()