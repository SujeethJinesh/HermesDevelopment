#!/usr/bin/env python3
"""T1.2 Acceptance Checker - Verifies PM < C and pass@1 parity."""

import json
import sys
from pathlib import Path


def check_acceptance():
    """Check if T1.2 acceptance criteria are met."""
    
    # Load metrics from both arms
    c_metrics_file = Path("runs/C/metrics.jsonl")
    pm_metrics_file = Path("runs/PM/metrics.jsonl")
    
    if not c_metrics_file.exists():
        print("❌ Missing C arm metrics at runs/C/metrics.jsonl")
        return False
        
    if not pm_metrics_file.exists():
        print("❌ Missing PM arm metrics at runs/PM/metrics.jsonl")
        return False
    
    # Calculate C arm stats
    c_total_bytes_in, c_total_bytes_out = 0, 0
    c_passed, c_total = 0, 0
    
    with open(c_metrics_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if 'bytes_in' in data and 'bytes_out' in data:
                    c_total_bytes_in += data['bytes_in']
                    c_total_bytes_out += data['bytes_out']
                    c_total += 1
                    if data.get('pass', False):
                        c_passed += 1
            except json.JSONDecodeError:
                continue
    
    # Calculate PM arm stats
    pm_total_bytes_in, pm_total_bytes_out = 0, 0
    pm_passed, pm_total = 0, 0
    pm_bytes_saved = 0
    pm_anchors = 0
    mcp_deref_times = []
    
    with open(pm_metrics_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if 'bytes_in' in data and 'bytes_out' in data:
                    pm_total_bytes_in += data['bytes_in']
                    pm_total_bytes_out += data['bytes_out']
                    pm_total += 1
                    if data.get('pass', False):
                        pm_passed += 1
                    pm_bytes_saved += data.get('bytes_saved', 0)
                    pm_anchors += data.get('mcp_anchors_created', 0)
                    if data.get('mcp_deref_ms_p95'):
                        mcp_deref_times.append(data['mcp_deref_ms_p95'])
            except json.JSONDecodeError:
                continue
    
    # Calculate averages
    if c_total == 0 or pm_total == 0:
        print(f"❌ Insufficient data: C={c_total} tasks, PM={pm_total} tasks")
        return False
    
    c_bytes_per_solve = (c_total_bytes_in + c_total_bytes_out) / c_total
    pm_bytes_per_solve = (pm_total_bytes_in + pm_total_bytes_out) / pm_total
    
    c_pass_rate = (c_passed / c_total) * 100
    pm_pass_rate = (pm_passed / pm_total) * 100
    
    # Calculate MCP deref p95
    mcp_p95 = None
    if mcp_deref_times:
        mcp_deref_times.sort()
        p95_idx = int(len(mcp_deref_times) * 0.95)
        mcp_p95 = mcp_deref_times[min(p95_idx, len(mcp_deref_times) - 1)]
    
    # Print results
    print("=" * 60)
    print("T1.2 Acceptance Check Results")
    print("=" * 60)
    print()
    
    print(f"C Arm (Protobuf Baseline):")
    print(f"  Tasks: {c_total}")
    print(f"  Bytes/solve: {c_bytes_per_solve:.0f} (in={c_total_bytes_in/c_total:.0f}, out={c_total_bytes_out/c_total:.0f})")
    print(f"  Pass@1: {c_pass_rate:.1f}%")
    print()
    
    print(f"PM Arm (Protobuf + MCP):")
    print(f"  Tasks: {pm_total}")
    print(f"  Bytes/solve: {pm_bytes_per_solve:.0f} (in={pm_total_bytes_in/pm_total:.0f}, out={pm_total_bytes_out/pm_total:.0f})")
    print(f"  Pass@1: {pm_pass_rate:.1f}%")
    print(f"  MCP anchors created: {pm_anchors}")
    print(f"  Bytes saved via MCP: {pm_bytes_saved:,}")
    if mcp_p95:
        print(f"  MCP deref p95: {mcp_p95:.1f}ms")
    print()
    
    # Check acceptance criteria
    print("Acceptance Criteria:")
    print("-" * 40)
    
    criteria_met = True
    
    # Criterion 1: PM bytes/solve < C bytes/solve
    if pm_bytes_per_solve < c_bytes_per_solve:
        reduction = 100 * (c_bytes_per_solve - pm_bytes_per_solve) / c_bytes_per_solve
        print(f"✅ Bytes: PM < C by {c_bytes_per_solve - pm_bytes_per_solve:.0f} bytes ({reduction:.1f}% reduction)")
    else:
        print(f"❌ Bytes: PM > C by {pm_bytes_per_solve - c_bytes_per_solve:.0f} bytes")
        criteria_met = False
    
    # Criterion 2: Pass@1 within ±2pp
    pass_delta = abs(pm_pass_rate - c_pass_rate)
    if pass_delta <= 2.0:
        print(f"✅ Pass@1: {pm_pass_rate:.1f}% vs {c_pass_rate:.1f}% (Δ={pm_pass_rate - c_pass_rate:+.1f}pp)")
    else:
        print(f"❌ Pass@1: {pm_pass_rate:.1f}% vs {c_pass_rate:.1f}% (Δ={pm_pass_rate - c_pass_rate:+.1f}pp > ±2pp)")
        criteria_met = False
    
    # Criterion 3: MCP deref p95 < 50ms (if available)
    if mcp_p95:
        if mcp_p95 < 50:
            print(f"✅ MCP deref p95: {mcp_p95:.1f}ms < 50ms")
        else:
            print(f"❌ MCP deref p95: {mcp_p95:.1f}ms > 50ms")
            criteria_met = False
    else:
        print(f"⚠️  MCP deref p95: No data (expected if no real repos)")
    
    print()
    print("=" * 60)
    
    if criteria_met:
        print("🎉 T1.2 ACCEPTANCE CRITERIA MET!")
        print()
        print("Next steps:")
        print("1. Commit all changes")
        print("2. Push to branch sujinesh/M1_F1_T12")
        print("3. Update PR #6 with acceptance evidence")
        print("4. Document in docs/M1/F1.1/T1.2_summary.md")
    else:
        print("❌ T1.2 NOT READY FOR ACCEPTANCE")
        print()
        print("Issues to address:")
        if pm_bytes_per_solve >= c_bytes_per_solve:
            print("- PM bytes still exceeds C - need to anchor more content")
            print("  Suggestion: Force anchor test logs >500B (currently >1KB)")
        if pass_delta > 2.0:
            print("- Pass@1 delta exceeds ±2pp - check patch resolution")
            print("  Note: 0% pass likely means git mirrors missing")
    
    print("=" * 60)
    
    return criteria_met


if __name__ == "__main__":
    success = check_acceptance()
    sys.exit(0 if success else 1)