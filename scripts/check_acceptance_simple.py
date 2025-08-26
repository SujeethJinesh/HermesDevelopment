#!/usr/bin/env python3
"""Check T1.2 acceptance criteria with exact spec compliance."""

import json
import numpy as np
import sys
from pathlib import Path


def p95(xs):
    """Calculate 95th percentile."""
    return float(np.percentile(xs, 95)) if xs else 0.0


def load_metrics(run_dir: str):
    """Load metrics from a run directory."""
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")
    
    rows = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(c_dir: str = "runs/C", pm_dir: str = "runs/PM"):
    """Check acceptance criteria for T1.2."""
    
    try:
        c_rows = load_metrics(c_dir)
        pm_rows = load_metrics(pm_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run hermetic evaluation first:")
        print("  ./prepare_hermetic_run.sh prep  # once, online")
        print("  ./prepare_hermetic_run.sh eval  # hermetic")
        return 1
    
    # Calculate aggregates
    def agg(rows, key):
        vals = [row[key] for row in rows if key in row]
        return np.mean(vals) if vals else 0.0
    
    c_bytes = agg(c_rows, "bytes_per_solve")
    pm_bytes = agg(pm_rows, "bytes_per_solve")
    c_pass = agg(c_rows, "pass_at_1")
    pm_pass = agg(pm_rows, "pass_at_1")
    
    # Calculate p95 latencies
    msg_vals = [r.get("message_path_ms", 0) for r in c_rows + pm_rows]
    mcp_vals = [r.get("mcp_deref_ms", 0) for r in pm_rows]  # Only PM uses MCP
    msg_p95 = p95(msg_vals)
    mcp_p95 = p95(mcp_vals)
    
    # Check acceptance criteria
    ok_bytes = pm_bytes < c_bytes
    ok_pass = abs(pm_pass - c_pass) <= 0.02  # Within 2pp
    ok_msg = msg_p95 < 35  # M1 target (20ms ideal, 35ms acceptable)
    ok_mcp = mcp_p95 < 50  # Spec requirement
    
    # Output JSON summary
    result = {
        "C_bytes": round(c_bytes, 1),
        "PM_bytes": round(pm_bytes, 1),
        "bytes_reduction": round(100 * (1 - pm_bytes/c_bytes), 1) if c_bytes > 0 else 0,
        "C_pass@1": round(c_pass, 3),
        "PM_pass@1": round(pm_pass, 3),
        "pass_delta_pp": round(abs(pm_pass - c_pass) * 100, 1),
        "message_path_ms_p95": round(msg_p95, 1),
        "mcp_deref_ms_p95": round(mcp_p95, 1),
        "acceptance": {
            "bytes": ok_bytes,
            "pass@1": ok_pass,
            "msg_p95": ok_msg,
            "mcp_p95": ok_mcp,
            "all": ok_bytes and ok_pass and ok_msg and ok_mcp
        }
    }
    
    print(json.dumps(result, indent=2))
    
    # Human-readable summary
    print("\n=== T1.2 Acceptance Status ===")
    print(f"{'✅' if ok_bytes else '❌'} Bytes: PM ({pm_bytes:.0f}) < C ({c_bytes:.0f})")
    print(f"{'✅' if ok_pass else '❌'} Pass@1: Delta {abs(pm_pass - c_pass)*100:.1f}pp ≤ 2pp")
    print(f"{'✅' if ok_msg else '❌'} Message p95: {msg_p95:.1f}ms < 35ms")
    print(f"{'✅' if ok_mcp else '❌'} MCP p95: {mcp_p95:.1f}ms < 50ms")
    
    if result["acceptance"]["all"]:
        print("\n✅ T1.2 ACCEPTED - All criteria met")
        return 0
    else:
        print("\n❌ T1.2 NOT ACCEPTED - Fix failing criteria above")
        return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))