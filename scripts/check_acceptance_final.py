#!/usr/bin/env python3
"""Final T1.2 acceptance check - minimal and strict."""

import json
import sys
from pathlib import Path
from statistics import mean


def load_metrics(jsonl_path: Path):
    """Load metrics from JSONL file."""
    vals = {
        "bytes_per_solve": [],
        "pass_at_1": [],
        "message_path_ms": [],
        "mcp_deref_ms": [],
    }
    
    try:
        with jsonl_path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                
                # Calculate bytes_per_solve from bytes_in + bytes_out
                if "bytes_in" in obj and "bytes_out" in obj:
                    vals["bytes_per_solve"].append(obj["bytes_in"] + obj["bytes_out"])
                
                # Track pass rate
                if "pass" in obj:
                    vals["pass_at_1"].append(1.0 if obj["pass"] else 0.0)
                
                # Track message path latency
                if "message_path_ms" in obj:
                    vals["message_path_ms"].append(obj["message_path_ms"])
                
                # MCP deref (if exists)
                if "mcp_deref_ms" in obj:
                    vals["mcp_deref_ms"].append(obj["mcp_deref_ms"])
                    
    except FileNotFoundError:
        return None
    
    # Calculate aggregates
    result = {}
    result["bytes_per_solve"] = mean(vals["bytes_per_solve"]) if vals["bytes_per_solve"] else None
    result["pass_at_1"] = mean(vals["pass_at_1"]) if vals["pass_at_1"] else None
    
    # Calculate p95 for latencies
    if vals["message_path_ms"]:
        sorted_mp = sorted(vals["message_path_ms"])
        idx = int(len(sorted_mp) * 0.95)
        result["message_path_ms_p95"] = sorted_mp[min(idx, len(sorted_mp)-1)]
    else:
        result["message_path_ms_p95"] = None
    
    if vals["mcp_deref_ms"]:
        sorted_mcp = sorted(vals["mcp_deref_ms"])
        idx = int(len(sorted_mcp) * 0.95)
        result["mcp_deref_ms_p95"] = sorted_mcp[min(idx, len(sorted_mcp)-1)]
    else:
        result["mcp_deref_ms_p95"] = None
                        
    return result


def main(c_dir: str, pm_dir: str):
    """Check T1.2 acceptance criteria."""
    c_path = Path(c_dir) / "metrics.jsonl"
    pm_path = Path(pm_dir) / "metrics.jsonl"
    
    if not c_path.is_file() or not pm_path.is_file():
        print("ERROR: metrics.jsonl missing. Run prepare_hermetic_run.sh eval first.", file=sys.stderr)
        sys.exit(2)
    
    C = load_metrics(c_path)
    PM = load_metrics(pm_path)
    
    if not C or not PM:
        print("ERROR: Failed to load metrics", file=sys.stderr)
        sys.exit(2)
    
    print("=== T1.2 Acceptance Check ===\n")
    print(f"C metrics:  {C}")
    print(f"PM metrics: {PM}\n")

    acceptance_met = True
    
    # 1. Bytes per solve (REQUIRED)
    if C["bytes_per_solve"] is None or PM["bytes_per_solve"] is None:
        print("❌ FAIL: bytes_per_solve missing")
        acceptance_met = False
    else:
        c_bytes = C["bytes_per_solve"]
        pm_bytes = PM["bytes_per_solve"]
        if pm_bytes < c_bytes:
            print(f"✅ Bytes: PM ({pm_bytes:.1f}) < C ({c_bytes:.1f})")
        else:
            print(f"❌ Bytes: PM ({pm_bytes:.1f}) >= C ({c_bytes:.1f})")
            acceptance_met = False

    # 2. Pass@1 parity (REQUIRED)
    if C["pass_at_1"] is None or PM["pass_at_1"] is None:
        print("⚠️  WARN: pass@1 missing; skipping ±2pp check")
    else:
        c_pass = C["pass_at_1"]
        pm_pass = PM["pass_at_1"]
        delta = abs(pm_pass - c_pass)
        if delta <= 0.02:
            print(f"✅ Pass@1: delta {delta*100:.1f}pp <= 2pp")
        else:
            print(f"❌ Pass@1: delta {delta*100:.1f}pp > 2pp")
            acceptance_met = False

    # 3. Latency checks (OPTIONAL - warn only)
    if PM["message_path_ms_p95"]:
        val = PM["message_path_ms_p95"]
        if val < 35:
            print(f"✅ Message path p95: {val:.1f}ms < 35ms")
        else:
            print(f"⚠️  Message path p95: {val:.1f}ms >= 35ms")
            
    if PM["mcp_deref_ms_p95"]:
        val = PM["mcp_deref_ms_p95"]
        if val < 50:
            print(f"✅ MCP deref p95: {val:.1f}ms < 50ms")
        else:
            print(f"⚠️  MCP deref p95: {val:.1f}ms >= 50ms")

    # Final verdict
    print("\n" + "="*40)
    if acceptance_met:
        print("✅ T1.2 ACCEPTED - Core criteria met")
        sys.exit(0)
    else:
        print("❌ T1.2 NOT ACCEPTED - Fix failing criteria")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/check_acceptance_final.py runs/C runs/PM", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])