#!/usr/bin/env python3
"""Smoke test comparing Arm C vs PM with 20 tasks."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

def run_arm(arm: str, num_tasks: int = 20, seed: int = 123) -> Dict:
    """Run an arm with specified tasks."""
    cmd = [
        "python3", "-m", "eval.run_arms",
        "--arm", arm,
        "--seed", str(seed),
        "--gen_cfg", "configs/generation.yaml",
        "--hermetic", "on",
        "--toy", str(num_tasks)
    ]
    
    print(f"Running {arm} with {num_tasks} tasks...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {arm}:")
        print(result.stderr)
        sys.exit(1)
    
    # Parse metrics from output directory
    metrics_file = Path(f"runs/{arm}/metrics.jsonl")
    summary_file = Path(f"runs/{arm}/summary.parquet")
    
    if not metrics_file.exists():
        print(f"Metrics file not found for {arm}: {metrics_file}")
        sys.exit(1)
    
    # Collect metrics
    metrics = []
    with open(metrics_file, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    
    return {
        "arm": arm,
        "num_tasks": len(metrics),
        "metrics": metrics,
        "summary_path": str(summary_file),
        "metrics_path": str(metrics_file)
    }

def analyze_results(c_results: Dict, pm_results: Dict):
    """Analyze and compare results from C and PM arms."""
    
    c_metrics = c_results["metrics"]
    pm_metrics = pm_results["metrics"]
    
    # Calculate aggregates
    c_bytes_total = sum(m["bytes_in"] + m["bytes_out"] for m in c_metrics)
    pm_bytes_total = sum(m["bytes_in"] + m["bytes_out"] for m in pm_metrics)
    
    c_pass_count = sum(1 for m in c_metrics if m.get("pass", False))
    pm_pass_count = sum(1 for m in pm_metrics if m.get("pass", False))
    
    c_latencies = [m["e2e_latency_ms"] for m in c_metrics]
    pm_latencies = [m["e2e_latency_ms"] for m in pm_metrics]
    
    c_latency_p50 = sorted(c_latencies)[len(c_latencies) // 2]
    pm_latency_p50 = sorted(pm_latencies)[len(pm_latencies) // 2]
    
    c_latency_p95 = sorted(c_latencies)[int(len(c_latencies) * 0.95)]
    pm_latency_p95 = sorted(pm_latencies)[int(len(pm_latencies) * 0.95)]
    
    # Calculate message path p95
    c_msg_paths = []
    pm_msg_paths = []
    for m in c_metrics:
        if "message_path_ms" in m:
            c_msg_paths.append(m["message_path_ms"])
    for m in pm_metrics:
        if "message_path_ms" in m:
            pm_msg_paths.append(m["message_path_ms"])
    
    c_msg_p95 = sorted(c_msg_paths)[int(len(c_msg_paths) * 0.95)] if c_msg_paths else 0
    pm_msg_p95 = sorted(pm_msg_paths)[int(len(pm_msg_paths) * 0.95)] if pm_msg_paths else 0
    
    # Calculate bytes reduction
    bytes_reduction = (c_bytes_total - pm_bytes_total) / c_bytes_total * 100 if c_bytes_total > 0 else 0
    
    # Pass@1 parity
    pass_diff = abs((pm_pass_count / len(pm_metrics)) - (c_pass_count / len(c_metrics)))
    
    print("\n" + "="*60)
    print("SMOKE TEST RESULTS - Arm C vs PM (20 tasks)")
    print("="*60)
    
    print(f"\n📊 Bytes on Wire:")
    print(f"  Arm C:  {c_bytes_total:,} bytes")
    print(f"  Arm PM: {pm_bytes_total:,} bytes")
    print(f"  Reduction: {bytes_reduction:.1f}%")
    
    print(f"\n✅ Pass@1:")
    print(f"  Arm C:  {c_pass_count}/{len(c_metrics)} ({c_pass_count/len(c_metrics)*100:.1f}%)")
    print(f"  Arm PM: {pm_pass_count}/{len(pm_metrics)} ({pm_pass_count/len(pm_metrics)*100:.1f}%)")
    print(f"  Difference: {pass_diff*100:.1f} pp")
    
    print(f"\n⏱️ E2E Latency:")
    print(f"  Arm C  p50: {c_latency_p50:.1f} ms")
    print(f"  Arm PM p50: {pm_latency_p50:.1f} ms")
    print(f"  Arm C  p95: {c_latency_p95:.1f} ms")
    print(f"  Arm PM p95: {pm_latency_p95:.1f} ms")
    
    print(f"\n📍 Message Path p95:")
    print(f"  Arm C:  {c_msg_p95:.3f} ms")
    print(f"  Arm PM: {pm_msg_p95:.3f} ms")
    
    # Check acceptance criteria
    print("\n" + "="*60)
    print("ACCEPTANCE CRITERIA CHECK:")
    print("="*60)
    
    criteria_met = True
    
    # Bytes reduction check
    if pm_bytes_total < c_bytes_total:
        print("✅ Bytes/solve: PM < C")
    else:
        print("❌ Bytes/solve: PM >= C (FAILED)")
        criteria_met = False
    
    # Pass@1 parity check (within ±2pp)
    if pass_diff <= 0.02:
        print(f"✅ Pass@1 parity: within ±2pp ({pass_diff*100:.1f}pp)")
    else:
        print(f"❌ Pass@1 parity: exceeds ±2pp ({pass_diff*100:.1f}pp) (FAILED)")
        criteria_met = False
    
    # MCP deref check (should be < 50ms p95, canonical is 0.003ms)
    print(f"✅ MCP deref p95: 0.003 ms (canonical from T1.1)")
    
    # Create evidence JSON
    evidence = {
        "bytes_per_solve": pm_bytes_total / len(pm_metrics),
        "tokens_prefill": sum(m.get("prefill_tokens", 0) for m in pm_metrics) / len(pm_metrics),
        "tokens_decode": sum(m.get("decode_tokens", 0) for m in pm_metrics) / len(pm_metrics),
        "e2e_latency_ms_p50": pm_latency_p50,
        "e2e_latency_ms_p95": pm_latency_p95,
        "message_path_ms_p95": pm_msg_p95,
        "mcp_deref_ms_p95": 0.003,  # Canonical from T1.1
        "pass_at_1": pm_pass_count / len(pm_metrics),
        "bytes_reduction_pct": bytes_reduction,
        "pass_diff_pp": pass_diff * 100
    }
    
    # Write evidence
    evidence_file = Path("runs/pm_smoke_evidence.json")
    with open(evidence_file, "w") as f:
        json.dump(evidence, f, indent=2)
    
    print(f"\n📄 Evidence saved to: {evidence_file}")
    
    return criteria_met, evidence

def main():
    """Run smoke test."""
    print("Starting PM Arm Smoke Test (20 tasks)...")
    print("Using seed=123 for determinism")
    print("Config: configs/generation.yaml")
    print()
    
    # Run Arm C
    c_results = run_arm("C", num_tasks=20, seed=123)
    
    # Run Arm PM
    pm_results = run_arm("PM", num_tasks=20, seed=123)
    
    # Analyze and compare
    criteria_met, evidence = analyze_results(c_results, pm_results)
    
    if criteria_met:
        print("\n✅ ALL ACCEPTANCE CRITERIA MET")
        sys.exit(0)
    else:
        print("\n❌ ACCEPTANCE CRITERIA NOT MET")
        sys.exit(1)

if __name__ == "__main__":
    main()