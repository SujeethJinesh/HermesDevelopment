#!/usr/bin/env python3
"""End-to-end integration tests for baseline agents."""

import json
import os
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.run_arms import ArmRunner


class TestAgentsE2E:
    """Test agents end-to-end through eval harness."""
    
    def test_arm_a_ten_tasks(self):
        """Test Arm A (JSON) runs 10 tasks E2E."""
        # Create runner
        runner = ArmRunner(
            arm="A",
            seed=123,
            gen_cfg_path="configs/generation.yaml",
            hermetic=True,
            toy_tasks=10
        )
        
        # Run evaluation
        runner.run()
        
        # Check metrics were generated
        assert len(runner.metrics) == 10
        
        # Check all tasks completed
        for m in runner.metrics:
            assert "task_id" in m
            assert "pass" in m
            assert "bytes_in" in m
            assert "bytes_out" in m
            assert "e2e_latency_ms" in m
            assert "message_path_ms" in m
        
        # Check metrics file exists
        metrics_file = Path("runs/A/metrics.jsonl")
        assert metrics_file.exists()
        
        # Check RTT file exists
        rtt_file = Path("runs/A/transport_rtts.jsonl")
        assert rtt_file.exists()
        
        # Load and check RTT data
        rtts = []
        with open(rtt_file) as f:
            for line in f:
                data = json.loads(line)
                rtts.append(data["rtt_ms"])
        
        # Should have 3 RTTs per task (planner, coder, tester)
        assert len(rtts) >= 30
        
        # Check p95 RTT
        rtts_sorted = sorted(rtts)
        p95_idx = int(len(rtts) * 0.95)
        p95 = rtts_sorted[p95_idx] if p95_idx < len(rtts) else rtts_sorted[-1]
        
        print(f"\nArm A RTT p95: {p95:.3f} ms")
        assert p95 < 10.0, f"RTT p95 {p95:.3f}ms exceeds 10ms"
        
        # Check summary parquet exists
        summary_file = Path("runs/A/summary.parquet")
        assert summary_file.exists()
    
    def test_arm_c_ten_tasks(self):
        """Test Arm C (Protobuf) runs 10 tasks E2E."""
        # Create runner
        runner = ArmRunner(
            arm="C",
            seed=456,
            gen_cfg_path="configs/generation.yaml",
            hermetic=True,
            toy_tasks=10
        )
        
        # Run evaluation
        runner.run()
        
        # Check metrics were generated
        assert len(runner.metrics) == 10
        
        # Check all tasks completed
        passed_count = sum(1 for m in runner.metrics if m.get("pass", False))
        print(f"\nArm C passed {passed_count}/10 tasks")
        
        # Check metrics file
        metrics_file = Path("runs/C/metrics.jsonl")
        assert metrics_file.exists()
        
        # Check message path latencies
        message_paths = [m["message_path_ms"] for m in runner.metrics]
        avg_path = sum(message_paths) / len(message_paths)
        print(f"Arm C avg message_path_ms: {avg_path:.3f}")
        
        # Check RTT measurements
        rtt_file = Path("runs/C/transport_rtts.jsonl")
        assert rtt_file.exists()
        
        rtts = []
        with open(rtt_file) as f:
            for line in f:
                data = json.loads(line)
                rtts.append(data["rtt_ms"])
        
        rtts_sorted = sorted(rtts)
        p95_idx = int(len(rtts) * 0.95)
        p95 = rtts_sorted[p95_idx] if p95_idx < len(rtts) else rtts_sorted[-1]
        
        print(f"Arm C RTT p95: {p95:.3f} ms")
        assert p95 < 10.0, f"RTT p95 {p95:.3f}ms exceeds 10ms"
    
    def test_determinism_same_seed(self):
        """Test same seed produces deterministic results."""
        # Run twice with same seed
        results = []
        
        for run in range(2):
            runner = ArmRunner(
                arm="C",
                seed=999,
                gen_cfg_path="configs/generation.yaml",
                hermetic=True,
                toy_tasks=3
            )
            runner.run()
            
            # Collect deterministic fields
            run_results = []
            for m in runner.metrics:
                run_results.append({
                    "task_id": m["task_id"],
                    "pass": m["pass"],
                    "bytes_in": m["bytes_in"],
                    "bytes_out": m["bytes_out"],
                    # Note: timing fields are non-deterministic
                })
            results.append(run_results)
        
        # Check deterministic fields match
        for i in range(len(results[0])):
            r1 = results[0][i]
            r2 = results[1][i]
            assert r1["task_id"] == r2["task_id"]
            assert r1["pass"] == r2["pass"]
            # Bytes should be deterministic for same content
            assert r1["bytes_in"] == r2["bytes_in"]
            assert r1["bytes_out"] == r2["bytes_out"]
    
    def test_hermetic_cleanup(self):
        """Test hermetic scratch directories are cleaned up."""
        # Check scratch is empty before
        scratch_dir = Path("scratch")
        if scratch_dir.exists():
            existing = list(scratch_dir.glob("toy-*"))
            assert len(existing) == 0, "Scratch not clean before test"
        
        # Run tasks
        runner = ArmRunner(
            arm="A",
            seed=777,
            gen_cfg_path="configs/generation.yaml",
            hermetic=True,
            toy_tasks=2
        )
        runner.run()
        
        # Check scratch is cleaned after
        if scratch_dir.exists():
            remaining = list(scratch_dir.glob("toy-*"))
            assert len(remaining) == 0, f"Scratch not cleaned: {remaining}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])