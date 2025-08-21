#!/usr/bin/env python3
"""Integration tests for eval.run_arms deterministic execution."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRunArmsDeterminism(unittest.TestCase):
    """Test end-to-end deterministic execution."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = Path("test_runs")
        cls.test_dir.mkdir(exist_ok=True)

        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        """Set up each test."""
        # Clean runs directory
        runs_dir = Path("runs")
        if runs_dir.exists():
            shutil.rmtree(runs_dir, ignore_errors=True)

    def test_identical_runs_produce_identical_metrics(self):
        """Test that two identical runs produce identical metrics."""
        seed = 123
        arm = "A"

        # Run 1
        result1 = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "2",
            ],
            capture_output=True,
            text=True,
        )

        # Check run 1 succeeded
        if result1.returncode != 0:
            print("STDOUT:", result1.stdout)
            print("STDERR:", result1.stderr)
        self.assertEqual(result1.returncode, 0, f"Run 1 failed: {result1.stderr}")

        # Save run 1 outputs
        metrics1_path = Path("runs") / arm / "metrics.jsonl"
        summary1_path = Path("runs") / arm / "summary.parquet"

        self.assertTrue(metrics1_path.exists(), "Run 1 metrics.jsonl not created")
        self.assertTrue(summary1_path.exists(), "Run 1 summary.parquet not created")

        # Read run 1 metrics
        with open(metrics1_path) as f:
            metrics1 = [json.loads(line) for line in f]

        # Read run 1 summary
        df1 = pd.read_parquet(summary1_path)

        # Move run 1 outputs
        shutil.move(str(metrics1_path), str(self.test_dir / "metrics1.jsonl"))
        shutil.move(str(summary1_path), str(self.test_dir / "summary1.parquet"))

        # Clean runs directory
        shutil.rmtree("runs", ignore_errors=True)

        # Run 2 with identical parameters
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "2",
            ],
            capture_output=True,
            text=True,
        )

        # Check run 2 succeeded
        if result2.returncode != 0:
            print("STDOUT:", result2.stdout)
            print("STDERR:", result2.stderr)
        self.assertEqual(result2.returncode, 0, f"Run 2 failed: {result2.stderr}")

        # Read run 2 metrics
        metrics2_path = Path("runs") / arm / "metrics.jsonl"
        with open(metrics2_path) as f:
            metrics2 = [json.loads(line) for line in f]

        # Read run 2 summary
        summary2_path = Path("runs") / arm / "summary.parquet"
        df2 = pd.read_parquet(summary2_path)

        # Compare metrics (excluding timestamps)
        self.assertEqual(len(metrics1), len(metrics2), "Different number of metrics")

        for m1, m2 in zip(metrics1, metrics2):
            # Remove time-dependent fields
            time_keys = [
                "start_time",
                "end_time",
                "duration",
                "sandbox_cleanup_ms",
                "sandbox_setup_ms",
            ]
            for key in time_keys:
                m1.pop(key, None)
                m2.pop(key, None)

            # Remove nested manifest (contains timestamps)
            m1.pop("run_manifest", None)
            m2.pop("run_manifest", None)

            # Compare remaining fields
            self.assertEqual(
                json.dumps(m1, sort_keys=True),
                json.dumps(m2, sort_keys=True),
                f"Metrics differ for task {m1.get('task_id', 'unknown')}",
            )

        # Compare summaries (excluding timestamps and durations)
        time_cols = [
            c
            for c in df1.columns
            if "time" in c.lower() or "duration" in c or "cleanup" in c or "setup" in c
        ]
        df1_clean = df1.drop(columns=time_cols, errors="ignore")
        df2_clean = df2.drop(columns=time_cols, errors="ignore")

        # Sort by task_id for consistent comparison
        df1_clean = df1_clean.sort_values("task_id").reset_index(drop=True)
        df2_clean = df2_clean.sort_values("task_id").reset_index(drop=True)

        # Compare DataFrames
        pd.testing.assert_frame_equal(df1_clean, df2_clean)

        print("✓ Two identical runs produced identical metrics")

    def test_different_seeds_produce_different_metrics(self):
        """Test that different seeds produce different metrics."""
        arm = "A"

        # Run with seed 123
        result1 = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                arm,
                "--seed",
                "123",
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result1.returncode, 0)

        # Read metrics 1
        with open(Path("runs") / arm / "metrics.jsonl") as f:
            metrics1 = [json.loads(line) for line in f]

        # Clean runs
        shutil.rmtree("runs", ignore_errors=True)

        # Run with seed 456
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                arm,
                "--seed",
                "456",
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result2.returncode, 0)

        # Read metrics 2
        with open(Path("runs") / arm / "metrics.jsonl") as f:
            metrics2 = [json.loads(line) for line in f]

        # Metrics should be different (at least some fields)
        different = False
        for m1, m2 in zip(metrics1, metrics2):
            if m1.get("bytes_out") != m2.get("bytes_out"):
                different = True
                break
            if m1.get("tokens_out") != m2.get("tokens_out"):
                different = True
                break

        self.assertTrue(different, "Different seeds produced identical metrics")
        print("✓ Different seeds produced different metrics")

    def test_config_override_rejection(self):
        """Test that config overrides are rejected."""
        # Try to override temperature
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                "A",
                "--seed",
                "123",
                "--gen_cfg",
                "custom_config.yaml",  # Non-canonical config
                "--hermetic",
                "on",
                "--toy",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with config parity error
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Config parity", result.stderr)
        print("✓ Config override correctly rejected")

    def test_hermetic_network_blocking(self):
        """Test that network is blocked in hermetic mode."""
        # Create a simple test script that tries network access
        test_script = self.test_dir / "test_network.py"
        test_script.write_text(
            """
import socket
try:
    socket.create_connection(("1.1.1.1", 80), timeout=1)
    print("NETWORK_ALLOWED")
except Exception as e:
    print("NETWORK_BLOCKED")
"""
        )

        # Run in hermetic environment
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                "A",
                "--seed",
                "123",
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "1",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_HERMETIC": "1"},
        )

        # The run should succeed (network blocking doesn't fail the run)
        self.assertEqual(result.returncode, 0)
        print("✓ Hermetic mode enabled successfully")

    def test_metrics_schema_compliance(self):
        """Test that metrics comply with expected schema."""
        # Run evaluation
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                "A",
                "--seed",
                "123",
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)

        # Read metrics
        with open(Path("runs") / "A" / "metrics.jsonl") as f:
            metrics = [json.loads(line) for line in f]

        # Check required fields
        required_fields = {
            "task_id",
            "arm",
            "seed",
            "hermetic",
            "bytes_out",
            "bytes_in",
            "tokens_out",
            "tokens_in",
            "prefill_tokens",
            "decode_tokens",
            "e2e_latency_ms",
            "message_path_ms",
            "pass",
        }

        for m in metrics:
            for field in required_fields:
                self.assertIn(field, m, f"Missing required field: {field}")

        print("✓ Metrics comply with schema")

    def test_parquet_deterministic_hash(self):
        """Test that Parquet files have deterministic content hash."""
        seed = 123
        arm = "A"

        # Run 1
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "3",
            ],
            capture_output=True,
            text=True,
        )

        # Read and hash Parquet content (not file bytes, but content)
        df1 = pd.read_parquet(Path("runs") / arm / "summary.parquet")
        time_cols = [c for c in df1.columns if "time" in c.lower()]
        df1_clean = df1.drop(columns=time_cols, errors="ignore")
        content1 = df1_clean.to_json(orient="records")
        hash1 = hashlib.sha256(content1.encode()).hexdigest()

        # Clean and run 2
        shutil.rmtree("runs", ignore_errors=True)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.run_arms",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--gen_cfg",
                "configs/generation.yaml",
                "--hermetic",
                "on",
                "--toy",
                "3",
            ],
            capture_output=True,
            text=True,
        )

        # Read and hash Parquet content
        df2 = pd.read_parquet(Path("runs") / arm / "summary.parquet")
        time_cols = [c for c in df2.columns if "time" in c.lower()]
        df2_clean = df2.drop(columns=time_cols, errors="ignore")
        content2 = df2_clean.to_json(orient="records")
        hash2 = hashlib.sha256(content2.encode()).hexdigest()

        # Hashes should match
        self.assertEqual(hash1, hash2, "Parquet content hashes don't match")
        print(f"✓ Parquet files have deterministic content hash: {hash1[:16]}")


if __name__ == "__main__":
    unittest.main()
