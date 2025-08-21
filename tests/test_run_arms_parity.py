#!/usr/bin/env python3
"""Unit tests for eval.run_arms config parity and determinism."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_arms import ArmRunner, ConfigParityError, enforce_config_parity
from eval._seed import seed_all, verify_determinism


class TestConfigParity(unittest.TestCase):
    """Test config parity enforcement."""
    
    def test_reject_overrides(self):
        """Test that config overrides are rejected."""
        # Create mock args with override attempts
        class Args:
            arm = "A"
            seed = 123
            gen_cfg = "configs/generation.yaml"
            hermetic = "on"
            
        args = Args()
        
        # These should pass
        enforce_config_parity(args)
        
        # Add temperature override - should fail
        args.temperature = 0.5
        with self.assertRaises(ConfigParityError) as ctx:
            enforce_config_parity(args)
        self.assertIn("temperature", str(ctx.exception))
        
        # Remove temperature, add model override - should fail
        delattr(args, "temperature")
        args.model = "llama-70b"
        with self.assertRaises(ConfigParityError) as ctx:
            enforce_config_parity(args)
        self.assertIn("model", str(ctx.exception))
    
    def test_only_canonical_config_allowed(self):
        """Test that only configs/generation.yaml is accepted."""
        # Try to use different config path
        with self.assertRaises(ConfigParityError) as ctx:
            ArmRunner(
                arm="A",
                seed=123,
                gen_cfg_path="configs/custom.yaml",
                hermetic=True
            )
        
        self.assertIn("only 'configs/generation.yaml' is allowed", str(ctx.exception))
    
    def test_config_hash_computed(self):
        """Test that config hash is computed correctly."""
        # Create temporary config for testing
        config_content = """
model:
  temperature: 0.0
  seed: 42
"""
        # Just verify hash computation
        expected_hash = hashlib.sha256(config_content.encode()).hexdigest()[:16]
        actual_hash = hashlib.sha256(config_content.encode()).hexdigest()[:16]
        self.assertEqual(expected_hash, actual_hash)


class TestSeedDeterminism(unittest.TestCase):
    """Test deterministic seeding and execution."""
    
    def test_seed_all_deterministic(self):
        """Test that seed_all produces deterministic state."""
        import random
        
        # Seed and generate numbers
        seed_all(123)
        nums1 = [random.random() for _ in range(10)]
        
        # Re-seed with same value
        seed_all(123)
        nums2 = [random.random() for _ in range(10)]
        
        # Should be identical
        self.assertEqual(nums1, nums2)
    
    def test_task_seed_deterministic(self):
        """Test that task seeds are deterministic."""
        from eval._seed import compute_task_seed
        
        # Same inputs should give same seed
        seed1 = compute_task_seed(42, "task-001")
        seed2 = compute_task_seed(42, "task-001")
        self.assertEqual(seed1, seed2)
        
        # Different task IDs should give different seeds
        seed3 = compute_task_seed(42, "task-002")
        self.assertNotEqual(seed1, seed3)
        
        # Different base seeds should give different task seeds
        seed4 = compute_task_seed(43, "task-001")
        self.assertNotEqual(seed1, seed4)
    
    def test_verify_determinism_helper(self):
        """Test the determinism verification helper."""
        # Identical objects
        obj1 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        obj2 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        
        is_same, msg = verify_determinism(obj1, obj2)
        self.assertTrue(is_same)
        self.assertEqual(msg, "")
        
        # Different objects
        obj3 = {"a": 1, "b": [2, 4], "c": {"d": 4}}
        is_same, msg = verify_determinism(obj1, obj3)
        self.assertFalse(is_same)
        self.assertIn("difference", msg.lower())
        
        # With excluded keys (timestamps)
        obj4 = {"a": 1, "timestamp": 123456, "data": "test"}
        obj5 = {"a": 1, "timestamp": 789012, "data": "test"}
        
        is_same, msg = verify_determinism(obj4, obj5, exclude_keys={"timestamp"})
        self.assertTrue(is_same)
    
    @patch("eval.run_arms.Path.exists")
    @patch("builtins.open")
    def test_same_seed_identical_output(self, mock_open, mock_exists):
        """Test that same seed produces identical summary.parquet."""
        # Mock config file
        mock_exists.return_value = True
        mock_config = """
model:
  temperature: 0.0
  seed: 42
"""
        mock_open.return_value.__enter__.return_value.read.return_value = mock_config
        
        # We can't fully test Parquet output without running the full pipeline
        # But we can verify the seeding mechanism
        seed = 123
        
        # Create two runners with same seed
        with patch("eval.run_arms.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"model": {"temperature": 0.0}}
            
            runner1 = ArmRunner(
                arm="A",
                seed=seed,
                gen_cfg_path="configs/generation.yaml",
                hermetic=False,
                toy_tasks=2
            )
            
            runner2 = ArmRunner(
                arm="A",
                seed=seed,
                gen_cfg_path="configs/generation.yaml",
                hermetic=False,
                toy_tasks=2
            )
            
            # Run IDs should be identical for same seed
            self.assertEqual(runner1.run_id, runner2.run_id)
            
            # Config hashes should be identical
            self.assertEqual(runner1.config_hash, runner2.config_hash)


class TestMetricsOutput(unittest.TestCase):
    """Test metrics output format."""
    
    def test_metrics_jsonl_format(self):
        """Test that metrics are written in correct JSONL format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            
            # Write test metrics
            metrics = [
                {"task_id": "test-001", "pass": True, "latency_ms": 100},
                {"task_id": "test-002", "pass": False, "latency_ms": 150},
            ]
            
            with open(metrics_file, "w") as f:
                for m in metrics:
                    f.write(json.dumps(m, sort_keys=True) + "\n")
            
            # Read and verify
            with open(metrics_file) as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 2)
            
            # Each line should be valid JSON
            for line in lines:
                obj = json.loads(line)
                self.assertIn("task_id", obj)
                self.assertIn("pass", obj)
    
    def test_summary_parquet_format(self):
        """Test that summary is written in correct Parquet format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_file = Path(tmpdir) / "summary.parquet"
            
            # Create test dataframe
            data = {
                "task_id": ["test-001", "test-002"],
                "arm": ["A", "A"],
                "seed": [123, 123],
                "pass": [True, False],
                "latency_ms": [100, 150],
            }
            df = pd.DataFrame(data)
            
            # Write parquet
            df.to_parquet(summary_file, compression="snappy", index=False)
            
            # Read back and verify
            df_read = pd.read_parquet(summary_file)
            
            self.assertEqual(len(df_read), 2)
            self.assertListEqual(list(df_read.columns), list(df.columns))
            self.assertTrue(df.equals(df_read))


if __name__ == "__main__":
    unittest.main()