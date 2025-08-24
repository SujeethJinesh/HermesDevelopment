"""Tests for hermetic SWE-bench Lite loader."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestSWEBenchLiteLoader:
    """Test SWE-bench Lite loader functionality."""
    
    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config file."""
        config = {
            "dataset": {
                "name": "princeton-nlp/SWE-bench_Lite",
                "revision": "test123",
                "local_path": str(tmp_path / "data"),
                "expected_counts": {
                    "dev": 23,
                    "test": 300,
                }
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path
    
    @pytest.fixture
    def mock_manifest(self, tmp_path):
        """Create mock manifest file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        
        manifest = {
            "dataset_name": "princeton-nlp/SWE-bench_Lite",
            "revision": "test123",
            "local_path": str(data_dir),
            "splits": {
                "dev": 23,
                "test": 300,
            },
            "sha256_dir": "abc123",
        }
        
        manifest_path = data_dir / "MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        return data_dir
    
    def test_hermetic_mode_requires_manifest(self, temp_config):
        """Test that hermetic mode requires prepared data."""
        # Set hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"
        
        try:
            from eval.datasets.swebench_lite import SWEBenchLiteLoader
            
            # Should raise error without manifest
            with pytest.raises(RuntimeError, match="Hermetic mode requires prepared data"):
                loader = SWEBenchLiteLoader(str(temp_config))
        finally:
            del os.environ["HERMES_HERMETIC"]
    
    def test_manifest_validation(self, temp_config, mock_manifest):
        """Test manifest validation in hermetic mode."""
        os.environ["HERMES_HERMETIC"] = "1"
        
        try:
            from eval.datasets.swebench_lite import SWEBenchLiteLoader
            
            # Should load successfully with valid manifest
            loader = SWEBenchLiteLoader(str(temp_config))
            assert loader.revision == "test123"
            assert loader.expected_counts["dev"] == 23
            assert loader.expected_counts["test"] == 300
        finally:
            del os.environ["HERMES_HERMETIC"]
    
    def test_count_validation(self, temp_config, mock_manifest):
        """Test that counts are validated against expected."""
        # Create wrong counts in manifest
        manifest_path = mock_manifest / "MANIFEST.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        manifest["splits"]["dev"] = 20  # Wrong count
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        os.environ["HERMES_HERMETIC"] = "1"
        
        try:
            from eval.datasets.swebench_lite import SWEBenchLiteLoader
            
            # Should raise error with wrong counts
            with pytest.raises(ValueError, match="expected 23 instances, got 20"):
                loader = SWEBenchLiteLoader(str(temp_config))
        finally:
            del os.environ["HERMES_HERMETIC"]
    
    @patch('eval.datasets.swebench_lite.load_from_disk')
    def test_hermetic_load_from_disk(self, mock_load, temp_config, mock_manifest):
        """Test that hermetic mode uses load_from_disk only."""
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 300
        mock_dataset.__iter__ = lambda self: iter([
            {"instance_id": f"task-{i}", "repo": "test"} for i in range(300)
        ])
        mock_load.return_value = mock_dataset
        
        # Create test split directory
        (mock_manifest / "test").mkdir()
        
        try:
            from eval.datasets.swebench_lite import SWEBenchLiteLoader
            
            loader = SWEBenchLiteLoader(str(temp_config))
            dataset = loader.load_split("test")
            
            # Verify load_from_disk was called
            mock_load.assert_called_once()
            assert len(dataset) == 300
        finally:
            del os.environ["HERMES_HERMETIC"]
    
    @patch('eval.datasets.swebench_lite.load_dataset')
    def test_network_blocked_in_hermetic(self, mock_load_dataset, temp_config, mock_manifest):
        """Test that network calls are blocked in hermetic mode."""
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Mock should not be called in hermetic mode
        mock_load_dataset.side_effect = RuntimeError("Network access attempted in hermetic mode!")
        
        # Create test split directory with mock data
        test_dir = mock_manifest / "test"
        test_dir.mkdir()
        
        # Use actual datasets format
        with patch('eval.datasets.swebench_lite.load_from_disk') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = lambda self: 300
            mock_load.return_value = mock_dataset
            
            try:
                from eval.datasets.swebench_lite import SWEBenchLiteLoader
                
                loader = SWEBenchLiteLoader(str(temp_config))
                dataset = loader.load_split("test")
                
                # Network load should NOT have been called
                mock_load_dataset.assert_not_called()
            finally:
                del os.environ["HERMES_HERMETIC"]
    
    def test_smoke20_deterministic_selection(self, temp_config, mock_manifest):
        """Test that smoke-20 selection is deterministic."""
        # Create mock dataset
        test_instances = [
            {"instance_id": f"django__django-{i:05d}", "repo": "django/django"}
            for i in range(100)
        ]
        
        with patch('eval.datasets.swebench_lite.load_from_disk') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = lambda self: len(test_instances)
            mock_dataset.__iter__ = lambda self: iter(test_instances)
            mock_load.return_value = mock_dataset
            
            # Create test directory
            (mock_manifest / "test").mkdir()
            
            os.environ["HERMES_HERMETIC"] = "1"
            
            try:
                from eval.datasets.swebench_lite import SWEBenchLiteLoader
                
                loader = SWEBenchLiteLoader(str(temp_config))
                
                # Get smoke-20 with same seed twice
                smoke1 = loader.get_smoke20("test", seed=123)
                smoke2 = loader.get_smoke20("test", seed=123)
                
                # Should be identical
                assert len(smoke1) == 20
                assert len(smoke2) == 20
                assert [t["instance_id"] for t in smoke1] == [t["instance_id"] for t in smoke2]
                
                # Different seed should give different selection
                smoke3 = loader.get_smoke20("test", seed=456)
                assert [t["instance_id"] for t in smoke1] != [t["instance_id"] for t in smoke3]
                
            finally:
                del os.environ["HERMES_HERMETIC"]
    
    def test_slice50_loading(self, temp_config, mock_manifest, tmp_path):
        """Test loading pre-registered 50 instances."""
        # Create slice50 file
        slice_file = tmp_path / "slice50.txt"
        instance_ids = [f"django__django-{i:05d}" for i in range(50)]
        with open(slice_file, "w") as f:
            for iid in instance_ids:
                f.write(f"{iid}\n")
        
        # Update config with slice file
        with open(temp_config) as f:
            config = yaml.safe_load(f)
        config["dataset"]["slice50_file"] = str(slice_file)
        with open(temp_config, "w") as f:
            yaml.dump(config, f)
        
        # Create mock dataset with more than 50 instances
        all_instances = [
            {"instance_id": f"django__django-{i:05d}", "repo": "django/django"}
            for i in range(100)
        ]
        
        with patch('eval.datasets.swebench_lite.load_from_disk') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = lambda self: len(all_instances)
            mock_dataset.__iter__ = lambda self: iter(all_instances)
            mock_load.return_value = mock_dataset
            
            # Create test directory
            (mock_manifest / "test").mkdir()
            
            os.environ["HERMES_HERMETIC"] = "1"
            
            try:
                from eval.datasets.swebench_lite import SWEBenchLiteLoader
                
                loader = SWEBenchLiteLoader(str(temp_config))
                slice50 = loader.get_slice50("test")
                
                # Should return exactly 50 in order
                assert len(slice50) == 50
                returned_ids = [t["instance_id"] for t in slice50]
                assert returned_ids == instance_ids
                
            finally:
                del os.environ["HERMES_HERMETIC"]
    
    def test_to_task_format(self, temp_config):
        """Test conversion to task format."""
        from eval.datasets.swebench_lite import SWEBenchLiteLoader
        
        # Don't need hermetic mode for this test
        loader = SWEBenchLiteLoader(str(temp_config))
        
        # Create SWE-bench instance
        instance = {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "hints_text": "Look at line 42",
            "test_patch": "diff content",
            "FAIL_TO_PASS": ["test_foo", "test_bar"],
            "PASS_TO_PASS": ["test_baz"],
            "environment_setup_commit": "def456",
            "created_at": "2024-01-01",
            "version": "1.0",
        }
        
        # Convert to task format
        task = loader.to_task_format(instance)
        
        # Verify fields
        assert task["task_id"] == "django__django-12345"
        assert task["repo"] == "django/django"
        assert task["base_commit"] == "abc123"
        assert task["problem_statement"] == "Fix the bug"
        assert task["fail_to_pass"] == ["test_foo", "test_bar"]
        assert task["pass_to_pass"] == ["test_baz"]