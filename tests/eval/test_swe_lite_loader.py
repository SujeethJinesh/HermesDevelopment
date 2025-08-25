#!/usr/bin/env python3
"""Tests for SWE-bench Lite loader."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from eval.datasets.swebench_lite import SWEBenchLiteLoader


class TestSWEBenchLiteLoader:
    """Test suite for SWE-bench Lite loader."""
    
    def test_loader_initialization(self):
        """Test loader initializes with correct defaults."""
        loader = SWEBenchLiteLoader()
        assert loader.DATASET_NAME == "SWE-bench/SWE-bench_Lite"
        assert loader.EXPECTED_DEV_COUNT == 23
        assert loader.EXPECTED_TEST_COUNT == 300
        assert len(loader.REQUIRED_COLUMNS) == 9
    
    def test_loader_with_revision(self):
        """Test loader accepts pinned revision."""
        loader = SWEBenchLiteLoader(revision="abc123")
        assert loader.revision == "abc123"
    
    @patch('eval.datasets.swebench_lite.load_dataset')
    def test_load_split_validates_counts(self, mock_load):
        """Test that loader validates row counts strictly."""
        # Mock dataset with wrong count
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 22  # Wrong count for dev
        mock_dataset.column_names = [
            "instance_id", "repo", "base_commit", "patch", "test_patch",
            "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"
        ]
        mock_dataset.__iter__ = lambda self: iter([])
        mock_load.return_value = mock_dataset
        
        loader = SWEBenchLiteLoader()
        
        # Should raise ValueError for wrong count
        with pytest.raises(ValueError, match="22 rows, expected 23"):
            loader.load_split("dev")
    
    @patch('eval.datasets.swebench_lite.load_dataset')
    def test_load_split_validates_columns(self, mock_load):
        """Test that loader validates required columns."""
        # Mock dataset missing columns
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 23
        mock_dataset.column_names = ["instance_id", "repo"]  # Missing columns
        mock_load.return_value = mock_dataset
        
        loader = SWEBenchLiteLoader()
        
        # Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_split("dev")
    
    @patch('eval.datasets.swebench_lite.load_dataset')
    def test_load_split_success(self, mock_load):
        """Test successful loading with correct data."""
        # Mock correct dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 300
        mock_dataset.column_names = [
            "instance_id", "repo", "base_commit", "patch", "test_patch",
            "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"
        ]
        
        # Mock iteration
        sample_instance = {
            "instance_id": "test_id",
            "repo": "test/repo",
            "base_commit": "abc123",
            "patch": "diff",
            "test_patch": "test diff",
            "problem_statement": "Fix bug",
            "FAIL_TO_PASS": ["test1"],
            "PASS_TO_PASS": ["test2"],
            "environment_setup_commit": "def456"
        }
        mock_dataset.__iter__ = lambda self: iter([sample_instance])
        mock_load.return_value = mock_dataset
        
        loader = SWEBenchLiteLoader()
        instances = loader.load_split("test")
        
        assert len(instances) == 1
        assert instances[0]["instance_id"] == "test_id"
    
    def test_load_instances_file(self, tmp_path):
        """Test loading instances from file."""
        # Create test file
        instances_file = tmp_path / "test_instances.txt"
        instances_file.write_text("""# Test instances
django__django-11001
django__django-11019
# Comment line
django__django-11039
""")
        
        loader = SWEBenchLiteLoader()
        
        # Mock the dataset
        with patch.object(loader, 'load_split') as mock_load:
            mock_load.return_value = [
                {"instance_id": "django__django-11001"},
                {"instance_id": "django__django-11019"},
                {"instance_id": "django__django-11039"},
                {"instance_id": "other_instance"}
            ]
            
            instances = loader.load_instances_file(str(instances_file), "test")
            
            assert len(instances) == 3
            assert instances[0]["instance_id"] == "django__django-11001"
            assert instances[2]["instance_id"] == "django__django-11039"
    
    def test_load_instances_file_missing(self):
        """Test error handling for missing file."""
        loader = SWEBenchLiteLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_instances_file("/nonexistent/file.txt", "test")
    
    def test_to_task_format(self):
        """Test conversion to task format excludes patch."""
        loader = SWEBenchLiteLoader()
        
        instance = {
            "instance_id": "test_id",
            "repo": "test/repo",
            "base_commit": "abc123",
            "patch": "SECRET PATCH",  # Should be excluded
            "problem_statement": "Fix bug",
            "test_patch": "test diff",
            "FAIL_TO_PASS": ["test1"],
            "PASS_TO_PASS": ["test2"],
            "environment_setup_commit": "def456"
        }
        
        task = loader.to_task_format(instance)
        
        # Check patch is NOT included
        assert "patch" not in task
        assert task["task_id"] == "test_id"
        assert task["repo"] == "test/repo"
        assert task["problem_statement"] == "Fix bug"
    
    def test_hermetic_mode(self):
        """Test hermetic mode sets offline flag."""
        with patch.dict(os.environ, {"HERMES_HERMETIC": "1"}):
            loader = SWEBenchLiteLoader()
            assert loader.hermetic
            assert os.environ.get("HF_DATASETS_OFFLINE") == "1"