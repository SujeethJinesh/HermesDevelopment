"""Tests for SWE-bench predictions bridge."""

import json
import tempfile
from pathlib import Path

import pytest

from eval.swebench_bridge import SWEBenchBridge


class TestSWEBenchBridge:
    """Test SWE-bench bridge functionality."""
    
    def test_write_predictions_format(self, tmp_path):
        """Test predictions are written in correct format."""
        predictions = [
            {
                "instance_id": "django__django-12345",
                "model_patch": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            },
            {
                "instance_id": "flask__flask-4567",
                "model_patch": "diff content here",
            },
        ]
        
        output_path = tmp_path / "predictions.jsonl"
        
        # Write predictions
        SWEBenchBridge.write_predictions(
            predictions, str(output_path), model_name="hermes-test"
        )
        
        # Verify file exists
        assert output_path.exists()
        
        # Read and verify format
        with open(output_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Check first prediction
        pred1 = json.loads(lines[0])
        assert pred1["instance_id"] == "django__django-12345"
        assert "model_patch" in pred1
        assert pred1["model_name"] == "hermes-test"
        
        # Check second prediction
        pred2 = json.loads(lines[1])
        assert pred2["instance_id"] == "flask__flask-4567"
        assert pred2["model_patch"] == "diff content here"
    
    def test_missing_required_fields(self, tmp_path):
        """Test error handling for missing fields."""
        # Missing instance_id
        predictions = [
            {
                "model_patch": "some patch",
            }
        ]
        
        output_path = tmp_path / "bad_predictions.jsonl"
        
        with pytest.raises(ValueError, match="Missing instance_id"):
            SWEBenchBridge.write_predictions(predictions, str(output_path))
        
        # Missing model_patch
        predictions = [
            {
                "instance_id": "test-123",
            }
        ]
        
        with pytest.raises(ValueError, match="Missing model_patch"):
            SWEBenchBridge.write_predictions(predictions, str(output_path))
    
    def test_validate_predictions_format(self, tmp_path):
        """Test validation of predictions file format."""
        # Write valid predictions
        valid_path = tmp_path / "valid.jsonl"
        with open(valid_path, "w") as f:
            f.write('{"instance_id": "test-1", "model_patch": "patch1"}\n')
            f.write('{"instance_id": "test-2", "model_patch": "patch2"}\n')
        
        assert SWEBenchBridge.validate_predictions_format(str(valid_path))
        
        # Invalid JSON
        invalid_json_path = tmp_path / "invalid_json.jsonl"
        with open(invalid_json_path, "w") as f:
            f.write('not json\n')
        
        assert not SWEBenchBridge.validate_predictions_format(str(invalid_json_path))
        
        # Missing field
        missing_field_path = tmp_path / "missing_field.jsonl"
        with open(missing_field_path, "w") as f:
            f.write('{"instance_id": "test-1"}\n')  # Missing model_patch
        
        assert not SWEBenchBridge.validate_predictions_format(str(missing_field_path))
        
        # Wrong type
        wrong_type_path = tmp_path / "wrong_type.jsonl"
        with open(wrong_type_path, "w") as f:
            f.write('{"instance_id": "test-1", "model_patch": 123}\n')  # Should be string
        
        assert not SWEBenchBridge.validate_predictions_format(str(wrong_type_path))
    
    def test_generate_harness_command(self, tmp_path):
        """Test generation of harness evaluation command."""
        predictions_path = "/path/to/predictions.jsonl"
        
        # Basic command
        cmd = SWEBenchBridge.generate_harness_command(
            predictions_path,
            run_id="test_run_123"
        )
        
        assert "python -m swebench.harness.run_evaluation" in cmd
        assert "--dataset_name princeton-nlp/SWE-bench_Lite" in cmd
        assert "--split test" in cmd
        assert "--predictions_path /path/to/predictions.jsonl" in cmd
        assert "--max_workers 8" in cmd
        assert "--run_id test_run_123" in cmd
        
        # With instance IDs file
        ids_file = tmp_path / "ids.txt"
        with open(ids_file, "w") as f:
            f.write("django__django-12345\n")
            f.write("flask__flask-6789\n")
        
        cmd = SWEBenchBridge.generate_harness_command(
            predictions_path,
            instance_ids_file=str(ids_file),
            max_workers=4
        )
        
        assert "--instance_ids django__django-12345,flask__flask-6789" in cmd
        assert "--max_workers 4" in cmd
    
    def test_empty_predictions(self, tmp_path):
        """Test handling of empty predictions list."""
        output_path = tmp_path / "empty.jsonl"
        
        # Should create empty file
        SWEBenchBridge.write_predictions([], str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size == 0
    
    def test_directory_creation(self, tmp_path):
        """Test that directories are created if needed."""
        output_path = tmp_path / "deep" / "nested" / "path" / "predictions.jsonl"
        
        predictions = [
            {
                "instance_id": "test-1",
                "model_patch": "patch",
            }
        ]
        
        # Should create directories
        SWEBenchBridge.write_predictions(predictions, str(output_path))
        
        assert output_path.exists()
        assert output_path.parent.exists()