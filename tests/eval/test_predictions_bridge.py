#!/usr/bin/env python3
"""Tests for SWE-bench predictions bridge."""

import json
import pytest
from pathlib import Path

from eval.swebench_bridge import SWEBenchBridge


class TestSWEBenchBridge:
    """Test suite for predictions bridge."""
    
    def test_write_predictions_format(self, tmp_path):
        """Test predictions are written in official format."""
        predictions = [
            {"instance_id": "test_1", "model_patch": "diff1"},
            {"instance_id": "test_2", "model_patch": "diff2"},
        ]
        
        output_file = tmp_path / "predictions.jsonl"
        bridge = SWEBenchBridge()
        bridge.write_predictions(predictions, str(output_file), "hermes-pm")
        
        # Read and validate
        with open(output_file) as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Check first prediction
        pred1 = json.loads(lines[0])
        assert set(pred1.keys()) == {"instance_id", "model_name_or_path", "model_patch"}
        assert pred1["instance_id"] == "test_1"
        assert pred1["model_name_or_path"] == "hermes-pm"
        assert pred1["model_patch"] == "diff1"
    
    def test_validate_predictions_correct(self, tmp_path):
        """Test validation passes for correct format."""
        # Write correct predictions
        pred_file = tmp_path / "good.jsonl"
        with open(pred_file, "w") as f:
            f.write(json.dumps({
                "instance_id": "test_1",
                "model_name_or_path": "hermes",
                "model_patch": "diff"
            }) + "\n")
        
        bridge = SWEBenchBridge()
        assert bridge.validate_predictions_format(str(pred_file))
    
    def test_validate_predictions_wrong_fields(self, tmp_path):
        """Test validation fails for wrong fields."""
        # Write predictions with extra field
        pred_file = tmp_path / "bad.jsonl"
        with open(pred_file, "w") as f:
            f.write(json.dumps({
                "instance_id": "test_1",
                "model_name_or_path": "hermes",
                "model_patch": "diff",
                "extra_field": "bad"  # Extra field not allowed
            }) + "\n")
        
        bridge = SWEBenchBridge()
        assert not bridge.validate_predictions_format(str(pred_file))
    
    def test_validate_predictions_missing_field(self, tmp_path):
        """Test validation fails for missing fields."""
        # Write predictions missing model_name_or_path
        pred_file = tmp_path / "bad.jsonl"
        with open(pred_file, "w") as f:
            f.write(json.dumps({
                "instance_id": "test_1",
                "model_patch": "diff"
                # Missing model_name_or_path
            }) + "\n")
        
        bridge = SWEBenchBridge()
        assert not bridge.validate_predictions_format(str(pred_file))
    
    def test_missing_instance_id_raises(self):
        """Test that missing instance_id raises error."""
        predictions = [
            {"model_patch": "diff"}  # Missing instance_id
        ]
        
        bridge = SWEBenchBridge()
        with pytest.raises(ValueError, match="Missing instance_id"):
            bridge.write_predictions(predictions, "out.jsonl", "hermes")