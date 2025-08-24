"""Bridge to official SWE-bench harness evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class SWEBenchBridge:
    """Bridge to write predictions in SWE-bench harness format."""
    
    @staticmethod
    def write_predictions(
        predictions: List[Dict[str, str]],
        output_path: str,
        model_name: Optional[str] = "hermes"
    ):
        """
        Write predictions in SWE-bench harness format.
        
        Args:
            predictions: List of dicts with 'instance_id' and 'model_patch'
            output_path: Path to write predictions.jsonl
            model_name: Optional model name to include
        
        Expected format per line:
        {"instance_id": "<id>", "model_patch": "<unified diff>", "model_name": "<name>"}
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pred in predictions:
                # Validate required fields
                if "instance_id" not in pred:
                    raise ValueError(f"Missing instance_id in prediction: {pred}")
                if "model_patch" not in pred:
                    raise ValueError(f"Missing model_patch for {pred['instance_id']}")
                
                # Build harness-compatible record
                record = {
                    "instance_id": pred["instance_id"],
                    "model_patch": pred["model_patch"],
                }
                
                # Add optional model name
                if model_name:
                    record["model_name"] = model_name
                
                # Write as JSONL
                f.write(json.dumps(record) + "\n")
        
        return output_path
    
    @staticmethod
    def generate_harness_command(
        predictions_path: str,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        instance_ids_file: Optional[str] = None,
        max_workers: int = 8,
        run_id: Optional[str] = None
    ) -> str:
        """
        Generate command to run official harness evaluation.
        
        Returns command string for run_evaluation.
        """
        cmd_parts = [
            "python -m swebench.harness.run_evaluation",
            f"--dataset_name {dataset_name}",
            f"--split {split}",
            f"--predictions_path {predictions_path}",
            f"--max_workers {max_workers}",
        ]
        
        if instance_ids_file:
            # Load IDs from file
            with open(instance_ids_file) as f:
                ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            cmd_parts.append(f"--instance_ids {','.join(ids)}")
        
        if run_id:
            cmd_parts.append(f"--run_id {run_id}")
        
        return " \\\n  ".join(cmd_parts)
    
    @staticmethod
    def validate_predictions_format(predictions_path: str) -> bool:
        """Validate predictions file matches harness format."""
        try:
            with open(predictions_path) as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Line {line_no}: Invalid JSON - {e}")
                        return False
                    
                    # Check required fields
                    if "instance_id" not in record:
                        print(f"Line {line_no}: Missing instance_id")
                        return False
                    
                    if "model_patch" not in record:
                        print(f"Line {line_no}: Missing model_patch")
                        return False
                    
                    # Validate patch is string
                    if not isinstance(record["model_patch"], str):
                        print(f"Line {line_no}: model_patch must be string")
                        return False
            
            return True
        
        except Exception as e:
            print(f"Error reading predictions: {e}")
            return False