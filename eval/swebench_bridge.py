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
        model_name: str = "hermes"
    ):
        """
        Write predictions in official SWE-bench harness format.
        
        The official format requires EXACTLY these three fields:
        - instance_id: The task identifier
        - model_name_or_path: Model identifier for tracking
        - model_patch: The generated patch as a diff string
        
        Args:
            predictions: List of dicts with 'instance_id' and 'model_patch'
            output_path: Path to write predictions.jsonl
            model_name: Model name (required for official format)
        
        Expected format per line:
        {"instance_id": "<id>", "model_name_or_path": "<name>", "model_patch": "<diff>"}
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
                
                # Build harness-compatible record with EXACTLY the required fields
                record = {
                    "instance_id": pred["instance_id"],
                    "model_name_or_path": model_name,  # Official field name
                    "model_patch": pred["model_patch"],
                }
                
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
        """Validate predictions file matches official harness format."""
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
                    
                    # Check for EXACTLY the required fields
                    required_fields = {"instance_id", "model_name_or_path", "model_patch"}
                    actual_fields = set(record.keys())
                    
                    if actual_fields != required_fields:
                        print(f"Line {line_no}: Invalid fields. Expected {required_fields}, got {actual_fields}")
                        return False
                    
                    # Validate types
                    if not isinstance(record["instance_id"], str):
                        print(f"Line {line_no}: instance_id must be string")
                        return False
                    
                    if not isinstance(record["model_name_or_path"], str):
                        print(f"Line {line_no}: model_name_or_path must be string")
                        return False
                    
                    if not isinstance(record["model_patch"], str):
                        print(f"Line {line_no}: model_patch must be string")
                        return False
            
            return True
        
        except Exception as e:
            print(f"Error reading predictions: {e}")
            return False