"""Official SWE-bench Lite loader with hermetic support."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

from datasets import load_dataset


class SWEBenchLiteLoader:
    """Official loader for SWE-bench Lite dataset from Hugging Face."""
    
    # Official dataset name and expected counts
    DATASET_NAME = "SWE-bench/SWE-bench_Lite"
    EXPECTED_DEV_COUNT = 23
    EXPECTED_TEST_COUNT = 300
    
    # Required columns per official schema
    REQUIRED_COLUMNS = [
        "instance_id", "repo", "base_commit", "patch", "test_patch",
        "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"
    ]
    
    def __init__(self, revision: Optional[str] = None):
        """Initialize with optional pinned revision.
        
        Args:
            revision: HF dataset revision (commit SHA or tag). If None, reads from
                     SWEBENCH_REVISION env var or configs/swebench_lite.yaml
        """
        # Load config for revision if not provided
        if revision is None:
            revision = os.environ.get("SWEBENCH_REVISION")
            if revision is None:
                try:
                    import yaml
                    with open("configs/swebench_lite.yaml") as f:
                        config = yaml.safe_load(f)
                        revision = config.get("dataset", {}).get("revision")
                except (FileNotFoundError, KeyError):
                    pass  # Use latest if no revision specified
        
        self.revision = revision
        self.expected_counts = {
            "dev": self.EXPECTED_DEV_COUNT,
            "test": self.EXPECTED_TEST_COUNT
        }
        
        # Check hermetic mode (respects HF_DATASETS_OFFLINE)
        self.hermetic = os.environ.get("HERMES_HERMETIC") == "1"
        if self.hermetic:
            # Ensure HF offline mode is set
            os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    def load_split(self, split: str = "test") -> List[Dict]:
        """Load a dataset split from Hugging Face.
        
        Args:
            split: Dataset split to load ("dev" or "test")
            
        Returns:
            List of instances as dictionaries
            
        Raises:
            ValueError: If split is invalid or dataset validation fails
        """
        if split not in self.expected_counts:
            raise ValueError(f"Invalid split '{split}'. Must be 'dev' or 'test'")
        
        # Load from Hugging Face (uses cache if available, respects HF_DATASETS_OFFLINE)
        dataset = load_dataset(
            self.DATASET_NAME,
            split=split,
            revision=self.revision,
            trust_remote_code=False  # Security: don't execute remote code
        )
        
        # Validate structure
        self._validate_dataset(dataset, split)
        
        # Convert to list of dicts
        return [dict(row) for row in dataset]
    
    def _validate_dataset(self, dataset, split: str):
        """Validate dataset has expected structure and size."""
        # Check row count
        actual_count = len(dataset)
        expected_count = self.expected_counts[split]
        if actual_count != expected_count:
            raise ValueError(
                f"Dataset validation failed: {split} split has {actual_count} rows, "
                f"expected {expected_count}"
            )
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(dataset.column_names)
        if missing_cols:
            raise ValueError(
                f"Dataset validation failed: missing required columns {missing_cols}"
            )
    
    def get_instances_by_ids(self, split: str, instance_ids: List[str]) -> List[Dict]:
        """Get specific instances by their IDs.
        
        Args:
            split: Dataset split
            instance_ids: List of instance IDs to retrieve
            
        Returns:
            List of matching instances in the order specified
            
        Raises:
            ValueError: If any instance ID is not found
        """
        all_instances = self.load_split(split)
        id_to_instance = {inst["instance_id"]: inst for inst in all_instances}
        
        instances = []
        missing_ids = []
        for inst_id in instance_ids:
            if inst_id in id_to_instance:
                instances.append(id_to_instance[inst_id])
            else:
                missing_ids.append(inst_id)
        
        if missing_ids:
            raise ValueError(
                f"Instance IDs not found in {split} split: {missing_ids}"
            )
        
        return instances
    
    def load_instances_file(self, path: str, split: str = "test") -> List[Dict]:
        """Load instances specified in a file.
        
        Args:
            path: Path to file with one instance ID per line
            split: Dataset split to load from
            
        Returns:
            List of instances in the order specified
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Instances file not found: {path}")
        
        # Load instance IDs from file
        instance_ids = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    instance_ids.append(line)
        
        if not instance_ids:
            raise ValueError(f"No instance IDs found in {path}")
        
        return self.get_instances_by_ids(split, instance_ids)
    
    def get_dataset_info(self) -> Dict:
        """Get metadata about the dataset.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            "dataset_name": self.DATASET_NAME,
            "revision": self.revision,
            "expected_counts": self.expected_counts,
            "required_columns": self.REQUIRED_COLUMNS,
            "hermetic": self.hermetic,
        }
    
    def to_task_format(self, instance: Dict) -> Dict:
        """Convert SWE-bench instance to our task format.
        
        IMPORTANT: Never expose gold patch to agents!
        """
        # Validate required fields
        required_fields = ["instance_id", "repo", "base_commit", "problem_statement"]
        for field in required_fields:
            if field not in instance:
                raise ValueError(f"Missing required field: {field}")
        
        # NEVER expose the gold patch to agents
        # The patch field exists in the dataset but must not be passed to agents
        result = {
            "task_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "problem_statement": instance["problem_statement"],
            "hints_text": instance.get("hints_text", ""),
            "test_patch": instance.get("test_patch", ""),
            "fail_to_pass": instance.get("FAIL_TO_PASS", []),
            "pass_to_pass": instance.get("PASS_TO_PASS", []),
            "environment_setup_commit": instance.get("environment_setup_commit"),
            "created_at": instance.get("created_at", ""),
            "version": instance.get("version", ""),
        }
        
        # Ensure patch is not accidentally included
        if "patch" in result:
            del result["patch"]
        
        return result