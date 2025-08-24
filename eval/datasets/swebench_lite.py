"""Hermetic SWE-bench Lite loader with deterministic selection."""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

try:
    from datasets import load_dataset, load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    # Fallback for when datasets library is not available
    def load_from_disk(path):
        """Load mock dataset from disk."""
        import pickle
        dataset_file = Path(path) / "dataset.pkl"
        if dataset_file.exists():
            with open(dataset_file, "rb") as f:
                return pickle.load(f)
        # Try JSON fallback
        json_file = Path(path) / "dataset.json"
        if json_file.exists():
            with open(json_file) as f:
                return json.load(f)
        raise FileNotFoundError(f"No dataset found at {path}")
    
    def load_dataset(name, split, revision):
        """Stub that raises error."""
        raise RuntimeError("datasets library not installed. Use mock data or install datasets.")


class SWEBenchLiteLoader:
    """Hermetic loader for SWE-bench Lite dataset."""
    
    def __init__(self, config_path: str = "configs/swebench_lite.yaml"):
        """Initialize with config."""
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["dataset"]
        
        self.name = self.config["name"]
        self.revision = self.config["revision"]
        self.local_path = Path(self.config["local_path"])
        self.expected_counts = self.config.get("expected_counts", {})
        
        # Check hermetic mode
        self.hermetic = os.environ.get("HERMES_HERMETIC") == "1"
        
        if self.hermetic:
            # Verify local data exists
            manifest_path = self.local_path / "MANIFEST.json"
            if not manifest_path.exists():
                raise RuntimeError(
                    f"Hermetic mode requires prepared data at {self.local_path}\n"
                    f"Run: bash scripts/prepare_swebench.sh"
                )
            
            # Validate manifest
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            if manifest["revision"] != self.revision:
                raise ValueError(
                    f"Manifest revision {manifest['revision']} != config {self.revision}"
                )
            
            # Validate counts
            for split, expected in self.expected_counts.items():
                actual = manifest["splits"].get(split, 0)
                if actual != expected:
                    raise ValueError(
                        f"Split {split}: expected {expected} instances, got {actual}"
                    )
    
    def load_split(self, split: str = "test"):
        """Load a dataset split (hermetic or network)."""
        if self.hermetic:
            # Load from disk only
            split_path = self.local_path / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split {split} not found at {split_path}")
            
            dataset = load_from_disk(str(split_path))
            # If it's a list (mock data), keep as is
            if isinstance(dataset, list):
                dataset = dataset
        else:
            # Development mode - allow network access
            dataset = load_dataset(
                self.name,
                split=split,
                revision=self.revision
            )
            
            # Cache to disk for consistency
            split_path = self.local_path / split
            split_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(split_path))
        
        # Validate count
        expected = self.expected_counts.get(split)
        if expected and len(dataset) != expected:
            raise ValueError(
                f"Split {split}: expected {expected} instances, got {len(dataset)}"
            )
        
        # Convert to list if it's a datasets object
        if not isinstance(dataset, list):
            dataset = list(dataset)
        
        return dataset
    
    def get_smoke20(self, split: str = "test", seed: Optional[int] = None) -> List[Dict]:
        """Get 20 instances for smoke testing (deterministic)."""
        dataset = self.load_split(split)
        
        # Sort by instance_id for stability
        instances = sorted(dataset, key=lambda x: x["instance_id"])
        
        if seed is not None:
            # Seed-based deterministic selection
            import random
            rng = random.Random(seed)
            
            # Create stable hash for each instance
            hashed = []
            for inst in instances:
                h = hashlib.sha256(f"{seed}{inst['instance_id']}".encode()).hexdigest()
                hashed.append((h, inst))
            
            # Sort by hash and take first 20
            hashed.sort(key=lambda x: x[0])
            return [inst for _, inst in hashed[:20]]
        else:
            # Default: first 20 by instance_id
            return instances[:20]
    
    def get_slice50(self, split: str = "test") -> List[Dict]:
        """Get pre-registered 50 instances for MVP-3."""
        slice_file = self.config.get("slice50_file")
        if not slice_file:
            raise ValueError("slice50_file not configured")
        
        # Load instance IDs
        with open(slice_file) as f:
            target_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        if len(target_ids) != 50:
            raise ValueError(f"Expected 50 IDs in {slice_file}, got {len(target_ids)}")
        
        # Load dataset and filter
        dataset = self.load_split(split)
        id_to_inst = {inst["instance_id"]: inst for inst in dataset}
        
        # Return in order specified by file
        result = []
        for tid in target_ids:
            if tid not in id_to_inst:
                raise ValueError(f"Instance {tid} not found in {split} split")
            result.append(id_to_inst[tid])
        
        return result
    
    def to_task_format(self, instance: Dict) -> Dict:
        """Convert SWE-bench instance to our task format."""
        return {
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