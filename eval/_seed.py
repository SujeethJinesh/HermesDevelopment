"""Deterministic seeding utilities for HERMES evaluation."""

import hashlib
import json
import os
import random
from typing import Any, Dict, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def seed_all(seed: int, verbose: bool = False) -> Dict[str, Any]:
    """Seed all random number generators for determinism.
    
    Args:
        seed: Random seed to use
        verbose: Whether to print seeding info
        
    Returns:
        Dictionary with seeding information
    """
    seeding_info = {
        "seed": seed,
        "python_random": True,
        "numpy": False,
        "torch": False,
        "env_pythonhashseed": False,
    }
    
    # Python's built-in random
    random.seed(seed)
    
    # NumPy if available
    if HAS_NUMPY:
        np.random.seed(seed)
        seeding_info["numpy"] = True
        if verbose:
            print(f"Seeded NumPy with {seed}")
    
    # PyTorch if available
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seeding_info["torch"] = True
        seeding_info["torch_cuda"] = torch.cuda.is_available()
        if verbose:
            print(f"Seeded PyTorch with {seed}")
    
    # Python hash seed for consistent dict ordering
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(seed)
        seeding_info["env_pythonhashseed"] = True
        if verbose:
            print(f"Set PYTHONHASHSEED={seed}")
    
    if verbose:
        print(f"Seeded all RNGs with seed={seed}")
    
    return seeding_info


def compute_task_seed(base_seed: int, task_id: str) -> int:
    """Compute deterministic seed for a specific task.
    
    Args:
        base_seed: Base random seed
        task_id: Task identifier
        
    Returns:
        Deterministic seed for the task
    """
    # Create stable hash from task_id and base seed
    seed_str = f"{base_seed}:{task_id}"
    seed_hash = hashlib.sha256(seed_str.encode()).hexdigest()
    
    # Convert first 8 hex chars to int (32-bit range)
    task_seed = int(seed_hash[:8], 16) % (2**31 - 1)
    
    return task_seed


def verify_determinism(
    obj1: Any,
    obj2: Any,
    exclude_keys: Optional[set] = None
) -> tuple[bool, str]:
    """Verify two objects are identical for determinism checking.
    
    Args:
        obj1: First object
        obj2: Second object
        exclude_keys: Keys to exclude from comparison (e.g., timestamps)
        
    Returns:
        Tuple of (is_identical, difference_message)
    """
    exclude_keys = exclude_keys or {"timestamp", "duration", "time"}
    
    def clean_dict(d: Dict) -> Dict:
        """Remove excluded keys from dict."""
        if not isinstance(d, dict):
            return d
        return {
            k: clean_dict(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if not any(ex in k.lower() for ex in exclude_keys)
        }
    
    # Clean objects if they're dicts
    if isinstance(obj1, dict):
        obj1 = clean_dict(obj1)
    if isinstance(obj2, dict):
        obj2 = clean_dict(obj2)
    
    # Compare JSON representations for complex objects
    try:
        json1 = json.dumps(obj1, sort_keys=True, default=str)
        json2 = json.dumps(obj2, sort_keys=True, default=str)
        
        if json1 == json2:
            return True, ""
        else:
            # Find first difference
            for i, (c1, c2) in enumerate(zip(json1, json2)):
                if c1 != c2:
                    context_start = max(0, i - 20)
                    context_end = min(len(json1), i + 20)
                    return False, (
                        f"First difference at position {i}:\n"
                        f"  obj1: ...{json1[context_start:context_end]}...\n"
                        f"  obj2: ...{json2[context_start:context_end]}..."
                    )
            
            # Different lengths
            return False, f"Different lengths: {len(json1)} vs {len(json2)}"
            
    except Exception as e:
        return False, f"Comparison failed: {e}"


def generate_deterministic_id(seed: int, prefix: str = "run") -> str:
    """Generate deterministic run ID from seed.
    
    Args:
        seed: Random seed
        prefix: Prefix for the ID
        
    Returns:
        Deterministic ID string
    """
    # Use seed to generate stable hash
    id_str = f"{prefix}_{seed}"
    id_hash = hashlib.sha256(id_str.encode()).hexdigest()[:12]
    
    return f"{prefix}_{seed}_{id_hash}"