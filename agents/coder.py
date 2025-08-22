#!/usr/bin/env python3
"""Minimal deterministic coder agent for MVP-0 F0.4."""

import hashlib
import json
from typing import Dict, List, Any


class Coder:
    """Minimal deterministic coder that generates toy patches."""
    
    def __init__(self, seed: int = 0):
        """Initialize coder with seed for determinism."""
        self.seed = seed
    
    def code(self, task: Dict[str, Any], plan_steps: List[str]) -> Dict[str, Any]:
        """Generate deterministic patch for task.
        
        Args:
            task: Task dictionary with task_id, file_path
            plan_steps: Steps from planner
            
        Returns:
            Code response with patch, files_changed, lines added/removed
        """
        # Deterministic hash from task_id and seed
        task_hash = hashlib.sha256(f"{task['task_id']}:{self.seed}".encode()).hexdigest()
        hash_val = int(task_hash[:8], 16)
        
        # Generate minimal deterministic patch
        file_path = task.get("file_path", "src/file.py")
        line_num = 10 + (hash_val % 50)  # Line 10-59
        
        # Deterministic changes based on hash
        lines_added = 1 + (hash_val % 3)  # 1-3 lines
        lines_removed = hash_val % 2  # 0-1 lines
        
        # Generate unified diff format patch
        patch_lines = [
            f"--- a/{file_path}",
            f"+++ b/{file_path}",
            f"@@ -{line_num},{lines_removed + 1} +{line_num},{lines_added + 1} @@"
        ]
        
        # Add removed lines
        if lines_removed > 0:
            patch_lines.append(f"-    old_line = 'removed'")
        
        # Add new lines
        for i in range(lines_added):
            fix_type = ["assert", "return", "if"][i % 3]
            patch_lines.append(f"+    {fix_type} fixed_{hash_val % 100}  # Fix for {task['task_id']}")
        
        patch = "\n".join(patch_lines)
        
        return {
            "patch": patch,
            "files_changed": [file_path],
            "lines_added": lines_added,
            "lines_removed": lines_removed
        }
    
    def code_json(self, task: Dict[str, Any], plan_steps: List[str]) -> str:
        """Generate code as JSON for Arm A."""
        result = self.code(task, plan_steps)
        # Add natural language wrapper for Arm A
        return json.dumps({
            "role": "coder",
            "task_id": task["task_id"],
            "message": f"Applied fix to {result['files_changed'][0]}",
            "patch": result["patch"],
            "summary": f"Added {result['lines_added']} lines, removed {result['lines_removed']} lines"
        }, sort_keys=True)