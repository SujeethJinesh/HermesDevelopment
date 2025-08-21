#!/usr/bin/env python3
"""Minimal deterministic planner agent for MVP-0 F0.4."""

import hashlib
import json
from typing import Dict, List, Any


class Planner:
    """Minimal deterministic planner that generates toy plans."""
    
    def __init__(self, seed: int = 0):
        """Initialize planner with seed for determinism."""
        self.seed = seed
    
    def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deterministic plan for task.
        
        Args:
            task: Task dictionary with task_id, repo, file_path, test_name, description
            
        Returns:
            Plan with steps, approach, and confidence
        """
        # Deterministic hash from task_id and seed
        task_hash = hashlib.sha256(f"{task['task_id']}:{self.seed}".encode()).hexdigest()
        hash_val = int(task_hash[:8], 16)
        
        # Deterministic steps based on hash
        num_steps = 2 + (hash_val % 3)  # 2-4 steps
        steps = []
        for i in range(num_steps):
            step_type = ["Analyze", "Identify", "Fix", "Verify"][i % 4]
            steps.append(f"{step_type} {task.get('file_path', 'file')} for {task.get('test_name', 'test')}")
        
        # Deterministic approach
        approaches = ["direct fix", "refactor first", "add missing logic", "patch edge case"]
        approach = approaches[hash_val % len(approaches)]
        
        # Deterministic confidence (60-95%)
        confidence = 60 + (hash_val % 36)
        
        return {
            "steps": steps,
            "approach": approach,
            "confidence": confidence
        }
    
    def plan_json(self, task: Dict[str, Any]) -> str:
        """Generate plan as JSON for Arm A."""
        plan = self.plan(task)
        # Add natural language wrapper for Arm A
        return json.dumps({
            "role": "planner",
            "task_id": task["task_id"],
            "plan": f"I will {plan['approach']} in {len(plan['steps'])} steps",
            "steps": plan["steps"],
            "confidence": f"{plan['confidence']}%"
        }, sort_keys=True)