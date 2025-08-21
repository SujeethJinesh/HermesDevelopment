#!/usr/bin/env python3
"""Minimal deterministic tester agent for MVP-0 F0.4."""

import hashlib
import json
from typing import Dict, List, Any


class Tester:
    """Minimal deterministic tester that generates test results."""
    
    def __init__(self, seed: int = 0):
        """Initialize tester with seed for determinism."""
        self.seed = seed
    
    def test(self, task: Dict[str, Any], patch: str) -> Dict[str, Any]:
        """Generate deterministic test results for task.
        
        Args:
            task: Task dictionary with task_id, test_name
            patch: Applied patch from coder
            
        Returns:
            Test response with passed, output, duration_ms, failures
        """
        # Deterministic hash from task_id and seed
        task_hash = hashlib.sha256(f"{task['task_id']}:{self.seed}".encode()).hexdigest()
        hash_val = int(task_hash[:8], 16)
        
        # Deterministic pass/fail (70% pass rate)
        passed = (hash_val % 10) < 7
        
        # Deterministic test duration (100-500ms)
        duration_ms = 100 + (hash_val % 401)
        
        # Generate test output
        test_name = task.get("test_name", "test_function")
        if passed:
            output = f"Running {test_name}...\n✓ All tests passed (1/1)"
            failures = []
        else:
            output = f"Running {test_name}...\n✗ Test failed"
            # Deterministic failure reason
            failure_types = [
                "AssertionError: Expected 42, got 41",
                "ValueError: Invalid parameter",
                "KeyError: 'missing_key'"
            ]
            failures = [failure_types[hash_val % len(failure_types)]]
            output += f"\n  {failures[0]}"
        
        return {
            "passed": passed,
            "output": output,
            "duration_ms": duration_ms,
            "failures": failures
        }
    
    def test_json(self, task: Dict[str, Any], patch: str) -> str:
        """Generate test result as JSON for Arm A."""
        result = self.test(task, patch)
        # Add natural language wrapper for Arm A
        status = "passed" if result["passed"] else "failed"
        return json.dumps({
            "role": "tester",
            "task_id": task["task_id"],
            "message": f"Test {status} in {result['duration_ms']}ms",
            "passed": result["passed"],
            "output": result["output"],
            "duration_ms": result["duration_ms"]
        }, sort_keys=True)