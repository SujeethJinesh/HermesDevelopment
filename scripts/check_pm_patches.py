#!/usr/bin/env python3
"""Check what PM arm is sending as patches."""

import json
from pathlib import Path

metrics_file = Path("runs/PM/metrics.jsonl")

print("Checking PM patches in last 3 tasks...")
print("=" * 60)

# Read last 3 entries
with open(metrics_file) as f:
    lines = f.readlines()
    last_entries = lines[-3:] if len(lines) >= 3 else lines

for i, line in enumerate(last_entries, 1):
    try:
        data = json.loads(line.strip())
        task_id = data.get("task_id", "unknown")
        passed = data.get("pass", False)
        
        print(f"\nTask {i}: {task_id}")
        print(f"  Passed: {passed}")
        
        # Check for MCP references
        mcp_refs = data.get("mcp_refs", [])
        if mcp_refs:
            print(f"  MCP refs created: {mcp_refs}")
        
        # Check error messages
        if "error" in data:
            print(f"  Error: {data['error'][:200]}")
            
    except json.JSONDecodeError as e:
        print(f"  Failed to parse: {e}")
        print(f"  Line preview: {line[:200]}")