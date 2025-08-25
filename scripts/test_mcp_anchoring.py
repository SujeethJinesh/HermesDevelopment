#!/usr/bin/env python3
"""Test MCP anchoring with large payloads."""

import json
from pathlib import Path

# Load existing test data
data_path = Path("data/swebench_lite/b8b14b3/test/dataset.json")
with open(data_path) as f:
    instances = json.load(f)

# Modify first instance to have a large problem statement (>32KB)
large_description = """
This is a critical issue affecting production systems. The problem manifests when users attempt to
process large datasets with the following characteristics:

""" + ("The system fails to handle edge cases properly when processing large batches of data. " * 500)

# This creates ~40KB of text
instances[0]["problem_statement"] = large_description
print(f"Modified first instance with {len(large_description)} chars (~{len(large_description.encode())//1024}KB)")

# Save back
with open(data_path, "w") as f:
    json.dump(instances, f, indent=2)

print("Updated test dataset with large problem statement for MCP anchoring test")