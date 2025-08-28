#!/usr/bin/env python3
"""Trace what patches PM generates to see their sizes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import MCPServer
from agents.pm_arm import PMAgent
from proto import baseline_pb2
import yaml


def trace_patches():
    """Generate patches and check their sizes."""
    
    # Load config
    with open("configs/generation.yaml") as f:
        config = yaml.safe_load(f)
    
    mcp_server = MCPServer()
    pm_agent = PMAgent(mcp_server=mcp_server, config=config)
    
    # Test with different configurations
    test_cases = [
        {
            "name": "Small task",
            "file_path": "test.py",
            "description": "Fix bug",
            "plan_steps": ["Fix the bug"]
        },
        {
            "name": "Normal task",  
            "file_path": "src/utils/helper.py",
            "description": "Fix the helper function to handle edge cases properly",
            "plan_steps": [
                "Analyze the failing test",
                "Identify edge case issue",
                "Implement proper validation",
                "Add error handling",
                "Test the fix"
            ]
        }
    ]
    
    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['name']}")
        print(f"{'='*60}")
        
        # Generate code
        code_req = baseline_pb2.CodeRequest(
            task_id="test-123",
            file_path=tc["file_path"],
            plan_steps=tc["plan_steps"]
        )
        
        code_resp = pm_agent.handle_code_request(code_req)
        
        patch = code_resp.patch
        is_mcp = patch.startswith("mcp://")
        patch_size = len(patch.encode()) if not is_mcp else 0
        
        print(f"Patch is MCP ref: {is_mcp}")
        print(f"Patch size: {patch_size} bytes")
        print(f"Threshold: {pm_agent.inline_max_bytes} bytes")
        
        if not is_mcp:
            print(f"Patch preview:\n{patch[:300]}")
            
            # Check why it wasn't anchored
            if patch_size < pm_agent.inline_max_bytes:
                print(f"→ Not anchored: {patch_size} < {pm_agent.inline_max_bytes} threshold")
            else:
                print(f"→ Should have been anchored! {patch_size} >= {pm_agent.inline_max_bytes}")
    
    print(f"\n{'='*60}")
    print("Stats:")
    print(f"  Anchors created: {pm_agent.anchors_created}")
    print(f"  Bytes saved: {pm_agent.bytes_saved}")


if __name__ == "__main__":
    trace_patches()