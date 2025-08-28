#!/usr/bin/env python3
"""Trace a single PM task to see what's happening."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable debug output
os.environ["HERMES_DEBUG"] = "1"

from mcp.server import MCPServer
from agents.pm_arm import PMAgent
from proto import baseline_pb2


def trace_pm_task():
    """Trace a single task through PM arm."""
    
    # Create PM agent
    mcp_server = MCPServer()
    pm_agent = PMAgent(mcp_server=mcp_server)
    
    # 1. Plan request
    plan_req = baseline_pb2.PlanRequest(
        task_id="test-123",
        repo="astropy/astropy",
        file_path="astropy/units/quantity.py",
        test_name="test_quantity",
        description="Fix quantity conversion"
    )
    
    print("=" * 60)
    print("1. PLAN PHASE")
    print("=" * 60)
    plan_resp = pm_agent.handle_plan_request(plan_req)
    print(f"Steps: {len(plan_resp.steps)}")
    print(f"Approach MCP ref: {'mcp://' in plan_resp.approach}")
    
    # 2. Code request
    code_req = baseline_pb2.CodeRequest(
        task_id="test-123",
        file_path="astropy/units/quantity.py",
        plan_steps=plan_resp.steps[:2]  # Just use first 2 steps
    )
    
    print("\n" + "=" * 60)
    print("2. CODE PHASE")
    print("=" * 60)
    code_resp = pm_agent.handle_code_request(code_req)
    print(f"Patch is MCP ref: {'mcp://' in code_resp.patch}")
    print(f"Patch value: {code_resp.patch[:100] if code_resp.patch else 'None'}")
    print(f"Files changed: {code_resp.files_changed}")
    
    # 3. Test request with MCP patch
    test_req = baseline_pb2.TestRequest(
        task_id="test-123",
        test_name="test_quantity",
        patch=code_resp.patch,  # This should be an MCP reference
        repo="astropy/astropy",
        base_commit="HEAD"
    )
    
    print("\n" + "=" * 60)
    print("3. TEST PHASE")
    print("=" * 60)
    print(f"Patch being tested: {test_req.patch[:100] if test_req.patch else 'None'}")
    
    # Check if MCP resolution works
    if test_req.patch.startswith("mcp://"):
        print(f"Resolving MCP ref: {test_req.patch}")
        resolved_data = pm_agent.mcp_client.resolve(test_req.patch)
        if resolved_data:
            print(f"  Resolved to {len(resolved_data)} bytes")
            print(f"  Content preview: {resolved_data[:200].decode('utf-8', errors='replace')}")
        else:
            print("  ERROR: Failed to resolve MCP reference!")
    
    # Run the test
    test_resp = pm_agent.handle_test_request(test_req)
    print(f"Test passed: {test_resp.passed}")
    print(f"Output is MCP ref: {'mcp://' in test_resp.output}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("4. STATS")
    print("=" * 60)
    stats = pm_agent.get_stats()
    print(f"Anchors created: {stats['anchors_created']}")
    print(f"Bytes saved: {stats['bytes_saved']}")


if __name__ == "__main__":
    trace_pm_task()