#!/usr/bin/env python3
"""Test a single PM task end-to-end to debug failures."""

import sys
import yaml
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import MCPServer
from agents.pm_arm import PMAgent
from proto import baseline_pb2


def test_single_task():
    """Test a single task through PM arm to see why tests fail."""
    
    # Load config
    with open("configs/generation.yaml") as f:
        config = yaml.safe_load(f)
    
    mcp_server = MCPServer()
    pm_agent = PMAgent(mcp_server=mcp_server, config=config)
    
    # 1. Plan
    plan_req = baseline_pb2.PlanRequest(
        task_id="test-123",
        repo="test/repo",
        file_path="test.py",
        test_name="test_function",
        description="Fix the test function"
    )
    
    print("1. Planning...")
    plan_resp = pm_agent.handle_plan_request(plan_req)
    print(f"   Steps: {len(plan_resp.steps)}")
    
    # 2. Code
    code_req = baseline_pb2.CodeRequest(
        task_id="test-123",
        file_path="test.py",
        plan_steps=plan_resp.steps[:2]
    )
    
    print("\n2. Coding...")
    code_resp = pm_agent.handle_code_request(code_req)
    is_mcp = code_resp.patch.startswith("mcp://")
    print(f"   Patch is MCP ref: {is_mcp}")
    
    if is_mcp:
        # Resolve the patch to see what it contains
        mcp_ref = code_resp.patch
        resolved_patch = pm_agent.mcp_client.resolve(mcp_ref)
        print(f"   MCP ref: {mcp_ref}")
        print(f"   Resolved to {len(resolved_patch)} bytes")
        print(f"   Patch content:\n{resolved_patch.decode()[:500]}")
    else:
        print(f"   Patch content:\n{code_resp.patch[:500]}")
    
    # 3. Test (with a real test file)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("""def hello():
    return "world"

def test_function():
    assert hello() == "hello world"
""")
        
        test_req = baseline_pb2.TestRequest(
            task_id="test-123",
            test_name="test_function",
            patch=code_resp.patch,  # Pass the MCP ref
            repo="test/repo",
            base_commit="HEAD"
        )
        
        print("\n3. Testing...")
        print(f"   Patch being tested: {test_req.patch[:100]}")
        
        # The test should resolve the MCP ref internally
        test_resp = pm_agent.handle_test_request(test_req)
        
        print(f"   Test passed: {test_resp.passed}")
        
        # Check if output is MCP ref
        if test_resp.output.startswith("mcp://"):
            print(f"   Output is MCP ref: {test_resp.output}")
            resolved_output = pm_agent.mcp_client.resolve(test_resp.output)
            if resolved_output:
                print(f"   Output (resolved):\n{resolved_output.decode()[:500]}")
        else:
            print(f"   Output:\n{test_resp.output[:500]}")
    
    print(f"\n4. Stats:")
    stats = pm_agent.get_stats()
    print(f"   Anchors created: {stats['anchors_created']}")
    print(f"   Bytes saved: {stats['bytes_saved']}")


if __name__ == "__main__":
    test_single_task()