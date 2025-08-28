#!/usr/bin/env python3
"""Debug script to test MCP resolution in PM arm."""

import sys
import hashlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import MCPServer
from mcp.client import MCPClient
from agents.pm_arm import PMAgent
from proto import baseline_pb2


def test_mcp_resolution():
    """Test that MCP references are properly resolved."""
    print("Testing MCP resolution in PM arm...")
    
    # Create PM agent
    mcp_server = MCPServer()
    pm_agent = PMAgent(mcp_server=mcp_server)
    
    # Test patch content
    patch_content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,5 +1,6 @@
 def hello():
-    return "world"
+    # Fixed implementation
+    return "hello world"
 
 def test_hello():
     assert hello() == "hello world"
"""
    
    # 1. Create a code request that should generate an MCP anchor
    code_req = baseline_pb2.CodeRequest(
        task_id="test-123",
        file_path="test.py",
        plan_steps=["Fix hello function"]
    )
    
    # Set up PM to create anchor by using a large patch
    large_patch = patch_content * 100  # Make it large enough to trigger anchoring
    
    # Manually create an anchor for testing
    sha256_hash = hashlib.sha256(large_patch.encode()).hexdigest()[:16]
    ref = f"mcp://pm/{sha256_hash}"
    success = mcp_server.put(ref, large_patch.encode(), ttl_s=3600)
    print(f"Created MCP anchor: {ref} (success={success})")
    
    # 2. Test if the anchor can be resolved
    resolved_data = pm_agent.mcp_client.resolve(ref)
    print(f"Resolution test: success={resolved_data is not None}, size={len(resolved_data) if resolved_data else 0}")
    
    if resolved_data:
        print(f"Resolved content matches: {resolved_data.decode()[:100] == large_patch[:100]}")
    
    # 3. Test the real tester's _load_patch_bytes method
    from agents.real_tester import RealTester
    tester = RealTester(mcp_client=pm_agent.mcp_client)
    
    try:
        loaded_bytes = tester._load_patch_bytes(ref)
        print(f"RealTester._load_patch_bytes: success! size={len(loaded_bytes)}")
        print(f"Content matches: {loaded_bytes.decode()[:100] == large_patch[:100]}")
    except Exception as e:
        print(f"RealTester._load_patch_bytes failed: {e}")
    
    # 4. Test full flow with TestRequest containing MCP ref
    test_req = baseline_pb2.TestRequest(
        task_id="test-123",
        test_name="test_hello",
        patch=ref,  # Pass MCP reference
        repo="test-repo",
        base_commit="HEAD"
    )
    
    print(f"\nTesting full flow with patch={ref}")
    try:
        # This should resolve the MCP reference and apply the patch
        test_resp = pm_agent.handle_test_request(test_req)
        print(f"Test response: passed={test_resp.passed}")
        print(f"Test output: {test_resp.output[:200] if test_resp.output else 'No output'}")
    except Exception as e:
        print(f"Full flow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_mcp_resolution()