#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug script to test PM patch resolution."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.pm_arm import PMAgent
from mcp.server import MCPServer

def test_patch_flow():
    """Test the patch flow with PM agent."""
    
    # Create PM agent with MCP
    mcp_server = MCPServer()
    config = {
        "mcp": {
            "inline_max_bytes": 1024,  # 1KB threshold to force anchoring
            "ttl_logs_hours": 24,
            "ttl_diffs_days": 7,
            "ttl_default_hours": 24
        }
    }
    
    pm_agent = PMAgent(mcp_server, config, scratch_dir=Path("/tmp/hermes_debug"))
    
    # Simulate a code request with patch
    from proto import baseline_pb2
    
    code_req = baseline_pb2.CodeRequest(
        task_id="test_task",
        file_path="test.py",
        plan_steps=["Fix the bug", "Apply patch"],
        seed=123
    )
    
    print("=" * 60)
    print("1. Generating patch with PM agent...")
    code_resp = pm_agent.handle_code_request(code_req)
    
    print("   Patch type: {}".format(type(code_resp.patch)))
    print("   Patch content: {}...".format(code_resp.patch[:100]))
    print("   Is MCP ref: {}".format(code_resp.patch.startswith('mcp://')))
    print("   Anchors created: {}".format(pm_agent.anchors_created))
    print("   Bytes saved: {}".format(pm_agent.bytes_saved))
    
    # Now test if real_tester can resolve it
    print("\n" + "=" * 60)
    print("2. Testing patch resolution in real_tester...")
    
    if code_resp.patch.startswith("mcp://"):
        print("   Resolving MCP ref: {}".format(code_resp.patch))
        try:
            patch_bytes = pm_agent.real_tester._load_patch_bytes(code_resp.patch)
            print("   ✓ Resolved to {} bytes".format(len(patch_bytes)))
            print("   First 100 chars: {}".format(patch_bytes[:100]))
        except Exception as e:
            print("   ✗ Failed to resolve: {}".format(e))
    else:
        print("   Patch was inlined, no MCP resolution needed")
    
    # Test with a larger patch to trigger anchoring
    print("\n" + "=" * 60)
    print("3. Testing with larger patch to trigger anchoring...")
    
    # Create a large patch (>1KB)
    large_patch = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,100 +1,150 @@
""" + ("+" * 50 + " # Large addition line\n") * 30  # Make it >1KB
    
    # Manually test anchoring
    patch_bytes = large_patch.encode('utf-8')
    print("   Large patch size: {} bytes".format(len(patch_bytes)))
    
    wire_repr, on_wire_bytes = pm_agent.anchor_if_beneficial(patch_bytes, "patches", 3600)
    print("   Wire representation: {}...".format(wire_repr[:100]))
    print("   On-wire bytes: {}".format(on_wire_bytes))
    print("   Is MCP ref: {}".format(wire_repr.startswith('mcp://')))
    
    if wire_repr.startswith("mcp://"):
        print("\n   Testing resolution of manual anchor...")
        try:
            resolved = pm_agent.mcp_client.resolve(wire_repr)
            if isinstance(resolved, tuple):
                ok, data = resolved
                if ok:
                    print("   ✓ Resolved to {} bytes".format(len(data)))
                else:
                    print("   ✗ Resolution failed")
            else:
                print("   ✓ Resolved to {} bytes".format(len(resolved)))
        except Exception as e:
            print("   ✗ Failed to resolve: {}".format(e))
    
    print("\n" + "=" * 60)
    print("Final stats:")
    print("  Total anchors created: {}".format(pm_agent.anchors_created))
    print("  Total bytes saved: {}".format(pm_agent.bytes_saved))
    
    # Check MCP server stats
    stats = mcp_server.get_stats()
    print("  MCP server entries: {}".format(stats['total_entries']))
    print("  MCP server bytes: {}".format(stats['total_bytes']))

if __name__ == "__main__":
    test_patch_flow()