#!/usr/bin/env python3
"""Test that MCP references are resolved before applying patches."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from agents.real_tester import RealTester


def test_apply_patch_with_mcp_ref(tmp_path):
    """Test that MCP refs are resolved to actual patch content."""
    
    # Create mock MCP client
    mcp_client = Mock()
    patch_content = b"""--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-print('old')
+print('new')
"""
    
    # Mock resolve to return patch content (MCPClient.resolve returns just data)
    mcp_client.resolve.return_value = patch_content
    
    # Create RealTester with MCP client
    tester = RealTester(scratch_dir=tmp_path, mcp_client=mcp_client)
    
    # Test resolving MCP reference
    mcp_ref = "mcp://pm/abc123"
    resolved = tester._load_patch_bytes(mcp_ref)
    
    # Verify MCP client was called
    mcp_client.resolve.assert_called_once_with(mcp_ref)
    
    # Verify correct content returned
    assert resolved == patch_content


def test_apply_patch_with_regular_content(tmp_path):
    """Test that regular patch content passes through unchanged."""
    
    # Create RealTester without MCP client
    tester = RealTester(scratch_dir=tmp_path, mcp_client=None)
    
    # Test with regular patch string
    patch_text = """--- a/bar.py
+++ b/bar.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
    
    # Load patch bytes
    resolved = tester._load_patch_bytes(patch_text)
    
    # Verify unchanged (just encoded)
    assert resolved == patch_text.encode("utf-8")


def test_apply_patch_mcp_ref_without_client_fails(tmp_path):
    """Test that MCP ref without client raises error."""
    
    # Create RealTester without MCP client
    tester = RealTester(scratch_dir=tmp_path, mcp_client=None)
    
    # Try to resolve MCP ref without client
    mcp_ref = "mcp://pm/xyz789"
    
    with pytest.raises(RuntimeError, match="no MCP client configured"):
        tester._load_patch_bytes(mcp_ref)


def test_apply_patch_mcp_resolve_failure(tmp_path):
    """Test that failed MCP resolution raises error."""
    
    # Create mock MCP client that fails
    mcp_client = Mock()
    mcp_client.resolve.return_value = None  # MCPClient.resolve returns None on failure
    
    # Create RealTester with failing MCP client
    tester = RealTester(scratch_dir=tmp_path, mcp_client=mcp_client)
    
    # Try to resolve MCP ref
    mcp_ref = "mcp://pm/bad_ref"
    
    with pytest.raises(RuntimeError, match="Failed to resolve MCP ref"):
        tester._load_patch_bytes(mcp_ref)


def test_full_patch_application_with_mcp(tmp_path):
    """Test end-to-end patch application with MCP resolution."""
    
    # Create a fake repo
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Create a file to patch
    test_file = repo_dir / "test.py"
    test_file.write_text("print('original')\n")
    
    # Create mock MCP client
    mcp_client = Mock()
    patch_content = b"""--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-print('original')
+print('modified')
"""
    mcp_client.resolve.return_value = patch_content  # MCPClient.resolve returns just data
    
    # Create RealTester with MCP client
    tester = RealTester(scratch_dir=tmp_path, mcp_client=mcp_client)
    
    # Apply patch via MCP ref
    mcp_ref = "mcp://pm/patch123"
    
    # Mock git command (since we may not have git in test env)
    import subprocess
    original_run = subprocess.run
    
    def mock_run(cmd, *args, **kwargs):
        if cmd[0] == "git" and cmd[1] == "apply":
            # Simulate successful patch application
            return Mock(returncode=0, stderr="")
        return original_run(cmd, *args, **kwargs)
    
    subprocess.run = mock_run
    try:
        tester._apply_patch(repo_dir, mcp_ref)
        
        # Verify MCP client was called
        mcp_client.resolve.assert_called_once_with(mcp_ref)
    finally:
        subprocess.run = original_run