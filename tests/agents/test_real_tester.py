"""Tests for RealTester agent."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from agents.pm_arm import PMAnchorManager, PMMetrics
from agents.real_tester import RealTester


class TestRealTester:
    """Test cases for RealTester agent."""

    def test_load_patch_bytes_inline_ref_strict(self):
        """Test strict MCP resolution - missing ref raises ValueError."""
        # Setup mock MCP client
        mock_mcp = MagicMock()
        mock_mcp.resolve_bytes.return_value = b""  # Empty/missing
        
        tester = RealTester(mock_mcp)
        
        # Test 1: Bytes input returns as-is
        patch_bytes = b"diff --git a/test.py b/test.py\n+fix"
        result = tester._load_patch_bytes(patch_bytes)
        assert result == patch_bytes
        
        # Test 2: String input gets encoded
        patch_str = "diff --git a/test.py b/test.py\n+fix"
        result = tester._load_patch_bytes(patch_str)
        assert result == patch_str.encode("utf-8")
        
        # Test 3: MCP ref with data succeeds
        mock_mcp.resolve_bytes.return_value = b"patch content"
        result = tester._load_patch_bytes("mcp://patches/abc123")
        assert result == b"patch content"
        mock_mcp.resolve_bytes.assert_called_with("mcp://patches/abc123")
        
        # Test 4: MCP ref missing/empty raises ValueError
        mock_mcp.resolve_bytes.return_value = b""
        with pytest.raises(ValueError, match="missing or empty"):
            tester._load_patch_bytes("mcp://patches/missing")

    def test_apply_patch_sequence(self):
        """Test git apply with fallback sequence."""
        # Create a temporary git repository
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo_path,
                check=True
            )
            
            # Create initial file
            test_file = repo_path / "test.py"
            test_file.write_text("def hello():\n    return 'world'\n")
            subprocess.run(["git", "add", "test.py"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=repo_path,
                check=True
            )
            
            # Create a valid patch
            patch_content = b"""diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 def hello():
-    return 'world'
+    # Fixed version
+    return 'hello world'
"""
            
            # Test patch application
            mock_mcp = MagicMock()
            tester = RealTester(mock_mcp)
            
            # Should succeed with one of the strategies
            tester.apply_patch(repo_path, patch_content)
            
            # Verify patch was applied
            updated_content = test_file.read_text()
            assert "hello world" in updated_content
            assert "# Fixed version" in updated_content
            
            # Verify .hermes.patch was created
            patch_file = repo_path / ".hermes.patch"
            assert patch_file.exists()
            assert patch_file.read_bytes() == patch_content

    def test_apply_patch_failure(self):
        """Test that apply_patch raises on all failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            
            # Create an invalid patch that will fail
            bad_patch = b"This is not a valid patch format\n"
            
            mock_mcp = MagicMock()
            tester = RealTester(mock_mcp)
            
            # Should raise RuntimeError after trying all strategies
            with pytest.raises(RuntimeError, match="Failed to apply patch"):
                tester.apply_patch(repo_path, bad_patch)

    def test_pytest_log_anchoring_small_vs_large(self):
        """Test that small logs stay inline, large logs get anchored."""
        # Setup mock MCP client
        mock_mcp = MagicMock()
        
        # Create PM manager with metrics
        pm_metrics = PMMetrics()
        pm_manager = PMAnchorManager(mock_mcp, pm_metrics)
        
        tester = RealTester(mock_mcp, pm_manager)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create a dummy test file
            test_file = repo_path / "test_example.py"
            test_file.write_text("""
def test_pass():
    assert True

def test_output():
    print("Test output")
    assert True
""")
            
            # Mock subprocess.run for pytest execution
            with patch("subprocess.run") as mock_run:
                # Test 1: Small output (< 1KB) stays inline
                small_output = b"." * 100  # 100 bytes
                mock_run.return_value = Mock(
                    stdout=small_output,
                    stderr=b"",
                    returncode=0
                )
                
                payload, anchored, code = tester.run_pytest_and_anchor_logs(repo_path)
                
                # Small log should stay inline
                assert anchored is False
                assert payload == small_output + b"\n"
                assert pm_metrics.inline_count == 1
                assert pm_metrics.anchor_count == 0
                
                # Test 2: Large output (> 1KB) gets anchored
                large_output = b"X" * 2000  # 2KB
                mock_run.return_value = Mock(
                    stdout=large_output,
                    stderr=b"Error" * 100,
                    returncode=1
                )
                
                payload, anchored, code = tester.run_pytest_and_anchor_logs(repo_path)
                
                # Large log should be anchored
                assert anchored is True
                assert isinstance(payload, str)
                assert payload.startswith("mcp://logs/")
                assert pm_metrics.anchor_count == 1
                assert pm_metrics.anchors_created == 1
                assert pm_metrics.bytes_saved > 0
                
                # Verify MCP client was called
                mock_mcp.put_if_absent.assert_called_once()
                call_args = mock_mcp.put_if_absent.call_args
                assert call_args[0][0].startswith("mcp://logs/")
                assert len(call_args[0][1]) > 1024  # Large data
                assert call_args[1]["ttl_s"] == 24 * 3600  # 24 hours for logs

    def test_test_repository_integration(self):
        """Test the full test_repository workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo_path,
                check=True
            )
            
            # Create initial file
            code_file = repo_path / "code.py"
            code_file.write_text("def add(a, b):\n    return a + b\n")
            
            test_file = repo_path / "test_code.py"
            test_file.write_text("""
from code import add

def test_add():
    assert add(2, 3) == 5
""")
            
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial"],
                cwd=repo_path,
                check=True
            )
            
            # Create a patch that breaks the test
            patch = """diff --git a/code.py b/code.py
index abc123..def456 100644
--- a/code.py
+++ b/code.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a - b  # Bug introduced
"""
            
            # Setup tester
            mock_mcp = MagicMock()
            tester = RealTester(mock_mcp)
            
            # Run test_repository
            result = tester.test_repository(repo_path, patch)
            
            # Tests should fail due to the bug
            assert result["success"] is False
            assert result["return_code"] != 0
            assert "logs" in result
            assert "logs_anchored" in result
            assert "pm_metrics" in result
            
            # Verify metrics were collected
            metrics = result["pm_metrics"]
            assert "anchors_created" in metrics
            assert "bytes_saved" in metrics
            assert "inline_count" in metrics
            assert "anchor_count" in metrics

    def test_test_repository_with_mcp_ref(self):
        """Test test_repository with MCP reference input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repo_path,
                check=True
            )
            
            # Create initial file
            test_file = repo_path / "test.py"
            test_file.write_text("x = 1\n")
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial"],
                cwd=repo_path,
                check=True
            )
            
            # Setup mock MCP that returns patch data
            patch_data = b"""diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
            mock_mcp = MagicMock()
            mock_mcp.resolve_bytes.return_value = patch_data
            
            tester = RealTester(mock_mcp)
            
            # Use MCP reference
            result = tester.test_repository(repo_path, "mcp://patches/test123")
            
            # Verify MCP was resolved
            mock_mcp.resolve_bytes.assert_called_with("mcp://patches/test123")
            
            # Verify patch was applied
            assert test_file.read_text() == "x = 2\n"