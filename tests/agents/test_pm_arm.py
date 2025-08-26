#!/usr/bin/env python3
"""Unit tests for PM agent with protobuf and MCP anchoring."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from proto import baseline_pb2
from agents.pm_arm import PMAgent
from mcp.server import MCPServer


class TestPMAgent(unittest.TestCase):
    """Test PM agent functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mcp_server = Mock(spec=MCPServer)
        self.config = {
            "mcp": {
                "inline_max_bytes": 1024,  # 1KB threshold for testing
                "ttl_logs_hours": 24,
                "ttl_diffs_days": 7,
                "ttl_default_hours": 24,
            }
        }
        self.pm_agent = PMAgent(
            mcp_server=self.mcp_server, 
            config=self.config,
            scratch_dir=Path("/tmp/test_scratch")
        )
        # Mock the MCP client
        self.pm_agent.mcp_client.put = Mock(return_value=(True, "OK"))

    def test_handle_test_request_with_all_fields(self):
        """Test that handle_test_request properly uses all protobuf fields."""
        # Create a test request with all fields
        test_req = baseline_pb2.TestRequest(
            task_id="test-123",
            test_name="test_function",
            patch="--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new",
            seed=42,
            repo="django/django",
            base_commit="abc123def",
            test_patch="test patch content"
        )
        
        # Mock RealTester to return controlled output
        mock_output = "Test output that is less than 1KB\n" * 10  # ~380 bytes
        with patch.object(
            self.pm_agent.real_tester, 
            'run_test_for_instance',
            return_value=(True, mock_output, 1234)
        ) as mock_run:
            
            # Call handle_test_request
            response = self.pm_agent.handle_test_request(test_req)
            
            # Verify RealTester was called with correct instance
            mock_run.assert_called_once()
            instance = mock_run.call_args[0][0]
            
            # Check all fields were passed correctly
            self.assertEqual(instance["repo"], "django/django")
            self.assertEqual(instance["base_commit"], "abc123def")
            self.assertEqual(instance["test_patch"], "test patch content")
            self.assertEqual(instance["patch"], test_req.patch)
            self.assertEqual(instance["FAIL_TO_PASS"], ["test_function"])
            
            # Check response
            self.assertTrue(response.passed)
            self.assertEqual(response.duration_ms, 1234)
            # Output should be inline (< 1KB threshold)
            self.assertEqual(response.output, mock_output)
            
    def test_handle_test_request_with_mcp_anchoring(self):
        """Test that large test output triggers MCP anchoring."""
        test_req = baseline_pb2.TestRequest(
            task_id="test-456",
            test_name="test_large_output",
            patch="",
            seed=42,
            repo="astropy/astropy",
            base_commit="def456abc",
            test_patch=""
        )
        
        # Create output > 1KB to trigger anchoring
        large_output = "A" * 2000  # 2KB of data
        
        with patch.object(
            self.pm_agent.real_tester,
            'run_test_for_instance',
            return_value=(False, large_output, 5678)
        ):
            response = self.pm_agent.handle_test_request(test_req)
            
            # Verify MCP anchor was created
            self.pm_agent.mcp_client.put.assert_called_once()
            call_args = self.pm_agent.mcp_client.put.call_args
            
            # Check the data being anchored
            self.assertEqual(call_args[0][1], large_output.encode())
            
            # Check response contains MCP reference
            self.assertFalse(response.passed)
            self.assertEqual(response.duration_ms, 5678)
            self.assertTrue(response.output.startswith("mcp://"))
            
            # Verify stats were updated
            self.assertEqual(self.pm_agent.anchors_created, 1)
            self.assertGreater(self.pm_agent.bytes_saved, 0)

    def test_handle_test_request_missing_fields(self):
        """Test graceful handling when optional fields are missing."""
        test_req = baseline_pb2.TestRequest(
            task_id="test-789",
            test_name="test_minimal",
            patch="",
            seed=42,
            # repo, base_commit, test_patch are not set
        )
        
        with patch.object(
            self.pm_agent.real_tester,
            'run_test_for_instance',
            return_value=(True, "OK", 100)
        ) as mock_run:
            
            response = self.pm_agent.handle_test_request(test_req)
            
            # Check defaults were used
            instance = mock_run.call_args[0][0]
            self.assertEqual(instance["repo"], "test-repo")  # default
            self.assertEqual(instance["base_commit"], "HEAD")  # default
            self.assertEqual(instance["test_patch"], "")  # default
            
    def test_handle_test_request_pytest_not_available(self):
        """Test handling when pytest is not available."""
        test_req = baseline_pb2.TestRequest(
            task_id="test-error",
            test_name="test_error",
            patch="",
            seed=42,
            repo="test/repo",
            base_commit="main",
        )
        
        with patch.object(
            self.pm_agent.real_tester,
            'run_test_for_instance',
            side_effect=RuntimeError("pytest not found")
        ):
            response = self.pm_agent.handle_test_request(test_req)
            
            # Should handle error gracefully
            self.assertFalse(response.passed)
            self.assertEqual(response.duration_ms, 0)
            self.assertIn("pytest not found", response.output)


if __name__ == "__main__":
    unittest.main()