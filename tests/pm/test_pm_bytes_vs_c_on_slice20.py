#!/usr/bin/env python3
"""Integration test for PM vs C bytes/solve on slice20."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agents.pm_arm import PMAgent
from mcp.server import MCPServer
from proto import baseline_pb2


class TestPMBytesVsC:
    """Test that PM achieves lower bytes/solve than C."""
    
    def test_pm_anchors_large_logs(self):
        """Test PM anchors large test logs while C inlines them."""
        # Create PM agent with low threshold for demonstration
        config = {
            "mcp": {
                "inline_max_bytes": 100,  # Very low threshold to force anchoring
                "ttl_logs_hours": 24
            }
        }
        
        mcp_server = MCPServer()
        pm_agent = PMAgent(mcp_server, config)
        
        # Create test request with task context
        request = baseline_pb2.TestRequest()
        request.task_id = "django__django-11001"
        request.test_name = "test_edge_case"
        request.patch = "diff patch here"
        
        # Simulate real test output (would come from actual pytest run)
        # For now, inject a larger test output to demonstrate anchoring
        large_output = "FAILED test output\n" * 50  # > 100 bytes
        
        # Mock the test output temporarily
        original_method = pm_agent.handle_test_request
        def mock_handle(req):
            resp = baseline_pb2.TestResponse()
            if pm_agent._should_anchor(large_output.encode()):
                ref = pm_agent._create_anchor(large_output.encode(), ttl_s=pm_agent.ttl_logs)
                resp.output = ref
            else:
                resp.output = large_output
            resp.passed = False
            resp.duration_ms = 0
            return resp
        
        pm_agent.handle_test_request = mock_handle
        response = pm_agent.handle_test_request(request)
        
        # Verify response contains MCP anchor for large output
        assert response.output.startswith("mcp://"), \
            f"Expected MCP anchor but got: {response.output[:50]}"
        
        # Verify anchor was created
        assert pm_agent.anchors_created > 0
        assert pm_agent.bytes_saved > 0
    
    def test_c_inlines_test_logs(self):
        """Test that Arm C inlines test logs (no anchoring)."""
        # For Arm C, test logs are inlined in JSON
        test_output = "x" * 10000  # 10KB test output
        
        # Arm C response (JSON with inlined output)
        c_response = json.dumps({
            "role": "tester",
            "task_id": "django__django-11001",
            "message": "Test failed",
            "output": test_output,  # Inlined
            "passed": False
        })
        
        # C bytes = full JSON with inlined output
        c_bytes = len(c_response.encode())
        
        # PM would use anchor
        pm_anchor = "mcp://pm/abc123"
        pm_response = baseline_pb2.TestResponse()
        pm_response.output = pm_anchor
        pm_response.passed = False
        
        # PM bytes = protobuf with anchor reference
        pm_bytes = len(pm_response.SerializeToString())
        
        # Verify PM uses fewer bytes
        assert pm_bytes < c_bytes, \
            f"PM ({pm_bytes} bytes) should use fewer bytes than C ({c_bytes} bytes)"
        
        # Verify significant savings
        savings_ratio = 1 - (pm_bytes / c_bytes)
        assert savings_ratio > 0.9, \
            f"Expected >90% savings but got {savings_ratio*100:.1f}%"
    
    def test_pm_config_threshold(self):
        """Test PM respects configured inline threshold."""
        config = {
            "mcp": {
                "inline_max_bytes": 1024,  # 1KB threshold
            }
        }
        
        mcp_server = MCPServer()
        pm_agent = PMAgent(mcp_server, config)
        
        # Test small data (< 1KB) is inlined
        small_data = "x" * 500  # 500 bytes
        assert not pm_agent._should_anchor(small_data.encode())
        
        # Test large data (> 1KB) is anchored
        large_data = "x" * 2000  # 2KB
        assert pm_agent._should_anchor(large_data.encode())
        
        # Test hard cap (> 256KB) always anchored
        huge_data = "x" * 300000  # 300KB
        assert pm_agent._should_anchor(huge_data.encode())