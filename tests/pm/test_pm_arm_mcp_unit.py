#!/usr/bin/env python3
"""Unit tests for PM Arm (Protobuf + MCP Anchors)."""

import pytest

from agents.pm_arm import PMAgent
from mcp.server import MCPServer
from proto import baseline_pb2


class TestPMArmUnit:
    """Unit tests for PM arm with MCP anchors."""

    def test_small_payload_not_anchored(self):
        """Test that small payloads (<256 bytes) are not anchored."""
        agent = PMAgent()
        
        # Small plan request
        request = baseline_pb2.PlanRequest(
            task_id="test-1",
            repo="test-repo",
            file_path="src/module.py",
            test_name="test_small",
            description="Fix small issue",
            seed=42
        )
        
        response = agent.handle_plan_request(request)
        
        # Approach should be inline (not an MCP reference)
        assert not response.approach.startswith("mcp://")
        assert agent.anchors_created == 0
        assert agent.bytes_saved == 0

    def test_large_payload_anchored(self):
        """Test that large payloads (>256 bytes) are anchored."""
        agent = PMAgent()
        
        # Request that will generate large output
        request = baseline_pb2.CodeRequest(
            task_id="test-2",
            file_path="src/large_module.py",
            plan_steps=["step1", "step2"],
            seed=42
        )
        
        response = agent.handle_code_request(request)
        
        # Patch should be an MCP reference
        assert response.patch.startswith("mcp://")
        assert agent.anchors_created == 1
        assert agent.bytes_saved > 0
        
        # Verify we can resolve the anchor
        resolved = agent.mcp_client.resolve(response.patch)
        assert resolved is not None
        assert b"--- a/" in resolved  # It's a diff

    def test_test_output_anchoring(self):
        """Test that test output is properly anchored."""
        agent = PMAgent()
        
        request = baseline_pb2.TestRequest(
            task_id="test-3",
            test_name="test_feature",
            patch="some patch content",
            seed=42
        )
        
        response = agent.handle_test_request(request)
        
        # Large test output should be anchored
        assert response.output.startswith("mcp://")
        assert response.passed is True
        assert response.duration_ms > 0
        
        # Resolve and verify
        resolved = agent.mcp_client.resolve(response.output)
        assert b"pytest" in resolved
        assert b"passed" in resolved

    def test_bytes_accounting(self):
        """Test that bytes saved are properly accounted."""
        agent = PMAgent()
        
        # Generate multiple large outputs
        for i in range(3):
            request = baseline_pb2.CodeRequest(
                task_id=f"test-{i}",
                file_path=f"src/module_{i}.py",
                plan_steps=["fix"],
                seed=i
            )
            agent.handle_code_request(request)
        
        stats = agent.get_stats()
        assert stats["anchors_created"] == 3
        assert stats["bytes_saved"] > 500  # Should save significant bytes
        
    def test_anchor_ttl_by_type(self):
        """Test that different content types get appropriate TTLs."""
        server = MCPServer()
        agent = PMAgent(mcp_server=server)
        
        # Test output (should get 24h TTL)
        test_req = baseline_pb2.TestRequest(
            task_id="ttl-test",
            test_name="test_ttl",
            patch="patch",
            seed=1
        )
        test_resp = agent.handle_test_request(test_req)
        
        # Code patch (should get 7d TTL)
        code_req = baseline_pb2.CodeRequest(
            task_id="ttl-code",
            file_path="src/file.py",
            plan_steps=["step"],
            seed=2
        )
        code_resp = agent.handle_code_request(code_req)
        
        # Check TTLs in server
        test_entry = server._anchors.get(test_resp.output)
        code_entry = server._anchors.get(code_resp.patch)
        
        assert test_entry.ttl_s == 24 * 3600  # 24 hours for logs
        assert code_entry.ttl_s == 7 * 24 * 3600  # 7 days for diffs

    def test_anchor_deref_performance(self):
        """Test that anchor dereferencing is fast."""
        import time
        
        agent = PMAgent()
        
        # Create an anchor
        request = baseline_pb2.CodeRequest(
            task_id="perf-test",
            file_path="src/perf.py",
            plan_steps=["optimize"],
            seed=99
        )
        response = agent.handle_code_request(request)
        ref = response.patch
        
        # Measure deref time
        times_ms = []
        for _ in range(100):
            start_ns = time.perf_counter_ns()
            data = agent.mcp_client.resolve(ref)
            end_ns = time.perf_counter_ns()
            assert data is not None
            times_ms.append((end_ns - start_ns) / 1e6)
        
        # Check p95 < 50ms (MCP requirement)
        times_ms.sort()
        p95 = times_ms[int(len(times_ms) * 0.95)]
        assert p95 < 50.0, f"Deref p95 {p95:.3f}ms exceeds 50ms limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])