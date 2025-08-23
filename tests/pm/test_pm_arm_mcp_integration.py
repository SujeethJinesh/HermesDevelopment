#!/usr/bin/env python3
"""Integration tests for PM Arm vs Arm C comparison."""

import time
from typing import Dict, Tuple

import pytest

from agents.pm_arm import PMAgent
from proto import baseline_pb2


class TestPMArmIntegration:
    """Integration tests comparing PM (with MCP) vs Arm C."""

    def _measure_arm_c_bytes(self, request, response) -> int:
        """Measure bytes for Arm C (inline Protobuf)."""
        # In Arm C, everything is inlined in the protobuf
        envelope = baseline_pb2.AgentEnvelope(
            task_id=request.task_id if hasattr(request, 'task_id') else "test",
            role="agent",
            content_type="application/x-protobuf",
            payload=response.SerializeToString(),
            trace_id="trace-123",
            span_id="span-456",
            timestamp_ns=time.time_ns()
        )
        return len(envelope.SerializeToString())
    
    def _measure_arm_pm_bytes(self, request, response) -> Tuple[int, int]:
        """Measure bytes for Arm PM (with MCP anchors).
        
        Returns (wire_bytes, total_with_deref)
        """
        # In PM, large content is replaced with mcp:// refs
        envelope = baseline_pb2.AgentEnvelope(
            task_id=request.task_id if hasattr(request, 'task_id') else "test",
            role="agent", 
            content_type="application/x-protobuf",
            payload=response.SerializeToString(),
            trace_id="trace-123",
            span_id="span-456",
            timestamp_ns=time.time_ns()
        )
        wire_bytes = len(envelope.SerializeToString())
        
        # For fair comparison, add the anchored content size
        # (but it's not sent over wire initially)
        total_bytes = wire_bytes
        for field in ['approach', 'patch', 'output']:
            if hasattr(response, field):
                value = getattr(response, field)
                if value.startswith("mcp://"):
                    # This would require a separate deref, not inline
                    total_bytes += 50  # Approximate deref overhead
                    
        return wire_bytes, total_bytes

    def test_plan_bytes_comparison(self):
        """Test that PM uses fewer bytes than C for planning."""
        pm_agent = PMAgent()
        
        request = baseline_pb2.PlanRequest(
            task_id="bytes-test-1",
            repo="test-repo",
            file_path="src/complex_module.py",
            test_name="test_complex_feature",
            description="Fix complex test failure with multiple edge cases",
            seed=42
        )
        
        # Arm C response (inline)
        c_response = baseline_pb2.PlanResponse()
        c_response.steps.extend([
            f"Step {i}: " + "x" * 50 for i in range(5)
        ])
        c_response.approach = "x" * 500  # Large approach text
        c_response.confidence = 85
        
        # Arm PM response (with anchors)
        pm_response = pm_agent.handle_plan_request(request)
        
        # Measure bytes
        c_bytes = self._measure_arm_c_bytes(request, c_response)
        pm_wire_bytes, pm_total_bytes = self._measure_arm_pm_bytes(request, pm_response)
        
        print(f"\nPlan comparison:")
        print(f"  Arm C: {c_bytes} bytes")
        print(f"  Arm PM (wire): {pm_wire_bytes} bytes")
        print(f"  Arm PM (total): {pm_total_bytes} bytes")
        print(f"  Reduction: {(1 - pm_wire_bytes/c_bytes)*100:.1f}%")
        
        # PM should use fewer bytes on wire
        assert pm_wire_bytes < c_bytes, "PM should use fewer bytes than C"

    def test_code_bytes_comparison(self):
        """Test that PM uses fewer bytes than C for code patches."""
        pm_agent = PMAgent()
        
        request = baseline_pb2.CodeRequest(
            task_id="bytes-test-2",
            file_path="src/large_file.py",
            plan_steps=["refactor", "optimize", "test"],
            seed=42
        )
        
        # Generate responses
        pm_response = pm_agent.handle_code_request(request)
        
        # For C, we'd have the full patch inline
        c_response = baseline_pb2.CodeResponse()
        c_response.patch = "--- a/file\n+++ b/file\n" + "@@ -1,100 +1,150 @@\n" + "x" * 1000
        c_response.files_changed.append("src/large_file.py")
        c_response.lines_added = 50
        c_response.lines_removed = 10
        
        # Measure
        c_bytes = self._measure_arm_c_bytes(request, c_response)
        pm_wire_bytes, pm_total_bytes = self._measure_arm_pm_bytes(request, pm_response)
        
        print(f"\nCode patch comparison:")
        print(f"  Arm C: {c_bytes} bytes")
        print(f"  Arm PM (wire): {pm_wire_bytes} bytes")
        print(f"  Reduction: {(1 - pm_wire_bytes/c_bytes)*100:.1f}%")
        
        assert pm_wire_bytes < c_bytes

    def test_test_output_bytes_comparison(self):
        """Test that PM uses fewer bytes than C for test output."""
        pm_agent = PMAgent()
        
        request = baseline_pb2.TestRequest(
            task_id="bytes-test-3",
            test_name="test_integration",
            patch="some patch",
            seed=42
        )
        
        # Generate responses
        pm_response = pm_agent.handle_test_request(request)
        
        # For C, full output inline
        c_response = baseline_pb2.TestResponse()
        c_response.output = "=" * 50 + "\n" + ("test output line\n" * 100) + "=" * 50
        c_response.passed = True
        c_response.duration_ms = 2340
        
        # Measure
        c_bytes = self._measure_arm_c_bytes(request, c_response)
        pm_wire_bytes, pm_total_bytes = self._measure_arm_pm_bytes(request, pm_response)
        
        print(f"\nTest output comparison:")
        print(f"  Arm C: {c_bytes} bytes")
        print(f"  Arm PM (wire): {pm_wire_bytes} bytes")
        print(f"  Reduction: {(1 - pm_wire_bytes/c_bytes)*100:.1f}%")
        
        assert pm_wire_bytes < c_bytes

    def test_end_to_end_task_bytes(self):
        """Test full task execution bytes (plan->code->test)."""
        pm_agent = PMAgent()
        total_c_bytes = 0
        total_pm_wire_bytes = 0
        
        # Step 1: Planning
        plan_req = baseline_pb2.PlanRequest(
            task_id="e2e-test",
            repo="test-repo",
            file_path="src/module.py",
            test_name="test_feature",
            description="Fix the test",
            seed=42
        )
        plan_resp = pm_agent.handle_plan_request(plan_req)
        
        # Simulate Arm C 
        c_plan_resp = baseline_pb2.PlanResponse()
        c_plan_resp.steps.extend(["step1", "step2", "step3"])
        c_plan_resp.approach = "x" * 500
        c_plan_resp.confidence = 85
        
        c_bytes = self._measure_arm_c_bytes(plan_req, c_plan_resp)
        pm_bytes, _ = self._measure_arm_pm_bytes(plan_req, plan_resp)
        total_c_bytes += c_bytes
        total_pm_wire_bytes += pm_bytes
        
        # Step 2: Coding
        code_req = baseline_pb2.CodeRequest(
            task_id="e2e-test",
            file_path="src/module.py",
            plan_steps=["step1", "step2"],
            seed=42
        )
        code_resp = pm_agent.handle_code_request(code_req)
        
        c_code_resp = baseline_pb2.CodeResponse()
        c_code_resp.patch = "x" * 800
        c_code_resp.files_changed.append("src/module.py")
        
        c_bytes = self._measure_arm_c_bytes(code_req, c_code_resp)
        pm_bytes, _ = self._measure_arm_pm_bytes(code_req, code_resp)
        total_c_bytes += c_bytes
        total_pm_wire_bytes += pm_bytes
        
        # Step 3: Testing
        test_req = baseline_pb2.TestRequest(
            task_id="e2e-test",
            test_name="test_feature",
            patch=code_resp.patch,  # PM sends ref, C sends full
            seed=42
        )
        test_resp = pm_agent.handle_test_request(test_req)
        
        c_test_resp = baseline_pb2.TestResponse()
        c_test_resp.output = "y" * 2000
        c_test_resp.passed = True
        
        c_bytes = self._measure_arm_c_bytes(test_req, c_test_resp)
        pm_bytes, _ = self._measure_arm_pm_bytes(test_req, test_resp)
        total_c_bytes += c_bytes
        total_pm_wire_bytes += pm_bytes
        
        # Summary
        print(f"\nEnd-to-end bytes comparison:")
        print(f"  Arm C total: {total_c_bytes} bytes")
        print(f"  Arm PM total: {total_pm_wire_bytes} bytes")
        print(f"  Reduction: {(1 - total_pm_wire_bytes/total_c_bytes)*100:.1f}%")
        print(f"  Anchors created: {pm_agent.anchors_created}")
        print(f"  Bytes saved: {pm_agent.bytes_saved}")
        
        # PM should achieve significant reduction
        assert total_pm_wire_bytes < total_c_bytes
        reduction = (1 - total_pm_wire_bytes/total_c_bytes)
        assert reduction > 0.20, f"Should achieve >20% reduction, got {reduction*100:.1f}%"

    def test_deref_latency_requirement(self):
        """Test that MCP deref meets p95 < 50ms requirement."""
        import statistics
        
        pm_agent = PMAgent()
        
        # Create several anchors
        refs = []
        for i in range(10):
            req = baseline_pb2.CodeRequest(
                task_id=f"latency-{i}",
                file_path=f"src/file_{i}.py",
                plan_steps=["fix"],
                seed=i
            )
            resp = pm_agent.handle_code_request(req)
            refs.append(resp.patch)
        
        # Measure deref latency
        times_ms = []
        for _ in range(100):
            for ref in refs:
                start_ns = time.perf_counter_ns()
                data = pm_agent.mcp_client.resolve(ref)
                end_ns = time.perf_counter_ns()
                assert data is not None
                times_ms.append((end_ns - start_ns) / 1e6)
        
        # Calculate p95
        times_ms.sort()
        p50 = times_ms[int(len(times_ms) * 0.50)]
        p95 = times_ms[int(len(times_ms) * 0.95)]
        mean = statistics.mean(times_ms)
        
        print(f"\nMCP deref latency:")
        print(f"  Samples: {len(times_ms)}")
        print(f"  P50: {p50:.3f}ms")
        print(f"  P95: {p95:.3f}ms")
        print(f"  Mean: {mean:.3f}ms")
        
        assert p95 < 50.0, f"Deref p95 {p95:.3f}ms exceeds 50ms requirement"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])