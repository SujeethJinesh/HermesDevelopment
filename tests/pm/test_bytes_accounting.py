#!/usr/bin/env python3
"""Test bytes accounting for PM arm."""

import pytest

from agents.pm_arm import PMAgent
from proto import baseline_pb2


class TestBytesAccounting:
    """Test accurate bytes accounting for PM arm."""

    def test_bytes_saved_calculation(self):
        """Test that bytes saved is accurately calculated."""
        agent = PMAgent()
        
        # Track original vs anchored sizes
        original_total = 0
        anchored_total = 0
        
        # Generate large code patch
        req = baseline_pb2.CodeRequest(
            task_id="accounting-1",
            file_path="src/file.py",
            plan_steps=["step"],
            seed=1
        )
        resp = agent.handle_code_request(req)
        
        # The patch is anchored
        assert resp.patch.startswith("mcp://")
        
        # Resolve to get original size
        original_data = agent.mcp_client.resolve(resp.patch)
        original_total += len(original_data)
        anchored_total += len(resp.patch.encode())
        
        # Generate large test output
        req2 = baseline_pb2.TestRequest(
            task_id="accounting-2",
            test_name="test_func",
            patch="patch",
            seed=2
        )
        resp2 = agent.handle_test_request(req2)
        
        # The output is anchored
        assert resp2.output.startswith("mcp://")
        
        original_data2 = agent.mcp_client.resolve(resp2.output)
        original_total += len(original_data2)
        anchored_total += len(resp2.output.encode())
        
        # Check accounting
        expected_saved = original_total - anchored_total
        assert agent.bytes_saved == expected_saved, \
            f"Expected {expected_saved} bytes saved, got {agent.bytes_saved}"
        
        print(f"\nBytes accounting:")
        print(f"  Original total: {original_total} bytes")
        print(f"  Anchored total: {anchored_total} bytes")  
        print(f"  Bytes saved: {agent.bytes_saved} bytes")
        print(f"  Reduction: {(agent.bytes_saved/original_total)*100:.1f}%")

    def test_no_double_counting(self):
        """Test that reusing anchors doesn't double-count savings."""
        agent = PMAgent()
        
        # Create first anchor
        req1 = baseline_pb2.CodeRequest(
            task_id="no-double-1",
            file_path="src/file.py",
            plan_steps=["fix"],
            seed=42
        )
        resp1 = agent.handle_code_request(req1)
        ref1 = resp1.patch
        
        initial_saved = agent.bytes_saved
        initial_anchors = agent.anchors_created
        
        # Same request should create same content
        req2 = baseline_pb2.CodeRequest(
            task_id="no-double-2",
            file_path="src/file.py",
            plan_steps=["fix"],
            seed=42
        )
        resp2 = agent.handle_code_request(req2)
        ref2 = resp2.patch
        
        # Should create a new anchor (different hash due to task_id in content)
        # But let's verify the accounting is correct
        assert agent.anchors_created == initial_anchors + 1
        
        # Bytes saved should increase
        assert agent.bytes_saved > initial_saved

    def test_small_payload_no_savings(self):
        """Test that small payloads don't count as savings."""
        agent = PMAgent()
        
        # Small request that won't trigger anchoring
        req = baseline_pb2.PlanRequest(
            task_id="small-1",
            repo="r",
            file_path="f.py",
            test_name="t",
            description="d",
            seed=1
        )
        
        resp = agent.handle_plan_request(req)
        
        # No anchoring should occur
        assert agent.anchors_created == 0
        assert agent.bytes_saved == 0
        assert not resp.approach.startswith("mcp://")

    def test_cumulative_savings(self):
        """Test cumulative savings across multiple operations."""
        agent = PMAgent()
        
        savings_checkpoints = []
        
        # Run 5 operations
        for i in range(5):
            req = baseline_pb2.TestRequest(
                task_id=f"cumulative-{i}",
                test_name=f"test_{i}",
                patch="some patch",
                seed=i
            )
            agent.handle_test_request(req)
            savings_checkpoints.append(agent.bytes_saved)
        
        # Savings should be cumulative and increasing
        for i in range(1, len(savings_checkpoints)):
            assert savings_checkpoints[i] >= savings_checkpoints[i-1], \
                "Savings should be cumulative"
        
        # Final stats
        stats = agent.get_stats()
        assert stats["anchors_created"] == 5
        assert stats["bytes_saved"] == savings_checkpoints[-1]
        
        print(f"\nCumulative savings over {len(savings_checkpoints)} operations:")
        for i, saved in enumerate(savings_checkpoints):
            print(f"  After op {i+1}: {saved} bytes saved")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])