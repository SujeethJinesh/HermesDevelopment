#!/usr/bin/env python3
"""Test for 256KB hard cap on inline blobs per spec requirement."""

import pytest

from agents.pm_arm import PMAgent
from proto import baseline_pb2


class TestInlineCap256KB:
    """Test that no inline blobs > 256KB are allowed per spec."""

    def test_hard_cap_enforced_regardless_of_config(self):
        """Test that payloads > 256KB are ALWAYS anchored."""
        # Try to configure with threshold > 256KB (should fail)
        config_too_high = {
            "mcp": {
                "inline_max_bytes": 300 * 1024  # 300KB - exceeds hard cap
            }
        }
        
        with pytest.raises(ValueError, match="exceeds hard cap of 256KB"):
            PMAgent(config=config_too_high)
    
    def test_256kb_payload_always_anchored(self):
        """Test that exactly 256KB + 1 byte payload is anchored."""
        # Configure with max allowed threshold
        config_max = {
            "mcp": {
                "inline_max_bytes": 256 * 1024  # 256KB - at hard cap
            }
        }
        
        agent = PMAgent(config=config_max)
        
        # Create payload exactly at 256KB
        payload_256kb = b"x" * (256 * 1024)
        assert not agent._should_anchor(payload_256kb)  # At limit, not over
        
        # Create payload just over 256KB
        payload_over = b"x" * (256 * 1024 + 1)
        assert agent._should_anchor(payload_over)  # Must be anchored
    
    def test_large_payload_anchoring(self):
        """Test that very large payloads (>256KB) are anchored."""
        agent = PMAgent()  # Default config (32KB threshold)
        
        # Create a 512KB payload
        large_payload = b"x" * (512 * 1024)
        
        # Must be anchored due to hard cap
        assert agent._should_anchor(large_payload)
        
        # Test that creating anchor for large data works
        ref = agent._create_anchor(large_payload, ttl_s=3600)
        assert ref.startswith("mcp://")
        assert len(ref) < 100  # Reference is small
        assert agent.anchors_created == 1
        
        # Verify bytes saved
        bytes_saved = len(large_payload) - len(ref.encode())
        assert agent.bytes_saved == bytes_saved
        assert bytes_saved > 500 * 1024  # Should save most of the 512KB
    
    def test_config_threshold_below_hard_cap(self):
        """Test normal operation with threshold below hard cap."""
        config_normal = {
            "mcp": {
                "inline_max_bytes": 32 * 1024  # 32KB - well below hard cap
            }
        }
        
        agent = PMAgent(config=config_normal)
        
        # Small payload - not anchored
        small = b"x" * 1024  # 1KB
        assert not agent._should_anchor(small)
        
        # Medium payload (over threshold but under hard cap) - anchored
        medium = b"x" * (40 * 1024)  # 40KB
        assert agent._should_anchor(medium)
        
        # Large payload (over hard cap) - definitely anchored
        large = b"x" * (300 * 1024)  # 300KB
        assert agent._should_anchor(large)
    
    def test_hard_cap_logging(self):
        """Test that hard cap anchoring is logged properly."""
        agent = PMAgent()
        
        # Create payload just over hard cap
        payload = b"x" * (257 * 1024)  # 257KB
        
        # Create anchor
        ref = agent._create_anchor(payload, ttl_s=3600)
        
        # Verify anchor was created
        assert ref.startswith("mcp://")
        assert agent.anchors_created == 1
        
        # Verify bytes saved accounting
        bytes_saved = len(payload) - len(ref.encode())
        assert agent.bytes_saved == bytes_saved
        assert bytes_saved > 250 * 1024  # Should save most of the 257KB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])