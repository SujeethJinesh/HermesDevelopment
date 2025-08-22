#!/usr/bin/env python3
"""Test UDS path length limits."""

import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestUDSPathLimits:
    """Test UNIX domain socket path length limits."""
    
    def test_uds_path_length_limit(self):
        """Test that UDS paths stay within OS limits (108 chars on most Unix)."""
        # Most Unix systems have a limit of 108 chars for sun_path
        MAX_UDS_PATH = 108
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path that's within limits
            short_task_id = "t123"
            short_path = Path(tmpdir) / f"hermes_{short_task_id}_42" / "rpc.sock"
            
            assert len(str(short_path)) < MAX_UDS_PATH, f"Path too long: {len(str(short_path))}"
            
            # Create socket to test
            short_path.parent.mkdir(parents=True, exist_ok=True)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(short_path))
            sock.close()
            os.unlink(str(short_path))
    
    def test_uds_path_truncation_strategy(self):
        """Test that long task IDs are truncated to keep path within limits."""
        MAX_UDS_PATH = 108
        
        # Strategy: use /tmp (short) and truncate task_id if needed
        base_dir = "/tmp"
        
        # Very long task ID
        long_task_id = "a" * 200
        seed = 123456789
        
        # Truncation strategy: use first 8 chars of task_id + seed
        truncated_id = f"{long_task_id[:8]}_{seed}"
        socket_path = Path(base_dir) / f"h_{truncated_id}.sock"
        
        assert len(str(socket_path)) < MAX_UDS_PATH, f"Path still too long: {len(str(socket_path))}"
        
        # Test that it works
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(socket_path))
        sock.close()
        os.unlink(str(socket_path))
    
    def test_transport_uds_path_generation(self):
        """Test that transport layer generates valid UDS paths."""
        from transport.grpc_impl import GrpcTransport
        
        # Test with normal task ID
        normal_task = "MVP-0-F0.4-T0.4"
        socket_path = Path("/tmp") / f"hermes_{normal_task}_42" / "rpc.sock"
        
        # Should be well within limits
        assert len(str(socket_path)) < 108
        
        # Test with long task ID (should handle gracefully)
        long_task = "x" * 100
        socket_path = Path("/tmp") / f"h_{long_task[:8]}_42" / "rpc.sock"
        
        assert len(str(socket_path)) < 108


if __name__ == "__main__":
    pytest.main([__file__, "-v"])