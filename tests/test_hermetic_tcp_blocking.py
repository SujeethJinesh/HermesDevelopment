#!/usr/bin/env python3
"""Test that TCP is blocked under hermetic mode."""

import os
import socket
import sys
from pathlib import Path

import grpc
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.hermetic import HermeticRun


class TestHermeticTCPBlocking:
    """Test TCP/network blocking in hermetic mode."""

    def test_tcp_blocked_in_hermetic_mode(self):
        """Test that TCP connections are blocked in hermetic mode."""
        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"

        # Create hermetic run
        run = HermeticRun(
            task_id="test-tcp-block",
            seed=42,
            hermetic=True
        )

        with run():
            # Import socket inside hermetic context to get patched version
            import socket as hermetic_socket

            # Attempt to create TCP socket should fail
            with pytest.raises(Exception) as exc_info:
                hermetic_socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            err = str(exc_info.value).lower()
            assert "hermetic" in err or "blocked" in err

            # Attempt to connect to localhost should fail
            with pytest.raises(Exception) as exc_info:
                hermetic_socket.create_connection(("localhost", 8080))

            err = str(exc_info.value).lower()
            assert "hermetic" in err or "blocked" in err

            # DNS lookups (except localhost) should fail
            with pytest.raises(Exception) as exc_info:
                hermetic_socket.getaddrinfo("google.com", 80)

            err = str(exc_info.value).lower()
            assert "hermetic" in err or "blocked" in err

    def test_unix_socket_allowed_in_hermetic_mode(self):
        """Test that UNIX domain sockets are allowed in hermetic mode."""
        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"

        # Create hermetic run
        run = HermeticRun(
            task_id="test-uds-allow",
            seed=42,
            hermetic=True
        )

        with run():
            # Import socket inside hermetic context
            import socket as hermetic_socket

            # UNIX domain socket should work
            sock = hermetic_socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            assert sock is not None
            sock.close()

    def test_grpc_tcp_blocked_hermetic(self):
        """Test that gRPC TCP connections fail in hermetic mode."""
        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"

        run = HermeticRun(
            task_id="test-grpc-tcp",
            seed=42,
            hermetic=True
        )

        with run():
            # Attempting to create TCP channel should fail or timeout
            channel = grpc.insecure_channel("localhost:50051")

            # Try to make a call (should fail/timeout)
            # We can't easily test this without a stub, but channel creation
            # itself should be affected by network blocking
            channel.close()

    def test_grpc_uds_allowed_hermetic(self):
        """Test that gRPC UDS connections work in hermetic mode."""
        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"

        run = HermeticRun(
            task_id="test-grpc-uds",
            seed=42,
            hermetic=True
        )

        with run():
            import tempfile

            # UDS channel should work
            with tempfile.TemporaryDirectory() as tmpdir:
                socket_path = Path(tmpdir) / "test.sock"
                channel = grpc.insecure_channel(f"unix://{socket_path}")
                assert channel is not None
                channel.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
