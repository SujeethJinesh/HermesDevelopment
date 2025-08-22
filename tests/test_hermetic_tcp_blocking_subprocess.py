#!/usr/bin/env python3
"""Test that TCP is blocked under hermetic mode using subprocess."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.hermetic import HermeticRun


class TestHermeticTCPBlockingSubprocess:
    """Test TCP/network blocking in hermetic mode using subprocess."""

    def test_tcp_blocked_in_hermetic_subprocess(self):
        """Test that TCP connections are blocked when running in hermetic venv subprocess."""
        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Create hermetic run
        run = HermeticRun(
            task_id="test-tcp-subprocess",
            seed=42,
            hermetic=True
        )
        
        # Setup hermetic environment
        run.setup()
        
        try:
            # Create test script
            test_script = run.scratch_base / "test_tcp.py"
            test_script.write_text("""
import socket
import sys

# Try to create TCP socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("ERROR: TCP socket created successfully (should have been blocked)")
    sys.exit(1)
except Exception as e:
    print(f"PASS: TCP blocked with: {e}")

# Try to connect to localhost
try:
    socket.create_connection(("localhost", 8080))
    print("ERROR: TCP connection succeeded (should have been blocked)")
    sys.exit(1)
except Exception as e:
    print(f"PASS: TCP connection blocked with: {e}")

# Try DNS lookup
try:
    socket.getaddrinfo("google.com", 80)
    print("ERROR: DNS lookup succeeded (should have been blocked)")
    sys.exit(1)
except Exception as e:
    print(f"PASS: DNS lookup blocked with: {e}")

print("SUCCESS: All TCP operations properly blocked")
sys.exit(0)
""")
            
            # Run script in hermetic venv
            result = subprocess.run(
                [str(run.venv_path / "bin" / "python"), str(test_script)],
                capture_output=True,
                text=True,
                env={**os.environ, "HERMES_HERMETIC": "1"}
            )
            
            print("=== Subprocess stdout ===")
            print(result.stdout)
            print("=== Subprocess stderr ===")
            print(result.stderr)
            
            # Check that all operations were blocked
            assert result.returncode == 0, f"Test failed: {result.stdout} {result.stderr}"
            assert "PASS: TCP blocked" in result.stdout
            assert "PASS: TCP connection blocked" in result.stdout
            assert "PASS: DNS lookup blocked" in result.stdout
            assert "SUCCESS: All TCP operations properly blocked" in result.stdout
            
        finally:
            run._cleanup()
    
    def test_unix_socket_allowed_in_hermetic_subprocess(self):
        """Test that UNIX domain sockets work in hermetic venv subprocess."""
        # Enable hermetic mode
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Create hermetic run
        run = HermeticRun(
            task_id="test-uds-subprocess",
            seed=42,
            hermetic=True
        )
        
        # Setup hermetic environment
        run.setup()
        
        try:
            # Create test script
            test_script = run.scratch_base / "test_uds.py"
            test_script.write_text("""
import socket
import sys
import os

# Try to create UNIX socket
try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    print("PASS: UNIX socket created successfully")
    sock.close()
except Exception as e:
    print(f"ERROR: UNIX socket failed: {e}")
    sys.exit(1)

# Try to bind and connect via UDS
socket_path = "/tmp/test_hermetic_uds.sock"
if os.path.exists(socket_path):
    os.unlink(socket_path)

try:
    # Create server socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    print(f"PASS: UDS server bound to {socket_path}")
    server.close()
    os.unlink(socket_path)
except Exception as e:
    print(f"ERROR: UDS bind failed: {e}")
    sys.exit(1)

print("SUCCESS: UNIX domain sockets work correctly")
sys.exit(0)
""")
            
            # Run script in hermetic venv
            result = subprocess.run(
                [str(run.venv_path / "bin" / "python"), str(test_script)],
                capture_output=True,
                text=True,
                env={**os.environ, "HERMES_HERMETIC": "1"}
            )
            
            print("=== Subprocess stdout ===")
            print(result.stdout)
            print("=== Subprocess stderr ===")
            print(result.stderr)
            
            # Check that UDS operations succeeded
            assert result.returncode == 0, f"Test failed: {result.stdout} {result.stderr}"
            assert "PASS: UNIX socket created" in result.stdout
            assert "PASS: UDS server bound" in result.stdout
            assert "SUCCESS: UNIX domain sockets work correctly" in result.stdout
            
        finally:
            run._cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])