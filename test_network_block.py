#!/usr/bin/env python3
"""Test network blocking in hermetic mode."""

import socket
import os

# Simulate hermetic network block
if os.environ.get('HERMES_HERMETIC') == '1':
    print("HERMES_HERMETIC=1 detected, attempting outbound connection...")
    try:
        # Try to connect to a public DNS server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect(('8.8.8.8', 53))
        sock.close()
        print("ERROR: Outbound connection succeeded (should be blocked)")
    except (socket.timeout, socket.error, ConnectionRefusedError, OSError) as e:
        print(f"✓ Outbound network blocked: {type(e).__name__}: {e}")
else:
    print("Not in hermetic mode")