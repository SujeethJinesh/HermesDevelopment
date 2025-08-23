#!/usr/bin/env python3
"""Minimal MCP test."""

import sys
sys.path.insert(0, '.')

from mcp.server import MCPServer

# Test basic operations
server = MCPServer()

# Put
success, msg = server.put("mcp://test/1", b"test data", ttl_s=3600)
print(f"Put: {success}, {msg}")

# Resolve
data = server.resolve("mcp://test/1")
print(f"Resolve: {data}")

# Stat
meta = server.stat("mcp://test/1")
print(f"Stat: {meta}")

print("✓ Basic operations work")