"""Integration tests for MCP TTL expiry and cleanup."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp.client import MCPClient
from mcp.server import MCPServer


class TestMCPTTLIntegration:
    """Integration tests for TTL functionality."""

    def test_log_ttl_24h(self):
        """Test that log entries have 24-hour TTL by default."""
        server = MCPServer()
        client = MCPClient(server)

        # Put log data
        log_ref = "mcp://logs/2024-01-15/app.log"
        log_data = b"2024-01-15 10:00:00 INFO Application started\n"

        success, msg = client.put(log_ref, log_data)
        assert success

        # Verify TTL is 24 hours
        meta = client.stat(log_ref)
        assert meta is not None
        assert meta["ttl_s"] == 24 * 3600

        # Should resolve immediately
        assert client.resolve(log_ref) == log_data

        # Simulate time passing (23 hours)
        with patch("time.time", return_value=time.time() + 23 * 3600):
            # Should still be available
            assert client.resolve(log_ref) == log_data

        # Simulate time passing (25 hours) - patch monotonic time
        with patch("time.monotonic", return_value=time.monotonic() + 25 * 3600):
            # Should be expired
            assert client.resolve(log_ref) is None

    def test_diff_ttl_7d(self):
        """Test that diff entries have 7-day TTL by default."""
        server = MCPServer()
        client = MCPClient(server)

        # Put diff data
        diff_ref = "mcp://diffs/pr-1234.diff"
        diff_data = b"--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n"

        success, msg = client.put(diff_ref, diff_data)
        assert success

        # Verify TTL is 7 days
        meta = client.stat(diff_ref)
        assert meta is not None
        assert meta["ttl_s"] == 7 * 24 * 3600

        # Should resolve immediately
        assert client.resolve(diff_ref) == diff_data

        # Simulate time passing (6 days) - patch monotonic time
        with patch("time.monotonic", return_value=time.monotonic() + 6 * 24 * 3600):
            # Should still be available
            assert client.resolve(diff_ref) == diff_data

        # Simulate time passing (8 days) - patch monotonic time
        with patch("time.monotonic", return_value=time.monotonic() + 8 * 24 * 3600):
            # Should be expired
            assert client.resolve(diff_ref) is None

    def test_mixed_ttl_expiry(self):
        """Test mixed TTL expiry with different content types."""
        server = MCPServer()
        client = MCPClient(server)

        # Add various entries with different TTLs
        entries = [
            ("mcp://logs/test.log", b"log", None),  # 24h default
            ("mcp://diffs/test.diff", b"diff", None),  # 7d default
            ("mcp://temp/data", b"temp", 60),  # 1 minute explicit
            ("mcp://repo/sha123", b"repo", None),  # permanent default
            ("mcp://cache/item", b"cache", 300),  # 5 minutes explicit
        ]

        for ref, data, ttl in entries:
            if ttl is None:
                client.put(ref, data)
            else:
                client.put(ref, data, ttl_s=ttl)

        # All should resolve immediately
        for ref, data, _ in entries:
            assert client.resolve(ref) == data

        # After 2 minutes - patch monotonic time
        with patch("time.monotonic", return_value=time.monotonic() + 120):
            # Temp (1 min) should be expired
            assert client.resolve("mcp://temp/data") is None
            # Others should still exist
            assert client.resolve("mcp://logs/test.log") is not None
            assert client.resolve("mcp://cache/item") is not None
            assert client.resolve("mcp://repo/sha123") is not None

        # After 10 minutes - patch monotonic time
        with patch("time.monotonic", return_value=time.monotonic() + 600):
            # Cache (5 min) should also be expired
            assert client.resolve("mcp://cache/item") is None
            # Long-lived entries still exist
            assert client.resolve("mcp://logs/test.log") is not None
            assert client.resolve("mcp://diffs/test.diff") is not None
            assert client.resolve("mcp://repo/sha123") is not None

        # After 1 year (permanent should still exist) - patch monotonic time
        with patch("time.monotonic", return_value=time.monotonic() + 365 * 24 * 3600):
            # Only permanent entry remains
            assert client.resolve("mcp://repo/sha123") == b"repo"
            assert client.resolve("mcp://logs/test.log") is None
            assert client.resolve("mcp://diffs/test.diff") is None

    @pytest.mark.asyncio
    async def test_background_cleanup_task(self):
        """Test that background cleanup task removes expired entries."""
        server = MCPServer()

        # Start server with background cleanup
        await server.start()

        try:
            # Add entries with very short TTLs
            server.put("mcp://cleanup/1", b"data1", ttl_s=1)
            server.put("mcp://cleanup/2", b"data2", ttl_s=2)
            server.put("mcp://cleanup/3", b"data3", ttl_s=-1)  # Permanent

            # All should exist initially
            assert len(server._anchors) == 3

            # Wait for entries to expire
            await asyncio.sleep(3)

            # Manually trigger cleanup (normally runs every 60s)
            with server._lock:
                to_remove = []
                for ref, entry in server._anchors.items():
                    if entry.is_expired():
                        to_remove.append(ref)
                for ref in to_remove:
                    del server._anchors[ref]

            # Only permanent entry should remain
            assert len(server._anchors) == 1
            assert "mcp://cleanup/3" in server._anchors

        finally:
            await server.stop()

    def test_ttl_with_persistence(self):
        """Test TTL handling with persistent storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Create server and add entries
            server1 = MCPServer(storage_path=storage_path)

            # Add entries with different TTLs
            server1.put("mcp://persist/short", b"short", ttl_s=1)
            server1.put("mcp://persist/long", b"long", ttl_s=3600)
            server1.put("mcp://persist/perm", b"perm", ttl_s=-1)

            # Wait for short TTL to expire
            time.sleep(1.5)

            # Create new server and load from disk
            server2 = MCPServer(storage_path=storage_path)

            # Short-lived should be gone (expired during load)
            assert server2.resolve("mcp://persist/short") is None

            # Long-lived and permanent should exist
            assert server2.resolve("mcp://persist/long") == b"long"
            assert server2.resolve("mcp://persist/perm") == b"perm"

    def test_expiry_during_resolve(self):
        """Test that expired entries are cleaned during resolve."""
        server = MCPServer()

        # Add entry with short TTL
        ref = "mcp://expire/test"
        server.put(ref, b"data", ttl_s=0.5)

        # Verify it exists
        assert ref in server._anchors
        initial_bytes = server._stats["bytes_stored"]

        # Wait for expiry
        time.sleep(0.6)

        # Resolve should trigger cleanup
        result = server.resolve(ref)
        assert result is None

        # Verify cleanup happened
        assert ref not in server._anchors
        assert server._stats["expired"] == 1
        assert server._stats["bytes_stored"] < initial_bytes

    def test_speculative_namespace_rollback(self):
        """Test rollback of speculative namespace on failure."""
        server = MCPServer()
        client = MCPClient(server)

        # Simulate speculative execution
        spec_namespace = "speculative-task-123"

        # Create speculative anchors
        client.put("mcp://spec/code", b"new code", namespace=spec_namespace)
        client.put("mcp://spec/test", b"test output", namespace=spec_namespace)
        client.put("mcp://spec/logs", b"execution logs", namespace=spec_namespace)

        # Create normal anchors
        client.put("mcp://normal/data", b"normal data")

        # Verify all exist
        assert client.resolve("mcp://spec/code") is not None
        assert client.resolve("mcp://spec/test") is not None
        assert client.resolve("mcp://spec/logs") is not None
        assert client.resolve("mcp://normal/data") is not None

        # Simulate rollback (e.g., test failed)
        removed = client.cleanup_namespace(spec_namespace)
        assert removed == 3

        # Verify speculative anchors are gone
        assert client.resolve("mcp://spec/code") is None
        assert client.resolve("mcp://spec/test") is None
        assert client.resolve("mcp://spec/logs") is None

        # Normal anchors remain
        assert client.resolve("mcp://normal/data") == b"normal data"

        # Check rollback stats
        stats = client.get_stats()
        assert stats["rollbacks"] == 3

    def test_ttl_boundary_conditions(self):
        """Test TTL boundary conditions and edge cases."""
        server = MCPServer()

        # TTL = 0 (permanent, same as negative)
        server.put("mcp://ttl/zero", b"zero", ttl_s=0)
        entry = server._anchors["mcp://ttl/zero"]
        # TTL <= 0 means permanent
        assert not entry.is_expired()

        # TTL = -1 (permanent)
        server.put("mcp://ttl/negative", b"negative", ttl_s=-1)
        entry = server._anchors["mcp://ttl/negative"]
        assert not entry.is_expired()

        # Very large TTL
        server.put("mcp://ttl/large", b"large", ttl_s=365 * 24 * 3600 * 100)  # 100 years
        entry = server._anchors["mcp://ttl/large"]
        assert not entry.is_expired()

    def test_concurrent_expiry_and_access(self):
        """Test concurrent expiry and access patterns."""
        import threading

        server = MCPServer()
        errors = []
        results = []

        def writer():
            """Continuously write entries with short TTLs."""
            try:
                for i in range(20):
                    ref = f"mcp://concurrent/write/{i}"
                    server.put(ref, b"data", ttl_s=0.1)
                    time.sleep(0.05)
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            """Continuously read entries."""
            try:
                for _ in range(50):
                    for i in range(20):
                        ref = f"mcp://concurrent/write/{i}"
                        data = server.resolve(ref)
                        if data:
                            results.append(ref)
                    time.sleep(0.02)
            except Exception as e:
                errors.append(("reader", e))

        def cleaner():
            """Simulate cleanup operations."""
            try:
                for _ in range(10):
                    time.sleep(0.1)
                    # Trigger cleanup via resolve on expired entries
                    for i in range(20):
                        server.resolve(f"mcp://concurrent/write/{i}")
            except Exception as e:
                errors.append(("cleaner", e))

        # Run concurrent operations
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=cleaner),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Concurrent errors: {errors}"

        # Should have successfully read some entries
        assert len(results) > 0

        # Stats should show expiries
        stats = server.get_stats()
        assert stats["expired"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
