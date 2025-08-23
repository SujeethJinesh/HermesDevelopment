"""Deterministic TTL tests with short timeouts for reliable CI/CD."""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp.client import MCPClient
from mcp.server import MCPServer


class TestDeterministicTTL:
    """Deterministic TTL tests using short timeouts and monotonic time."""

    def test_ttl_expiry_200ms(self):
        """Test TTL expiry at 200ms boundary."""
        server = MCPServer()
        client = MCPClient(server)

        ref = "mcp://test/short_ttl"
        data = b"expires quickly"

        # Put with 200ms TTL
        success, msg = client.put(ref, data, ttl_s=0.2)
        assert success

        # Should resolve immediately
        assert client.resolve(ref) == data

        # Should still resolve at 150ms
        start = time.perf_counter()
        while time.perf_counter() - start < 0.15:
            time.sleep(0.01)
        assert client.resolve(ref) == data

        # Should NOT resolve after 250ms
        while time.perf_counter() - start < 0.25:
            time.sleep(0.01)
        assert client.resolve(ref) is None

        # Verify expiry was recorded
        stats = server.get_stats()
        assert stats["expired"] >= 1

    def test_ttl_500ms_boundary(self):
        """Test TTL expiry at 500ms boundary."""
        server = MCPServer()
        client = MCPClient(server)

        ref = "mcp://test/medium_ttl"
        data = b"half second lifetime"

        # Put with 500ms TTL
        success, msg = client.put(ref, data, ttl_s=0.5)
        assert success

        # Track time with monotonic clock
        start = time.perf_counter()

        # Should resolve at 100ms intervals up to 400ms
        for checkpoint in [0.1, 0.2, 0.3, 0.4]:
            while time.perf_counter() - start < checkpoint:
                time.sleep(0.01)
            resolved = client.resolve(ref)
            assert resolved == data, f"Failed at {checkpoint}s"

        # Should NOT resolve after 600ms
        while time.perf_counter() - start < 0.6:
            time.sleep(0.01)
        assert client.resolve(ref) is None

    def test_mixed_ttl_expiry_order(self):
        """Test that entries expire in correct order."""
        server = MCPServer()
        client = MCPClient(server)

        # Create entries with staggered TTLs
        entries = [
            ("mcp://ttl/100ms", b"first", 0.1),
            ("mcp://ttl/300ms", b"second", 0.3),
            ("mcp://ttl/500ms", b"third", 0.5),
            ("mcp://ttl/permanent", b"forever", -1),
        ]

        for ref, data, ttl in entries:
            client.put(ref, data, ttl_s=ttl)

        start = time.perf_counter()

        # At 50ms, all should exist
        while time.perf_counter() - start < 0.05:
            time.sleep(0.01)
        for ref, data, _ in entries:
            assert client.resolve(ref) == data

        # At 150ms, first should be gone
        while time.perf_counter() - start < 0.15:
            time.sleep(0.01)
        assert client.resolve("mcp://ttl/100ms") is None
        assert client.resolve("mcp://ttl/300ms") == b"second"
        assert client.resolve("mcp://ttl/500ms") == b"third"
        assert client.resolve("mcp://ttl/permanent") == b"forever"

        # At 350ms, first two should be gone
        while time.perf_counter() - start < 0.35:
            time.sleep(0.01)
        assert client.resolve("mcp://ttl/100ms") is None
        assert client.resolve("mcp://ttl/300ms") is None
        assert client.resolve("mcp://ttl/500ms") == b"third"
        assert client.resolve("mcp://ttl/permanent") == b"forever"

        # At 550ms, only permanent remains
        while time.perf_counter() - start < 0.55:
            time.sleep(0.01)
        assert client.resolve("mcp://ttl/100ms") is None
        assert client.resolve("mcp://ttl/300ms") is None
        assert client.resolve("mcp://ttl/500ms") is None
        assert client.resolve("mcp://ttl/permanent") == b"forever"

    def test_ttl_classes_with_mocked_time(self):
        """Test TTL classes (logs/diffs/repo) with time mocking."""
        server = MCPServer()
        client = MCPClient(server)

        # Put entries with default TTLs
        client.put("mcp://logs/test.log", b"log data")
        client.put("mcp://diffs/pr.diff", b"diff data")
        client.put("mcp://repo/sha256abc", b"repo data")

        # Verify TTLs were set correctly
        assert server._anchors["mcp://logs/test.log"].ttl_s == 24 * 3600
        assert server._anchors["mcp://diffs/pr.diff"].ttl_s == 7 * 24 * 3600
        assert server._anchors["mcp://repo/sha256abc"].ttl_s == -1

        # For testing, override with short TTLs
        server._anchors["mcp://logs/test.log"].ttl_s = 0.2
        server._anchors["mcp://diffs/pr.diff"].ttl_s = 0.4

        # Reset creation times to now
        now = time.time()
        server._anchors["mcp://logs/test.log"].created_at = now
        server._anchors["mcp://diffs/pr.diff"].created_at = now

        # Test expiry boundaries
        start = time.perf_counter()

        # At 100ms, all exist
        while time.perf_counter() - start < 0.1:
            time.sleep(0.01)
        assert client.resolve("mcp://logs/test.log") is not None
        assert client.resolve("mcp://diffs/pr.diff") is not None
        assert client.resolve("mcp://repo/sha256abc") is not None

        # At 250ms, logs expired
        while time.perf_counter() - start < 0.25:
            time.sleep(0.01)

        # Need to update time.time() for expiry check
        with patch("time.time", return_value=now + 0.25):
            assert client.resolve("mcp://logs/test.log") is None
            assert client.resolve("mcp://diffs/pr.diff") is not None
            assert client.resolve("mcp://repo/sha256abc") is not None

        # At 450ms, diffs also expired
        while time.perf_counter() - start < 0.45:
            time.sleep(0.01)

        with patch("time.time", return_value=now + 0.45):
            assert client.resolve("mcp://logs/test.log") is None
            assert client.resolve("mcp://diffs/pr.diff") is None
            assert client.resolve("mcp://repo/sha256abc") is not None

    def test_no_stale_reads_after_expiry(self):
        """Ensure no stale reads from any cache layer after expiry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))
            client = MCPClient(server)

            ref = "mcp://cache/test"
            data = b"cached data"

            # Put with 300ms TTL
            client.put(ref, data, ttl_s=0.3)

            # Resolve multiple times to potentially populate any caches
            for _ in range(10):
                assert client.resolve(ref) == data

            # Force persist to disk
            server._persist_to_disk()

            # Wait for expiry
            start = time.perf_counter()
            while time.perf_counter() - start < 0.35:
                time.sleep(0.01)

            # Should not resolve from memory
            assert client.resolve(ref) is None

            # Should not resolve even after re-reading from disk
            server2 = MCPServer(storage_path=Path(tmpdir))
            client2 = MCPClient(server2)
            assert client2.resolve(ref) is None

            # Verify no stale data in stats
            _ = server.get_stats()  # Just verify it runs, no checks needed
            assert ref not in server._anchors

    def test_ttl_update_on_overwrite(self):
        """Test that TTL is updated when overwriting an entry."""
        server = MCPServer()
        client = MCPClient(server)

        ref = "mcp://update/ttl"

        # Initial put with 200ms TTL
        client.put(ref, b"v1", ttl_s=0.2)
        created1 = server._anchors[ref].created_at

        # Wait 100ms
        time.sleep(0.1)

        # Overwrite with 500ms TTL
        client.put(ref, b"v2", ttl_s=0.5)
        created2 = server._anchors[ref].created_at

        # Creation time should be updated
        assert created2 > created1

        # Should still be alive after original TTL expires
        time.sleep(0.15)  # Total 250ms from original
        assert client.resolve(ref) == b"v2"

        # Should expire based on new TTL
        time.sleep(0.4)  # Total 550ms from update
        assert client.resolve(ref) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
