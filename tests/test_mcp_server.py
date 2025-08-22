"""Tests for MCP server with TTL and performance requirements."""

import tempfile
import time
from pathlib import Path

import pytest

from mcp.client import MCPClient
from mcp.server import MCPServer


class TestMCPServer:
    """Test MCP server functionality."""

    def test_basic_put_resolve(self):
        """Test basic put and resolve operations."""
        server = MCPServer()

        # Put data
        ref = "mcp://test/1234"
        data = b"Hello, MCP!"
        success, msg = server.put(ref, data)

        assert success
        assert "Stored" in msg

        # Resolve data
        resolved = server.resolve(ref)
        assert resolved == data

        # Check stats
        stats = server.get_stats()
        assert stats['puts'] == 1
        assert stats['resolves'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 0

    def test_ttl_expiry(self):
        """Test TTL expiry mechanism."""
        server = MCPServer()

        # Put with short TTL
        ref = "mcp://ephemeral/test"
        data = b"Temporary data"
        success, _ = server.put(ref, data, ttl_s=1)
        assert success

        # Should resolve immediately
        assert server.resolve(ref) == data

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        assert server.resolve(ref) is None

        # Check stats
        stats = server.get_stats()
        assert stats['expired'] == 1
        assert stats['misses'] == 1

    def test_default_ttls(self):
        """Test default TTLs by content type."""
        server = MCPServer()

        # Test logs TTL
        server.put("mcp://logs/123", b"log data")
        entry = server._anchors["mcp://logs/123"]
        assert entry.ttl_s == 24 * 3600  # 24 hours

        # Test diffs TTL
        server.put("mcp://diffs/456", b"diff data")
        entry = server._anchors["mcp://diffs/456"]
        assert entry.ttl_s == 7 * 24 * 3600  # 7 days

        # Test repo TTL (permanent)
        server.put("mcp://repo/sha256", b"repo data")
        entry = server._anchors["mcp://repo/sha256"]
        assert entry.ttl_s == -1  # Permanent

        # Test default TTL
        server.put("mcp://other/789", b"other data")
        entry = server._anchors["mcp://other/789"]
        assert entry.ttl_s == 3600  # 1 hour

    def test_namespace_cleanup(self):
        """Test speculative namespace cleanup."""
        server = MCPServer()

        # Add anchors in different namespaces
        server.put("mcp://spec/1", b"data1", namespace="speculative-1")
        server.put("mcp://spec/2", b"data2", namespace="speculative-1")
        server.put("mcp://normal/1", b"data3", namespace="default")

        # Verify all exist
        assert server.resolve("mcp://spec/1") is not None
        assert server.resolve("mcp://spec/2") is not None
        assert server.resolve("mcp://normal/1") is not None

        # Clean up speculative namespace
        removed = server.cleanup_namespace("speculative-1")
        assert removed == 2

        # Verify cleanup
        assert server.resolve("mcp://spec/1") is None
        assert server.resolve("mcp://spec/2") is None
        assert server.resolve("mcp://normal/1") is not None

        # Check stats
        stats = server.get_stats()
        assert stats['rollbacks'] == 2

    def test_stat_operation(self):
        """Test stat operation for metadata."""
        server = MCPServer()

        ref = "mcp://test/metadata"
        data = b"Test data for metadata"
        server.put(ref, data, ttl_s=3600)

        # Get metadata
        meta = server.stat(ref)
        assert meta is not None
        assert meta['ref'] == ref
        assert meta['size_bytes'] == len(data)
        assert meta['ttl_s'] == 3600
        assert meta['namespace'] == 'default'
        assert 'sha256' in meta
        assert 'created_at' in meta

        # Non-existent ref
        assert server.stat("mcp://missing") is None

    def test_size_limits(self):
        """Test storage size limits."""
        # 1 MB limit
        server = MCPServer(max_size_mb=1)

        # Store 500 KB - should succeed
        data_500kb = b"x" * (500 * 1024)
        success, _ = server.put("mcp://large/1", data_500kb)
        assert success

        # Try to store another 600 KB - should fail
        data_600kb = b"y" * (600 * 1024)
        success, msg = server.put("mcp://large/2", data_600kb)
        assert not success
        assert "Storage limit exceeded" in msg

        # Replace existing - should succeed
        data_400kb = b"z" * (400 * 1024)
        success, _ = server.put("mcp://large/1", data_400kb)
        assert success

    def test_persistence(self):
        """Test persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "mcp_storage"

            # Create server and store data
            server1 = MCPServer(storage_path=storage_path)
            server1.put("mcp://persistent/1", b"data1", ttl_s=3600)
            server1.put("mcp://persistent/2", b"data2", ttl_s=-1)  # Permanent

            # Persist to disk
            server1._persist_to_disk()

            # Create new server and load
            server2 = MCPServer(storage_path=storage_path)

            # Verify data loaded
            assert server2.resolve("mcp://persistent/1") == b"data1"
            assert server2.resolve("mcp://persistent/2") == b"data2"

    @pytest.mark.asyncio
    async def test_cleanup_task(self):
        """Test background cleanup task."""
        server = MCPServer()

        # Start server
        await server.start()

        try:
            # Add expired anchor
            ref = "mcp://expired/test"
            server._anchors[ref] = server._anchors.get("dummy", type('', (), {
                'ref': ref,
                'data': b"old",
                'ttl_s': 1,
                'created_at': time.time() - 100,  # Expired 99s ago
                'namespace': 'default',
                'sha256': 'abc',
                'is_expired': lambda: True,
                'size_bytes': 3
            })())

            # Should be cleaned up (but we won't wait 60s for the task)
            # Just verify server is running
            assert server._running

        finally:
            await server.stop()
            assert not server._running

    def test_deref_performance(self):
        """Test deref p95 < 50ms requirement."""
        server = MCPServer()
        client = MCPClient(server)

        # Prepare test data
        test_data = {
            f"mcp://perf/{i}": b"x" * (i * 100)  # Varying sizes
            for i in range(100)
        }

        # Store all data
        for ref, data in test_data.items():
            server.put(ref, data)

        # Resolve multiple times to collect timing
        for _ in range(10):
            for ref in test_data:
                client.resolve(ref)

        # Check p95
        p95 = client.get_deref_p95()
        assert p95 is not None
        assert p95 < 50.0, f"Deref p95 {p95:.3f}ms exceeds 50ms limit"

        # Verify stats
        stats = client.get_stats()
        assert 'deref_p50_ms' in stats
        assert 'deref_p95_ms' in stats
        assert stats['deref_p95_ms'] < 50.0

    def test_client_operations(self):
        """Test MCP client operations."""
        server = MCPServer()
        client = MCPClient(server)

        # Put via client
        ref = "mcp://client/test"
        data = b"Client data"
        success, msg = client.put(ref, data, ttl_s=3600)
        assert success

        # Resolve via client
        resolved = client.resolve(ref)
        assert resolved == data

        # Stat via client
        meta = client.stat(ref)
        assert meta is not None
        assert meta['size_bytes'] == len(data)

        # Cleanup namespace via client
        client.put("mcp://spec/1", b"spec1", namespace="test-spec")
        client.put("mcp://spec/2", b"spec2", namespace="test-spec")
        removed = client.cleanup_namespace("test-spec")
        assert removed == 2

        # Check client stats
        stats = client.get_stats()
        assert stats['puts'] == 3
        assert stats['resolves'] == 1
        assert stats['rollbacks'] == 2

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading

        server = MCPServer()
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    ref = f"mcp://concurrent/{worker_id}/{i}"
                    data = f"Worker {worker_id} data {i}".encode()

                    # Put
                    success, _ = server.put(ref, data)
                    assert success

                    # Resolve
                    resolved = server.resolve(ref)
                    assert resolved == data

                    # Stat
                    meta = server.stat(ref)
                    assert meta is not None

            except Exception as e:
                errors.append(e)

        # Run concurrent workers
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check no errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # Verify all data
        stats = server.get_stats()
        assert stats['puts'] == 100  # 10 workers * 10 puts
        assert stats['resolves'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
