"""Unit tests for MCP server core functionality."""

import hashlib
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp.server import AnchorEntry, MCPServer
from mcp.storage import ContentAddressedStorage


class TestAnchorEntry:
    """Test AnchorEntry data class."""

    def test_anchor_entry_creation(self):
        """Test creating an anchor entry."""
        data = b"test data"
        entry = AnchorEntry(
            ref="mcp://test/1",
            data=data,
            ttl_s=3600,
            created_at=time.time(),
            namespace="default"
        )
        
        assert entry.ref == "mcp://test/1"
        assert entry.data == data
        assert entry.ttl_s == 3600
        assert entry.namespace == "default"
        assert entry.sha256 == hashlib.sha256(data).hexdigest()
        assert entry.size_bytes == len(data)
        
    def test_expiry_check(self):
        """Test TTL expiry checking."""
        # Non-expired entry
        entry = AnchorEntry(
            ref="mcp://test/1",
            data=b"data",
            ttl_s=3600,
            created_at=time.time(),
        )
        assert not entry.is_expired()
        
        # Expired entry - manually set monotonic time in the past
        entry_expired = AnchorEntry(
            ref="mcp://test/2",
            data=b"data",
            ttl_s=1,
            created_at=time.time() - 10,  # Created 10s ago with 1s TTL
        )
        # Override monotonic time to simulate expiry
        entry_expired.created_at_monotonic = time.monotonic() - 10
        assert entry_expired.is_expired()
        
        # Permanent entry (TTL <= 0)
        entry_permanent = AnchorEntry(
            ref="mcp://test/3",
            data=b"data",
            ttl_s=-1,
            created_at=time.time() - 100000,
        )
        assert not entry_permanent.is_expired()


class TestMCPServerCore:
    """Test MCP server core operations."""
    
    def test_put_and_resolve(self):
        """Test basic put and resolve operations."""
        server = MCPServer()
        
        ref = "mcp://test/basic"
        data = b"Hello, MCP!"
        
        # Put operation
        success, msg = server.put(ref, data, ttl_s=3600)
        assert success
        assert "Stored" in msg
        assert str(len(data)) in msg
        
        # Resolve operation
        resolved = server.resolve(ref)
        assert resolved == data
        
        # Verify stats
        stats = server.get_stats()
        assert stats["puts"] == 1
        assert stats["resolves"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["bytes_stored"] == len(data)
        
    def test_missing_key_resolve(self):
        """Test resolving non-existent key."""
        server = MCPServer()
        
        result = server.resolve("mcp://missing/key")
        assert result is None
        
        stats = server.get_stats()
        assert stats["resolves"] == 1
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        
    def test_overwrite_entry(self):
        """Test overwriting an existing entry."""
        server = MCPServer()
        
        ref = "mcp://test/overwrite"
        data1 = b"original data"
        data2 = b"updated data with more bytes"
        
        # First put
        success, _ = server.put(ref, data1)
        assert success
        assert server._stats["bytes_stored"] == len(data1)
        
        # Overwrite
        success, _ = server.put(ref, data2)
        assert success
        
        # Verify new data
        resolved = server.resolve(ref)
        assert resolved == data2
        
        # Verify storage size updated correctly
        assert server._stats["bytes_stored"] == len(data2)
        assert server._stats["puts"] == 2
        
    def test_stat_operation(self):
        """Test stat operation for metadata retrieval."""
        server = MCPServer()
        
        ref = "mcp://test/stat"
        data = b"stat test data"
        ttl_s = 7200
        
        # Put data
        before_put = time.time()
        server.put(ref, data, ttl_s=ttl_s)
        after_put = time.time()
        
        # Get metadata
        meta = server.stat(ref)
        
        assert meta is not None
        assert meta["ref"] == ref
        assert meta["size_bytes"] == len(data)
        assert meta["sha256"] == hashlib.sha256(data).hexdigest()
        assert meta["ttl_s"] == ttl_s
        assert meta["namespace"] == "default"
        assert before_put <= meta["created_at"] <= after_put
        assert meta["expires_at"] == meta["created_at"] + ttl_s
        
        # Non-existent ref
        assert server.stat("mcp://missing") is None
        
    def test_stat_expired_entry(self):
        """Test stat returns None for expired entries."""
        server = MCPServer()
        
        ref = "mcp://test/expired"
        server.put(ref, b"data", ttl_s=0.1)
        
        # Should exist immediately
        assert server.stat(ref) is not None
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should return None when expired
        assert server.stat(ref) is None


class TestMCPServerTTL:
    """Test TTL-related functionality."""
    
    def test_default_ttl_inference(self):
        """Test automatic TTL inference from ref patterns."""
        server = MCPServer()
        
        # Logs pattern - 24 hours
        server.put("mcp://logs/2024/app.log", b"log data")
        assert server._anchors["mcp://logs/2024/app.log"].ttl_s == 24 * 3600
        
        # Diffs pattern - 7 days
        server.put("mcp://diffs/pr-123.diff", b"diff data")
        assert server._anchors["mcp://diffs/pr-123.diff"].ttl_s == 7 * 24 * 3600
        
        # Repo/SHA pattern - permanent
        server.put("mcp://repo/sha256abc", b"repo data")
        assert server._anchors["mcp://repo/sha256abc"].ttl_s == -1
        
        # Default pattern - 1 hour
        server.put("mcp://temp/file", b"temp data")
        assert server._anchors["mcp://temp/file"].ttl_s == 3600
        
    def test_ttl_expiry_cleanup(self):
        """Test that expired entries are cleaned up on resolve."""
        server = MCPServer()
        
        ref = "mcp://test/ttl"
        data = b"temporary"
        
        # Put with very short TTL
        server.put(ref, data, ttl_s=0.1)
        
        # Should resolve immediately
        assert server.resolve(ref) == data
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should not resolve after expiry
        assert server.resolve(ref) is None
        
        # Verify cleanup happened
        assert ref not in server._anchors
        stats = server.get_stats()
        assert stats["expired"] == 1
        
    def test_permanent_entries(self):
        """Test permanent entries with TTL <= 0."""
        server = MCPServer()
        
        ref = "mcp://permanent/data"
        data = b"forever"
        
        # Create permanent entry
        server.put(ref, data, ttl_s=-1)
        
        # Verify it's marked as permanent
        entry = server._anchors[ref]
        assert entry.ttl_s == -1
        assert not entry.is_expired()
        
        # Should still exist after "long" time
        with patch('time.time', return_value=time.time() + 1000000):
            assert server.resolve(ref) == data
            assert not entry.is_expired()


class TestMCPServerNamespaces:
    """Test namespace functionality."""
    
    def test_namespace_isolation(self):
        """Test that namespaces are properly isolated."""
        server = MCPServer()
        
        # Add entries in different namespaces
        server.put("mcp://ns/1", b"data1", namespace="ns1")
        server.put("mcp://ns/2", b"data2", namespace="ns2")
        server.put("mcp://ns/3", b"data3", namespace="default")
        
        # All should be resolvable
        assert server.resolve("mcp://ns/1") == b"data1"
        assert server.resolve("mcp://ns/2") == b"data2"
        assert server.resolve("mcp://ns/3") == b"data3"
        
        # Check namespace assignment
        assert server._anchors["mcp://ns/1"].namespace == "ns1"
        assert server._anchors["mcp://ns/2"].namespace == "ns2"
        assert server._anchors["mcp://ns/3"].namespace == "default"
        
    def test_namespace_cleanup(self):
        """Test cleaning up a specific namespace."""
        server = MCPServer()
        
        # Add entries in speculative namespace
        server.put("mcp://spec/1", b"spec1", namespace="speculative-123")
        server.put("mcp://spec/2", b"spec2", namespace="speculative-123")
        server.put("mcp://spec/3", b"spec3", namespace="speculative-123")
        
        # Add entries in other namespaces
        server.put("mcp://normal/1", b"normal1", namespace="default")
        server.put("mcp://other/1", b"other1", namespace="other")
        
        # Clean up speculative namespace
        removed = server.cleanup_namespace("speculative-123")
        assert removed == 3
        
        # Verify speculative entries are gone
        assert server.resolve("mcp://spec/1") is None
        assert server.resolve("mcp://spec/2") is None
        assert server.resolve("mcp://spec/3") is None
        
        # Verify other namespaces untouched
        assert server.resolve("mcp://normal/1") == b"normal1"
        assert server.resolve("mcp://other/1") == b"other1"
        
        # Check stats
        stats = server.get_stats()
        assert stats["rollbacks"] == 3
        
    def test_cleanup_nonexistent_namespace(self):
        """Test cleaning up a namespace that doesn't exist."""
        server = MCPServer()
        
        server.put("mcp://test/1", b"data", namespace="exists")
        
        # Clean up non-existent namespace
        removed = server.cleanup_namespace("does-not-exist")
        assert removed == 0
        
        # Original data still there
        assert server.resolve("mcp://test/1") == b"data"


class TestMCPServerLimits:
    """Test size limits and constraints."""
    
    def test_storage_size_limit(self):
        """Test enforcement of storage size limits."""
        # 1 MB limit
        server = MCPServer(max_size_mb=1)
        
        # Store 600 KB - should succeed
        data_600kb = b"x" * (600 * 1024)
        success, msg = server.put("mcp://large/1", data_600kb)
        assert success
        
        # Try to store another 500 KB - should fail
        data_500kb = b"y" * (500 * 1024)
        success, msg = server.put("mcp://large/2", data_500kb)
        assert not success
        assert "Storage limit exceeded" in msg
        
        # Replace existing with smaller - should succeed
        data_300kb = b"z" * (300 * 1024)
        success, msg = server.put("mcp://large/1", data_300kb)
        assert success
        
        # Now we have room for the 500 KB
        success, msg = server.put("mcp://large/2", data_500kb)
        assert success
        
        # Verify total size
        assert server._stats["bytes_stored"] == len(data_300kb) + len(data_500kb)
        
    def test_large_payload_handling(self):
        """Test handling of large payloads up to 256 KB."""
        server = MCPServer(max_size_mb=10)
        
        # Test various sizes up to 256 KB
        sizes = [1024, 10240, 102400, 262144]  # 1KB, 10KB, 100KB, 256KB
        
        for size in sizes:
            ref = f"mcp://large/{size}"
            data = b"L" * size
            
            success, msg = server.put(ref, data)
            assert success, f"Failed to store {size} bytes"
            
            resolved = server.resolve(ref)
            assert resolved == data
            assert len(resolved) == size
            
    def test_concurrent_size_tracking(self):
        """Test that concurrent operations track size correctly."""
        import threading
        
        server = MCPServer(max_size_mb=10)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    ref = f"mcp://concurrent/{worker_id}/{i}"
                    data = f"Worker {worker_id} data {i}".encode() * 100
                    
                    success, _ = server.put(ref, data)
                    assert success
                    
                    # Occasionally overwrite
                    if i % 3 == 0:
                        new_data = b"overwritten" * 50
                        success, _ = server.put(ref, new_data)
                        assert success
                        
            except Exception as e:
                errors.append(e)
                
        # Run concurrent workers
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        assert len(errors) == 0
        
        # Verify size tracking is consistent
        actual_size = sum(len(entry.data) for entry in server._anchors.values())
        assert server._stats["bytes_stored"] == actual_size


class TestContentAddressedStorage:
    """Test content-addressed storage utilities."""
    
    def test_content_addressed_storage(self):
        """Test content-addressed storage operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ContentAddressedStorage(Path(tmpdir))
            
            # Write content
            data = b"test content for CAS"
            sha256 = storage.write_content(data)
            expected_sha = hashlib.sha256(data).hexdigest()
            assert sha256 == expected_sha
            
            # Read content
            read_data = storage.read_content(sha256)
            assert read_data == data
            
            # Write same content again (should be idempotent)
            sha256_2 = storage.write_content(data)
            assert sha256_2 == sha256
            
            # Delete content
            assert storage.delete_content(sha256)
            assert storage.read_content(sha256) is None
            
            # Delete non-existent
            assert not storage.delete_content("nonexistent")
            
    def test_index_persistence(self):
        """Test index save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ContentAddressedStorage(Path(tmpdir))
            
            # Save index
            index = {
                "mcp://test/1": {
                    "sha256": "abc123",
                    "ttl_s": 3600,
                    "created_at": time.time()
                }
            }
            storage.save_index(index)
            
            # Load index
            loaded = storage.load_index()
            assert loaded == index
            
            # Create new storage instance and verify persistence
            storage2 = ContentAddressedStorage(Path(tmpdir))
            loaded2 = storage2.load_index()
            assert loaded2 == index
            
    def test_storage_size_calculation(self):
        """Test storage size calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ContentAddressedStorage(Path(tmpdir))
            
            # Initially empty
            assert storage.get_storage_size() == 0
            
            # Add content
            data1 = b"x" * 1024
            data2 = b"y" * 2048
            
            storage.write_content(data1)
            storage.write_content(data2)
            
            total_size = storage.get_storage_size()
            assert total_size == len(data1) + len(data2)
            
    def test_orphaned_cleanup(self):
        """Test cleanup of orphaned content files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ContentAddressedStorage(Path(tmpdir))
            
            # Write content
            data1 = b"keep this"
            data2 = b"delete this"
            data3 = b"also delete"
            
            sha1 = storage.write_content(data1)
            sha2 = storage.write_content(data2)
            sha3 = storage.write_content(data3)
            
            # Clean up orphaned (only sha1 is valid)
            valid_hashes = {sha1}
            cleaned = storage.cleanup_orphaned(valid_hashes)
            assert cleaned == 2
            
            # Verify cleanup
            assert storage.read_content(sha1) == data1
            assert storage.read_content(sha2) is None
            assert storage.read_content(sha3) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])