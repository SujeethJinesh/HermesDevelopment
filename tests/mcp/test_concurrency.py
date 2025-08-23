"""Concurrency tests for MCP server - verifying thread-safe operations."""

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from typing import Tuple

import pytest

from mcp.client import MCPClient
from mcp.server import MCPServer


class TestMCPConcurrency:
    """Test concurrent access patterns for MCP server."""

    def test_concurrent_puts_same_ref(self):
        """Test concurrent puts to the same reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))

            ref = "mcp://concurrent/shared"
            results = []
            errors = []

            def worker(worker_id: int):
                """Worker that puts unique data."""
                try:
                    client = MCPClient(server)
                    # Each worker writes unique data
                    data = f"Worker-{worker_id}-data".encode() * 100
                    success, msg = client.put(ref, data)
                    if success:
                        results.append((worker_id, hashlib.sha256(data).hexdigest()))
                except Exception as e:
                    errors.append((worker_id, str(e)))

            # Launch 32 concurrent workers
            threads = []
            for i in range(32):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Should have no errors
            assert len(errors) == 0, f"Errors occurred: {errors}"

            # All puts should succeed (last-write-wins semantics)
            assert len(results) == 32

            # Final state should be one of the written values
            final_data = server.resolve(ref)
            assert final_data is not None
            final_sha = hashlib.sha256(final_data).hexdigest()

            # Verify it matches one of the worker's data
            worker_shas = [sha for _, sha in results]
            assert final_sha in worker_shas

    def test_concurrent_puts_distinct_refs(self):
        """Test concurrent puts to distinct references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))

            results = []
            errors = []

            def worker(worker_id: int):
                """Worker that puts to its own ref."""
                try:
                    client = MCPClient(server)
                    ref = f"mcp://concurrent/worker_{worker_id}"
                    data = f"Worker {worker_id} unique data".encode() * 50

                    success, msg = client.put(ref, data)
                    assert success

                    # Immediately verify our write
                    resolved = client.resolve(ref)
                    assert resolved == data

                    results.append((worker_id, ref, hashlib.sha256(data).hexdigest()))
                except Exception as e:
                    errors.append((worker_id, str(e)))

            # Launch 32 concurrent workers
            threads = []
            for i in range(32):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Should have no errors
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 32

            # Verify all data is intact
            client = MCPClient(server)
            for worker_id, ref, expected_sha in results:
                data = client.resolve(ref)
                assert data is not None, f"Lost data for {ref}"
                actual_sha = hashlib.sha256(data).hexdigest()
                assert actual_sha == expected_sha, f"Data corruption for {ref}"

    def test_no_torn_writes(self):
        """Verify no torn writes under concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))

            # Use a marker pattern to detect torn writes
            marker = b"COMPLETE"

            def make_data(worker_id: int, size: int = 10000) -> bytes:
                """Create data with integrity markers."""
                # Format: START|worker_id|payload|worker_id|END
                payload = f"W{worker_id}".encode() * (size // 3)
                return (
                    b"START|"
                    + f"{worker_id}|".encode()
                    + payload
                    + f"|{worker_id}|".encode()
                    + marker
                )

            def verify_data(data: bytes) -> Tuple[bool, int]:
                """Verify data integrity, return (is_valid, worker_id)."""
                if not data.startswith(b"START|") or not data.endswith(marker):
                    return False, -1

                try:
                    # Extract worker ID from start
                    parts = data.split(b"|")
                    worker_id = int(parts[1])
                    # Verify matching ID before end marker
                    end_id = int(parts[-2])
                    return worker_id == end_id, worker_id
                except Exception:
                    return False, -1

            errors = []
            success_count = 0

            def writer(worker_id: int):
                """Continuously write with integrity checks."""
                nonlocal success_count
                try:
                    client = MCPClient(server)
                    for iteration in range(10):
                        ref = f"mcp://torn/test_{iteration % 5}"  # Reuse refs
                        data = make_data(worker_id)

                        success, _ = client.put(ref, data)
                        assert success

                        # Small delay to increase contention
                        time.sleep(0.001)

                        # Read back and verify
                        read_data = client.resolve(ref)
                        is_valid, found_id = verify_data(read_data)

                        if not is_valid:
                            errors.append(f"Torn write detected at {ref}")
                        else:
                            success_count += 1

                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")

            # Run concurrent writers
            threads = []
            for i in range(16):
                t = threading.Thread(target=writer, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # No torn writes should occur
            assert len(errors) == 0, f"Torn writes or errors: {errors}"
            assert success_count > 0, "No successful writes"
            print(f"✓ {success_count} writes verified without tearing")

    def test_concurrent_resolve_during_writes(self):
        """Test resolves return consistent data during concurrent writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))

            ref = "mcp://consistency/test"
            inconsistencies = []

            # Pre-populate
            initial_data = b"INITIAL" * 1000
            server.put(ref, initial_data)

            def writer():
                """Continuously update the ref."""
                client = MCPClient(server)
                for i in range(50):
                    data = f"UPDATE_{i}".encode() * 500
                    client.put(ref, data)
                    time.sleep(0.002)  # 2ms between writes

            def reader(reader_id: int):
                """Continuously read and verify consistency."""
                client = MCPClient(server)
                for _ in range(100):
                    data = client.resolve(ref)
                    if data:
                        # Verify data is either initial or a complete update
                        if data == initial_data:
                            continue  # OK
                        elif data.startswith(b"UPDATE_"):
                            # Check it's a complete update (not torn)
                            parts = data.split(b"UPDATE_")
                            if len(parts) > 1:
                                # All parts should have same number
                                numbers = []
                                for part in parts[1:]:
                                    if part:
                                        try:
                                            num = int(part.split(b"UPDATE_")[0].decode())
                                            numbers.append(num)
                                        except Exception:
                                            pass

                                # All numbers should be the same (not mixed)
                                if numbers and len(set(numbers)) > 1:
                                    inconsistencies.append(
                                        f"Reader {reader_id}: Mixed updates {set(numbers)}"
                                    )
                        else:
                            inconsistencies.append(f"Reader {reader_id}: Unexpected data prefix")

                    time.sleep(0.001)

            # Start writer and multiple readers
            threads = []

            writer_thread = threading.Thread(target=writer)
            threads.append(writer_thread)
            writer_thread.start()

            for i in range(8):
                reader_thread = threading.Thread(target=reader, args=(i,))
                threads.append(reader_thread)
                reader_thread.start()

            for t in threads:
                t.join()

            # Should have no inconsistencies
            assert len(inconsistencies) == 0, f"Inconsistencies found: {inconsistencies[:5]}"
            print("✓ All reads returned consistent data during concurrent writes")

    def test_namespace_isolation_under_concurrency(self):
        """Test namespace cleanup doesn't affect other namespaces under load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))

            errors = []

            def namespace_worker(namespace: str, worker_id: int):
                """Worker operating in a specific namespace."""
                try:
                    client = MCPClient(server)

                    # Write data in our namespace
                    for i in range(10):
                        ref = f"mcp://ns/{namespace}/item_{worker_id}_{i}"
                        data = f"{namespace}:{worker_id}:{i}".encode() * 100
                        success, _ = client.put(ref, data, namespace=namespace)
                        assert success

                    # Verify our data exists
                    for i in range(10):
                        ref = f"mcp://ns/{namespace}/item_{worker_id}_{i}"
                        data = client.resolve(ref)
                        assert data is not None

                except Exception as e:
                    errors.append((namespace, worker_id, str(e)))

            def cleanup_worker(namespace: str):
                """Worker that cleans up a namespace."""
                try:
                    time.sleep(0.05)  # Let writers get started
                    client = MCPClient(server)
                    removed = client.cleanup_namespace(namespace)
                    print(f"Cleaned {removed} items from {namespace}")
                except Exception as e:
                    errors.append((namespace, "cleanup", str(e)))

            # Create workers in different namespaces
            threads = []

            # Regular namespace workers (should persist)
            for i in range(8):
                t = threading.Thread(target=namespace_worker, args=("persistent", i))
                threads.append(t)
                t.start()

            # Speculative namespace workers (will be cleaned)
            for i in range(8):
                t = threading.Thread(target=namespace_worker, args=("speculative", i))
                threads.append(t)
                t.start()

            # Cleanup speculative namespace mid-flight
            cleanup_thread = threading.Thread(target=cleanup_worker, args=("speculative",))
            threads.append(cleanup_thread)
            cleanup_thread.start()

            for t in threads:
                t.join()

            # Check persistent namespace is intact
            client = MCPClient(server)
            persistent_count = 0
            speculative_count = 0

            for ref in list(server._anchors.keys()):
                if "persistent" in ref:
                    data = client.resolve(ref)
                    if data:
                        persistent_count += 1
                elif "speculative" in ref:
                    data = client.resolve(ref)
                    if data:
                        speculative_count += 1

            print(f"Persistent items: {persistent_count}, Speculative items: {speculative_count}")

            # Persistent namespace should have data
            assert persistent_count > 0, "Persistent namespace was affected"
            # Speculative should be cleaned (might have some if written after cleanup)
            assert speculative_count < 80, "Cleanup didn't work"

    def test_stat_consistency_during_updates(self):
        """Test stat() returns consistent metadata during concurrent updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServer(storage_path=Path(tmpdir))

            ref = "mcp://stat/test"
            inconsistencies = []

            def updater():
                """Update data and metadata."""
                client = MCPClient(server)
                for i in range(20):
                    data = f"Version_{i}".encode() * (100 + i * 10)
                    client.put(ref, data, ttl_s=3600 + i)
                    time.sleep(0.005)

            def stat_checker(checker_id: int):
                """Check stat consistency."""
                client = MCPClient(server)
                for _ in range(40):
                    meta = client.stat(ref)
                    if meta:
                        # If we can stat, we should be able to resolve
                        data = client.resolve(ref)
                        if data is None:
                            inconsistencies.append(
                                f"Checker {checker_id}: stat exists but resolve failed"
                            )
                        elif len(data) != meta["size_bytes"]:
                            inconsistencies.append(f"Checker {checker_id}: size mismatch")
                        elif hashlib.sha256(data).hexdigest() != meta["sha256"]:
                            inconsistencies.append(f"Checker {checker_id}: SHA mismatch")

                    time.sleep(0.003)

            # Run updater and checkers concurrently
            threads = []

            updater_thread = threading.Thread(target=updater)
            threads.append(updater_thread)
            updater_thread.start()

            for i in range(4):
                checker_thread = threading.Thread(target=stat_checker, args=(i,))
                threads.append(checker_thread)
                checker_thread.start()

            for t in threads:
                t.join()

            # Should have no inconsistencies
            assert len(inconsistencies) == 0, f"Stat inconsistencies: {inconsistencies[:5]}"
            print("✓ Stat metadata remained consistent during updates")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
