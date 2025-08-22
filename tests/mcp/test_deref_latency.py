"""Microbenchmark for MCP deref latency (p95 < 50ms requirement)."""

import json
import statistics
import time
from typing import List

import pytest

from mcp.server import MCPServer
from mcp.client import MCPClient


class TestMCPDerefLatency:
    """Test MCP dereference latency performance."""
    
    def test_deref_p95_requirement(self):
        """Test that deref p95 < 50ms on local storage."""
        # Use filesystem-backed storage for realistic measurements
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            server = MCPServer(storage_path=storage_path)  # Enable filesystem storage
            client = MCPClient(server)
            
            # Number of samples (N >= 2000 as required by reviewer)
            N = 2000
            warmup_runs = 100
        
            # Prepare test data with varying sizes
            test_refs = []
            for i in range(100):
                ref = f"mcp://perf/item_{i}"
                # Vary data size: 100 bytes to 10KB
                size = 100 + (i * 100)
                data = b"x" * size
                client.put(ref, data, ttl_s=3600)
                test_refs.append(ref)
                
            # Force flush to disk
            server._persist_to_disk()
            
            # Warmup runs (excluded from metrics)
            print("\nWarming up...")
            for _ in range(warmup_runs):
                for ref in test_refs[:10]:  # Just use first 10 for warmup
                    client.resolve(ref)
                    
            # Clear any warmup timings
            client._deref_times_ms = []
            
            # Actual benchmark runs
            print(f"Running {N} deref operations...")
            deref_times_ms = []
            
            for run in range(N):
                # Pick a random-ish ref
                ref = test_refs[run % len(test_refs)]
                
                # Time the deref operation
                start_ns = time.perf_counter_ns()
                data = client.resolve(ref)
                end_ns = time.perf_counter_ns()
                
                assert data is not None, f"Failed to resolve {ref}"
                
                # Record timing in milliseconds
                deref_ms = (end_ns - start_ns) / 1e6
                deref_times_ms.append(deref_ms)
            
            # Calculate percentiles
            sorted_times = sorted(deref_times_ms)
            p50_idx = int(len(sorted_times) * 0.50)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            
            p50 = sorted_times[min(p50_idx, len(sorted_times) - 1)]
            p95 = sorted_times[min(p95_idx, len(sorted_times) - 1)]
            p99 = sorted_times[min(p99_idx, len(sorted_times) - 1)]
            
            mean = statistics.mean(deref_times_ms)
            stdev = statistics.stdev(deref_times_ms) if len(deref_times_ms) > 1 else 0
            
            # Print results
            print(f"\n=== MCP Deref Latency Results (Filesystem-backed) ===")
            print(f"Storage: {storage_path}")
            print(f"Samples: {N} (warmup: {warmup_runs} excluded)")
            print(f"Mean: {mean:.3f}ms")
            print(f"Stdev: {stdev:.3f}ms")
            print(f"Min: {min(deref_times_ms):.3f}ms")
            print(f"Max: {max(deref_times_ms):.3f}ms")
            print(f"P50: {p50:.3f}ms")
            print(f"P95: {p95:.3f}ms")
            print(f"P99: {p99:.3f}ms")
            
            # Generate metrics for evidence pack
            import platform
            metrics = {
                "mcp_deref_ms_p50": round(p50, 3),
                "mcp_deref_ms_p95": round(p95, 3),
                "mcp_deref_ms_p99": round(p99, 3),
                "mcp_deref_ms_mean": round(mean, 3),
                "mcp_deref_ms_stdev": round(stdev, 3),
                "mcp_deref_samples": N,
                "mcp_deref_warmup": warmup_runs,
                "os_fingerprint": f"{platform.system()}-{platform.release()}-{platform.machine()}-Python{platform.python_version()}",
                "storage_backend": "filesystem",
            }
            
            # Save metrics to file for evidence pack
            with open("mcp_deref_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
                
            # Assert p95 < 50ms requirement
            assert p95 < 50.0, f"Deref p95 {p95:.3f}ms exceeds 50ms limit"
            
            # Also check using client's built-in p95 calculation
            client_p95 = client.get_deref_p95()
            assert client_p95 is not None
            print(f"Client-reported P95: {client_p95:.3f}ms")
        
    def test_deref_with_different_sizes(self):
        """Test deref latency across different payload sizes."""
        server = MCPServer()
        client = MCPClient(server)
        
        # Test different sizes
        test_cases = [
            (100, "100B"),
            (1024, "1KB"),
            (10240, "10KB"),
            (102400, "100KB"),
            (262144, "256KB"),
        ]
        
        results = {}
        
        for size, label in test_cases:
            ref = f"mcp://size/{label}"
            data = b"S" * size
            client.put(ref, data)
            
            # Warmup
            for _ in range(5):
                client.resolve(ref)
                
            # Measure
            times = []
            for _ in range(100):
                start_ns = time.perf_counter_ns()
                resolved = client.resolve(ref)
                end_ns = time.perf_counter_ns()
                
                assert resolved == data
                times.append((end_ns - start_ns) / 1e6)
                
            # Calculate stats
            p95 = sorted(times)[int(len(times) * 0.95)]
            results[label] = {
                "size_bytes": size,
                "p50_ms": sorted(times)[int(len(times) * 0.50)],
                "p95_ms": p95,
                "mean_ms": statistics.mean(times),
            }
            
        # Print size-based results
        print("\n=== Deref Latency by Size ===")
        for label, stats in results.items():
            print(f"{label:>6}: P50={stats['p50_ms']:.3f}ms, "
                  f"P95={stats['p95_ms']:.3f}ms, "
                  f"Mean={stats['mean_ms']:.3f}ms")
            
        # All sizes should meet p95 < 50ms
        for label, stats in results.items():
            assert stats["p95_ms"] < 50.0, \
                f"Size {label} p95 {stats['p95_ms']:.3f}ms exceeds limit"
                
    def test_deref_under_load(self):
        """Test deref latency under concurrent load."""
        import threading
        
        server = MCPServer()
        
        # Pre-populate with data
        for i in range(200):
            ref = f"mcp://load/item_{i}"
            data = b"L" * (1000 + i * 10)
            server.put(ref, data)
            
        # Shared results
        all_times: List[float] = []
        lock = threading.Lock()
        
        def worker(worker_id: int):
            """Worker thread performing derefs."""
            client = MCPClient(server)
            local_times = []
            
            for i in range(50):
                ref = f"mcp://load/item_{(worker_id * 50 + i) % 200}"
                
                start_ns = time.perf_counter_ns()
                data = client.resolve(ref)
                end_ns = time.perf_counter_ns()
                
                if data:
                    deref_ms = (end_ns - start_ns) / 1e6
                    local_times.append(deref_ms)
                    
            with lock:
                all_times.extend(local_times)
                
        # Run concurrent workers
        threads = []
        num_workers = 10
        
        print(f"\nRunning {num_workers} concurrent workers...")
        start_time = time.time()
        
        for i in range(num_workers):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        elapsed = time.time() - start_time
        
        # Calculate stats
        sorted_times = sorted(all_times)
        p50 = sorted_times[int(len(sorted_times) * 0.50)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        
        print(f"Completed in {elapsed:.2f}s")
        print(f"Total operations: {len(all_times)}")
        print(f"Throughput: {len(all_times)/elapsed:.1f} ops/sec")
        print(f"Under load - P50: {p50:.3f}ms, P95: {p95:.3f}ms")
        
        # Even under load, should meet p95 < 50ms
        assert p95 < 50.0, f"Under load p95 {p95:.3f}ms exceeds limit"
        
    def test_deref_cache_effects(self):
        """Test cache effects on deref latency."""
        server = MCPServer()
        client = MCPClient(server)
        
        # Create a frequently accessed item
        hot_ref = "mcp://cache/hot"
        hot_data = b"HOT" * 100
        client.put(hot_ref, hot_data)
        
        # Create many cold items
        for i in range(1000):
            ref = f"mcp://cache/cold_{i}"
            data = f"COLD_{i}".encode() * 10
            client.put(ref, data)
            
        # Access pattern: 90% hot, 10% cold
        times_hot = []
        times_cold = []
        
        for i in range(500):
            if i % 10 == 0:
                # Cold access
                ref = f"mcp://cache/cold_{i}"
                start_ns = time.perf_counter_ns()
                client.resolve(ref)
                end_ns = time.perf_counter_ns()
                times_cold.append((end_ns - start_ns) / 1e6)
            else:
                # Hot access
                start_ns = time.perf_counter_ns()
                client.resolve(hot_ref)
                end_ns = time.perf_counter_ns()
                times_hot.append((end_ns - start_ns) / 1e6)
                
        # Calculate stats
        p95_hot = sorted(times_hot)[int(len(times_hot) * 0.95)]
        p95_cold = sorted(times_cold)[int(len(times_cold) * 0.95)]
        
        print("\n=== Cache Effects ===")
        print(f"Hot item P95: {p95_hot:.3f}ms (n={len(times_hot)})")
        print(f"Cold items P95: {p95_cold:.3f}ms (n={len(times_cold)})")
        
        # Both should meet requirements
        assert p95_hot < 50.0
        assert p95_cold < 50.0
        
        # Hot should generally be faster (cache effects)
        # This is informational, not a hard requirement
        if p95_hot < p95_cold:
            print("✓ Hot item shows cache benefit")
        

def main():
    """Run the microbenchmark and print results."""
    test = TestMCPDerefLatency()
    
    # Run main p95 test
    test.test_deref_p95_requirement()
    
    # Run size-based test
    test.test_deref_with_different_sizes()
    
    # Run load test
    test.test_deref_under_load()
    
    # Run cache test
    test.test_deref_cache_effects()
    
    print("\n✓ All MCP deref latency tests passed!")
    print("✓ P95 < 50ms requirement met")
    

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])