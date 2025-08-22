#!/usr/bin/env python3
"""Simple MCP deref latency benchmark."""

import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from mcp.server import MCPServer
from mcp.client import MCPClient


def run_benchmark():
    """Run a simple deref benchmark."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create server with filesystem storage
        server = MCPServer(storage_path=Path(tmpdir))
        client = MCPClient(server)
        
        # Prepare test data
        print("Preparing test data...")
        test_refs = []
        for i in range(100):
            ref = f"mcp://perf/item_{i}"
            size = 1000 + (i * 100)  # 1KB to 10KB
            data = b"x" * size
            client.put(ref, data, ttl_s=3600)
            test_refs.append(ref)
        
        # Force persist
        server._persist_to_disk()
        
        # Warmup
        print("Warming up (100 ops)...")
        for _ in range(100):
            client.resolve(test_refs[0])
        
        # Benchmark
        N = 2000
        print(f"Running {N} deref operations...")
        times_ms = []
        
        for i in range(N):
            ref = test_refs[i % len(test_refs)]
            
            start_ns = time.perf_counter_ns()
            data = client.resolve(ref)
            end_ns = time.perf_counter_ns()
            
            if data is None:
                print(f"ERROR: Failed to resolve {ref}")
                continue
                
            times_ms.append((end_ns - start_ns) / 1e6)
        
        # Calculate stats
        times_ms.sort()
        p50 = times_ms[int(len(times_ms) * 0.50)]
        p95 = times_ms[int(len(times_ms) * 0.95)]
        p99 = times_ms[int(len(times_ms) * 0.99)]
        
        print(f"\nResults:")
        print(f"  Samples: {len(times_ms)}")
        print(f"  Mean: {statistics.mean(times_ms):.3f}ms")
        print(f"  P50: {p50:.3f}ms")
        print(f"  P95: {p95:.3f}ms")
        print(f"  P99: {p99:.3f}ms")
        
        # Save metrics
        metrics = {
            "mcp_deref_ms_p50": round(p50, 3),
            "mcp_deref_ms_p95": round(p95, 3),
            "mcp_deref_ms_p99": round(p99, 3),
            "mcp_deref_ms_mean": round(statistics.mean(times_ms), 3),
            "mcp_deref_ms_stdev": round(statistics.stdev(times_ms), 3),
            "mcp_deref_samples": len(times_ms),
            "mcp_deref_warmup": 100,
            "os_fingerprint": f"{platform.system()}-{platform.release()}-{platform.machine()}-Python{platform.python_version()}",
            "storage_backend": "memory_with_disk_persistence",
        }
        
        with open("runs/mcp/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to runs/mcp/metrics.json")
        
        # Verify p95 < 50ms requirement
        assert p95 < 50.0, f"P95 {p95:.3f}ms exceeds 50ms requirement"
        print(f"✓ P95 requirement met ({p95:.3f}ms < 50ms)")


if __name__ == "__main__":
    run_benchmark()