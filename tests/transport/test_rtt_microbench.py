#!/usr/bin/env python3
"""RTT microbenchmark for gRPC over UNIX domain sockets."""

import statistics
import time
from pathlib import Path
import tempfile
import os

import pytest
from transport.grpc_impl import GrpcTransport


def test_local_grpc_uds_rtt_p95():
    """Test that local gRPC UDS RTT p95 is under 10ms (goal) or 20ms (acceptable).
    
    This is a pure transport microbench with no application logic.
    """
    # Create temporary socket path
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "test.sock"
        
        # Create transport for Arm C (simplest protobuf)
        transport = GrpcTransport(str(socket_path), arm="C", seed=42)
        
        try:
            # Start server
            transport.start_server()
            time.sleep(0.1)  # Give server time to start
            
            # Connect client
            transport.connect_client()
            
            # Warm up (first few calls are slower)
            for _ in range(10):
                transport.ping(b"warmup")
            
            # Measure RTT for small messages
            samples = []
            for i in range(1000):
                # Use tiny protobuf ping
                data = f"ping_{i}".encode()
                
                t0 = time.perf_counter_ns()
                result = transport.ping(data)  # Simple echo
                t1 = time.perf_counter_ns()
                
                # Convert to milliseconds
                rtt_ms = (t1 - t0) / 1e6
                samples.append(rtt_ms)
            
            # Calculate statistics
            p50 = statistics.median(samples)
            p95 = statistics.quantiles(samples, n=100)[94]
            p99 = statistics.quantiles(samples, n=100)[98]
            mean = statistics.mean(samples)
            
            print(f"\nRTT Statistics (n={len(samples)}):")
            print(f"  Mean: {mean:.3f} ms")
            print(f"  P50:  {p50:.3f} ms")
            print(f"  P95:  {p95:.3f} ms")
            print(f"  P99:  {p99:.3f} ms")
            
            # Assert p95 is under acceptable limit
            assert p95 < 20.0, f"RTT p95 ({p95:.3f}ms) exceeds acceptable 20ms"
            
            # Check if we meet the goal
            if p95 < 10.0:
                print("✓ Goal achieved: p95 < 10ms")
            else:
                print("✗ Goal not met (p95 < 10ms), but within acceptable range")
            
        finally:
            transport.cleanup()


def test_rtt_with_various_payload_sizes():
    """Test RTT with different payload sizes."""
    sizes = [10, 100, 1024, 10240, 102400]  # 10B to 100KB
    
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "test.sock"
        transport = GrpcTransport(str(socket_path), arm="C", seed=42)
        
        try:
            transport.start_server()
            time.sleep(0.1)
            transport.connect_client()
            
            # Warmup
            for _ in range(10):
                transport.ping(b"warmup")
            
            results = {}
            for size in sizes:
                data = b"x" * size
                samples = []
                
                for _ in range(100):
                    t0 = time.perf_counter_ns()
                    result = transport.ping(data)
                    t1 = time.perf_counter_ns()
                    samples.append((t1 - t0) / 1e6)
                
                results[size] = {
                    "mean": statistics.mean(samples),
                    "p50": statistics.median(samples),
                    "p95": statistics.quantiles(samples, n=100)[94],
                }
            
            print("\nRTT vs Payload Size:")
            print("Size (B)  Mean (ms)  P50 (ms)  P95 (ms)")
            for size, stats in results.items():
                print(f"{size:8}  {stats['mean']:9.3f}  {stats['p50']:8.3f}  {stats['p95']:8.3f}")
            
            # All should be under 50ms even for 100KB
            for size, stats in results.items():
                assert stats["p95"] < 50.0, f"RTT p95 for {size}B exceeds 50ms"
                
        finally:
            transport.cleanup()


if __name__ == "__main__":
    # Run directly for quick testing
    test_local_grpc_uds_rtt_p95()
    test_rtt_with_various_payload_sizes()