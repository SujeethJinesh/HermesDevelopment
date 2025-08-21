#!/usr/bin/env python3
"""RTT microbenchmark for gRPC transport over UNIX domain sockets."""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proto import baseline_pb2
from transport.grpc_impl import GrpcTransport


class TestTransportRTT:
    """Test transport RTT meets requirements."""

    def test_grpc_uds_rtt_p95(self):
        """Test gRPC over UDS has RTT p95 < 10ms."""
        # Create temp directory for socket
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "test.sock"

            # Create transport
            transport = GrpcTransport(str(socket_path), arm="C", seed=42)

            try:
                # Start server
                transport.start_server()
                time.sleep(0.2)  # Give server time to fully start

                # Connect client
                transport.connect_client()

                # Warm up with 10 calls
                for i in range(10):
                    req = baseline_pb2.PlanRequest(
                        task_id=f"warmup-{i}",
                        repo="test",
                        file_path="test.py",
                        test_name="test",
                        description="warmup",
                        seed=i,
                    )
                    transport.call_agent(
                        f"warmup-{i}",
                        "planner",
                        req.SerializeToString(),
                        "application/x-protobuf",
                        "warmup",
                    )

                # Measure RTT for 1500 calls
                rtts = []
                num_calls = 1500

                for i in range(num_calls):
                    # Create minimal request
                    req = baseline_pb2.PlanRequest(
                        task_id=f"bench-{i}",
                        repo="test",
                        file_path="test.py",
                        test_name="test",
                        description="benchmark",
                        seed=i,
                    )

                    # Measure RTT
                    _, rtt_ms = transport.call_agent(
                        f"bench-{i}",
                        "planner",
                        req.SerializeToString(),
                        "application/x-protobuf",
                        f"trace-{i}",
                    )

                    rtts.append(rtt_ms)

                # Calculate statistics
                rtts_sorted = sorted(rtts)
                p50 = statistics.median(rtts)
                p95_idx = int(num_calls * 0.95)
                p95 = rtts_sorted[p95_idx]
                p99_idx = int(num_calls * 0.99)
                p99 = rtts_sorted[p99_idx]
                mean = statistics.mean(rtts)

                print(f"\nRTT Statistics (n={num_calls}):")
                print(f"  Mean: {mean:.3f} ms")
                print(f"  P50:  {p50:.3f} ms")
                print(f"  P95:  {p95:.3f} ms")
                print(f"  P99:  {p99:.3f} ms")
                print(f"  Min:  {min(rtts):.3f} ms")
                print(f"  Max:  {max(rtts):.3f} ms")

                # Assert p95 < 10ms
                assert p95 < 10.0, f"RTT p95 {p95:.3f}ms exceeds 10ms limit"

                # Save RTT data
                output_dir = Path("runs/test")
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / "transport_rtts_test.jsonl", "w") as f:
                    for i, rtt in enumerate(rtts):
                        f.write(json.dumps({"call_id": i, "rtt_ms": rtt}) + "\n")

            finally:
                transport.stop()

    def test_json_vs_protobuf_rtt(self):
        """Compare RTT for JSON (Arm A) vs Protobuf (Arm C)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test both arms
            results = {}

            for arm in ["A", "C"]:
                socket_path = Path(tmpdir) / f"test_{arm}.sock"
                transport = GrpcTransport(str(socket_path), arm=arm, seed=42)

                try:
                    transport.start_server()
                    time.sleep(0.2)
                    transport.connect_client()

                    # Warm up
                    for i in range(10):
                        if arm == "A":
                            payload = json.dumps(
                                {"task_id": f"warmup-{i}", "file_path": "test.py"}
                            ).encode("utf-8")
                            content_type = "application/json"
                        else:
                            req = baseline_pb2.PlanRequest(
                                task_id=f"warmup-{i}",
                                repo="test",
                                file_path="test.py",
                                test_name="test",
                                description="warmup",
                                seed=i,
                            )
                            payload = req.SerializeToString()
                            content_type = "application/x-protobuf"

                        transport.call_agent(
                            f"warmup-{i}", "planner", payload, content_type, "warmup"
                        )

                    # Measure 500 calls
                    rtts = []
                    for i in range(500):
                        if arm == "A":
                            payload = json.dumps(
                                {
                                    "task_id": f"bench-{i}",
                                    "file_path": "test.py",
                                    "description": "benchmark",
                                }
                            ).encode("utf-8")
                            content_type = "application/json"
                        else:
                            req = baseline_pb2.PlanRequest(
                                task_id=f"bench-{i}",
                                repo="test",
                                file_path="test.py",
                                test_name="test",
                                description="benchmark",
                                seed=i,
                            )
                            payload = req.SerializeToString()
                            content_type = "application/x-protobuf"

                        _, rtt_ms = transport.call_agent(
                            f"bench-{i}", "planner", payload, content_type, f"trace-{i}"
                        )
                        rtts.append(rtt_ms)

                    results[arm] = {
                        "mean": statistics.mean(rtts),
                        "p50": statistics.median(rtts),
                        "p95": sorted(rtts)[int(len(rtts) * 0.95)],
                    }

                finally:
                    transport.stop()

            # Print comparison
            print("\nArm A (JSON) vs Arm C (Protobuf) RTT:")
            for arm in ["A", "C"]:
                r = results[arm]
                print(
                    f"  Arm {arm}: mean={r['mean']:.3f}ms, "
                    f"p50={r['p50']:.3f}ms, p95={r['p95']:.3f}ms"
                )

            # Both should be under 10ms p95
            assert results["A"]["p95"] < 10.0
            assert results["C"]["p95"] < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
