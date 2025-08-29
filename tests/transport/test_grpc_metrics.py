"""Tests for gRPC transport metrics tracking."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from transport.grpc_impl import GrpcTransport, TransportMetrics


class TestTransportMetrics:
    """Test cases for transport metrics collection."""

    def test_metrics_recording(self):
        """Test basic metrics recording."""
        metrics = TransportMetrics()
        
        # Record some data
        metrics.record_bytes_out(100)
        metrics.record_bytes_out(200)
        metrics.record_bytes_in(150)
        metrics.record_message_path(5.0)
        metrics.record_message_path(10.0)
        metrics.record_message_path(15.0)
        
        # Get stats
        stats = metrics.get_stats()
        
        assert stats["bytes_out_total"] == 300
        assert stats["bytes_in_total"] == 150
        assert stats["messages_sent"] == 2
        assert stats["messages_received"] == 1
        assert stats["message_path_ms_p50"] == 10.0
        assert stats["message_path_ms_p95"] >= 15.0

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        metrics = TransportMetrics()
        
        # Add 100 measurements (0-99ms)
        for i in range(100):
            metrics.record_message_path(float(i))
        
        stats = metrics.get_stats()
        
        # Check percentiles
        assert stats["message_path_ms_p50"] == 50.0
        assert stats["message_path_ms_p95"] == 95.0
        assert stats["message_path_ms_p99"] == 99.0

    def test_memory_limit(self):
        """Test that metrics list is capped at 1000 entries."""
        metrics = TransportMetrics()
        
        # Add 1500 measurements
        for i in range(1500):
            metrics.record_message_path(float(i))
        
        # Should only keep last 1000
        assert len(metrics.message_path_ms_list) == 1000
        
        # First value should be 500 (indices 0-499 dropped)
        assert min(metrics.message_path_ms_list) == 500.0

    def test_grpc_transport_metrics(self):
        """Test metrics collection in gRPC transport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "test.sock"
            
            # Create transport
            transport = GrpcTransport(str(socket_path), arm="C", seed=42)
            
            # Start server
            transport.start_server()
            
            # Connect client
            transport.connect_client()
            
            # Make a few calls
            for i in range(3):
                result, rtt_ms = transport.call_agent(
                    task_id=f"test_{i}",
                    role="planner",
                    payload=b'{"task": "test"}',
                    content_type="application/json"
                )
                
                # Check result
                assert result.ok
                assert result.bytes_in > 0
                assert result.bytes_out > 0
                assert result.message_path_ms >= 0
                assert rtt_ms > 0
            
            # Get metrics
            metrics = transport.get_metrics()
            
            # Check server-side metrics
            assert metrics["messages_received"] >= 3
            assert metrics["bytes_in_total"] > 0
            assert metrics["bytes_out_total"] > 0
            assert metrics["message_path_ms_p50"] is not None
            assert metrics["message_path_ms_p95"] is not None
            
            # Cleanup
            transport.stop()

    def test_arm_c_protobuf_metrics(self):
        """Test metrics for Arm C (Protobuf) messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "test.sock"
            
            # Create transport for Arm C
            transport = GrpcTransport(str(socket_path), arm="C", seed=42)
            
            # Start server
            transport.start_server()
            
            # Connect client
            transport.connect_client()
            
            # Create protobuf payload
            from proto import baseline_pb2
            
            plan_req = baseline_pb2.PlanRequest(
                task_id="test_proto",
                repo="test/repo",
                file_path="test.py",
                test_name="test_function",
                description="Test task"
            )
            
            payload = plan_req.SerializeToString()
            
            # Make call
            result, rtt_ms = transport.call_agent(
                task_id="test_proto",
                role="planner",
                payload=payload,
                content_type="application/x-protobuf"
            )
            
            # Check result
            assert result.ok
            assert result.content_type == "application/x-protobuf"
            
            # Parse response
            plan_resp = baseline_pb2.PlanResponse()
            plan_resp.ParseFromString(result.payload)
            
            # Check metrics in result
            assert result.bytes_in == len(payload)
            assert result.bytes_out == len(result.payload)
            assert result.message_path_ms >= 0
            
            # Get transport metrics
            metrics = transport.get_metrics()
            assert metrics["messages_received"] >= 1
            assert metrics["message_path_ms_p50"] is not None
            
            # Cleanup
            transport.stop()

    def test_message_path_p95_threshold(self):
        """Test that message path p95 is measurable and reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "test.sock"
            
            transport = GrpcTransport(str(socket_path), arm="C", seed=42)
            transport.start_server()
            transport.connect_client()
            
            # Make 100 calls to get good p95
            rtts = []
            for i in range(100):
                result, rtt_ms = transport.call_agent(
                    task_id=f"perf_{i}",
                    role="planner",
                    payload=b'{"task": "perf test"}',
                    content_type="application/json"
                )
                rtts.append(result.message_path_ms)
            
            # Get metrics
            metrics = transport.get_metrics()
            
            # Check p95 is reasonable (< 20ms for local)
            p95 = metrics["message_path_ms_p95"]
            assert p95 is not None
            assert p95 < 20.0  # Should be fast on local socket
            
            # p95 should be >= p50
            assert metrics["message_path_ms_p95"] >= metrics["message_path_ms_p50"]
            
            transport.stop()