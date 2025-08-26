#!/usr/bin/env python3
"""Demonstrate PM < C bytes/solve with simulated large test outputs.

This demo shows how MCP anchoring reduces bytes on wire without
requiring full repository clones. For production, use real repos.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict
from proto import baseline_pb2
from mcp.server import MCPServer
from mcp.client import MCPClient


def generate_large_test_output(size_kb: float = 50) -> str:
    """Generate realistic pytest output of specified size."""
    lines = [
        "=" * 80,
        "test session starts",
        "=" * 80,
        "platform darwin -- Python 3.11.6, pytest-7.4.3",
        "collected 100 items",
        "",
    ]
    
    # Add test results with verbose output
    for i in range(20):
        lines.extend([
            f"\ntest_module.py::TestClass::test_case_{i:03d} ",
            f"PASSED" if i % 3 else "FAILED",
            "",
        ])
        
        # Add captured output to reach target size
        for j in range(int(size_kb * 50)):
            lines.append(f"  [LOG {j:04d}] Processing item {j}: status={'OK' if j%2 else 'WARN'}")
    
    lines.extend([
        "",
        "=" * 80,
        f"{'80 passed, 20 failed' if size_kb > 10 else '5 passed'}",
        "=" * 80,
    ])
    
    return "\n".join(lines)


def demo_c_arm(test_output: str) -> Dict:
    """Simulate C arm: inline everything."""
    # Create protobuf response with inline output
    response = baseline_pb2.TestResponse()
    response.passed = True
    response.output = test_output  # Inline the entire output
    response.duration_ms = 1234
    
    # Measure bytes
    serialized = response.SerializeToString()
    
    return {
        "arm": "C",
        "bytes_on_wire": len(serialized),
        "output_size": len(test_output),
        "method": "inline"
    }


def demo_pm_arm(test_output: str, threshold_kb: float = 1.0) -> Dict:
    """Simulate PM arm: anchor large outputs."""
    # Initialize MCP
    mcp_server = MCPServer()
    mcp_client = MCPClient(mcp_server)
    
    # Check if output should be anchored
    output_bytes = test_output.encode()
    threshold_bytes = int(threshold_kb * 1024)
    
    if len(output_bytes) > threshold_bytes:
        # Create MCP anchor
        sha256 = hashlib.sha256(output_bytes).hexdigest()[:16]
        ref = f"mcp://pm/{sha256}"
        
        # Store in MCP (simulated)
        success, msg = mcp_client.put(ref, output_bytes, ttl_s=86400)
        if not success:
            raise RuntimeError(f"MCP put failed: {msg}")
        
        # Response contains only reference
        response = baseline_pb2.TestResponse()
        response.passed = True
        response.output = ref  # Just the reference, not the data
        response.duration_ms = 1234
        
        method = "anchored"
    else:
        # Inline small outputs
        response = baseline_pb2.TestResponse()
        response.passed = True
        response.output = test_output
        response.duration_ms = 1234
        
        method = "inline"
    
    # Measure bytes
    serialized = response.SerializeToString()
    
    return {
        "arm": "PM",
        "bytes_on_wire": len(serialized),
        "output_size": len(test_output),
        "method": method
    }


def main():
    """Run demo comparison."""
    print("=== PM vs C Bytes Comparison Demo ===\n")
    
    # Test with different output sizes
    test_sizes_kb = [0.5, 2, 10, 50, 100]
    
    results = []
    for size_kb in test_sizes_kb:
        print(f"\nTest output size: {size_kb} KB")
        
        # Generate test output
        test_output = generate_large_test_output(size_kb)
        actual_size = len(test_output) / 1024
        print(f"  Actual size: {actual_size:.1f} KB")
        
        # Run both arms
        c_result = demo_c_arm(test_output)
        pm_result = demo_pm_arm(test_output, threshold_kb=1.0)
        
        # Calculate savings
        savings_bytes = c_result["bytes_on_wire"] - pm_result["bytes_on_wire"]
        savings_pct = 100 * savings_bytes / c_result["bytes_on_wire"] if c_result["bytes_on_wire"] > 0 else 0
        
        print(f"  C arm:  {c_result['bytes_on_wire']:,} bytes (inline)")
        print(f"  PM arm: {pm_result['bytes_on_wire']:,} bytes ({pm_result['method']})")
        print(f"  Savings: {savings_bytes:,} bytes ({savings_pct:.1f}%)")
        
        if pm_result["bytes_on_wire"] < c_result["bytes_on_wire"]:
            print(f"  ✅ PM < C")
        else:
            print(f"  ❌ PM >= C")
        
        results.append({
            "size_kb": size_kb,
            "c_bytes": c_result["bytes_on_wire"],
            "pm_bytes": pm_result["bytes_on_wire"],
            "savings_bytes": savings_bytes,
            "savings_pct": savings_pct,
            "pm_method": pm_result["method"]
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Average across all sizes
    avg_c = sum(r["c_bytes"] for r in results) / len(results)
    avg_pm = sum(r["pm_bytes"] for r in results) / len(results)
    avg_savings_pct = 100 * (avg_c - avg_pm) / avg_c if avg_c > 0 else 0
    
    print(f"\nAverage bytes/solve:")
    print(f"  C arm:  {avg_c:.0f} bytes")
    print(f"  PM arm: {avg_pm:.0f} bytes")
    print(f"  Savings: {avg_savings_pct:.1f}%")
    
    if avg_pm < avg_c:
        print(f"\n✅ PM < C on average (T1.2 acceptance criterion would be MET)")
    else:
        print(f"\n❌ PM >= C on average (T1.2 acceptance criterion NOT MET)")
    
    # When anchoring kicks in
    anchored = [r for r in results if r["pm_method"] == "anchored"]
    if anchored:
        print(f"\nAnchoring triggered for outputs > 1 KB:")
        for r in anchored:
            print(f"  {r['size_kb']} KB: {r['savings_pct']:.1f}% reduction")


if __name__ == "__main__":
    main()