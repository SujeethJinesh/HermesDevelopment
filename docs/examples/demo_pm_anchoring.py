#!/usr/bin/env python3
"""Demo script showing PM anchoring behavior."""

import sys
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.pm_arm import PMAnchorManager, PMMetrics


class MockMCPClient:
    """Mock MCP client for demo."""
    
    def __init__(self):
        self.stored = {}
    
    def put_if_absent(self, ref: str, data: bytes, ttl_s: int) -> None:
        if ref not in self.stored:
            self.stored[ref] = data
            print(f"  → Stored {len(data)} bytes at {ref} with TTL={ttl_s}s")
    
    def resolve_bytes(self, ref: str) -> bytes:
        return self.stored.get(ref, b"")


def main():
    """Demonstrate PM anchoring."""
    print("PM Anchoring Demonstration")
    print("=" * 60)
    
    # Setup
    client = MockMCPClient()
    metrics = PMMetrics()
    manager = PMAnchorManager(client, metrics)
    
    # Test 1: Small log (stays inline)
    print("\n1. Small log (<1KB):")
    small_log = b"Test passed.\n" * 10  # 130 bytes
    result, anchored = manager.maybe_anchor(small_log, "logs")
    print(f"  Size: {len(small_log)} bytes")
    print(f"  Anchored: {anchored}")
    print(f"  Result type: {type(result).__name__}")
    
    # Test 2: Large log (gets anchored)
    print("\n2. Large log (>1KB):")
    large_log = b"X" * 2000  # 2KB
    result, anchored = manager.maybe_anchor(large_log, "logs")
    print(f"  Size: {len(large_log)} bytes")
    print(f"  Anchored: {anchored}")
    if isinstance(result, str):
        print(f"  Result: {result}")
    
    # Test 3: Large patch (>4KB gets anchored)
    print("\n3. Medium patch (3KB, stays inline):")
    medium_patch = b"diff --git a/file.py b/file.py\n" + b"Y" * 3000
    result, anchored = manager.maybe_anchor(medium_patch, "patches")
    print(f"  Size: {len(medium_patch)} bytes")
    print(f"  Anchored: {anchored}")
    
    print("\n4. Large patch (5KB, gets anchored):")
    large_patch = b"diff --git a/file.py b/file.py\n" + b"Z" * 5000
    result, anchored = manager.maybe_anchor(large_patch, "patches")
    print(f"  Size: {len(large_patch)} bytes")
    print(f"  Anchored: {anchored}")
    if isinstance(result, str):
        print(f"  Result: {result}")
    
    # Show final metrics
    print("\n" + "=" * 60)
    print("PM Metrics Summary:")
    print(f"  Anchors created: {metrics.anchors_created}")
    print(f"  Bytes saved: {metrics.bytes_saved}")
    print(f"  Inline count: {metrics.inline_count}")
    print(f"  Anchor count: {metrics.anchor_count}")
    
    # Calculate savings
    if metrics.bytes_saved > 0:
        print(f"\n  → Saved {metrics.bytes_saved:,} bytes by anchoring")


if __name__ == "__main__":
    main()