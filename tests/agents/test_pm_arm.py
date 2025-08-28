"""Unit tests for PM arm benefit-aware anchoring."""

from __future__ import annotations

from agents.pm_arm import HARD_CAP, PMAnchorManager, PMMetrics


class MockMCPClient:
    """Mock MCP client for testing."""

    def __init__(self):
        self.put_calls = []
        self.stored_data = {}

    def put_if_absent(self, ref: str, data: bytes, ttl_s: int) -> None:
        """Mock put_if_absent implementation."""
        self.put_calls.append((ref, data, ttl_s))
        if ref not in self.stored_data:
            self.stored_data[ref] = data

    def resolve_bytes(self, ref: str) -> bytes:
        """Mock resolve_bytes implementation."""
        return self.stored_data.get(ref, b"")


class TestPMAnchorManager:
    """Test cases for PM anchor manager."""

    def test_none_payload_returns_empty_inline(self):
        """Test that None payload returns empty bytes inline."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        result, anchored = manager.maybe_anchor(None, "logs")

        assert result == b""
        assert anchored is False
        assert metrics.anchors_created == 0
        assert metrics.bytes_saved == 0
        assert metrics.inline_count == 0  # None doesn't count as inline
        assert len(client.put_calls) == 0

    def test_tiny_patch_stays_inline(self):
        """Test that tiny patches (≤500B) stay inline due to benefit analysis."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create a 300-byte patch
        patch_data = b"diff --git a/file.py b/file.py\n" + b"x" * 269
        assert len(patch_data) == 300

        result, anchored = manager.maybe_anchor(patch_data, "patches")

        assert result == patch_data
        assert anchored is False
        assert metrics.inline_count == 1
        assert metrics.anchors_created == 0
        assert len(client.put_calls) == 0

    def test_log_just_under_1kb_stays_inline(self):
        """Test that logs just under 1KB stay inline."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create 1023-byte log (just under 1KB)
        log_data = b"Log entry: " + b"x" * 1012
        assert len(log_data) == 1023

        result, anchored = manager.maybe_anchor(log_data, "logs")

        assert result == log_data
        assert anchored is False
        assert metrics.inline_count == 1
        assert metrics.anchors_created == 0

    def test_log_just_over_1kb_gets_anchored(self):
        """Test that logs just over 1KB get anchored."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create 1025-byte log (just over 1KB)
        log_data = b"Log entry: " + b"x" * 1014
        assert len(log_data) == 1025

        result, anchored = manager.maybe_anchor(log_data, "logs")

        assert isinstance(result, str)
        assert result.startswith("mcp://logs/")
        assert anchored is True
        assert metrics.anchor_count == 1
        assert metrics.anchors_created == 1
        assert metrics.bytes_saved > 0
        assert len(client.put_calls) == 1

        # Verify TTL was set correctly (24 hours for logs)
        _, _, ttl = client.put_calls[0]
        assert ttl == 24 * 3600

    def test_patch_under_4kb_stays_inline(self):
        """Test that patches under 4KB stay inline."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create 3KB patch
        patch_data = b"diff --git a/file.py b/file.py\n" + b"x" * 3041
        assert len(patch_data) == 3072  # 3KB

        result, anchored = manager.maybe_anchor(patch_data, "patches")

        assert result == patch_data
        assert anchored is False
        assert metrics.inline_count == 1

    def test_patch_over_4kb_gets_anchored(self):
        """Test that patches over 4KB get anchored."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create 5KB patch
        patch_data = b"diff --git a/file.py b/file.py\n" + b"x" * 5089
        assert len(patch_data) == 5120  # 5KB

        result, anchored = manager.maybe_anchor(patch_data, "patches")

        assert isinstance(result, str)
        assert result.startswith("mcp://patches/")
        assert anchored is True
        assert metrics.anchor_count == 1
        assert len(client.put_calls) == 1

        # Verify TTL (7 days for patches)
        _, _, ttl = client.put_calls[0]
        assert ttl == 7 * 24 * 3600

    def test_huge_payload_always_anchored(self):
        """Test that payloads >= 256KB are always anchored."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create exactly 256KB payload
        huge_data = b"x" * HARD_CAP
        assert len(huge_data) == 256 * 1024

        result, anchored = manager.maybe_anchor(huge_data, "logs")

        assert isinstance(result, str)
        assert result.startswith("mcp://logs/")
        assert anchored is True
        assert metrics.anchor_count == 1
        assert metrics.bytes_saved > 0
        assert len(client.put_calls) == 1

    def test_bytes_saved_calculation(self):
        """Test that bytes_saved is calculated correctly."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create 10KB log that will be anchored
        log_data = b"x" * 10240
        assert len(log_data) == 10240

        result, anchored = manager.maybe_anchor(log_data, "logs")

        assert anchored is True
        ref_len = len(result.encode("utf-8"))

        # bytes_saved = inline_len - ref_len
        expected_saved = 10240 - ref_len
        assert metrics.bytes_saved == expected_saved

    def test_anchor_ref_length_vs_inline_comparison(self):
        """Test that anchoring decision correctly compares ref length vs inline."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create data where ref would be longer than inline
        # MCP ref format: "mcp://KIND/16CHARHASH" = ~28-30 chars minimum
        short_data = b"x" * 25  # 25 bytes - shorter than ref

        result, anchored = manager.maybe_anchor(short_data, "logs")

        # Should stay inline even if above threshold, because ref is longer
        assert result == short_data
        assert anchored is False

    def test_put_if_absent_called_once_per_ref(self):
        """Test that put_if_absent is called exactly once per unique ref."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create same data twice
        data = b"x" * 5000

        # First call
        ref1, anchored1 = manager.maybe_anchor(data, "logs")
        assert anchored1 is True
        assert len(client.put_calls) == 1

        # Second call with same data
        ref2, anchored2 = manager.maybe_anchor(data, "logs")
        assert anchored2 is True
        assert ref1 == ref2  # Same ref for same data
        assert len(client.put_calls) == 2  # Called again (idempotent)

    def test_different_content_types_use_correct_ttls(self):
        """Test that different content types use their correct TTLs."""
        client = MockMCPClient()
        manager = PMAnchorManager(client)

        # Test logs (24 hours)
        log_data = b"x" * 2000
        manager.maybe_anchor(log_data, "logs")
        assert client.put_calls[-1][2] == 24 * 3600

        # Test diffs (7 days)
        diff_data = b"y" * 2000
        manager.maybe_anchor(diff_data, "diffs")
        assert client.put_calls[-1][2] == 7 * 24 * 3600

        # Test patches (7 days)
        patch_data = b"z" * 5000
        manager.maybe_anchor(patch_data, "patches")
        assert client.put_calls[-1][2] == 7 * 24 * 3600

    def test_custom_ttl_override(self):
        """Test that custom TTL overrides default."""
        client = MockMCPClient()
        manager = PMAnchorManager(client)

        data = b"x" * 2000
        custom_ttl = 3600  # 1 hour

        manager.maybe_anchor(data, "logs", ttl_s=custom_ttl)

        assert client.put_calls[-1][2] == custom_ttl

    def test_realistic_log_blob_anchoring(self):
        """Test anchoring of realistic 5-50KB log blob."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create realistic 25KB log
        log_lines = []
        for i in range(500):
            log_lines.append(f"2024-01-15 12:34:{i:02d} INFO Processing item {i}\n")
        log_data = "".join(log_lines).encode("utf-8")
        assert 5000 < len(log_data) < 50000  # In realistic range

        result, anchored = manager.maybe_anchor(log_data, "logs")

        assert anchored is True
        assert result.startswith("mcp://logs/")
        assert metrics.bytes_saved > 0
        assert metrics.anchor_count == 1

    def test_realistic_small_patch_stays_inline(self):
        """Test that realistic 200-500B patches stay inline."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Create realistic small patch
        patch = """diff --git a/src/main.py b/src/main.py
index abc123..def456 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ def process():
     for item in items:
-        result = compute(item)
+        result = compute_v2(item)
         output.append(result)
     return output
"""
        patch_data = patch.encode("utf-8")
        assert 200 <= len(patch_data) <= 500

        result, anchored = manager.maybe_anchor(patch_data, "patches")

        assert result == patch_data
        assert anchored is False
        assert metrics.inline_count == 1
        assert metrics.bytes_saved == 0

    def test_metrics_accumulation(self):
        """Test that metrics accumulate correctly across multiple operations."""
        client = MockMCPClient()
        metrics = PMMetrics()
        manager = PMAnchorManager(client, metrics)

        # Inline small patch
        manager.maybe_anchor(b"x" * 100, "patches")
        assert metrics.inline_count == 1
        assert metrics.anchor_count == 0

        # Anchor large log
        manager.maybe_anchor(b"y" * 2000, "logs")
        assert metrics.inline_count == 1
        assert metrics.anchor_count == 1

        # Inline another small item
        manager.maybe_anchor(b"z" * 200, "diffs")
        assert metrics.inline_count == 2
        assert metrics.anchor_count == 1

        # Anchor another large item
        manager.maybe_anchor(b"w" * 10000, "logs")
        assert metrics.inline_count == 2
        assert metrics.anchor_count == 2
        assert metrics.anchors_created == 2

        # Bytes saved should be positive
        assert metrics.bytes_saved > 0
