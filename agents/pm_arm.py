"""PM Arm: Protobuf + MCP benefit-aware anchoring implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

# Hard capacity threshold - always anchor if data >= this size
HARD_CAP = 256 * 1024  # 256 KB

# Default thresholds for benefit-aware anchoring
DEFAULT_THRESHOLDS = {
    "logs": 1024,  # 1 KB for logs
    "diffs": 1024,  # 1 KB for diffs
    "patches": 4096,  # 4 KB for patches
}

# TTL values per content type
DEFAULT_TTLS = {
    "logs": 24 * 3600,  # 24 hours for logs
    "diffs": 7 * 24 * 3600,  # 7 days for diffs
    "patches": 7 * 24 * 3600,  # 7 days for patches
}


@dataclass
class PMMetrics:
    """Metrics for PM arm anchoring operations."""

    anchors_created: int = 0
    bytes_saved: int = 0
    inline_count: int = 0
    anchor_count: int = 0


class PMAnchorManager:
    """Manages benefit-aware MCP anchoring for PM arm."""

    def __init__(self, mcp_client, metrics: Optional[PMMetrics] = None):
        """Initialize PM anchor manager.

        Args:
            mcp_client: MCP client for storage operations
            metrics: Optional metrics object for tracking stats
        """
        self.mcp_client = mcp_client
        self.metrics = metrics or PMMetrics()
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        self.ttls = DEFAULT_TTLS.copy()

    def maybe_anchor(
        self, data: bytes, kind: str, ttl_s: Optional[int] = None
    ) -> Tuple[str | bytes, bool]:
        """Determine whether to anchor data or keep inline based on benefit analysis.

        Args:
            data: The data to potentially anchor
            kind: Type of content (logs, diffs, patches)
            ttl_s: Optional TTL in seconds (uses default if not specified)

        Returns:
            Tuple of (reference_or_data, was_anchored)
            - If anchored: (mcp://reference, True)
            - If inline: (original_data, False)
        """
        # Handle None/empty data
        if data is None:
            return b"", False

        inline_len = len(data)

        # Always anchor if data exceeds hard capacity
        if inline_len >= HARD_CAP:
            ref = self._create_and_store_anchor(data, kind, ttl_s)
            return ref, True

        # Calculate potential reference
        sha16 = hashlib.sha256(data).hexdigest()[:16]
        ref = f"mcp://{kind}/{sha16}"
        ref_len = len(ref.encode("utf-8"))

        # Get threshold for this content type
        threshold = self.thresholds.get(kind, 1024)

        # Benefit-aware decision:
        # 1. Must be above threshold for this content type
        # 2. Reference must be smaller than inline data
        if inline_len >= threshold and ref_len < inline_len:
            # Store and return reference
            if ttl_s is None:
                ttl_s = self.ttls.get(kind, 24 * 3600)

            self.mcp_client.put_if_absent(ref, data, ttl_s=ttl_s)

            # Update metrics
            self.metrics.anchors_created += 1
            self.metrics.anchor_count += 1
            self.metrics.bytes_saved += inline_len - ref_len

            return ref, True

        # Keep inline - either below threshold or no benefit
        self.metrics.inline_count += 1
        return data, False

    def _create_and_store_anchor(self, data: bytes, kind: str, ttl_s: Optional[int] = None) -> str:
        """Create and store an anchor for data that must be anchored.

        Args:
            data: Data to anchor
            kind: Content type
            ttl_s: Optional TTL

        Returns:
            MCP reference string
        """
        sha16 = hashlib.sha256(data).hexdigest()[:16]
        ref = f"mcp://{kind}/{sha16}"

        if ttl_s is None:
            ttl_s = self.ttls.get(kind, 24 * 3600)

        self.mcp_client.put_if_absent(ref, data, ttl_s=ttl_s)

        # Update metrics
        ref_len = len(ref.encode("utf-8"))
        self.metrics.anchors_created += 1
        self.metrics.anchor_count += 1
        self.metrics.bytes_saved += len(data) - ref_len

        return ref

    def set_threshold(self, kind: str, threshold: int) -> None:
        """Update threshold for a content type.

        Args:
            kind: Content type
            threshold: New threshold in bytes
        """
        self.thresholds[kind] = threshold

    def set_ttl(self, kind: str, ttl_s: int) -> None:
        """Update TTL for a content type.

        Args:
            kind: Content type
            ttl_s: New TTL in seconds
        """
        self.ttls[kind] = ttl_s
