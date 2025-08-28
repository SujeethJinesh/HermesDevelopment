"""MCP Client for interacting with the MCP server."""

import logging
import time
from typing import Dict, Optional, Tuple

from mcp.server import MCPServer

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for MCP operations with performance tracking."""

    def __init__(self, server: MCPServer):
        """Initialize MCP client.

        Args:
            server: MCP server instance to connect to
        """
        self.server = server
        self._deref_times_ms = []  # For p95 tracking

    def put(
        self, ref: str, data: bytes, ttl_s: Optional[int] = None, namespace: str = "default"
    ) -> Tuple[bool, str]:
        """Store data at the given reference.

        Args:
            ref: Reference key (e.g., "mcp://logs/1234")
            data: Binary data to store
            ttl_s: Time-to-live in seconds
            namespace: Namespace for grouping

        Returns:
            (success, message)
        """
        return self.server.put(ref, data, ttl_s, namespace)

    def resolve(self, ref: str) -> Optional[bytes]:
        """Resolve a reference to its data with timing.

        Args:
            ref: Reference key to resolve

        Returns:
            Data bytes if found, None otherwise
        """
        start_ns = time.perf_counter_ns()

        try:
            data = self.server.resolve(ref)

            # Track timing for p95 calculation
            deref_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._deref_times_ms.append(deref_ms)

            # Keep only last 1000 measurements
            if len(self._deref_times_ms) > 1000:
                self._deref_times_ms = self._deref_times_ms[-1000:]

            return data

        except Exception as e:
            logger.error(f"Failed to resolve {ref}: {e}")
            return None
    
    def resolve_bytes(self, ref: str) -> bytes:
        """Strict resolution that always returns bytes or raises.
        
        Args:
            ref: Reference key to resolve
            
        Returns:
            Raw bytes
            
        Raises:
            RuntimeError: If resolution fails
            TypeError: If data is not bytes
        """
        data = self.resolve(ref)
        
        # Handle tuple return for backward compatibility
        if isinstance(data, tuple):
            ok, payload = data
            if not ok:
                raise RuntimeError(f"MCP resolve failed: {ref}")
            data = payload
        
        # Ensure we have data
        if data is None:
            raise RuntimeError(f"MCP resolve returned None: {ref}")
        
        # Must be bytes
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"MCP client must return raw bytes, got {type(data)}")
        
        return bytes(data)

    def stat(self, ref: str) -> Optional[Dict]:
        """Get metadata about an anchor.

        Args:
            ref: Reference key to check

        Returns:
            Metadata dict or None if not found
        """
        return self.server.stat(ref)

    def cleanup_namespace(self, namespace: str) -> int:
        """Clean up all anchors in a namespace.

        Args:
            namespace: Namespace to clean up

        Returns:
            Number of anchors removed
        """
        return self.server.cleanup_namespace(namespace)

    def get_deref_p95(self) -> Optional[float]:
        """Calculate p95 deref time in milliseconds.

        Returns:
            P95 time in ms, or None if no data
        """
        if not self._deref_times_ms:
            return None

        sorted_times = sorted(self._deref_times_ms)
        p95_idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(p95_idx, len(sorted_times) - 1)]

    def get_stats(self) -> Dict:
        """Get client and server statistics.

        Returns:
            Combined statistics dict
        """
        stats = self.server.get_stats()

        # Add client-side metrics
        if self._deref_times_ms:
            sorted_times = sorted(self._deref_times_ms)
            p50_idx = int(len(sorted_times) * 0.50)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)

            stats.update(
                {
                    "deref_p50_ms": sorted_times[min(p50_idx, len(sorted_times) - 1)],
                    "deref_p95_ms": sorted_times[min(p95_idx, len(sorted_times) - 1)],
                    "deref_p99_ms": sorted_times[min(p99_idx, len(sorted_times) - 1)],
                    "deref_samples": len(self._deref_times_ms),
                }
            )

        return stats
