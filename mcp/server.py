"""MCP Server with TTL support and speculative namespace cleanup."""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class AnchorEntry:
    """An anchor entry with TTL and metadata."""

    ref: str
    data: bytes
    ttl_s: int
    created_at: float  # Wall time for human-readable metadata
    namespace: str = "default"
    sha256: str = field(init=False)
    created_at_monotonic: float = field(init=False)  # Monotonic for TTL calculations

    def __post_init__(self):
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        # Store both monotonic and wall time
        if not hasattr(self, 'created_at_monotonic'):
            self.created_at_monotonic = time.monotonic()

    def is_expired(self) -> bool:
        """Check if this entry has expired using monotonic time."""
        if self.ttl_s <= 0:  # Permanent if TTL <= 0
            return False
        return time.monotonic() > (self.created_at_monotonic + self.ttl_s)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class MCPServer:
    """MCP Server for managing artifact anchors with TTL and namespaces."""

    # Default TTLs by content type
    DEFAULT_TTLS = {
        "logs": 24 * 3600,  # 24 hours
        "diffs": 7 * 24 * 3600,  # 7 days
        "repo": -1,  # Permanent (pinned by SHA)
        "default": 3600,  # 1 hour
    }

    def __init__(self, storage_path: Optional[Path] = None, max_size_mb: int = 1024):
        """Initialize MCP server.

        Args:
            storage_path: Path for persistent storage (None = memory only)
            max_size_mb: Maximum total storage size in MB
        """
        self.storage_path = storage_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._anchors: Dict[str, AnchorEntry] = {}
        self._lock = Lock()
        self._cleanup_task = None
        self._running = False

        # Stats for monitoring
        self._stats = {
            "puts": 0,
            "resolves": 0,
            "hits": 0,
            "misses": 0,
            "expired": 0,
            "rollbacks": 0,
            "bytes_stored": 0,
        }

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self):
        """Load persisted anchors from disk."""
        if not self.storage_path:
            return

        metadata_file = self.storage_path / "metadata.json"
        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for ref, info in metadata.items():
                data_file = self.storage_path / info["sha256"]
                if data_file.exists():
                    with open(data_file, "rb") as f:
                        data = f.read()
                    entry = AnchorEntry(
                        ref=ref,
                        data=data,
                        ttl_s=info["ttl_s"],
                        created_at=info["created_at"],
                        namespace=info.get("namespace", "default"),
                    )
                    # Restore monotonic time if available, else compute from wall time
                    if "created_at_monotonic" in info:
                        entry.created_at_monotonic = info["created_at_monotonic"]
                    else:
                        # Estimate monotonic based on elapsed wall time
                        elapsed = time.time() - info["created_at"]
                        entry.created_at_monotonic = time.monotonic() - elapsed
                    
                    if not entry.is_expired():
                        self._anchors[ref] = entry
                        self._stats["bytes_stored"] += entry.size_bytes
        except Exception as e:
            logger.warning(f"Failed to load persisted anchors: {e}")

    def _persist_to_disk(self):
        """Persist current anchors to disk with atomic writes."""
        if not self.storage_path:
            return

        # Take snapshot under lock
        with self._lock:
            snapshot = {
                ref: entry for ref, entry in self._anchors.items() 
                if not entry.is_expired()
            }
        
        # Perform I/O outside lock
        metadata = {}
        for ref, entry in snapshot.items():
            # Write data file atomically
            data_file = self.storage_path / entry.sha256
            if not data_file.exists():
                self._atomic_write(data_file, entry.data)

            # Add to metadata
            metadata[ref] = {
                "sha256": entry.sha256,
                "ttl_s": entry.ttl_s,
                "created_at": entry.created_at,
                "created_at_monotonic": entry.created_at_monotonic,
                "namespace": entry.namespace,
            }

        # Write metadata atomically
        metadata_file = self.storage_path / "metadata.json"
        self._atomic_write(metadata_file, json.dumps(metadata, indent=2).encode())
    
    def _atomic_write(self, path: Path, data: bytes):
        """Write data atomically using temp file + fsync + rename."""
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=path.parent,
            delete=False
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        
        # Atomic rename
        tmp_path.rename(path)
        
        # Sync directory to ensure rename is persisted
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    
    def _infer_ttl_from_ref(self, ref: str) -> Optional[int]:
        """Infer TTL from ref using strict parsing.
        
        Args:
            ref: Reference like mcp://<type>/path
            
        Returns:
            TTL in seconds or None if invalid format
        """
        try:
            parsed = urlparse(ref)
            if parsed.scheme != 'mcp':
                return None
            
            # Extract type from path
            path_parts = parsed.path.lstrip('/').split('/')
            if not path_parts or not path_parts[0]:
                # Use netloc as type if no path (e.g., mcp://logs)
                ref_type = parsed.netloc
            else:
                # Use first path component as type
                ref_type = path_parts[0] if not parsed.netloc else parsed.netloc
            
            # Map type to TTL
            type_to_ttl = {
                'logs': self.DEFAULT_TTLS['logs'],
                'log': self.DEFAULT_TTLS['logs'],
                'diffs': self.DEFAULT_TTLS['diffs'],
                'diff': self.DEFAULT_TTLS['diffs'],
                'repo': self.DEFAULT_TTLS['repo'],
                'sha': self.DEFAULT_TTLS['repo'],
                'sha256': self.DEFAULT_TTLS['repo'],
            }
            
            return type_to_ttl.get(ref_type, self.DEFAULT_TTLS['default'])
            
        except Exception:
            return None

    def put(
        self, ref: str, data: bytes, ttl_s: Optional[int] = None, namespace: str = "default"
    ) -> Tuple[bool, str]:
        """Store data at the given reference.

        Args:
            ref: Reference key (e.g., "mcp://logs/1234")
            data: Binary data to store
            ttl_s: Time-to-live in seconds (None = use default)
            namespace: Namespace for grouping (e.g., "speculative")

        Returns:
            (success, message)
        """
        # Determine TTL using strict parsing
        if ttl_s is None:
            ttl_s = self._infer_ttl_from_ref(ref)
            if ttl_s is None:
                return False, f"Invalid ref format: {ref}. Expected mcp://<type>/..."

        with self._lock:
            # Check size limits
            new_size = len(data)
            current_size = self._stats["bytes_stored"]

            # If replacing, subtract old size
            if ref in self._anchors:
                current_size -= self._anchors[ref].size_bytes

            if current_size + new_size > self.max_size_bytes:
                return False, f"Storage limit exceeded ({self.max_size_bytes} bytes)"

            # Create entry with both wall and monotonic time
            entry = AnchorEntry(
                ref=ref, 
                data=data, 
                ttl_s=ttl_s, 
                created_at=time.time(),  # Wall time for metadata
                namespace=namespace
            )
            # Monotonic time is set in __post_init__

            # Update storage
            if ref in self._anchors:
                self._stats["bytes_stored"] -= self._anchors[ref].size_bytes
            self._anchors[ref] = entry
            self._stats["bytes_stored"] += new_size
            self._stats["puts"] += 1

            # Persist if configured
            self._persist_to_disk()

            return True, f"Stored {new_size} bytes at {ref} (TTL: {ttl_s}s)"

    def resolve(self, ref: str) -> Optional[bytes]:
        """Resolve a reference to its data.

        Args:
            ref: Reference key to resolve

        Returns:
            Data bytes if found and not expired, None otherwise
        """
        start_ns = time.perf_counter_ns()

        with self._lock:
            self._stats["resolves"] += 1

            if ref not in self._anchors:
                self._stats["misses"] += 1
                return None

            entry = self._anchors[ref]
            if entry.is_expired():
                # Clean up expired entry
                del self._anchors[ref]
                self._stats["bytes_stored"] -= entry.size_bytes
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1

            # Log deref time for p95 tracking
            deref_ms = (time.perf_counter_ns() - start_ns) / 1e6
            logger.debug(f"MCP deref {ref}: {deref_ms:.3f}ms")

            return entry.data

    def stat(self, ref: str) -> Optional[Dict]:
        """Get metadata about an anchor without resolving data.

        Args:
            ref: Reference key to check

        Returns:
            Metadata dict or None if not found/expired
        """
        with self._lock:
            if ref not in self._anchors:
                return None

            entry = self._anchors[ref]
            if entry.is_expired():
                return None

            return {
                "ref": ref,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
                "created_at": entry.created_at,
                "ttl_s": entry.ttl_s,
                "expires_at": entry.created_at + entry.ttl_s if entry.ttl_s > 0 else None,
                "namespace": entry.namespace,
            }

    def cleanup_namespace(self, namespace: str) -> int:
        """Remove all anchors in a namespace (for speculative rollback).

        Args:
            namespace: Namespace to clean up

        Returns:
            Number of anchors removed
        """
        with self._lock:
            to_remove = []
            bytes_freed = 0

            for ref, entry in self._anchors.items():
                if entry.namespace == namespace:
                    to_remove.append(ref)
                    bytes_freed += entry.size_bytes

            for ref in to_remove:
                del self._anchors[ref]

            self._stats["bytes_stored"] -= bytes_freed
            self._stats["rollbacks"] += len(to_remove)

            # Persist changes
            self._persist_to_disk()

            logger.info(
                f"Cleaned up {len(to_remove)} anchors in namespace '{namespace}' "
                f"({bytes_freed} bytes)"
            )

            return len(to_remove)

    async def cleanup_expired(self):
        """Background task to clean up expired anchors."""
        while self._running:
            try:
                with self._lock:
                    to_remove = []
                    bytes_freed = 0

                    for ref, entry in self._anchors.items():
                        if entry.is_expired():
                            to_remove.append(ref)
                            bytes_freed += entry.size_bytes

                    for ref in to_remove:
                        del self._anchors[ref]

                    if to_remove:
                        self._stats["bytes_stored"] -= bytes_freed
                        self._stats["expired"] += len(to_remove)
                        self._persist_to_disk()
                        logger.info(
                            f"Cleaned up {len(to_remove)} expired anchors " f"({bytes_freed} bytes)"
                        )

                # Run cleanup every 60 seconds
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

    async def start(self):
        """Start the MCP server and background tasks."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self.cleanup_expired())
        logger.info(f"MCP server started (max size: {self.max_size_bytes} bytes)")

    async def stop(self):
        """Stop the MCP server and cleanup."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Final persist
        self._persist_to_disk()
        logger.info("MCP server stopped")

    def get_stats(self) -> Dict:
        """Get server statistics."""
        with self._lock:
            return {
                **self._stats,
                "anchors_count": len(self._anchors),
                "namespaces": len(set(e.namespace for e in self._anchors.values())),
            }
