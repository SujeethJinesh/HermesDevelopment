"""MCP Server with TTL support and speculative namespace cleanup."""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
from threading import Lock
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnchorEntry:
    """An anchor entry with TTL and metadata."""
    ref: str
    data: bytes
    ttl_s: int
    created_at: float
    namespace: str = "default"
    sha256: str = field(init=False)
    
    def __post_init__(self):
        self.sha256 = hashlib.sha256(self.data).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl_s <= 0:  # Permanent if TTL <= 0
            return False
        return time.time() > (self.created_at + self.ttl_s)
    
    @property
    def size_bytes(self) -> int:
        return len(self.data)


class MCPServer:
    """MCP Server for managing artifact anchors with TTL and namespaces."""
    
    # Default TTLs by content type
    DEFAULT_TTLS = {
        "logs": 24 * 3600,     # 24 hours
        "diffs": 7 * 24 * 3600, # 7 days  
        "repo": -1,            # Permanent (pinned by SHA)
        "default": 3600        # 1 hour
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
            "bytes_stored": 0
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
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for ref, info in metadata.items():
                data_file = self.storage_path / info['sha256']
                if data_file.exists():
                    with open(data_file, 'rb') as f:
                        data = f.read()
                    entry = AnchorEntry(
                        ref=ref,
                        data=data,
                        ttl_s=info['ttl_s'],
                        created_at=info['created_at'],
                        namespace=info.get('namespace', 'default')
                    )
                    if not entry.is_expired():
                        self._anchors[ref] = entry
                        self._stats['bytes_stored'] += entry.size_bytes
        except Exception as e:
            logger.warning(f"Failed to load persisted anchors: {e}")
    
    def _persist_to_disk(self):
        """Persist current anchors to disk."""
        if not self.storage_path:
            return
            
        metadata = {}
        for ref, entry in self._anchors.items():
            if not entry.is_expired():
                # Write data file
                data_file = self.storage_path / entry.sha256
                if not data_file.exists():
                    with open(data_file, 'wb') as f:
                        f.write(entry.data)
                
                # Add to metadata
                metadata[ref] = {
                    'sha256': entry.sha256,
                    'ttl_s': entry.ttl_s,
                    'created_at': entry.created_at,
                    'namespace': entry.namespace
                }
        
        # Write metadata
        metadata_file = self.storage_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def put(self, ref: str, data: bytes, ttl_s: Optional[int] = None,
            namespace: str = "default") -> Tuple[bool, str]:
        """Store data at the given reference.
        
        Args:
            ref: Reference key (e.g., "mcp://logs/1234")
            data: Binary data to store
            ttl_s: Time-to-live in seconds (None = use default)
            namespace: Namespace for grouping (e.g., "speculative")
        
        Returns:
            (success, message)
        """
        # Determine TTL
        if ttl_s is None:
            # Infer from ref pattern
            if "logs" in ref:
                ttl_s = self.DEFAULT_TTLS["logs"]
            elif "diff" in ref:
                ttl_s = self.DEFAULT_TTLS["diffs"]
            elif "repo" in ref or "sha" in ref:
                ttl_s = self.DEFAULT_TTLS["repo"]
            else:
                ttl_s = self.DEFAULT_TTLS["default"]
        
        with self._lock:
            # Check size limits
            new_size = len(data)
            current_size = self._stats['bytes_stored']
            
            # If replacing, subtract old size
            if ref in self._anchors:
                current_size -= self._anchors[ref].size_bytes
            
            if current_size + new_size > self.max_size_bytes:
                return False, f"Storage limit exceeded ({self.max_size_bytes} bytes)"
            
            # Create entry
            entry = AnchorEntry(
                ref=ref,
                data=data,
                ttl_s=ttl_s,
                created_at=time.time(),
                namespace=namespace
            )
            
            # Update storage
            if ref in self._anchors:
                self._stats['bytes_stored'] -= self._anchors[ref].size_bytes
            self._anchors[ref] = entry
            self._stats['bytes_stored'] += new_size
            self._stats['puts'] += 1
            
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
            self._stats['resolves'] += 1
            
            if ref not in self._anchors:
                self._stats['misses'] += 1
                return None
            
            entry = self._anchors[ref]
            if entry.is_expired():
                # Clean up expired entry
                del self._anchors[ref]
                self._stats['bytes_stored'] -= entry.size_bytes
                self._stats['expired'] += 1
                self._stats['misses'] += 1
                return None
            
            self._stats['hits'] += 1
            
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
                "namespace": entry.namespace
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
            
            self._stats['bytes_stored'] -= bytes_freed
            self._stats['rollbacks'] += len(to_remove)
            
            # Persist changes
            self._persist_to_disk()
            
            logger.info(f"Cleaned up {len(to_remove)} anchors in namespace '{namespace}' "
                       f"({bytes_freed} bytes)")
            
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
                        self._stats['bytes_stored'] -= bytes_freed
                        self._stats['expired'] += len(to_remove)
                        self._persist_to_disk()
                        logger.info(f"Cleaned up {len(to_remove)} expired anchors "
                                   f"({bytes_freed} bytes)")
                
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
                "namespaces": len(set(e.namespace for e in self._anchors.values()))
            }