"""Storage utilities for content-addressed storage with atomic writes."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional


class ContentAddressedStorage:
    """Content-addressed storage with atomic writes and fsync."""

    def __init__(self, base_path: Path):
        """Initialize storage.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = base_path / "index.json"
        self.content_path = base_path / "content"
        self.content_path.mkdir(exist_ok=True)
        
    def _get_content_path(self, sha256: str) -> Path:
        """Get path for content by SHA-256.
        
        Args:
            sha256: Content hash
            
        Returns:
            Path to content file
        """
        # Use first 2 chars as directory for better filesystem performance
        subdir = self.content_path / sha256[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / sha256
        
    def write_content(self, data: bytes) -> str:
        """Write content atomically with SHA-256.
        
        Args:
            data: Binary data to write
            
        Returns:
            SHA-256 hash of content
        """
        sha256 = hashlib.sha256(data).hexdigest()
        target_path = self._get_content_path(sha256)
        
        # Skip if already exists
        if target_path.exists():
            return sha256
            
        # Atomic write: write to temp, fsync, rename
        with tempfile.NamedTemporaryFile(
            mode='wb', 
            dir=target_path.parent, 
            delete=False
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
            
        # Atomic rename
        tmp_path.rename(target_path)
        
        # Sync directory to ensure rename is persisted
        dir_fd = os.open(str(target_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
            
        return sha256
        
    def read_content(self, sha256: str) -> Optional[bytes]:
        """Read content by SHA-256.
        
        Args:
            sha256: Content hash
            
        Returns:
            Content bytes or None if not found
        """
        content_path = self._get_content_path(sha256)
        if not content_path.exists():
            return None
            
        with open(content_path, 'rb') as f:
            return f.read()
            
    def delete_content(self, sha256: str) -> bool:
        """Delete content by SHA-256.
        
        Args:
            sha256: Content hash
            
        Returns:
            True if deleted, False if not found
        """
        content_path = self._get_content_path(sha256)
        if not content_path.exists():
            return False
            
        content_path.unlink()
        return True
        
    def save_index(self, index: dict) -> None:
        """Save index atomically.
        
        Args:
            index: Index dictionary to save
        """
        # Atomic write for index
        with tempfile.NamedTemporaryFile(
            mode='w', 
            dir=self.base_path, 
            delete=False,
            suffix='.json'
        ) as tmp:
            json.dump(index, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
            
        # Atomic rename
        tmp_path.rename(self.index_path)
        
        # Sync directory
        dir_fd = os.open(str(self.base_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
            
    def load_index(self) -> dict:
        """Load index from disk.
        
        Returns:
            Index dictionary or empty dict if not found
        """
        if not self.index_path.exists():
            return {}
            
        with open(self.index_path, 'r') as f:
            return json.load(f)
            
    def get_storage_size(self) -> int:
        """Get total storage size in bytes.
        
        Returns:
            Total size of all content files
        """
        total = 0
        for path in self.content_path.rglob('*'):
            if path.is_file():
                total += path.stat().st_size
        return total
        
    def cleanup_orphaned(self, valid_hashes: set) -> int:
        """Clean up orphaned content files.
        
        Args:
            valid_hashes: Set of valid SHA-256 hashes
            
        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        for path in self.content_path.rglob('*'):
            if path.is_file():
                sha256 = path.name
                if sha256 not in valid_hashes:
                    path.unlink()
                    cleaned += 1
        return cleaned