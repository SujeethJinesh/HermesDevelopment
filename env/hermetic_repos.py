#!/usr/bin/env python3
"""Hermetic repository management for offline SWE-bench evaluation."""

from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path


class HermeticRepoManager:
    """Manages offline repo checkouts from local mirrors."""
    
    def __init__(self, mirrors_dir: str | Path):
        """Initialize with path to local git mirrors directory."""
        self.mirrors_dir = Path(mirrors_dir)
    
    def checkout(self, repo: str, commit: str, dest_dir: str | Path) -> None:
        """Checkout a repo at specific commit from local mirror.
        
        Args:
            repo: Repository in 'org/repo' format from SWE-bench
            commit: Commit SHA to checkout
            dest_dir: Destination directory for checkout
            
        Raises:
            RuntimeError: If mirror is missing or checkout fails
        """
        # Convert org/repo to org__repo for mirror directory
        mirror = self.mirrors_dir / f"{repo.replace('/', '__')}.git"
        
        if not mirror.exists():
            raise RuntimeError(f"Missing mirror for {repo}: {mirror}")
        
        dest = Path(dest_dir)
        if dest.exists():
            shutil.rmtree(dest)
        
        # Clone from local mirror (shared objects, no network)
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(mirror), str(dest)],
            check=True
        )
        
        # Checkout specific commit
        subprocess.run(
            ["git", "-C", str(dest), "checkout", commit],
            check=True
        )