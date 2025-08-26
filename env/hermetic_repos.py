#!/usr/bin/env python3
"""Hermetic repository management for offline SWE-bench evaluation.

This module provides functions to create fresh worktrees from local mirrors/bundles
during offline evaluation. No network access is required once mirrors are prepared.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _run(cmd, cwd=None, env=None):
    """Run command and raise on failure."""
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout


def _slug_to_safe_dir(slug: str) -> str:
    """Convert SWE-bench slug 'owner/repo' to safe directory name."""
    return slug.replace("/", "__")


@dataclass
class Checkout:
    """Represents a checked-out repository."""
    path: Path
    base_commit: str


class HermeticRepoManager:
    """
    Offline git checkout manager, backed by local bare mirrors.

    Mirrors directory layout (default: ./.mirrors):
      .mirrors/
        owner__repo.git/    # bare mirror (created online by a prep script)

    Usage:
      mgr = HermeticRepoManager()
      co = mgr.checkout(repo_slug='django/django', base_commit='abc123...')
      # co.path now contains a working tree at that commit, with no network access.
    """

    def __init__(self, mirrors_dir: Optional[Path] = None, work_root: Optional[Path] = None):
        self.mirrors_dir = Path(
            mirrors_dir or os.environ.get("HERMES_MIRRORS_DIR", ".mirrors")
        ).resolve()
        self.work_root = Path(
            work_root or os.environ.get("HERMES_CHECKOUTS_DIR", "scratch/hermetic_repos")
        ).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)

    def _mirror_path(self, repo_slug: str) -> Path:
        """Get path to local mirror for a repository."""
        safe = _slug_to_safe_dir(repo_slug)
        return self.mirrors_dir / f"{safe}.git"

    def _ensure_clean(self, path: Path):
        """Ensure repository is clean and won't fetch from network."""
        # Ensure we won't accidentally fetch from network:
        try:
            _run(["git", "config", "--local", "--unset-all", "remote.origin.url"], cwd=path)
        except RuntimeError:
            pass  # OK if no remote set
        # Make sure no leftover changes:
        _run(["git", "reset", "--hard"], cwd=path)
        _run(["git", "clean", "-xdf"], cwd=path)

    def checkout(self, repo_slug: str, base_commit: str) -> Checkout:
        """Checkout a repository at a specific commit from local mirror.
        
        Args:
            repo_slug: Repository in 'owner/repo' format
            base_commit: Commit SHA to checkout
            
        Returns:
            Checkout object with path to working tree
            
        Raises:
            RuntimeError: If mirror is missing or checkout fails
        """
        mirror = self._mirror_path(repo_slug)
        if not mirror.exists():
            raise RuntimeError(
                f"Hermetic mirror not found for {repo_slug}: {mirror}\n"
                "Run the online prep script first to create local mirrors."
            )
        safe = _slug_to_safe_dir(repo_slug)
        dest = self.work_root / f"{safe}-{base_commit[:12]}"

        if dest.exists():
            # Re-use existing checkout (faster), ensure clean & reset to desired commit
            self._ensure_clean(dest)
        else:
            # Clone without touching network; reference the mirror directly
            _run(["git", "clone", "--no-checkout", str(mirror), str(dest)])

        # Detach to the exact commit (no fetch)
        _run(["git", "reset", "--hard", base_commit], cwd=dest)
        return Checkout(path=dest, base_commit=base_commit)
    
    def checkout_repo(self, repo: str, base_commit: str, work_dir: Path,
                      apply_patch: Optional[str] = None):
        """Compatibility method for existing RealTester interface.
        
        Returns:
            Tuple of (success, repo_path, message)
        """
        try:
            co = self.checkout(repo_slug=repo, base_commit=base_commit)
            # Move to requested work_dir if needed
            final_path = work_dir / "repo"
            if co.path != final_path:
                if final_path.exists():
                    shutil.rmtree(final_path)
                shutil.move(str(co.path), str(final_path))
                co.path = final_path
            
            if apply_patch:
                self.apply_patch(str(co.path), apply_patch)
                
            return True, co.path, "Checkout successful"
        except Exception as e:
            return False, work_dir / "repo", str(e)
    
    def apply_patch(self, repo_path: str, patch_text: str):
        """Apply a patch to a repository."""
        if not patch_text:
            return
        p = subprocess.run(
            ["git", "apply", "-"],
            input=patch_text.encode("utf-8"),
            cwd=repo_path,
            capture_output=True
        )
        if p.returncode != 0:
            raise RuntimeError(f"Patch apply failed: {p.stderr.decode('utf-8', 'ignore')}")
    
    def verify_offline(self) -> bool:
        """Verify we're in offline mode."""
        return (os.environ.get("HF_DATASETS_OFFLINE") == "1" and 
                os.environ.get("HERMES_HERMETIC") == "1")

    def destroy(self, checkout: Checkout):
        """Clean up a checkout."""
        if checkout.path.exists():
            shutil.rmtree(checkout.path, ignore_errors=True)