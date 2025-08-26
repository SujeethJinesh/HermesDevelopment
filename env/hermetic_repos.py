#!/usr/bin/env python3
"""Hermetic repository management for offline SWE-bench evaluation.

This module provides functions to create fresh worktrees from local mirrors/bundles
during offline evaluation. No network access is required once mirrors are prepared.
"""

from __future__ import annotations
import os
import subprocess
import shutil
import pathlib
import tempfile
import json
from typing import Optional, Tuple


class HermeticRepoManager:
    """
    Manages hermetic repository checkouts from local mirrors.
    Resolves SWE-bench repos offline from local bare mirrors prepared by
    scripts/prepare_swebench_repos.py. No network, deterministic paths.
    """
    
    def __init__(self, mirror_root: Optional[str] = None, work_root: Optional[str] = None):
        """Initialize with mirror and work directories.
        
        Args:
            mirror_root: Root directory containing git mirrors
            work_root: Root directory for temporary worktrees
        """
        self.mirror_root = mirror_root or os.environ.get("HERMES_REPO_MIRRORS", ".mirrors")
        self.work_root = work_root or os.environ.get("HERMES_WORKTREES", "scratch/repos")
        pathlib.Path(self.work_root).mkdir(parents=True, exist_ok=True)
        
        # Also support manifest-based lookups for compatibility
        self.manifest_path = pathlib.Path("data/repos_manifest.json")
        self.manifest = self._load_manifest() if self.manifest_path.exists() else {}
    
    def _load_manifest(self) -> dict:
        """Load repos manifest if it exists."""
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except Exception:
            return {}
    
    def verify_offline(self) -> bool:
        """Verify we're in offline mode (no network access allowed)."""
        # Check HF offline mode
        if os.environ.get("HF_DATASETS_OFFLINE") != "1":
            print("Warning: HF_DATASETS_OFFLINE not set to 1")
            return False
        
        # Check hermetic flag
        if os.environ.get("HERMES_HERMETIC") != "1":
            print("Warning: HERMES_HERMETIC not set to 1")
            return False
        
        return True
    
    def checkout(self, repo: str, commit: str) -> str:
        """Checkout a repository at a specific commit from local mirror.
        
        Args:
            repo: Repository name in 'owner/name' or 'owner__name' format
            commit: SHA or ref to checkout
            
        Returns:
            Absolute path to the worktree with the requested commit
            
        Raises:
            RuntimeError: If mirror is missing or checkout fails
        """
        # Normalize repo format (support both / and __)
        if "/" in repo:
            owner, name = repo.split("/", 1)
        else:
            owner, name = repo.split("__", 1)
        
        # Try different mirror locations
        bare_candidates = [
            pathlib.Path(self.mirror_root) / f"{owner}/{name}.git",
            pathlib.Path(self.mirror_root) / f"{owner}__{name}.git",
            pathlib.Path(self.mirror_root) / f"{repo.replace('/', '__')}.git",
        ]
        
        bare = None
        for candidate in bare_candidates:
            if candidate.exists():
                bare = candidate
                break
        
        if not bare:
            # Check manifest for bundle/mirror paths
            repo_key = f"{owner}/{name}"
            if self.manifest and "repos" in self.manifest:
                for repo_info in self.manifest["repos"]:
                    if repo_info.get("repo") == repo_key:
                        if "mirror_path" in repo_info:
                            bare = pathlib.Path(repo_info["mirror_path"])
                            break
            
            if not bare or not bare.exists():
                raise RuntimeError(
                    f"Missing mirror for {repo}. Tried:\n" +
                    "\n".join(f"  - {c}" for c in bare_candidates) +
                    f"\nRun scripts/prepare_swebench_repos.py to create mirrors."
                )
        
        # Create a throwaway worktree (no network)
        dst = pathlib.Path(self.work_root) / f"{owner}__{name}__{commit[:12]}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=False)
        
        try:
            # Clone from local mirror and checkout commit
            subprocess.run(
                ["git", "clone", "--no-checkout", str(bare), str(dst)],
                check=True,
                capture_output=True,
                text=True
            )
            subprocess.run(
                ["git", "-C", str(dst), "checkout", commit],
                check=True,
                capture_output=True,
                text=True
            )
            return str(dst)
        except subprocess.CalledProcessError as e:
            # Clean up on failure
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            raise RuntimeError(
                f"Failed to checkout {repo}@{commit}:\n{e.stderr if e.stderr else str(e)}"
            )
    
    def checkout_repo(self, repo: str, base_commit: str, work_dir: pathlib.Path,
                      apply_patch: Optional[str] = None) -> Tuple[bool, pathlib.Path, str]:
        """Compatibility method matching the interface expected by RealTester.
        
        Args:
            repo: Repository name (e.g., "django/django")
            base_commit: Commit SHA to checkout
            work_dir: Directory for the worktree
            apply_patch: Optional patch to apply after checkout
            
        Returns:
            Tuple of (success, repo_path, message)
        """
        try:
            # Use our checkout method
            repo_path = self.checkout(repo, base_commit)
            
            # Move to requested work_dir if different
            final_path = work_dir / "repo"
            if str(repo_path) != str(final_path):
                if final_path.exists():
                    shutil.rmtree(final_path)
                shutil.move(repo_path, str(final_path))
                repo_path = str(final_path)
            
            # Apply patch if provided
            if apply_patch:
                try:
                    self.apply_patch(repo_path, apply_patch)
                    message = "Checkout and patch successful"
                except Exception as e:
                    # Non-fatal: patch may not apply cleanly
                    message = f"Checkout OK, patch warning: {e}"
            else:
                message = "Checkout successful"
            
            return True, pathlib.Path(repo_path), message
            
        except Exception as e:
            return False, work_dir / "repo", f"Checkout failed: {e}"
    
    def apply_patch(self, repo_path: str, patch_text: str) -> None:
        """Apply a patch to a repository.
        
        Args:
            repo_path: Path to the repository
            patch_text: Patch content to apply
            
        Raises:
            RuntimeError: If patch application fails
        """
        if not patch_text:
            return
        
        p = subprocess.run(
            ["git", "-C", repo_path, "apply", "-p0", "-v"],
            input=patch_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if p.returncode != 0:
            stderr = p.stderr.decode("utf-8", "ignore")
            # Check if it's a real error or just a warning
            if "error:" in stderr.lower():
                raise RuntimeError(f"Patch apply failed:\n{stderr}")
    
    def cleanup(self, repo_path: str) -> None:
        """Clean up a worktree.
        
        Args:
            repo_path: Path to remove
        """
        if os.path.isdir(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)