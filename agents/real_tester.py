#!/usr/bin/env python3
"""Real test runner for SWE-bench Lite evaluation.

This module runs actual pytest commands against checked-out repositories
to generate real test output, not synthetic logs.
"""

import subprocess
import tempfile
import shutil
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union


class RealTester:
    """Runs real tests against SWE-bench repositories."""
    
    def __init__(self, scratch_dir: Optional[Path] = None, mcp_client=None):
        """Initialize the real tester.
        
        Args:
            scratch_dir: Directory for temporary checkouts and test runs
            mcp_client: Optional MCP client for resolving anchors
        """
        self.scratch_dir = scratch_dir or Path(tempfile.mkdtemp(prefix="hermes_test_"))
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.mcp_client = mcp_client
    
    def run_test_for_instance(
        self,
        instance: Dict,
        apply_patch: bool = False
    ) -> Tuple[bool, str, int]:
        """Run tests for a SWE-bench instance.
        
        Args:
            instance: SWE-bench instance with repo, base_commit, test_patch, etc.
            apply_patch: If True, apply the solution patch before running tests
        
        Returns:
            Tuple of (passed, output, duration_ms)
        """
        repo_name = instance["repo"]
        base_commit = instance["base_commit"]
        test_patch = instance["test_patch"]
        patch = instance.get("patch", "")
        fail_to_pass = instance.get("FAIL_TO_PASS", [])
        
        # Create working directory for this instance
        work_dir = self.scratch_dir / f"{repo_name.replace('/', '_')}_{base_commit[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Clone repository at base commit
            repo_path = self._checkout_repo(
                repo_name,
                base_commit,
                work_dir,
                instance.get("environment_setup_commit")
            )
            
            # Apply test patch to create failing test
            if test_patch:
                self._apply_patch(repo_path, test_patch)
            
            # Optionally apply solution patch
            if apply_patch and patch:
                self._apply_patch(repo_path, patch)
            
            # Run tests and capture output
            passed, output, duration_ms = self._run_pytest_real(
                repo_path,
                fail_to_pass
            )
            
            return passed, output, duration_ms
            
        finally:
            # Clean up working directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _checkout_repo(
        self,
        repo_name: str,
        base_commit: str,
        work_dir: Path,
        setup_commit: Optional[str] = None
    ) -> Path:
        """Clone and checkout repository at specific commit.
        
        Args:
            repo_name: Repository name (e.g., "django/django")
            base_commit: Commit SHA to checkout
            work_dir: Working directory for the checkout
            setup_commit: Optional environment setup commit
        
        Returns:
            Path to the checked out repository
        """
        # Use hermetic repo manager if available and in hermetic mode
        if os.environ.get("HERMES_HERMETIC") == "1":
            try:
                from env.hermetic_repos import HermeticRepoManager
                # Use environment variable or default location
                mirrors_dir = Path(os.environ.get("GIT_MIRRORS_ROOT", Path.home() / ".hermes" / "git_mirrors"))
                
                # Verify offline mode by checking environment
                if os.environ.get("HF_DATASETS_OFFLINE") != "1":
                    raise RuntimeError(
                        "Hermetic mode requires offline execution. "
                        "Set HF_DATASETS_OFFLINE=1 and HERMES_HERMETIC=1"
                    )
                
                # Check if mirror exists before checkout
                manager = HermeticRepoManager(mirrors_dir)
                if not manager.repo_exists(repo_name, base_commit):
                    raise RuntimeError(
                        f"Mirror for {repo_name}@{base_commit} not found in {mirrors_dir}. "
                        f"Run: python scripts/prepare_swebench_repos.py --instances_file configs/swebench_lite_slice20.txt"
                    )
                
                # Checkout from local mirror
                repo_path = work_dir / repo_name.replace('/', '_')
                manager.checkout(repo_name, base_commit, repo_path)
                return repo_path
                    
            except ImportError as e:
                # Hard fail if hermetic infrastructure not available
                raise RuntimeError(
                    f"Hermetic mode requires env.hermetic_repos module: {e}\n"
                    f"Ensure env/hermetic_repos.py exists and imports correctly."
                )
            except FileNotFoundError as e:
                # Hard fail if manifest or mirrors missing
                raise RuntimeError(
                    f"Hermetic repos not prepared: {e}\n"
                    f"Run scripts/prepare_swebench_repos.py to create local mirrors."
                )
        
        # No simulation allowed - must have real repo access
        raise RuntimeError(
            f"Cannot checkout {repo_name}@{base_commit}: "
            f"Hermetic mode not enabled. Set HERMES_HERMETIC=1 and prepare local mirrors."
        )
    
    def _load_patch_bytes(self, patch_or_ref: Union[str, bytes]) -> bytes:
        """Hardened MCP resolution - always returns bytes or raises.
        
        Args:
            patch_or_ref: Patch content, MCP reference, or bytes
            
        Returns:
            Actual content as bytes
        """
        if isinstance(patch_or_ref, bytes):
            return patch_or_ref
        
        if isinstance(patch_or_ref, str) and patch_or_ref.startswith("mcp://"):
            if not self.mcp_client:
                raise RuntimeError(f"MCP ref provided but no MCP client configured: {patch_or_ref}")
            
            # Time the MCP dereference for production metrics
            t0 = time.perf_counter_ns()
            
            # Use resolve_bytes for strict bytes return
            data = self.mcp_client.resolve_bytes(patch_or_ref)
            
            t1 = time.perf_counter_ns()
            deref_ms = (t1 - t0) / 1_000_000
            
            # Store timing for later aggregation
            if not hasattr(self, 'mcp_deref_timings'):
                self.mcp_deref_timings = []
            self.mcp_deref_timings.append(deref_ms)
            
            return data
        
        # Plain text patch
        return patch_or_ref.encode("utf-8")
    
    def _resolve_bytes(self, maybe_ref: Union[str, bytes, bytearray]) -> bytes:
        """Legacy method for compatibility - delegates to _load_patch_bytes."""
        return self._load_patch_bytes(maybe_ref)
    
    def apply_patch(self, repo_path: Path, patch_or_ref: Union[str, bytes]) -> None:
        """Apply a patch to the repository with multiple strip level fallbacks.
        
        Args:
            repo_path: Path to the repository
            patch_or_ref: Patch content or MCP reference to apply
        """
        import sys
        if not patch_or_ref:
            print("[REAL_TESTER] No patch to apply", file=sys.stderr)
            return
        
        print(f"[REAL_TESTER] Applying patch, is MCP ref: {isinstance(patch_or_ref, str) and patch_or_ref.startswith('mcp://')}", file=sys.stderr)
        
        # Always resolve MCP refs before applying - unified path
        patch_bytes = self._load_patch_bytes(patch_or_ref)
        print(f"[REAL_TESTER] Resolved to {len(patch_bytes)} bytes", file=sys.stderr)
        
        # Try common strip levels (p0, p1, p2) to handle different patch formats
        for strip_level in [0, 1, 2]:
            if self._git_apply_check(repo_path, patch_bytes, strip_level):
                self._git_apply(repo_path, patch_bytes, strip_level)
                print(f"[REAL_TESTER] Patch applied successfully with -p{strip_level}", file=sys.stderr)
                return
        
        # All strip levels failed
        raise RuntimeError("git apply failed for all strip levels (p0, p1, p2)")
    
    def _git_apply_check(self, repo_path: Path, patch_bytes: bytes, strip_level: int) -> bool:
        """Check if patch applies cleanly with given strip level."""
        result = subprocess.run(
            ["git", "apply", f"-p{strip_level}", "--check", "-"],
            cwd=repo_path,
            input=patch_bytes,
            capture_output=True
        )
        return result.returncode == 0
    
    def _git_apply(self, repo_path: Path, patch_bytes: bytes, strip_level: int) -> None:
        """Apply patch with given strip level."""
        result = subprocess.run(
            ["git", "apply", f"-p{strip_level}", "-"],
            cwd=repo_path,
            input=patch_bytes,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"git apply -p{strip_level} failed: {result.stderr}")
    
    def _apply_patch(self, repo_path: Path, patch: str) -> None:
        """Legacy method - delegates to apply_patch."""
        self.apply_patch(repo_path, patch)
    
    def run_pytest_and_anchor_logs(self, repo_path: Path, inline_max: int = 1024) -> str:
        """Run pytest, return either inline string or mcp://logs/... if size > threshold.
        
        Args:
            repo_path: Path to the repository
            inline_max: Maximum size before anchoring
            
        Returns:
            Either the log content or MCP reference
        """
        # Run pytest and capture output
        cmd = ["python", "-m", "pytest", "-xvs", "--tb=short"]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            timeout=300
        )
        
        # Combine stdout and stderr
        full_log = b"=== STDOUT ===\n" + result.stdout + b"\n=== STDERR ===\n" + result.stderr
        
        # Anchor if too large
        if len(full_log) > inline_max:
            ref = f"mcp://logs/{hashlib.sha256(full_log).hexdigest()[:16]}"
            if self.mcp_client:
                ok, _ = self.mcp_client.put(ref, full_log, ttl_s=24*3600)  # 24h TTL for logs
                if not ok:
                    raise RuntimeError("Failed to anchor pytest logs")
            return ref
        
        return full_log.decode("utf-8", errors="replace")
    
    def _run_pytest_real(
        self,
        repo_path: Path,
        test_files: List[str]
    ) -> Tuple[bool, str, int]:
        """Actually run pytest and capture output.
        
        Args:
            repo_path: Path to the repository
            test_files: List of test files to run
        
        Returns:
            Tuple of (all_passed, output, duration_ms)
        """
        start_time = time.perf_counter()
        
        # Default to running all tests if none specified
        if not test_files:
            test_files = []
        
        # Prepare pytest command
        cmd = ["python", "-m", "pytest", "-xvs", "--tb=short"]
        if test_files:
            cmd.extend(test_files)
        
        try:
            # Run pytest for real
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            
            # Combine stdout and stderr for full output
            output = f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}"
            
            # Check if tests passed
            passed = result.returncode == 0
            
        except subprocess.TimeoutExpired as e:
            output = f"Test execution timed out after 60 seconds\nPartial output:\n{e.stdout or ''}\n{e.stderr or ''}"
            passed = False
            
        except FileNotFoundError:
            # Pytest must be available for real testing
            raise RuntimeError(
                "pytest not found. Install pytest to run real tests. "
                "Cannot use synthetic fallback in production."
            )
            
        except Exception as e:
            output = f"Test execution failed with error: {str(e)}"
            passed = False
        
        # Calculate duration
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        
        return passed, output, duration_ms
    
    
    def get_mcp_deref_p95(self) -> Optional[float]:
        """Get p95 MCP dereference time in ms."""
        if not hasattr(self, 'mcp_deref_timings') or not self.mcp_deref_timings:
            return None
        sorted_times = sorted(self.mcp_deref_timings)
        p95_idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
        return sorted_times[p95_idx]
    
    def cleanup(self):
        """Clean up all temporary files."""
        if self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir, ignore_errors=True)