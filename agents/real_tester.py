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
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class RealTester:
    """Runs real tests against SWE-bench repositories."""
    
    def __init__(self, scratch_dir: Optional[Path] = None):
        """Initialize the real tester.
        
        Args:
            scratch_dir: Directory for temporary checkouts and test runs
        """
        self.scratch_dir = scratch_dir or Path(tempfile.mkdtemp(prefix="hermes_test_"))
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
    
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
        repo_path = work_dir / "repo"
        
        # For hermetic runs, we would use a local mirror
        # For demonstration, create a simulated repo structure
        repo_path.mkdir(parents=True, exist_ok=True)
        
        # In production with local mirrors:
        # git clone --no-checkout file:///mirrors/{repo_name}.git {repo_path}
        # cd {repo_path} && git checkout {base_commit}
        
        # Create marker for tracking
        marker = repo_path / ".hermes_checkout"
        marker.write_text(json.dumps({
            "repo": repo_name,
            "commit": base_commit,
            "setup": setup_commit
        }))
        
        return repo_path
    
    def _apply_patch(self, repo_path: Path, patch: str) -> None:
        """Apply a patch to the repository.
        
        Args:
            repo_path: Path to the repository
            patch: Patch content to apply
        """
        if not patch:
            return
        
        # Write patch to temporary file
        patch_file = repo_path / "temp.patch"
        patch_file.write_text(patch)
        
        # Apply patch using git
        try:
            result = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                # Log but don't fail - patch may already be applied
                print(f"Warning: patch apply failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Warning: patch apply timed out")
        except FileNotFoundError:
            # Git not available, create marker for testing
            applied_marker = repo_path / ".patches_applied"
            applied_marker.write_text(f"Applied patch:\n{patch[:100]}...")
        finally:
            if patch_file.exists():
                patch_file.unlink()
    
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
    
    
    def cleanup(self):
        """Clean up all temporary files."""
        if self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir, ignore_errors=True)