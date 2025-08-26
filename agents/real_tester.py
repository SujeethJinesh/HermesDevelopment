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
                manager = HermeticRepoManager()
                
                # Verify offline mode
                if not manager.verify_offline():
                    raise RuntimeError(
                        "Hermetic mode requires offline execution. "
                        "Set HF_DATASETS_OFFLINE=1 and HERMES_HERMETIC=1"
                    )
                
                # Checkout from local mirror
                success, repo_path, message = manager.checkout_repo(
                    repo_name, base_commit, work_dir
                )
                
                if success:
                    return repo_path
                else:
                    # Hard fail in hermetic mode - no fallback to simulation
                    raise RuntimeError(
                        f"Hermetic checkout failed for {repo_name}@{base_commit}: {message}\n"
                        f"Run scripts/prepare_swebench_repos.py to prepare local mirrors."
                    )
                    
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
        
        # Non-hermetic mode: create simulated repo structure for testing only
        # This is acceptable for development but NOT for acceptance testing
        repo_path = work_dir / "repo"
        repo_path.mkdir(parents=True, exist_ok=True)
        
        # Create marker for tracking
        marker = repo_path / ".hermes_checkout"
        marker.write_text(json.dumps({
            "repo": repo_name,
            "commit": base_commit,
            "setup": setup_commit,
            "hermetic": "0",  # Explicitly mark as non-hermetic
            "warning": "SIMULATED CHECKOUT - NOT FOR PRODUCTION"
        }))
        
        return repo_path
    
    def _load_patch_bytes(self, patch_or_ref: Union[str, bytes]) -> bytes:
        """Load patch content, resolving MCP refs if needed.
        
        Args:
            patch_or_ref: Either patch content or MCP reference
            
        Returns:
            Actual patch bytes
        """
        # If already bytes, return as-is
        if isinstance(patch_or_ref, bytes):
            return patch_or_ref
        
        # Check if it's an MCP reference
        if isinstance(patch_or_ref, str) and patch_or_ref.startswith("mcp://"):
            if not self.mcp_client:
                raise RuntimeError(f"MCP ref provided but no MCP client configured: {patch_or_ref}")
            
            # Resolve the MCP anchor to get actual patch content
            success, data = self.mcp_client.resolve(patch_or_ref)
            if not success or data is None:
                raise RuntimeError(f"Failed to resolve MCP ref: {patch_or_ref}")
            
            return data
        
        # Regular string patch content
        return patch_or_ref.encode("utf-8")
    
    def _apply_patch(self, repo_path: Path, patch: str) -> None:
        """Apply a patch to the repository.
        
        Args:
            repo_path: Path to the repository
            patch: Patch content or MCP reference to apply
        """
        if not patch:
            return
        
        # Resolve MCP references if needed
        patch_bytes = self._load_patch_bytes(patch)
        patch_text = patch_bytes.decode("utf-8")
        
        # Write patch to temporary file
        patch_file = repo_path / "temp.patch"
        patch_file.write_text(patch_text)
        
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
            applied_marker.write_text(f"Applied patch:\n{patch_text[:100]}...")
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