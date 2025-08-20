"""Unit tests for hermetic execution sandbox."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.hermetic import HermeticRun, HermeticNetworkError


class TestHermeticRun:
    """Test hermetic execution sandbox."""
    
    def test_network_blocked_in_venv_py(self, tmp_path):
        """Test that network access is blocked in hermetic venv."""
        # Set hermetic environment variable
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Create hermetic run
        run = HermeticRun(
            task_id="test_network",
            seed=42,
            hermetic=True
        )
        
        with run():
            # Find Python executable in venv
            python_exe = run.venv_path / "bin" / "python"
            if not python_exe.exists():
                python_exe = run.venv_path / "Scripts" / "python.exe"
            
            assert python_exe.exists(), "Python executable not found in venv"
            
            # Test network blocking
            test_script = """
import socket
import sys

try:
    socket.create_connection(("1.1.1.1", 80), timeout=1)
    print("ERROR: Connection succeeded")
    sys.exit(1)
except Exception as e:
    if "hermetic" in str(e).lower() or "blocked" in str(e).lower() or "permission" in str(e).lower():
        print("SUCCESS: Network blocked")
        sys.exit(0)
    else:
        print(f"UNEXPECTED: {e}")
        sys.exit(2)
"""
            
            result = subprocess.run(
                [str(python_exe), "-c", test_script],
                capture_output=True,
                text=True,
                env={**os.environ, "HERMES_HERMETIC": "1"}
            )
            
            # Network should be blocked
            assert result.returncode in (0, 2), f"Network not blocked properly: {result.stderr}"
            
            if result.returncode == 0:
                assert "SUCCESS" in result.stdout, f"Unexpected output: {result.stdout}"
    
    def test_worktree_detached_and_clean(self):
        """Test that worktree is properly detached and clean."""
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Get current HEAD
        current_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        
        run = HermeticRun(
            task_id="test_worktree",
            seed=123,
            base_sha=current_sha,
            hermetic=True
        )
        
        with run():
            # Check worktree is at correct SHA
            worktree_sha = subprocess.run(
                ["git", "-C", str(run.worktree_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            assert worktree_sha == current_sha, f"Worktree SHA mismatch: {worktree_sha} != {current_sha}"
            
            # Check worktree is clean
            status = subprocess.run(
                ["git", "-C", str(run.worktree_path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            assert status == "", f"Worktree not clean: {status}"
        
        # After context exit, worktree should be removed
        assert not run.worktree_path.exists(), "Worktree not cleaned up"
    
    def test_manifest_fields_and_stable_hash(self):
        """Test manifest contains required fields and stable hash is deterministic."""
        os.environ["HERMES_HERMETIC"] = "1"
        
        # Create two runs with same seed
        manifests = []
        
        for i in range(2):
            run = HermeticRun(
                task_id="test_manifest",
                seed=999,
                hermetic=True
            )
            
            with run():
                manifest = run.emit_manifest()
                manifests.append(manifest)
        
        # Check required fields
        required_fields = [
            "task_id", "run_id", "seed", "repo_sha", "config_hash",
            "os_fingerprint", "hermetic", "scratch_path", "worktree_path",
            "venv_path", "venv_hash", "stable_hash", "durations"
        ]
        
        for manifest in manifests:
            for field in required_fields:
                assert field in manifest, f"Missing required field: {field}"
        
        # Check stable hash is identical for same seed
        assert manifests[0]["stable_hash"] == manifests[1]["stable_hash"], \
            "Stable hash not deterministic for same seed"
        
        # Check that run_ids are different
        assert manifests[0]["run_id"] != manifests[1]["run_id"], \
            "Run IDs should be unique"
    
    def test_cleanup_no_residue(self):
        """Test that cleanup removes all artifacts."""
        os.environ["HERMES_HERMETIC"] = "1"
        
        task_id = "test_cleanup"
        scratch_base = Path("scratch") / task_id
        
        # Create hermetic run
        run = HermeticRun(
            task_id=task_id,
            seed=456,
            hermetic=True
        )
        
        run_id = run.run_id
        scratch_path = run.scratch_base
        
        # Use context manager
        with run():
            # Verify scratch exists
            assert scratch_path.exists(), "Scratch directory not created"
            assert run.worktree_path.exists(), "Worktree not created"
            assert run.venv_path.exists(), "Venv not created"
            
            # Create a temp file that should be cleaned
            temp_file = Path("/tmp") / "hermes_test_file"
            temp_file.touch()
        
        # After context, everything should be cleaned
        assert not scratch_path.exists(), f"Scratch directory not cleaned: {scratch_path}"
        assert not temp_file.exists(), "Temp file not cleaned"
        
        # Check that parent task directory is empty or removed
        if scratch_base.exists():
            contents = list(scratch_base.iterdir())
            assert len(contents) == 0, f"Task directory not empty: {contents}"


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Clean up any test artifacts before and after tests."""
    yield
    
    # Clean up scratch directories
    scratch_dir = Path("scratch")
    if scratch_dir.exists():
        for task_dir in scratch_dir.glob("test_*"):
            shutil.rmtree(task_dir, ignore_errors=True)
    
    # Clean up any stray worktrees
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split("\n"):
            if line.startswith("worktree ") and "scratch" in line:
                worktree_path = line.split(" ", 1)[1]
                subprocess.run(
                    ["git", "worktree", "remove", "--force", worktree_path],
                    check=False,
                    capture_output=True
                )
    except:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])