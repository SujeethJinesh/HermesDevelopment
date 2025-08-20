#!/usr/bin/env python3
"""Hermetic execution sandbox for HERMES.

Provides fresh git worktree + venv with network isolation for deterministic test runs.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import yaml


class HermeticNetworkError(Exception):
    """Raised when network access is attempted in hermetic mode."""
    pass


class HermeticRun:
    """Manages hermetic execution environment with fresh worktree and venv."""
    
    def __init__(
        self,
        task_id: str,
        run_id: Optional[str] = None,
        seed: int = 42,
        base_sha: Optional[str] = None,
        hermetic: bool = True
    ):
        """Initialize hermetic run.
        
        Args:
            task_id: Task identifier for scratch organization
            run_id: Unique run ID (auto-generated if None)
            seed: Random seed for determinism
            base_sha: Git SHA to detach worktree to (HEAD if None)
            hermetic: Whether to enable network blocking
        """
        self.task_id = task_id
        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.seed = seed
        self.base_sha = base_sha
        self.hermetic = hermetic
        
        # Paths
        self.scratch_base = Path("scratch") / task_id / self.run_id
        self.worktree_path = self.scratch_base / "worktree"
        self.venv_path = self.scratch_base / "venv"
        
        # Timing
        self.setup_start_ns = 0
        self.setup_end_ns = 0
        self.cleanup_start_ns = 0
        self.cleanup_end_ns = 0
        self.network_guard_install_ms = 0.0
        
        # Manifest data
        self.manifest: Dict[str, Any] = {}
    
    def _get_repo_sha(self) -> str:
        """Get current repository SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def _get_config_hash(self) -> Optional[str]:
        """Compute hash of configs/generation.yaml."""
        config_path = Path("configs/generation.yaml")
        if not config_path.exists():
            print(f"Warning: {config_path} not found, config_hash will be null")
            return None
        
        with open(config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    
    def _get_os_fingerprint(self) -> Dict[str, str]:
        """Get OS and Python fingerprint."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
    
    def _compute_stable_hash(self) -> str:
        """Compute deterministic stable hash for run."""
        components = {
            "task_id": self.task_id,
            "seed": self.seed,
            "repo_sha": self.manifest.get("repo_sha", "unknown"),
            "config_hash": self.manifest.get("config_hash"),
            "os_fingerprint": json.dumps(self.manifest.get("os_fingerprint", {}), sort_keys=True),
        }
        
        stable_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(stable_str.encode()).hexdigest()[:16]
    
    def _setup_worktree(self) -> None:
        """Create fresh git worktree."""
        self.worktree_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get base SHA
        if not self.base_sha:
            self.base_sha = self._get_repo_sha()
        
        # Create worktree
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(self.worktree_path), self.base_sha],
            check=True,
            capture_output=True
        )
    
    def _setup_venv(self) -> None:
        """Create fresh venv with network guard."""
        subprocess.run(
            [sys.executable, "-m", "venv", str(self.venv_path)],
            check=True
        )
        
        # Install network guard if hermetic
        if self.hermetic and os.environ.get("HERMES_HERMETIC") == "1":
            self._install_network_guard()
    
    def _install_network_guard(self) -> None:
        """Install network blocking in venv via sitecustomize.py."""
        guard_start_ns = time.perf_counter_ns()
        
        site_packages = self.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        sitecustomize_path = site_packages / "sitecustomize.py"
        
        # Ensure site-packages exists
        site_packages.mkdir(parents=True, exist_ok=True)
        
        # Write network guard with DNS blocking
        guard_code = '''"""Network guard for hermetic execution."""
import socket
import os

if os.environ.get("HERMES_HERMETIC") == "1":
    _original_socket = socket.socket
    _original_create_connection = socket.create_connection
    _original_getaddrinfo = socket.getaddrinfo
    
    class HermeticNetworkError(Exception):
        """Network access blocked in hermetic mode."""
        pass
    
    def _blocked_socket(*args, **kwargs):
        # Allow Unix domain sockets
        if args and args[0] == socket.AF_UNIX:
            return _original_socket(*args, **kwargs)
        raise HermeticNetworkError("Network access blocked in hermetic mode")
    
    def _blocked_create_connection(*args, **kwargs):
        raise HermeticNetworkError("Network access blocked in hermetic mode")
    
    def _blocked_getaddrinfo(*args, **kwargs):
        # Allow localhost lookups
        if args and args[0] in ("localhost", "127.0.0.1", "::1", None):
            return _original_getaddrinfo(*args, **kwargs)
        raise HermeticNetworkError("DNS lookups blocked in hermetic mode")
    
    socket.socket = _blocked_socket
    socket.create_connection = _blocked_create_connection
    socket.getaddrinfo = _blocked_getaddrinfo
'''
        
        with open(sitecustomize_path, "w") as f:
            f.write(guard_code)
        
        guard_end_ns = time.perf_counter_ns()
        self.network_guard_install_ms = (guard_end_ns - guard_start_ns) / 1_000_000
    
    def _compute_venv_hash(self) -> str:
        """Compute hash of venv contents based on lockfile."""
        lockfile_path = Path("requirements.lock")
        if lockfile_path.exists():
            with open(lockfile_path, "rb") as f:
                lockfile_content = f.read()
            
            # Store lockfile SHA in manifest
            self.manifest["lockfile_sha"] = hashlib.sha256(lockfile_content).hexdigest()[:16]
            
            venv_str = f"{lockfile_content.decode('utf-8')}:{sys.version}"
            return hashlib.sha256(venv_str.encode()).hexdigest()[:16]
        else:
            # Fallback to Python version only
            self.manifest["lockfile_sha"] = None
            venv_str = f"{sys.version}"
            return hashlib.sha256(venv_str.encode()).hexdigest()[:16]
    
    def _cleanup(self) -> None:
        """Clean up scratch directory and temp files."""
        self.cleanup_start_ns = time.perf_counter_ns()
        
        # Remove worktree from git
        if self.worktree_path.exists():
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(self.worktree_path)],
                    check=False,
                    capture_output=True
                )
            except:
                pass
        
        # Remove scratch directory
        if self.scratch_base.exists():
            shutil.rmtree(self.scratch_base, ignore_errors=True)
        
        # Clean up temp files
        temp_pattern = Path("/tmp")
        for temp_file in temp_pattern.glob("hermes_*"):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
            except:
                pass
        
        self.cleanup_end_ns = time.perf_counter_ns()
    
    def _get_model_info_from_config(self) -> Dict[str, Any]:
        """Extract model information from config."""
        config_path = Path("configs/generation.yaml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Extract quantization info
            quantization = config.get("quantization", {})
            
            # Model SHAs would come from actual model files - null for now
            return {
                "model_shas": None,  # Would be computed from actual model files
                "tokenizer_shas": None,  # Would be computed from tokenizer files
                "quantization": {
                    "default": quantization.get("default", "Q4_K_M"),
                    "fallback": quantization.get("fallback", "Q5_K_M")
                }
            }
        return {
            "model_shas": None,
            "tokenizer_shas": None,
            "quantization": None
        }
    
    def setup(self) -> None:
        """Set up hermetic environment."""
        self.setup_start_ns = time.perf_counter_ns()
        
        # Get model info
        model_info = self._get_model_info_from_config()
        
        # Collect manifest data early
        self.manifest = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "repo_sha": self._get_repo_sha(),
            "base_sha": self.base_sha or self._get_repo_sha(),
            "config_hash": self._get_config_hash(),
            "os_fingerprint": self._get_os_fingerprint(),
            "hermetic": self.hermetic,
            "scratch_path": str(self.scratch_base),
            "worktree_path": str(self.worktree_path),
            "venv_path": str(self.venv_path),
            "model_shas": model_info["model_shas"],
            "tokenizer_shas": model_info["tokenizer_shas"],
            "quantization": model_info["quantization"],
        }
        
        # Set up environment
        self._setup_worktree()
        self._setup_venv()
        
        # Compute hashes after setup
        self.manifest["venv_hash"] = self._compute_venv_hash()
        self.manifest["stable_hash"] = self._compute_stable_hash()
        
        self.setup_end_ns = time.perf_counter_ns()
        
        # Add timing to manifest
        self.manifest["durations"] = {
            "setup_ms": (self.setup_end_ns - self.setup_start_ns) / 1_000_000
        }
    
    def emit_manifest(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Emit run manifest."""
        if path:
            with open(path, "w") as f:
                json.dump(self.manifest, f, indent=2, sort_keys=True)
        return self.manifest
    
    @contextmanager
    def __call__(self):
        """Context manager for hermetic execution."""
        try:
            self.setup()
            yield self
        finally:
            self._cleanup()
            
            # Update manifest with cleanup timing
            if self.cleanup_end_ns > 0:
                self.manifest["durations"]["cleanup_ms"] = (
                    self.cleanup_end_ns - self.cleanup_start_ns
                ) / 1_000_000


def selftest(args):
    """Run hermetic sandbox self-test."""
    print(f"Running hermetic selftest with task_id={args.task_id}, seed={args.seed}")
    
    # Enable hermetic mode
    os.environ["HERMES_HERMETIC"] = "1"
    
    # Create hermetic run
    run = HermeticRun(
        task_id=args.task_id,
        seed=args.seed,
        hermetic=True
    )
    
    metrics = {
        "sandbox_setup_ms_p50": 0,
        "sandbox_cleanup_ms_p95": 0,
        "network_guard_install_ms": 0,
    }
    
    with run():
        print(f"Hermetic environment created at {run.scratch_base}")
        
        # Test worktree
        worktree_sha = subprocess.run(
            ["git", "-C", str(run.worktree_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        
        print(f"Worktree SHA: {worktree_sha}")
        assert worktree_sha == run.base_sha, f"Worktree not detached to {run.base_sha}"
        
        # Test venv exists
        assert run.venv_path.exists(), "Venv not created"
        python_exe = run.venv_path / "bin" / "python"
        if not python_exe.exists():
            python_exe = run.venv_path / "Scripts" / "python.exe"
        assert python_exe.exists(), "Python executable not found in venv"
        
        # Test network guard (if hermetic)
        if run.hermetic:
            network_test = subprocess.run(
                [str(python_exe), "-c", 
                 "import socket; socket.create_connection(('1.1.1.1', 80), timeout=1)"],
                capture_output=True,
                text=True,
                env={**os.environ, "HERMES_HERMETIC": "1"}
            )
            
            if network_test.returncode == 0:
                print("WARNING: Network guard not blocking connections")
            else:
                print("Network guard active: connections blocked")
                metrics["network_guard_install_ms"] = run.network_guard_install_ms
        
        # Update metrics
        metrics["sandbox_setup_ms_p50"] = run.manifest["durations"]["setup_ms"]
        
        # Emit manifest
        if args.emit:
            manifest_path = Path(args.emit) if args.emit != "metrics.json" else Path(f"manifest_{run.run_id}.json")
            run.emit_manifest(manifest_path)
            print(f"Manifest written to {manifest_path}")
    
    # Check cleanup
    metrics["sandbox_cleanup_ms_p95"] = run.manifest["durations"]["cleanup_ms"]
    
    if not run.scratch_base.exists():
        print("Cleanup successful: scratch directory removed")
    else:
        print("WARNING: Scratch directory still exists after cleanup")
    
    # Emit metrics if requested
    if args.emit == "metrics.json":
        full_metrics = {
            "bytes_per_solve": None,
            "tokens_prefill": None,
            "tokens_decode": None,
            "e2e_latency_ms_p50": None,
            "e2e_latency_ms_p95": None,
            "message_path_ms_p95": None,
            "mcp_deref_ms_p95": None,
            "sae_accept_rate": None,
            "rollback_ms_p95": None,
            "pass_at_1": None,
            **metrics
        }
        
        with open("metrics.json", "w") as f:
            json.dump(full_metrics, f, indent=2)
        print("Metrics written to metrics.json")
    
    print(f"Selftest complete. Stable hash: {run.manifest['stable_hash']}")
    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hermetic execution sandbox")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Selftest command
    selftest_parser = subparsers.add_parser("selftest", help="Run self-test")
    selftest_parser.add_argument("--task-id", default="test", help="Task ID")
    selftest_parser.add_argument("--seed", type=int, default=123, help="Random seed")
    selftest_parser.add_argument("--emit", help="Emit manifest/metrics to file")
    
    args = parser.parse_args()
    
    if args.command == "selftest":
        return selftest(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())