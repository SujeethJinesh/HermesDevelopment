"""Integration tests for hermetic execution sandbox."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestHermeticIntegration:
    """Integration tests for hermetic sandbox."""

    def test_hermetic_selftest_determinism(self):
        """Test that selftest produces deterministic manifests."""
        # Clean up any existing manifests
        for f in Path(".").glob("manifest_*.json"):
            f.unlink()

        # Run selftest twice with same seed
        cmd_base = [
            sys.executable,
            "-m",
            "env.hermetic",
            "selftest",
            "--task-id",
            "MVP-0-F0.1-T0.1",
            "--seed",
            "123",
        ]

        # First run
        result1 = subprocess.run(
            cmd_base + ["--emit", "manifest_run1.json"],
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_HERMETIC": "1"},
        )

        assert result1.returncode == 0, f"First selftest failed: {result1.stderr}"
        assert Path("manifest_run1.json").exists(), "First manifest not created"

        # Second run
        result2 = subprocess.run(
            cmd_base + ["--emit", "manifest_run2.json"],
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_HERMETIC": "1"},
        )

        assert result2.returncode == 0, f"Second selftest failed: {result2.stderr}"
        assert Path("manifest_run2.json").exists(), "Second manifest not created"

        # Load and normalize manifests
        with open("manifest_run1.json") as f:
            manifest1 = json.load(f)

        with open("manifest_run2.json") as f:
            manifest2 = json.load(f)

        # Remove time-dependent fields
        for manifest in [manifest1, manifest2]:
            manifest.pop("run_id", None)
            manifest.pop("durations", None)

        # Check stable hash matches
        assert (
            manifest1["stable_hash"] == manifest2["stable_hash"]
        ), "Stable hashes don't match for same seed"

        # Check other fields match
        assert manifest1["task_id"] == manifest2["task_id"]
        assert manifest1["seed"] == manifest2["seed"]
        assert manifest1["repo_sha"] == manifest2["repo_sha"]
        assert manifest1["config_hash"] == manifest2["config_hash"]

        # Clean up
        Path("manifest_run1.json").unlink(missing_ok=True)
        Path("manifest_run2.json").unlink(missing_ok=True)

    def test_hermetic_metrics_generation(self):
        """Test that selftest generates proper metrics."""
        # Clean up any existing metrics
        Path("metrics.json").unlink(missing_ok=True)

        # Run selftest with metrics emission
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "env.hermetic",
                "selftest",
                "--task-id",
                "test-metrics",
                "--seed",
                "42",
                "--emit",
                "metrics.json",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_HERMETIC": "1"},
        )

        assert result.returncode == 0, f"Selftest failed: {result.stderr}"
        assert Path("metrics.json").exists(), "Metrics file not created"

        # Load and verify metrics
        with open("metrics.json") as f:
            metrics = json.load(f)

        # Check standard keys are present
        standard_keys = [
            "bytes_per_solve",
            "tokens_prefill",
            "tokens_decode",
            "e2e_latency_ms_p50",
            "e2e_latency_ms_p95",
            "message_path_ms_p95",
            "mcp_deref_ms_p95",
            "sae_accept_rate",
            "rollback_ms_p95",
            "pass_at_1",
        ]

        for key in standard_keys:
            assert key in metrics, f"Missing standard metric: {key}"
            assert metrics[key] is None, f"Standard metric {key} should be null"

        # Check hermetic-specific metrics
        assert "sandbox_setup_ms_p50" in metrics
        assert "sandbox_cleanup_ms_p95" in metrics
        assert "network_guard_install_ms" in metrics

        assert metrics["sandbox_setup_ms_p50"] > 0, "Setup time should be positive"
        assert metrics["sandbox_cleanup_ms_p95"] > 0, "Cleanup time should be positive"

        # Clean up
        Path("metrics.json").unlink(missing_ok=True)

    def test_hermetic_cleanup_verification(self):
        """Test that hermetic run cleans up all artifacts."""
        task_id = "test-cleanup-verification"
        scratch_base = Path("scratch") / task_id

        # Run selftest
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "env.hermetic",
                "selftest",
                "--task-id",
                task_id,
                "--seed",
                "789",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_HERMETIC": "1"},
        )

        assert result.returncode == 0, f"Selftest failed: {result.stderr}"

        # Check cleanup - scratch directory should not exist or be empty
        if scratch_base.exists():
            contents = list(scratch_base.iterdir())
            assert len(contents) == 0, f"Scratch not cleaned: {contents}"

        # Check no temp files remain
        temp_files = list(Path("/tmp").glob("hermes_*"))
        assert len(temp_files) == 0, f"Temp files not cleaned: {temp_files}"

        # Check no stray worktrees
        worktree_result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"], capture_output=True, text=True, check=True
        )

        for line in worktree_result.stdout.split("\n"):
            if line.startswith("worktree "):
                assert "scratch" not in line, f"Stray worktree found: {line}"


@pytest.fixture(autouse=True)
def cleanup_integration_artifacts():
    """Clean up integration test artifacts."""
    yield

    # Clean up manifests
    for f in Path(".").glob("manifest_*.json"):
        f.unlink(missing_ok=True)

    # Clean up metrics
    Path("metrics.json").unlink(missing_ok=True)

    # Clean up scratch directories
    scratch_dir = Path("scratch")
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
