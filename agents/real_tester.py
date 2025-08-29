"""RealTester agent with strict MCP deref, robust git apply, and pytest execution."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from agents.pm_arm import PMAnchorManager
from mcp.client import MCPClient

logger = logging.getLogger(__name__)


class RealTester:
    """Test agent that applies patches and runs pytest with log anchoring."""

    def __init__(self, mcp_client: MCPClient, pm_manager: Optional[PMAnchorManager] = None):
        """Initialize RealTester.

        Args:
            mcp_client: MCP client for resolving references
            pm_manager: PM anchor manager for log anchoring
        """
        self.mcp_client = mcp_client
        self.pm_manager = pm_manager or PMAnchorManager(mcp_client)

    def _load_patch_bytes(self, patch_or_ref: Union[str, bytes]) -> bytes:
        """Load patch content as bytes with strict MCP resolution.

        Args:
            patch_or_ref: Either raw bytes, MCP reference, or string content

        Returns:
            Patch content as bytes

        Raises:
            ValueError: If MCP reference is missing or empty
        """
        if isinstance(patch_or_ref, bytes):
            # Already bytes, return as-is
            return patch_or_ref

        if isinstance(patch_or_ref, str):
            if patch_or_ref.startswith("mcp://"):
                # MCP reference - strict resolution
                logger.debug(f"Resolving MCP reference: {patch_or_ref}")
                data = self.mcp_client.resolve_bytes(patch_or_ref)
                
                if not data:
                    raise ValueError(f"MCP reference {patch_or_ref} is missing or empty")
                
                return data
            else:
                # String content - encode to bytes
                return patch_or_ref.encode("utf-8")
        
        # Fallback for other types
        return str(patch_or_ref).encode("utf-8")

    def apply_patch(self, repo_dir: Union[str, Path], patch_bytes: bytes) -> None:
        """Apply patch to repository with robust fallback sequence.

        Tries git apply with various strategies:
        1. --3way with -p0, -p1, -p2 (three-way merge)
        2. Regular apply with -p0, -p1, -p2 (fallback)

        Args:
            repo_dir: Repository directory path
            patch_bytes: Patch content as bytes

        Raises:
            RuntimeError: If all apply attempts fail
        """
        repo_path = Path(repo_dir).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository directory does not exist: {repo_path}")

        # Write patch to temporary file
        patch_file = repo_path / ".hermes.patch"
        patch_file.write_bytes(patch_bytes)
        logger.debug(f"Wrote patch to {patch_file} ({len(patch_bytes)} bytes)")

        # Try different git apply strategies
        strategies = [
            # Three-way merge attempts (more robust)
            ["git", "apply", "--3way", "-p0", str(patch_file)],
            ["git", "apply", "--3way", "-p1", str(patch_file)],
            ["git", "apply", "--3way", "-p2", str(patch_file)],
            # Regular apply fallbacks
            ["git", "apply", "-p0", str(patch_file)],
            ["git", "apply", "-p1", str(patch_file)],
            ["git", "apply", "-p2", str(patch_file)],
        ]

        last_stderr = ""
        for cmd in strategies:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Successfully applied patch with: {' '.join(cmd[2:])}")
                return
            except subprocess.CalledProcessError as e:
                last_stderr = e.stderr
                logger.debug(f"Failed with {' '.join(cmd[2:])}: {e.stderr}")
                continue

        # All strategies failed
        raise RuntimeError(
            f"Failed to apply patch with all strategies. Last error:\n{last_stderr}"
        )

    def run_pytest_and_anchor_logs(
        self, repo_dir: Union[str, Path]
    ) -> Tuple[Union[str, bytes], bool, int]:
        """Run pytest and optionally anchor logs if large.

        Args:
            repo_dir: Repository directory to run tests in

        Returns:
            Tuple of:
            - Log content (bytes if inline, mcp:// ref if anchored)
            - Whether logs were anchored
            - pytest return code
        """
        repo_path = Path(repo_dir).resolve()
        
        # Run pytest with quiet mode and capture output
        cmd = ["pytest", "-q"]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=False,  # Get bytes directly
            timeout=300  # 5 minute timeout
        )

        # Combine stdout and stderr
        log_bytes = result.stdout + b"\n" + result.stderr
        
        logger.debug(f"pytest output: {len(log_bytes)} bytes, return code: {result.returncode}")

        # Use PM manager to maybe anchor logs (1KB threshold)
        payload_or_ref, anchored = self.pm_manager.maybe_anchor(
            log_bytes, 
            kind="logs", 
            ttl_s=None  # Use default TTL for logs
        )

        return payload_or_ref, anchored, result.returncode

    def test_repository(
        self, 
        repo_dir: Union[str, Path],
        patch_or_ref: Union[str, bytes]
    ) -> dict:
        """Apply patch and run tests on repository.

        Args:
            repo_dir: Repository directory
            patch_or_ref: Patch content or MCP reference

        Returns:
            Dictionary with test results including logs and metrics
        """
        # Load patch bytes with strict MCP resolution
        patch_bytes = self._load_patch_bytes(patch_or_ref)
        
        # Apply patch
        self.apply_patch(repo_dir, patch_bytes)
        
        # Run tests and anchor logs if needed
        log_payload, logs_anchored, return_code = self.run_pytest_and_anchor_logs(repo_dir)
        
        # Get PM metrics
        pm_metrics = {
            "anchors_created": self.pm_manager.metrics.anchors_created,
            "bytes_saved": self.pm_manager.metrics.bytes_saved,
            "inline_count": self.pm_manager.metrics.inline_count,
            "anchor_count": self.pm_manager.metrics.anchor_count,
        }
        
        return {
            "success": return_code == 0,
            "return_code": return_code,
            "logs": log_payload,
            "logs_anchored": logs_anchored,
            "pm_metrics": pm_metrics,
        }