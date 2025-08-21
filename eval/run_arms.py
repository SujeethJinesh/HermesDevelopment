#!/usr/bin/env python3
"""HERMES evaluation harness with strict config parity and deterministic execution.

Provides single entrypoint for running Arms with hermetic execution.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.hermetic import HermeticRun
from eval._seed import compute_task_seed, generate_deterministic_id, seed_all


class ConfigParityError(Exception):
    """Raised when config parity is violated."""

    pass


class ArmRunner:
    """Manages evaluation runs with strict config parity."""

    VALID_ARMS = {"A", "C", "PM", "D1", "D1_SAE"}

    def __init__(
        self,
        arm: str,
        seed: int,
        gen_cfg_path: str,
        hermetic: bool = True,
        toy_tasks: Optional[int] = None,
    ):
        """Initialize arm runner.

        Args:
            arm: Arm to run (A, C, PM, D1, D1_SAE)
            seed: Random seed for determinism
            gen_cfg_path: Path to generation config (must be configs/generation.yaml)
            hermetic: Whether to run in hermetic mode
            toy_tasks: Number of toy tasks to run (for testing)
        """
        if arm not in self.VALID_ARMS:
            raise ValueError(f"Invalid arm: {arm}. Must be one of {self.VALID_ARMS}")

        self.arm = arm
        self.seed = seed
        self.hermetic = hermetic
        self.toy_tasks = toy_tasks

        # Enforce config parity - only accept the canonical config
        if gen_cfg_path != "configs/generation.yaml":
            raise ConfigParityError(
                f"Config parity violation: only 'configs/generation.yaml' is allowed, "
                f"got '{gen_cfg_path}'. No overrides permitted."
            )

        self.gen_cfg_path = Path(gen_cfg_path)
        if not self.gen_cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {self.gen_cfg_path}")

        # Load and hash config
        with open(self.gen_cfg_path, "r") as f:
            self.config = yaml.safe_load(f)
            f.seek(0)
            config_content = f.read()
            self.config_hash = hashlib.sha256(config_content.encode()).hexdigest()[:16]

        # Create run ID
        self.run_id = generate_deterministic_id(seed, f"arm_{arm}")

        # Output paths
        self.output_dir = Path("runs") / arm
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics: List[Dict[str, Any]] = []
        self.warmup_count = 5  # First 5 inferences are warmup
        self.inference_count = 0
        self.token_timings: List[float] = []

    def _get_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks to run.

        Returns:
            List of task dictionaries
        """
        if self.toy_tasks:
            # Create toy tasks for testing
            tasks = []
            for i in range(self.toy_tasks):
                tasks.append(
                    {
                        "task_id": f"toy-{i:03d}",
                        "repo": "test-repo",
                        "file_path": f"src/test_{i}.py",
                        "test_name": f"test_function_{i}",
                        "description": f"Toy task {i} for testing",
                    }
                )
            return tasks

        # Would load real SWE-bench tasks here
        # For now, return empty list as we're focusing on the harness
        return []

    def _run_task_hermetic(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single task in hermetic environment.

        Args:
            task: Task dictionary

        Returns:
            Task metrics
        """
        task_id = task["task_id"]
        task_seed = compute_task_seed(self.seed, task_id)

        # Create hermetic run
        hermetic_run = HermeticRun(
            task_id=task_id,
            run_id=f"{self.run_id}_{task_id}",
            seed=task_seed,
            hermetic=self.hermetic,
        )

        metrics = {
            "task_id": task_id,
            "arm": self.arm,
            "seed": task_seed,
            "hermetic": self.hermetic,
            "start_time": time.time(),
        }

        with hermetic_run():
            # Seed RNGs within hermetic environment
            seed_info = seed_all(task_seed, verbose=False)
            metrics["seed_info"] = seed_info

            # Simulate task execution (would call actual arm here)
            # For now, generate deterministic mock metrics
            import random

            # Mock inference timing (excluding warmup)
            inference_time = 0.1 + random.random() * 0.05
            self.inference_count += 1
            if self.inference_count > self.warmup_count:
                self.token_timings.append(inference_time)

            # Mock deterministic metrics based on seed
            metrics.update(
                {
                    "bytes_out": 1000 + (task_seed % 500),
                    "bytes_in": 800 + (task_seed % 300),
                    "tokens_out": 250 + (task_seed % 100),
                    "tokens_in": 200 + (task_seed % 80),
                    "prefill_tokens": 150 + (task_seed % 50),
                    "decode_tokens": 100 + (task_seed % 30),
                    "e2e_latency_ms": 2000 + (task_seed % 1000),
                    "message_path_ms": 5 + (task_seed % 3),
                    "pass": (task_seed % 3) != 0,  # Deterministic pass/fail
                    "sandbox_setup_ms": hermetic_run.manifest["durations"]["setup_ms"],
                    "run_manifest": hermetic_run.manifest,
                }
            )

        # Add cleanup timing
        metrics["sandbox_cleanup_ms"] = hermetic_run.manifest["durations"].get("cleanup_ms", 0)
        metrics["end_time"] = time.time()
        metrics["duration"] = metrics["end_time"] - metrics["start_time"]

        return metrics

    def run(self) -> None:
        """Run evaluation for the arm."""
        print(f"Starting evaluation for arm {self.arm}")
        print(f"  Seed: {self.seed}")
        print(f"  Config: {self.gen_cfg_path} (hash: {self.config_hash})")
        print(f"  Hermetic: {self.hermetic}")
        print(f"  Run ID: {self.run_id}")

        # Set hermetic env var if needed
        if self.hermetic:
            os.environ["HERMES_HERMETIC"] = "1"

        # Seed global RNGs
        seed_all(self.seed, verbose=True)

        # Get tasks
        tasks = self._get_tasks()
        if not tasks:
            print("No tasks to run")
            return

        print(f"Running {len(tasks)} tasks...")

        # Run each task
        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] Running {task['task_id']}...")
            metrics = self._run_task_hermetic(task)
            self.metrics.append(metrics)

            # Write metrics incrementally
            self._write_metrics_jsonl(metrics)

        # Write final summary
        self._write_summary_parquet()

        # Compute aggregate metrics
        self._print_summary()

    def _write_metrics_jsonl(self, metrics: Dict[str, Any]) -> None:
        """Write metrics to JSONL file.

        Args:
            metrics: Task metrics to write
        """
        metrics_file = self.output_dir / "metrics.jsonl"

        # Remove non-serializable fields for JSONL
        clean_metrics = {
            k: v for k, v in metrics.items() if k != "run_manifest"  # Too large for JSONL
        }

        with open(metrics_file, "a") as f:
            f.write(json.dumps(clean_metrics, sort_keys=True) + "\n")

    def _write_summary_parquet(self) -> None:
        """Write summary to Parquet file."""
        if not self.metrics:
            return

        # Create DataFrame
        df = pd.DataFrame(self.metrics)

        # Remove complex nested fields that don't fit well in Parquet
        if "run_manifest" in df.columns:
            df = df.drop(columns=["run_manifest"])
        if "seed_info" in df.columns:
            df = df.drop(columns=["seed_info"])

        # Add metadata
        df["arm"] = self.arm
        df["run_id"] = self.run_id
        df["config_hash"] = self.config_hash
        df["seed"] = self.seed

        # Sort columns for consistent output
        df = df.reindex(sorted(df.columns), axis=1)

        # Write Parquet
        summary_file = self.output_dir / "summary.parquet"
        df.to_parquet(summary_file, compression="snappy", index=False)

        print(f"Summary written to {summary_file}")

    def _print_summary(self) -> None:
        """Print evaluation summary."""
        if not self.metrics:
            return

        # Calculate aggregates
        total_tasks = len(self.metrics)
        passed_tasks = sum(1 for m in self.metrics if m.get("pass", False))
        pass_rate = passed_tasks / total_tasks if total_tasks > 0 else 0

        # Calculate tokens/s (excluding warmup)
        if self.token_timings:
            avg_inference_time = sum(self.token_timings) / len(self.token_timings)
            tokens_per_sec = 1.0 / avg_inference_time * 100  # Mock 100 tokens per inference
        else:
            tokens_per_sec = 0

        # Calculate percentiles
        latencies = [m["e2e_latency_ms"] for m in self.metrics]
        p50_latency = pd.Series(latencies).quantile(0.5) if latencies else 0
        p95_latency = pd.Series(latencies).quantile(0.95) if latencies else 0

        print("\n" + "=" * 60)
        print(f"Evaluation Summary for Arm {self.arm}")
        print("=" * 60)
        print(f"  Total tasks: {total_tasks}")
        print(f"  Passed: {passed_tasks}/{total_tasks} ({pass_rate:.1%})")
        print(f"  E2E latency p50: {p50_latency:.0f} ms")
        print(f"  E2E latency p95: {p95_latency:.0f} ms")
        print(f"  Tokens/s (post-warmup): {tokens_per_sec:.1f}")
        print(f"  Config hash: {self.config_hash}")
        print(f"  Run ID: {self.run_id}")
        print("=" * 60)


def enforce_config_parity(args: argparse.Namespace) -> None:
    """Enforce strict config parity - no overrides allowed.

    Args:
        args: Command-line arguments

    Raises:
        ConfigParityError: If any config override is attempted
    """
    # Check for common override attempts
    override_attrs = [
        "temperature",
        "top_p",
        "max_tokens",
        "model",
        "num_beams",
        "do_sample",
        "repetition_penalty",
    ]

    for attr in override_attrs:
        if hasattr(args, attr):
            raise ConfigParityError(
                f"Config override attempted: --{attr.replace('_', '-')}. "
                f"Only configs/generation.yaml is allowed. No overrides permitted."
            )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HERMES evaluation harness with strict config parity"
    )

    # Required arguments
    parser.add_argument(
        "--arm", choices=["A", "C", "PM", "D1", "D1_SAE"], required=True, help="Arm to evaluate"
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed for determinism")
    parser.add_argument(
        "--gen_cfg",
        default="configs/generation.yaml",
        help="Generation config path (must be configs/generation.yaml)",
    )

    # Optional arguments
    parser.add_argument(
        "--hermetic",
        choices=["on", "off"],
        default="on",
        help="Enable hermetic execution (default: on)",
    )
    parser.add_argument("--toy", type=int, metavar="N", help="Run N toy tasks for testing")

    # Parse arguments
    args = parser.parse_args()

    # Enforce config parity
    try:
        enforce_config_parity(args)
    except ConfigParityError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Create and run evaluation
    try:
        runner = ArmRunner(
            arm=args.arm,
            seed=args.seed,
            gen_cfg_path=args.gen_cfg,
            hermetic=(args.hermetic == "on"),
            toy_tasks=args.toy,
        )
        runner.run()
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
