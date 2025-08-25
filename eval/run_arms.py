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
from eval.datasets.swebench_lite import SWEBenchLiteLoader
from eval.swebench_bridge import SWEBenchBridge
from proto import baseline_pb2
from transport.grpc_impl import GrpcTransport


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
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        instances_file: Optional[str] = None,
    ):
        """Initialize arm runner.

        Args:
            arm: Arm to run (A, C, PM, D1, D1_SAE)
            seed: Random seed for determinism
            gen_cfg_path: Path to generation config (must be configs/generation.yaml)
            hermetic: Whether to run in hermetic mode
            dataset: Dataset to use (swebench_lite)
            split: Dataset split (dev or test)
            instances_file: Path to file with instance IDs to run
        """
        if arm not in self.VALID_ARMS:
            raise ValueError(f"Invalid arm: {arm}. Must be one of {self.VALID_ARMS}")

        self.arm = arm
        self.seed = seed
        self.hermetic = hermetic
        self.dataset = dataset
        self.split = split or "test"
        self.instances_file = instances_file

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
            self.config_hash = hashlib.sha256(config_content.encode()).hexdigest()

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
        self.rtt_measurements: List[float] = []  # Track RTT for transport

    def _get_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks to run.

        Returns:
            List of task dictionaries
        """
        # Dataset is required
        if not self.dataset:
            raise ValueError(
                "Dataset must be specified. Use --dataset swebench_lite"
            )
        
        # Load SWE-bench Lite if specified
        if self.dataset == "swebench_lite":
            loader = SWEBenchLiteLoader()
            
            if self.instances_file:
                # Load specific instances from file
                instances = loader.load_instances_file(self.instances_file, self.split)
            else:
                # Full split
                dataset = loader.load_split(self.split)
                instances = list(dataset)
            
            # Convert to task format
            return [loader.to_task_format(inst) for inst in instances]
        
        # No dataset specified
        return []

    def _run_agents_grpc(
        self, task: Dict[str, Any], task_seed: int, hermetic_run
    ) -> Dict[str, Any]:
        """Run agents via gRPC for Arms A, C, and PM.

        Args:
            task: Task dictionary
            task_seed: Deterministic seed for this task
            hermetic_run: HermeticRun instance with scratch path

        Returns:
            Metrics dictionary
        """
        # Socket path - use scratch directory for automatic cleanup
        # This ensures the socket is removed when hermetic run completes
        socket_path = hermetic_run.scratch_base / "grpc.sock"

        # Ensure parent directory exists (scratch_base should already exist)
        socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Create transport with config for PM arm
        if self.arm == "PM":
            transport = GrpcTransport(
                str(socket_path), arm=self.arm, seed=task_seed, config=self.config
            )
        else:
            transport = GrpcTransport(str(socket_path), arm=self.arm, seed=task_seed)

        try:
            # Start server
            transport.start_server()
            time.sleep(0.1)  # Give server time to start

            # Connect client
            transport.connect_client()

            # Track metrics
            total_bytes_in = 0
            total_bytes_out = 0
            message_paths = []
            rtts = []

            # Prepare task data
            task_data = json.dumps(
                {
                    "task_id": task["task_id"],
                    "repo": task.get("repo", "test-repo"),
                    "file_path": task.get("file_path", "src/test.py"),
                    "test_name": task.get("test_name", "test_function"),
                    "description": task.get("description", "Fix the test"),
                }
            ).encode("utf-8")

            # Track E2E timing with monotonic clock
            e2e_start_ns = time.perf_counter_ns()

            # 1. Call Planner
            if self.arm == "A":
                # Arm A: JSON
                result, rtt = transport.call_agent(
                    task["task_id"],
                    "planner",
                    task_data,
                    "application/json",
                    f"trace_{task['task_id']}",
                )
                plan_data = json.loads(result.payload)["steps"] if result.ok else []
            else:
                # Arm C and PM: Protobuf
                plan_req = baseline_pb2.PlanRequest(
                    task_id=task["task_id"],
                    repo=task.get("repo", "test-repo"),
                    file_path=task.get("file_path", "src/test.py"),
                    test_name=task.get("test_name", "test_function"),
                    description=task.get("description", "Fix the test"),
                    seed=task_seed,
                )
                result, rtt = transport.call_agent(
                    task["task_id"],
                    "planner",
                    plan_req.SerializeToString(),
                    "application/x-protobuf",
                    f"trace_{task['task_id']}",
                )
                if result.ok:
                    plan_resp = baseline_pb2.PlanResponse()
                    plan_resp.ParseFromString(result.payload)
                    plan_data = list(plan_resp.steps)
                else:
                    plan_data = []

            total_bytes_in += result.bytes_in
            total_bytes_out += result.bytes_out
            message_paths.append(result.message_path_ms)
            rtts.append(rtt)

            # 2. Call Coder
            if self.arm == "A":
                # Arm A: JSON with plan steps
                code_data = json.dumps(
                    {
                        "task_id": task["task_id"],
                        "file_path": task.get("file_path", "src/test.py"),
                        "plan_steps": plan_data,
                    }
                ).encode("utf-8")
                result, rtt = transport.call_agent(
                    task["task_id"],
                    "coder",
                    code_data,
                    "application/json",
                    f"trace_{task['task_id']}",
                )
                patch = json.loads(result.payload)["patch"] if result.ok else ""
            else:
                # Arm C and PM: Protobuf
                code_req = baseline_pb2.CodeRequest(
                    task_id=task["task_id"],
                    file_path=task.get("file_path", "src/test.py"),
                    plan_steps=plan_data,
                    seed=task_seed,
                )
                result, rtt = transport.call_agent(
                    task["task_id"],
                    "coder",
                    code_req.SerializeToString(),
                    "application/x-protobuf",
                    f"trace_{task['task_id']}",
                )
                if result.ok:
                    code_resp = baseline_pb2.CodeResponse()
                    code_resp.ParseFromString(result.payload)
                    
                    # For PM arm, check if patch is an MCP reference
                    if self.arm == "PM" and code_resp.patch.startswith("mcp://"):
                        # Track that we got an anchor
                        if "mcp_refs" not in self.metrics[-1] if self.metrics else {}:
                            self.metrics.append({"mcp_refs": []})
                        self.metrics[-1]["mcp_refs"].append(code_resp.patch)
                    
                    patch = code_resp.patch
                else:
                    patch = ""

            total_bytes_in += result.bytes_in
            total_bytes_out += result.bytes_out
            message_paths.append(result.message_path_ms)
            rtts.append(rtt)

            # 3. Call Tester
            if self.arm == "A":
                # Arm A: JSON
                test_data = json.dumps(
                    {
                        "task_id": task["task_id"],
                        "test_name": task.get("test_name", "test_function"),
                        "patch": patch,
                    }
                ).encode("utf-8")
                result, rtt = transport.call_agent(
                    task["task_id"],
                    "tester",
                    test_data,
                    "application/json",
                    f"trace_{task['task_id']}",
                )
                passed = json.loads(result.payload)["passed"] if result.ok else False
            else:
                # Arm C and PM: Protobuf
                test_req = baseline_pb2.TestRequest(
                    task_id=task["task_id"],
                    test_name=task.get("test_name", "test_function"),
                    patch=patch,
                    seed=task_seed,
                )
                result, rtt = transport.call_agent(
                    task["task_id"],
                    "tester",
                    test_req.SerializeToString(),
                    "application/x-protobuf",
                    f"trace_{task['task_id']}",
                )
                if result.ok:
                    test_resp = baseline_pb2.TestResponse()
                    test_resp.ParseFromString(result.payload)
                    
                    # For PM arm, check if output is an MCP reference
                    if self.arm == "PM" and test_resp.output.startswith("mcp://"):
                        # Track that we got an anchor
                        if "mcp_refs" not in self.metrics[-1] if self.metrics else {}:
                            self.metrics.append({"mcp_refs": []})
                        self.metrics[-1]["mcp_refs"].append(test_resp.output)
                    
                    passed = test_resp.passed
                else:
                    passed = False

            total_bytes_in += result.bytes_in
            total_bytes_out += result.bytes_out
            message_paths.append(result.message_path_ms)
            rtts.append(rtt)

            # Store RTT measurements
            self.rtt_measurements.extend(rtts)

            # Write RTT data to file
            rtt_file = self.output_dir / "transport_rtts.jsonl"
            with open(rtt_file, "a") as f:
                for r in rtts:
                    f.write(json.dumps({"task_id": task["task_id"], "rtt_ms": r}) + "\n")

            # Calculate e2e latency properly with monotonic clock
            # E2E = total time from first request to last response
            e2e_end_ns = time.perf_counter_ns()
            e2e_latency_ms = (e2e_end_ns - e2e_start_ns) / 1_000_000  # Convert ns to ms

            # Calculate p95 message path (processing time inside agents)
            if message_paths:
                # Filter out zeros and calculate p95
                valid_paths = [p for p in message_paths if p > 0]
                if valid_paths:
                    sorted_paths = sorted(valid_paths)
                    p95_idx = min(int(len(sorted_paths) * 0.95), len(sorted_paths) - 1)
                    message_path_p95 = sorted_paths[p95_idx]
                else:
                    # All zeros - use minimum measurable time
                    message_path_p95 = 0.001  # 1 microsecond minimum
            else:
                message_path_p95 = 0.001

            return {
                "bytes_in": total_bytes_in,
                "bytes_out": total_bytes_out,
                "e2e_latency_ms": e2e_latency_ms,
                "message_path_ms": message_path_p95,
                "pass": passed,
                "sandbox_setup_ms": hermetic_run.manifest["durations"]["setup_ms"],
                # Capture full manifest with scratch listing
                "run_manifest": hermetic_run.emit_manifest(),
                # Mock token counts for now (would come from actual LLM)
                "tokens_out": 250 + (task_seed % 100),
                "tokens_in": 200 + (task_seed % 80),
                "prefill_tokens": 150 + (task_seed % 50),
                "decode_tokens": 100 + (task_seed % 30),
            }

        finally:
            transport.stop()

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
            "task_seed": task_seed,
            "hermetic": self.hermetic,
            "start_time": time.time(),
        }

        with hermetic_run():
            # Seed RNGs within hermetic environment
            seed_info = seed_all(task_seed, verbose=False)
            metrics["seed_info"] = seed_info

            # Run actual agents for Arms A, C, and PM
            if self.arm in ["A", "C", "PM"]:
                agent_metrics = self._run_agents_grpc(task, task_seed, hermetic_run)
                metrics.update(agent_metrics)
            else:
                # Original mock for other arms
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
                        # Capture full manifest with scratch listing
                        "run_manifest": hermetic_run.emit_manifest(),
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
        df["global_seed"] = self.seed  # Global seed used for the run
        # task_seed is already preserved in each metric dict from _run_task

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
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        choices=["swebench_lite"],
        required=True,
        help="Dataset to use (swebench_lite)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--instances_file",
        type=str,
        help="Path to file with instance IDs to run (one per line)",
    )

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
            dataset=args.dataset,
            split=args.split,
            instances_file=args.instances_file,
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
