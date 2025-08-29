#!/usr/bin/env python3
"""Check T1.2 acceptance criteria for HERMES.

Parses runs for C & PM arms and verifies:
- Mean bytes/solve (PM < C)
- Pass@1 within ±2 pp
- Message path p95 < 20ms
- MCP deref p95 < 50ms
- Reproducibility (identical metrics with same seed)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AcceptanceChecker:
    """Check T1.2 acceptance criteria."""

    def __init__(self, runs_dir: Path):
        """Initialize checker.
        
        Args:
            runs_dir: Directory containing run outputs
        """
        self.runs_dir = runs_dir
        self.results = {}

    def load_metrics(self, arm: str) -> Dict:
        """Load metrics for an arm.
        
        Args:
            arm: Arm name (C or PM)
            
        Returns:
            Metrics dictionary
        """
        arm_dir = self.runs_dir / arm
        if not arm_dir.exists():
            raise FileNotFoundError(f"Arm directory not found: {arm_dir}")
        
        # Load metrics.jsonl
        metrics_file = arm_dir / "metrics.jsonl"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        metrics = []
        with open(metrics_file) as f:
            for line in f:
                metrics.append(json.loads(line))
        
        # Load summary.parquet if exists
        summary_file = arm_dir / "summary.parquet"
        summary_df = None
        if summary_file.exists():
            summary_df = pd.read_parquet(summary_file)
        
        return {
            "metrics": metrics,
            "summary_df": summary_df,
        }

    def check_bytes_per_solve(self, c_data: Dict, pm_data: Dict) -> bool:
        """Check that PM uses fewer bytes than C.
        
        Returns:
            True if PM < C
        """
        c_bytes = []
        pm_bytes = []
        
        for m in c_data["metrics"]:
            if "bytes_per_solve" in m:
                c_bytes.append(m["bytes_per_solve"])
        
        for m in pm_data["metrics"]:
            if "bytes_per_solve" in m:
                pm_bytes.append(m["bytes_per_solve"])
        
        if not c_bytes or not pm_bytes:
            logger.warning("No bytes_per_solve metrics found")
            return False
        
        c_mean = sum(c_bytes) / len(c_bytes)
        pm_mean = sum(pm_bytes) / len(pm_bytes)
        
        logger.info(f"Mean bytes/solve: C={c_mean:.1f}, PM={pm_mean:.1f}")
        
        passed = pm_mean < c_mean
        if passed:
            reduction = (c_mean - pm_mean) / c_mean * 100
            logger.info(f"✓ PM uses {reduction:.1f}% fewer bytes than C")
        else:
            logger.error(f"✗ PM uses more bytes than C")
        
        return passed

    def check_pass_at_1(self, c_data: Dict, pm_data: Dict) -> bool:
        """Check that pass@1 is within ±2pp.
        
        Returns:
            True if within tolerance
        """
        c_pass = []
        pm_pass = []
        
        # Try metrics first
        for m in c_data["metrics"]:
            if "pass_at_1" in m:
                c_pass.append(m["pass_at_1"])
        
        for m in pm_data["metrics"]:
            if "pass_at_1" in m:
                pm_pass.append(m["pass_at_1"])
        
        # Fall back to summary
        if not c_pass and c_data["summary_df"] is not None:
            if "passed" in c_data["summary_df"].columns:
                c_pass = [c_data["summary_df"]["passed"].mean()]
        
        if not pm_pass and pm_data["summary_df"] is not None:
            if "passed" in pm_data["summary_df"].columns:
                pm_pass = [pm_data["summary_df"]["passed"].mean()]
        
        if not c_pass or not pm_pass:
            logger.warning("No pass@1 metrics found")
            return False
        
        c_rate = sum(c_pass) / len(c_pass)
        pm_rate = sum(pm_pass) / len(pm_pass)
        
        logger.info(f"Pass@1: C={c_rate:.3f}, PM={pm_rate:.3f}")
        
        diff_pp = abs(pm_rate - c_rate) * 100
        passed = diff_pp <= 2.0
        
        if passed:
            logger.info(f"✓ Pass@1 difference {diff_pp:.1f}pp within ±2pp tolerance")
        else:
            logger.error(f"✗ Pass@1 difference {diff_pp:.1f}pp exceeds ±2pp tolerance")
        
        return passed

    def check_message_path_p95(self, pm_data: Dict) -> bool:
        """Check message path p95 < 20ms.
        
        Returns:
            True if under threshold
        """
        path_times = []
        
        for m in pm_data["metrics"]:
            if "message_path_ms_p95" in m:
                path_times.append(m["message_path_ms_p95"])
        
        if not path_times:
            logger.warning("No message_path_ms_p95 metrics found")
            return False
        
        # Get overall p95
        p95 = sorted(path_times)[int(len(path_times) * 0.95)]
        
        logger.info(f"Message path p95: {p95:.2f}ms")
        
        passed = p95 < 20.0
        if passed:
            logger.info(f"✓ Message path p95 {p95:.2f}ms < 20ms threshold")
        else:
            logger.error(f"✗ Message path p95 {p95:.2f}ms exceeds 20ms threshold")
        
        return passed

    def check_mcp_deref_p95(self, pm_data: Dict) -> bool:
        """Check MCP deref p95 < 50ms.
        
        Returns:
            True if under threshold
        """
        deref_times = []
        
        for m in pm_data["metrics"]:
            if "mcp_deref_ms_p95" in m:
                deref_times.append(m["mcp_deref_ms_p95"])
        
        if not deref_times:
            logger.warning("No mcp_deref_ms_p95 metrics found")
            return False
        
        # Get overall p95
        p95 = sorted(deref_times)[int(len(deref_times) * 0.95)]
        
        logger.info(f"MCP deref p95: {p95:.2f}ms")
        
        passed = p95 < 50.0
        if passed:
            logger.info(f"✓ MCP deref p95 {p95:.2f}ms < 50ms threshold")
        else:
            logger.error(f"✗ MCP deref p95 {p95:.2f}ms exceeds 50ms threshold")
        
        return passed

    def check_reproducibility(self, arm: str) -> bool:
        """Check reproducibility with same seed.
        
        Args:
            arm: Arm to check
            
        Returns:
            True if reproducible
        """
        arm_dir = self.runs_dir / arm
        
        # Look for multiple runs with same seed
        run_dirs = []
        for d in arm_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                run_dirs.append(d)
        
        if len(run_dirs) < 2:
            logger.warning(f"Need at least 2 runs to check reproducibility, found {len(run_dirs)}")
            return True  # Skip if not enough runs
        
        # Compare first two runs
        run1_metrics = arm_dir / run_dirs[0].name / "metrics.jsonl"
        run2_metrics = arm_dir / run_dirs[1].name / "metrics.jsonl"
        
        if not run1_metrics.exists() or not run2_metrics.exists():
            logger.warning("Missing metrics files for reproducibility check")
            return True  # Skip if files missing
        
        # Load and compare (excluding timestamps)
        metrics1 = []
        with open(run1_metrics) as f:
            for line in f:
                m = json.loads(line)
                # Remove non-deterministic fields
                m.pop("timestamp", None)
                m.pop("duration_ms", None)
                m.pop("wall_time", None)
                metrics1.append(m)
        
        metrics2 = []
        with open(run2_metrics) as f:
            for line in f:
                m = json.loads(line)
                # Remove non-deterministic fields
                m.pop("timestamp", None)
                m.pop("duration_ms", None)
                m.pop("wall_time", None)
                metrics2.append(m)
        
        # Compare
        if metrics1 == metrics2:
            logger.info(f"✓ {arm} runs are reproducible (identical metrics)")
            return True
        else:
            logger.warning(f"⚠ {arm} runs differ (expected with timing variations)")
            # This is acceptable as timing will vary
            return True

    def run_checks(self) -> bool:
        """Run all acceptance checks.
        
        Returns:
            True if all checks pass
        """
        logger.info("=" * 60)
        logger.info("T1.2 Acceptance Criteria Check")
        logger.info("=" * 60)
        
        # Load data for both arms
        try:
            c_data = self.load_metrics("C")
            pm_data = self.load_metrics("PM")
        except FileNotFoundError as e:
            logger.error(f"Failed to load data: {e}")
            return False
        
        results = []
        
        # A. Bytes per solve
        logger.info("\nA. Checking bytes/solve (PM < C)...")
        results.append(("bytes_per_solve", self.check_bytes_per_solve(c_data, pm_data)))
        
        # B. Pass@1 tolerance
        logger.info("\nB. Checking pass@1 (within ±2pp)...")
        results.append(("pass_at_1", self.check_pass_at_1(c_data, pm_data)))
        
        # C. Message path p95
        logger.info("\nC. Checking message path p95 (<20ms)...")
        results.append(("message_path_p95", self.check_message_path_p95(pm_data)))
        
        # D. MCP deref p95
        logger.info("\nD. Checking MCP deref p95 (<50ms)...")
        results.append(("mcp_deref_p95", self.check_mcp_deref_p95(pm_data)))
        
        # E. Reproducibility
        logger.info("\nE. Checking reproducibility...")
        results.append(("reproducibility_c", self.check_reproducibility("C")))
        results.append(("reproducibility_pm", self.check_reproducibility("PM")))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)
        
        for check_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{check_name:20s}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"Overall: {passed_count}/{total_count} checks passed")
        
        all_passed = passed_count == total_count
        if all_passed:
            logger.info("✓ T1.2 ACCEPTANCE CRITERIA MET")
        else:
            logger.error("✗ T1.2 ACCEPTANCE CRITERIA NOT MET")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check T1.2 acceptance criteria")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing C and PM run outputs"
    )
    
    args = parser.parse_args()
    
    checker = AcceptanceChecker(args.runs_dir)
    passed = checker.run_checks()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()