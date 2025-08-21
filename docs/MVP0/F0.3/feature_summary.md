# MVP-0 F0.3: Evaluation Harness & Parity

## Overview
Feature F0.3 implements a deterministic evaluation harness with strict configuration parity enforcement for benchmarking HERMES Arms.

## Completed Tasks

### T0.3: Deterministic Evaluation Harness ✅
**Status**: COMPLETED (2025-08-21)

**Implemented**:
- `eval/run_arms.py`: Main harness with CLI, config parity, and hermetic execution
- `eval/_seed.py`: Deterministic seeding utilities for all RNGs
- Full integration with `env.hermetic` for isolated execution
- Metrics emission to both JSONL and Parquet formats
- Comprehensive test coverage (unit + integration)

**Key Achievements**:
- ✅ **Determinism Verified**: Two identical runs (seed=123) produce identical output (hash: `2f1cc621cd315b82`)
- ✅ **Config Parity Enforced**: Only `configs/generation.yaml` accepted, all overrides rejected
- ✅ **Hermetic Execution**: Network-blocked, isolated worktree + venv per task
- ✅ **Metrics Schema**: Complete Parquet schema with all required fields

**Evidence**:
```bash
# Determinism proof
✓ Hash run1: 2f1cc621cd315b82
✓ Hash run2: 2f1cc621cd315b82
✓ Hashes match: True

# Config parity enforcement
$ python3 -m eval.run_arms --arm A --seed 123 --gen_cfg configs/custom.yaml
ERROR: Config parity violation: only 'configs/generation.yaml' is allowed

# Test results
Unit tests: 9/9 passing
Integration tests: 1/1 passing (determinism verified)
```

## Feature Status: COMPLETE ✅

All F0.3 tasks completed successfully. The evaluation harness is ready for integration with real Arms and SWE-bench tasks.

## Files Added/Modified
- `eval/run_arms.py` (385 lines)
- `eval/_seed.py` (147 lines)
- `eval/__init__.py` (3 lines)
- `tests/test_run_arms_parity.py` (216 lines)
- `tests/integration/test_run_arms_determinism.py` (374 lines)
- `docs/MVP0/F0.3/T0.3_summary.md`
- `docs/MVP0/F0.3/feature_summary.md`

## Next Steps
1. Implement real Arms (A, C, PM, D1, D1_SAE)
2. Integrate with SWE-bench Lite dataset
3. Run full benchmark suite with all Arms
4. Collect baseline metrics for comparison