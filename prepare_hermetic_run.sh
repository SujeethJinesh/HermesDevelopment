#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1

prep() {
  echo "==> Preparing dataset cache and git mirrors (one-time, online)…"
  python3 scripts/prepare_swebench_data.py --dataset SWE-bench/SWE-bench_Lite --splits dev test
  python3 scripts/prepare_swebench_repos.py --instances_file configs/swebench_lite_slice20.txt
  echo "✅ Preparation complete"
}

eval_offline() {
  echo "==> Running hermetic evaluation (offline)…"
  export HERMES_HERMETIC=1
  export HF_DATASETS_OFFLINE=1
  export HF_HOME="$ROOT/.hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HERMES_MIRRORS_DIR="$ROOT/.mirrors"
  mkdir -p "$HF_DATASETS_CACHE"

  echo "Running C arm..."
  python3 -m eval.run_arms --arm C \
    --seed 12345 \
    --dataset swebench_lite --split test \
    --instances_file configs/swebench_lite_slice20.txt \
    --gen_cfg configs/generation.yaml --hermetic on

  echo "Running PM arm..."
  python3 -m eval.run_arms --arm PM \
    --seed 12345 \
    --dataset swebench_lite --split test \
    --instances_file configs/swebench_lite_slice20.txt \
    --gen_cfg configs/generation.yaml --hermetic on

  echo "✅ Evaluation complete"
}

accept() {
  echo "==> Checking acceptance criteria…"
  python3 scripts/check_acceptance_final.py runs/C runs/PM
}

case "${1:-}" in
  prep) prep ;;
  eval) eval_offline ;;
  accept) accept ;;
  *)
    echo "Usage: $0 {prep|eval|accept}"
    echo ""
    echo "  prep   - Download dataset and create repo mirrors (online)"
    echo "  eval   - Run hermetic evaluation (offline)"
    echo "  accept - Check acceptance criteria"
    exit 2
    ;;
esac