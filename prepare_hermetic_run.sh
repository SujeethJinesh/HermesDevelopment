#!/bin/bash
# Prepare and run hermetic evaluation for T1.2 acceptance
# This script documents the exact steps needed for a clean hermetic run

set -e

echo "=== T1.2 Hermetic Evaluation Preparation ==="
echo ""
echo "This script will guide you through preparing and running hermetic evaluation."
echo "The preparation phase requires network access (one-time, ~30 min)."
echo "The evaluation phase runs completely offline."
echo ""

# Check if we're in preparation or evaluation mode
if [ "$1" = "eval" ]; then
    # EVALUATION MODE (offline)
    echo "=== EVALUATION MODE (offline) ==="
    
    # Set hermetic environment
    export HERMES_HERMETIC=1
    export HF_DATASETS_OFFLINE=1
    export HF_HOME="$PWD/.hf"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HERMES_REPO_MIRRORS=".mirrors"
    export HERMES_WORKTREES="scratch/repos"
    
    echo "Environment variables set:"
    echo "  HERMES_HERMETIC=$HERMES_HERMETIC"
    echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
    echo "  HF_HOME=$HF_HOME"
    echo "  HERMES_REPO_MIRRORS=$HERMES_REPO_MIRRORS"
    echo ""
    
    # Check prerequisites
    echo "Checking prerequisites..."
    
    if [ ! -d "$HF_DATASETS_CACHE" ]; then
        echo "❌ Dataset cache not found at $HF_DATASETS_CACHE"
        echo "   Run: $0 prep"
        exit 1
    fi
    
    if [ ! -d "$HERMES_REPO_MIRRORS" ]; then
        echo "❌ Repo mirrors not found at $HERMES_REPO_MIRRORS"
        echo "   Run: $0 prep"
        exit 1
    fi
    
    if [ ! -f "configs/swebench_lite_slice20.txt" ]; then
        echo "❌ Slice20 config not found"
        exit 1
    fi
    
    echo "✅ Prerequisites checked"
    echo ""
    
    # Clean previous runs
    echo "Cleaning previous runs..."
    rm -rf runs/C runs/PM
    mkdir -p runs/C runs/PM
    
    # Run C arm
    echo "Running C arm evaluation..."
    echo "Command: python3 -m eval.run_arms --arm C --seed 12345 --dataset swebench_lite \\"
    echo "  --instances_file configs/swebench_lite_slice20.txt --gen_cfg configs/generation.yaml"
    echo ""
    
    python3 -m eval.run_arms \
        --arm C \
        --seed 12345 \
        --dataset swebench_lite \
        --instances_file configs/swebench_lite_slice20.txt \
        --gen_cfg configs/generation.yaml \
        --hermetic on || echo "C arm completed with status $?"
    
    echo ""
    echo "Running PM arm evaluation..."
    echo "Command: python3 -m eval.run_arms --arm PM --seed 12345 --dataset swebench_lite \\"
    echo "  --instances_file configs/swebench_lite_slice20.txt --gen_cfg configs/generation.yaml"
    echo ""
    
    python3 -m eval.run_arms \
        --arm PM \
        --seed 12345 \
        --dataset swebench_lite \
        --instances_file configs/swebench_lite_slice20.txt \
        --gen_cfg configs/generation.yaml \
        --hermetic on || echo "PM arm completed with status $?"
    
    echo ""
    echo "=== Evaluation Complete ==="
    echo ""
    echo "Checking acceptance criteria..."
    python3 scripts/check_acceptance.py
    
elif [ "$1" = "prep" ]; then
    # PREPARATION MODE (requires network)
    echo "=== PREPARATION MODE (requires network) ==="
    echo ""
    echo "This will download ~2GB of data and take ~30 minutes."
    echo "Press Ctrl+C to cancel, or Enter to continue..."
    read -r
    
    # Prepare dataset cache
    echo "1. Preparing SWE-bench Lite dataset cache..."
    export HF_HOME="$PWD/.hf"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    mkdir -p "$HF_DATASETS_CACHE"
    
    python3 -c "
from datasets import load_dataset
print('Downloading SWE-bench/SWE-bench_Lite...')
ds = load_dataset('SWE-bench/SWE-bench_Lite')
print(f'Cached: dev={len(ds[\"dev\"])}, test={len(ds[\"test\"])} instances')
"
    
    # Prepare repo mirrors
    echo ""
    echo "2. Preparing repository mirrors for slice20..."
    export HERMES_REPO_MIRRORS=".mirrors"
    mkdir -p "$HERMES_REPO_MIRRORS"
    
    if [ -f "scripts/prepare_swebench_repos.py" ]; then
        python3 scripts/prepare_swebench_repos.py \
            --instances_file configs/swebench_lite_slice20.txt \
            --mirror_root "$HERMES_REPO_MIRRORS"
    else
        echo "Warning: scripts/prepare_swebench_repos.py not found"
        echo "You'll need to manually create repo mirrors"
    fi
    
    echo ""
    echo "=== Preparation Complete ==="
    echo ""
    echo "Dataset cached at: $HF_DATASETS_CACHE"
    echo "Repo mirrors at: $HERMES_REPO_MIRRORS"
    echo ""
    echo "Now run hermetic evaluation with: $0 eval"
    
else
    # USAGE
    echo "Usage:"
    echo "  $0 prep   # Prepare dataset and repo mirrors (requires network)"
    echo "  $0 eval   # Run hermetic evaluation (offline)"
    echo ""
    echo "Steps:"
    echo "  1. Run '$0 prep' once to download data (~30 min, requires network)"
    echo "  2. Run '$0 eval' to execute hermetic evaluation (offline)"
    echo "  3. Results will be in runs/C/ and runs/PM/"
    echo "  4. Run 'python3 scripts/check_acceptance.py' to verify criteria"
fi