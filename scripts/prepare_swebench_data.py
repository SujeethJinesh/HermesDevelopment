"""One-time cache preparation script for SWE-bench Lite dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from typing import Dict, List

from datasets import load_dataset


def sha256_file(p: pathlib.Path) -> str:
    """Calculate SHA256 hash of a file.
    
    Args:
        p: Path to file
    
    Returns:
        Hex-encoded SHA256 hash
    """
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    """Pre-fill HF cache with SWE-bench Lite and verify offline readability."""
    ap = argparse.ArgumentParser(
        description="Prepare SWE-bench Lite dataset cache for offline usage"
    )
    ap.add_argument(
        "--dataset",
        default="SWE-bench/SWE-bench_Lite",
        help="Dataset name on HuggingFace Hub",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "test"],
        help="Splits to download and cache",
    )
    args = ap.parse_args()

    # Ensure local HF cache
    hf_home = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".hf"))
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    print(f"Using HF_HOME: {hf_home}")
    print(f"Using HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")

    # Online fetch: ensure HF_DATASETS_OFFLINE is NOT set
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    print(f"\nDownloading dataset: {args.dataset}")
    
    counts: Dict[str, int] = {}
    for s in args.splits:
        print(f"  Loading split: {s}")
        ds = load_dataset(args.dataset, split=s)  # downloads if missing
        counts[s] = ds.num_rows
        print(f"    → {ds.num_rows} instances")

    # Verify offline readability
    print("\nVerifying offline access...")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    for s in args.splits:
        print(f"  Re-loading split {s} in offline mode...")
        ds = load_dataset(args.dataset, split=s)
        assert ds.num_rows == counts[s], f"Offline load failed for split {s}"
        print(f"    → OK ({ds.num_rows} instances)")

    # Emit prep manifest (paths may vary by HF version; best-effort)
    manifest = {
        "dataset": args.dataset,
        "splits": counts,
        "env": {
            "HF_HOME": os.environ["HF_HOME"],
            "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
            "HF_DATASETS_OFFLINE": os.environ["HF_DATASETS_OFFLINE"],
        },
    }
    
    man_path = pathlib.Path(hf_home) / "swebench_lite_prep_manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {man_path}")
    print(f"Manifest contents:\n{json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()