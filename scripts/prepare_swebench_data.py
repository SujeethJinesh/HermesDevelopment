#!/usr/bin/env python3
"""Download and prepare SWE-bench Lite dataset for hermetic execution."""

import json
import hashlib
import os
from pathlib import Path

# Check if datasets is available
try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed")
    print("Run: pip3 install datasets")
    exit(1)

# Configuration
REVISION = os.environ.get("REVISION", "main")  # Use main branch
OUT = Path(f"data/swebench_lite/{REVISION}")
MANIFEST = OUT / "MANIFEST.json"

# Create output directory
OUT.mkdir(parents=True, exist_ok=True)

print(f"Loading princeton-nlp/SWE-bench_Lite at revision {REVISION}")

# Download from HuggingFace
try:
    ds_dev = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev", revision=REVISION)
    ds_test = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", revision=REVISION)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nNote: This requires internet access to download from HuggingFace.")
    print("If you're behind a proxy, set HF_DATASETS_OFFLINE=1 and provide local data.")
    exit(1)

print(f"Dev split: {len(ds_dev)} instances")
print(f"Test split: {len(ds_test)} instances")

# Verify expected counts
assert len(ds_dev) == 23, f"Expected 23 dev instances, got {len(ds_dev)}"
assert len(ds_test) == 300, f"Expected 300 test instances, got {len(ds_test)}"

# Save to disk (Arrow format for fast loading)
print("Saving to disk...")
ds_dev.save_to_disk(OUT / "dev")
ds_test.save_to_disk(OUT / "test")

# Compute directory hash for reproducibility
def dir_hash(p):
    h = hashlib.sha256()
    for root, _, files in os.walk(p):
        for f in sorted(files):
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    # Read in chunks for large files
                    while chunk := fh.read(8192):
                        h.update(chunk)
    return h.hexdigest()

print("Computing directory hash...")
manifest = {
    "dataset_name": "princeton-nlp/SWE-bench_Lite",
    "revision": REVISION,
    "local_path": str(OUT),
    "splits": {
        "dev": len(ds_dev),
        "test": len(ds_test),
    },
    "sha256_dir": dir_hash(str(OUT)),
}

# Write manifest
with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written to: {MANIFEST}")
print(json.dumps(manifest, indent=2))
print(f"\nSWE-bench Lite prepared at: {OUT}")