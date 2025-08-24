#!/usr/bin/env python3
"""Create mock SWE-bench Lite data for testing without network access."""

import json
import hashlib
import os
from pathlib import Path
import pickle

# Configuration
REVISION = "b8b14b3"  # Mock revision
OUT = Path(f"data/swebench_lite/{REVISION}")
MANIFEST = OUT / "MANIFEST.json"

# Create output directory
OUT.mkdir(parents=True, exist_ok=True)

print(f"Creating mock SWE-bench Lite data at revision {REVISION}")

# Create mock instances
def create_mock_instance(idx, split):
    """Create a mock SWE-bench instance."""
    repos = ["django/django", "flask/flask", "requests/requests", "scikit-learn/scikit-learn"]
    repo = repos[idx % len(repos)]
    
    return {
        "instance_id": f"{repo.replace('/', '__')}-{10000 + idx:05d}",
        "repo": repo,
        "base_commit": f"abc{idx:03x}",
        "problem_statement": f"Fix issue #{idx} in {repo}: Test failure in module",
        "hints_text": f"Look at line {42 + idx}",
        "test_patch": f"--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old{idx}\n+new{idx}",
        "FAIL_TO_PASS": [f"test_feature_{idx}", f"test_integration_{idx}"],
        "PASS_TO_PASS": [f"test_existing_{idx}"],
        "environment_setup_commit": f"setup{idx:03x}",
        "created_at": "2024-01-01T00:00:00Z",
        "version": "1.0",
    }

# Create mock datasets
dev_instances = [create_mock_instance(i, "dev") for i in range(23)]
test_instances = [create_mock_instance(i, "test") for i in range(300)]

print(f"Dev split: {len(dev_instances)} instances")
print(f"Test split: {len(test_instances)} instances")

# Save in a simple format that mimics datasets library
def save_mock_dataset(instances, path):
    """Save mock dataset in a format we can load."""
    path.mkdir(parents=True, exist_ok=True)
    
    # Save as both pickle and JSON for compatibility
    with open(path / "dataset.pkl", "wb") as f:
        pickle.dump(instances, f)
    
    with open(path / "dataset.json", "w") as f:
        json.dump(instances, f, indent=2)
    
    # Create state.json to mimic datasets format
    with open(path / "state.json", "w") as f:
        json.dump({"_data_files": [], "_split": path.name}, f)

print("Saving to disk...")
save_mock_dataset(dev_instances, OUT / "dev")
save_mock_dataset(test_instances, OUT / "test")

# Compute directory hash
def dir_hash(p):
    h = hashlib.sha256()
    for root, _, files in os.walk(p):
        for f in sorted(files):
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    h.update(fh.read(min(8192, os.path.getsize(fp))))
    return h.hexdigest()

print("Computing directory hash...")
manifest = {
    "dataset_name": "princeton-nlp/SWE-bench_Lite",
    "revision": REVISION,
    "local_path": str(OUT),
    "splits": {
        "dev": len(dev_instances),
        "test": len(test_instances),
    },
    "sha256_dir": dir_hash(str(OUT)),
}

# Write manifest
with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written to: {MANIFEST}")
print(json.dumps(manifest, indent=2))
print(f"\nMock SWE-bench Lite data created at: {OUT}")
print("\nNOTE: This is mock data for testing. For real evaluation, install 'datasets' library and run prepare_swebench.sh")