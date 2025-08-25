#!/usr/bin/env bash
set -euo pipefail

# Pin to a stable revision
REVISION="${REVISION:-b8b14b3}"  # SWE-bench_Lite stable revision
OUT="data/swebench_lite/${REVISION}"
MANIFEST="${OUT}/MANIFEST.json"

mkdir -p "${OUT}"

python3 - <<'PY'
from datasets import load_dataset
from pathlib import Path
import json, hashlib, os

rev = os.environ["REVISION"]
out = Path(os.environ["OUT"])

# Download from HF once (internet allowed in this script only)
print(f"Loading princeton-nlp/SWE-bench_Lite at revision {rev}")
ds_dev  = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev",  revision=rev)
ds_test = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", revision=rev)

print(f"Dev split: {len(ds_dev)} instances")
print(f"Test split: {len(ds_test)} instances")

# Save to disk (Arrow format)
ds_dev.save_to_disk(out / "dev")
ds_test.save_to_disk(out / "test")

def dir_hash(p):
    h = hashlib.sha256()
    for root, _, files in os.walk(p):
        for f in sorted(files):
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()

manifest = {
    "dataset_name": "princeton-nlp/SWE-bench_Lite",
    "revision": rev,
    "local_path": str(out),
    "splits": {
        "dev":  len(ds_dev),
        "test": len(ds_test),
    },
    "sha256_dir": dir_hash(str(out)),
}
(out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("\nManifest written to:", out / "MANIFEST.json")
print(json.dumps(manifest, indent=2))
PY

echo "SWE-bench Lite prepared at: ${OUT}"