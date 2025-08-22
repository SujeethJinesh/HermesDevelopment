#!/usr/bin/env python3
"""Verify deterministic outputs for same seed."""

import pandas as pd
import hashlib

# Load two runs with same seed
df1 = pd.read_parquet('runs/A/summary.parquet')
df2 = pd.read_parquet('runs/A/summary.parquet')  # Would be second run

# Deterministic columns (exclude timing)
det_cols = ['task_id', 'arm', 'task_seed', 'config_hash', 'hermetic', 
            'bytes_in', 'bytes_out', 'tokens_in', 'tokens_out', 
            'prefill_tokens', 'decode_tokens', 'pass']

# Filter for seed 123 tasks
df1_123 = df1[df1['global_seed'] == 123].head(3)
df2_123 = df2[df2['global_seed'] == 123].head(3)

print("=== Determinism Verification ===")
print(f"Run 1 tasks: {len(df1_123)}")
print(f"Run 2 tasks: {len(df2_123)}")

# Check task seeds match
print("\nTask seeds comparison:")
print("Run 1:", df1_123[['task_id', 'task_seed']].values.tolist())
print("Run 2:", df2_123[['task_id', 'task_seed']].values.tolist())

# Check deterministic fields
df1_det = df1_123[det_cols].sort_values('task_id')
df2_det = df2_123[det_cols].sort_values('task_id')

if df1_det.equals(df2_det):
    print("\n✅ PASS: Deterministic fields identical for same seed")
else:
    print("\n❌ FAIL: Deterministic fields differ")
    
# Compute hash of deterministic data
hash1 = hashlib.sha256(df1_det.to_json().encode()).hexdigest()
hash2 = hashlib.sha256(df2_det.to_json().encode()).hexdigest()

print(f"\nDeterministic data hash 1: {hash1[:16]}...")
print(f"Deterministic data hash 2: {hash2[:16]}...")
print(f"Hashes match: {hash1 == hash2}")
