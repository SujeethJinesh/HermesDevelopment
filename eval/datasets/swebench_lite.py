"""SWE-bench Lite dataset loader with strict split-size assertions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import datasets

DATASET_NAME = "SWE-bench/SWE-bench_Lite"
DEV_EXPECTED = 23
TEST_EXPECTED = 300


@dataclass(frozen=True)
class SWELiteInstance:
    """Strongly-typed instance from SWE-bench Lite dataset."""
    
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
    version: str
    problem_statement: str


def _load_split(name: str, cache_dir: Optional[str] = None) -> datasets.Dataset:
    """Load a single split from the SWE-bench Lite dataset."""
    return datasets.load_dataset(
        DATASET_NAME,
        split=name,
        cache_dir=cache_dir,
    )


def load_swebench_lite(
    cache_dir: Optional[str] = None
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Load SWE-bench Lite dev and test splits with strict size assertions.
    
    Args:
        cache_dir: Optional cache directory for datasets. If None, uses HF default.
    
    Returns:
        Tuple of (dev_dataset, test_dataset)
    
    Raises:
        AssertionError: If split sizes don't match expected counts.
    """
    dev = _load_split("dev", cache_dir)
    test = _load_split("test", cache_dir)
    
    assert dev.num_rows == DEV_EXPECTED, (
        f"SWE-bench Lite dev split mismatch: expected {DEV_EXPECTED}, got {dev.num_rows}. "
        "Counts must match the official dataset card."
    )
    assert test.num_rows == TEST_EXPECTED, (
        f"SWE-bench Lite test split mismatch: expected {TEST_EXPECTED}, got {test.num_rows}. "
        "Counts must match the official dataset card."
    )
    
    return dev, test


def iter_instances(ds: datasets.Dataset) -> Iterator[SWELiteInstance]:
    """Iterate over dataset rows as strongly-typed instances.
    
    Args:
        ds: A HuggingFace Dataset object
    
    Yields:
        SWELiteInstance objects with all required fields
    """
    for r in ds:
        yield SWELiteInstance(
            instance_id=r["instance_id"],
            repo=r["repo"],
            base_commit=r["base_commit"],
            patch=r["patch"],
            test_patch=r["test_patch"],
            FAIL_TO_PASS=r["FAIL_TO_PASS"],
            PASS_TO_PASS=r["PASS_TO_PASS"],
            environment_setup_commit=r["environment_setup_commit"],
            version=r["version"],
            problem_statement=r["problem_statement"],
        )