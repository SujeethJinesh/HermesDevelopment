"""Unit tests for SWE-bench Lite loader with mocked HF datasets."""

import pytest

from eval.datasets.swebench_lite import (
    DEV_EXPECTED,
    TEST_EXPECTED,
    iter_instances,
    load_swebench_lite,
)


class _StubDS:
    """Stub dataset for testing without network access."""

    def __init__(self, n):
        self.num_rows = n

    def __iter__(self):
        """Yield a single sample row with all required fields."""
        yield {
            "instance_id": "repo__name-123",
            "repo": "owner/name",
            "base_commit": "abc123",
            "patch": "diff --git a/file.py b/file.py\n...",
            "test_patch": "diff --git a/test.py b/test.py\n...",
            "FAIL_TO_PASS": '["test_function"]',
            "PASS_TO_PASS": '["test_other"]',
            "environment_setup_commit": "def456",
            "version": "1.0.0",
            "problem_statement": "Fix the issue with function X",
        }


def test_split_sizes_enforced(monkeypatch):
    """Test that correct split sizes pass validation."""

    def fake_load(name, split, cache_dir=None):
        return _StubDS(DEV_EXPECTED if split == "dev" else TEST_EXPECTED)

    monkeypatch.setattr("eval.datasets.swebench_lite.datasets.load_dataset", fake_load)
    dev, test = load_swebench_lite()
    assert dev.num_rows == DEV_EXPECTED
    assert test.num_rows == TEST_EXPECTED


def test_split_mismatch_raises(monkeypatch):
    """Test that incorrect split sizes raise RuntimeError with clear message."""

    def fake_load(name, split, cache_dir=None):
        # Return incorrect sizes
        return _StubDS(22 if split == "dev" else 299)

    monkeypatch.setattr("eval.datasets.swebench_lite.datasets.load_dataset", fake_load)

    with pytest.raises(RuntimeError) as exc_info:
        load_swebench_lite()

    # Verify error message mentions the official dataset card
    assert "official dataset card" in str(exc_info.value)


def test_iter_instances_shape(monkeypatch):
    """Test that iter_instances yields properly typed instances with all fields."""

    def fake_load(name, split, cache_dir=None):
        return _StubDS(DEV_EXPECTED if split == "dev" else TEST_EXPECTED)

    monkeypatch.setattr("eval.datasets.swebench_lite.datasets.load_dataset", fake_load)
    dev, _ = load_swebench_lite()

    # Get first instance
    row = next(iter(iter_instances(dev)))

    # Verify all required fields are present
    assert row.instance_id == "repo__name-123"
    assert row.repo == "owner/name"
    assert row.base_commit == "abc123"
    assert row.patch.startswith("diff")
    assert row.test_patch.startswith("diff")
    assert row.FAIL_TO_PASS == '["test_function"]'
    assert row.PASS_TO_PASS == '["test_other"]'
    assert row.environment_setup_commit == "def456"
    assert row.version == "1.0.0"
    assert row.problem_statement == "Fix the issue with function X"

    # Verify instance is frozen (dataclass)
    with pytest.raises(AttributeError):
        row.instance_id = "modified"


def test_cache_dir_parameter(monkeypatch):
    """Test that cache_dir parameter is passed through correctly."""
    cache_dir_used = None

    def fake_load(name, split, cache_dir=None):
        nonlocal cache_dir_used
        cache_dir_used = cache_dir
        return _StubDS(DEV_EXPECTED if split == "dev" else TEST_EXPECTED)

    monkeypatch.setattr("eval.datasets.swebench_lite.datasets.load_dataset", fake_load)

    # Test with explicit cache directory
    load_swebench_lite(cache_dir="/tmp/test_cache")
    assert cache_dir_used == "/tmp/test_cache"

    # Test with None (default)
    load_swebench_lite(cache_dir=None)
    assert cache_dir_used is None
