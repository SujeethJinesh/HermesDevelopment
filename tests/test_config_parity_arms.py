#!/usr/bin/env python3
"""Test config parity is enforced with Arms A/C."""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_arms import ArmRunner


class TestConfigParityArms:
    """Test that config parity is enforced for Arms A and C."""
    
    def test_arm_a_rejects_non_canonical_config(self):
        """Test that Arm A rejects non-canonical config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write a non-canonical config
            config = {
                "generation": {
                    "temperature": 0.5,  # Not allowed - override attempt
                    "max_tokens": 2000   # Not allowed - override attempt
                }
            }
            yaml.dump(config, f)
            config_path = f.name
        
        from eval.run_arms import ConfigParityError
        
        try:
            # Attempt to create runner with non-canonical config should fail
            with pytest.raises(ConfigParityError) as exc_info:
                runner = ArmRunner(
                    arm="A",
                    seed=123,
                    gen_cfg_path=config_path,
                    hermetic=True,
                    toy_tasks=1
                )
            
            assert "parity" in str(exc_info.value).lower() or "config" in str(exc_info.value).lower()
        finally:
            Path(config_path).unlink()
    
    def test_arm_c_rejects_non_canonical_config(self):
        """Test that Arm C rejects non-canonical config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write a non-canonical config
            config = {
                "generation": {
                    "top_p": 0.9,  # Not allowed - override attempt
                    "seed": 999    # Not allowed - override attempt
                }
            }
            yaml.dump(config, f)
            config_path = f.name
        
        from eval.run_arms import ConfigParityError
        
        try:
            # Attempt to create runner with non-canonical config should fail
            with pytest.raises(ConfigParityError) as exc_info:
                runner = ArmRunner(
                    arm="C",
                    seed=456,
                    gen_cfg_path=config_path,
                    hermetic=True,
                    toy_tasks=1
                )
            
            assert "parity" in str(exc_info.value).lower() or "config" in str(exc_info.value).lower()
        finally:
            Path(config_path).unlink()
    
    def test_arms_accept_canonical_config(self):
        """Test that both arms accept the canonical config."""
        # Use the actual canonical config
        canonical_path = "configs/generation.yaml"
        
        if not Path(canonical_path).exists():
            pytest.skip(f"Canonical config not found at {canonical_path}")
        
        # Arm A should accept canonical config
        runner_a = ArmRunner(
            arm="A",
            seed=123,
            gen_cfg_path=canonical_path,
            hermetic=False,  # Don't actually run
            toy_tasks=0
        )
        assert runner_a.config is not None
        
        # Arm C should accept canonical config
        runner_c = ArmRunner(
            arm="C",
            seed=456,
            gen_cfg_path=canonical_path,
            hermetic=False,  # Don't actually run
            toy_tasks=0
        )
        assert runner_c.config is not None
    
    def test_config_hash_deterministic(self):
        """Test that config hash is deterministic for same config."""
        canonical_path = "configs/generation.yaml"
        
        if not Path(canonical_path).exists():
            pytest.skip(f"Canonical config not found at {canonical_path}")
        
        # Create two runners with same config
        runner1 = ArmRunner(
            arm="A",
            seed=123,
            gen_cfg_path=canonical_path,
            hermetic=False,
            toy_tasks=0
        )
        
        runner2 = ArmRunner(
            arm="C",
            seed=456,
            gen_cfg_path=canonical_path,
            hermetic=False,
            toy_tasks=0
        )
        
        # Config hash should be identical (full SHA-256)
        assert runner1.config_hash == runner2.config_hash
        assert len(runner1.config_hash) == 64  # Full SHA-256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])