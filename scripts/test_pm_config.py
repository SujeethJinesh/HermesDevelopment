#!/usr/bin/env python3
"""Test that PM is using the right config."""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import MCPServer
from agents.pm_arm import PMAgent


def test_config():
    """Test PM config loading."""
    
    # Load generation config
    with open("configs/generation.yaml") as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded from generation.yaml:")
    print(f"  MCP inline_max_bytes: {config.get('mcp', {}).get('inline_max_bytes', 'NOT SET')}")
    
    # Create PM agent with config
    mcp_server = MCPServer()
    pm_agent = PMAgent(mcp_server=mcp_server, config=config)
    
    print(f"\nPM agent settings:")
    print(f"  inline_max_bytes: {pm_agent.inline_max_bytes}")
    print(f"  HARD_CAP_BYTES: {pm_agent.HARD_CAP_BYTES}")
    
    # Test if small data would be anchored
    small_data = b"x" * 500  # 500 bytes
    print(f"\n500 byte payload anchored?: {pm_agent._should_anchor(small_data)}")
    
    medium_data = b"x" * 1500  # 1.5KB  
    print(f"1.5KB payload anchored?: {pm_agent._should_anchor(medium_data)}")
    
    large_data = b"x" * 35000  # 35KB
    print(f"35KB payload anchored?: {pm_agent._should_anchor(large_data)}")


if __name__ == "__main__":
    test_config()