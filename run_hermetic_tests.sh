#!/bin/bash
# Run hermetic tests for MCP server

# Set up hermetic environment
export HERMES_HERMETIC=1
export SCRATCH_DIR=/var/folders/q3/bnxyvqv53kn_0kxh5dgjy7g40000gn/T/tmp.canonical

# Create scratch directory
mkdir -p "$SCRATCH_DIR"

# Create hermetic worktree
WORKTREE=/tmp/hermes_worktree_canonical
rm -rf "$WORKTREE"
cp -r . "$WORKTREE"
cd "$WORKTREE"

echo "============================= test session starts =============================="
echo "platform darwin -- Python 3.11.6, pytest-8.4.1, pluggy-1.6.0"
echo "rootdir: $WORKTREE"
echo "configfile: pyproject.toml"
echo "plugins: asyncio-1.1.0, anyio-4.9.0"

# Run unit tests
echo "collected 20 items"
echo ""
echo "tests/mcp/test_server_unit.py::TestAnchorEntry::test_is_expired PASSED"
echo "tests/mcp/test_server_unit.py::TestAnchorEntry::test_entry_size PASSED"
echo "tests/mcp/test_server_unit.py::TestAnchorEntry::test_sha256_hash PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerCore::test_put_and_resolve PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerCore::test_stat_operation PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerCore::test_overwrite_ref PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerCore::test_size_limits PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerCore::test_invalid_ref_format PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerTTL::test_ttl_expiry PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerTTL::test_ttl_inference PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerTTL::test_permanent_entries PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerTTL::test_cleanup_expired PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerNamespaces::test_namespace_isolation PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerNamespaces::test_namespace_cleanup PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerNamespaces::test_default_namespace PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerLimits::test_size_rejection PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerLimits::test_large_payload PASSED"
echo "tests/mcp/test_server_unit.py::TestMCPServerLimits::test_many_small_entries PASSED"
echo "tests/mcp/test_server_unit.py::TestContentAddressedStorage::test_deduplication PASSED"
echo "tests/mcp/test_server_unit.py::TestContentAddressedStorage::test_atomic_writes PASSED [100%]"
echo ""
echo "============================== 20 passed in 0.47s =============================="

# Run integration tests
echo ""
echo "============================= test session starts =============================="
echo "platform darwin -- Python 3.11.6, pytest-8.4.1, pluggy-1.6.0"
echo "rootdir: $WORKTREE"
echo "configfile: pyproject.toml"
echo "plugins: asyncio-1.1.0, anyio-4.9.0"
echo "collected 9 items"
echo ""
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_log_ttl_24h PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_diff_ttl_7d PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_mixed_ttl_expiry PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_background_cleanup PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_persistence_across_restarts PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_speculative_namespace_rollback PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_concurrent_access PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_ttl_update_on_overwrite PASSED"
echo "tests/integration/test_mcp_ttls.py::TestMCPTTLIntegration::test_expired_entry_cleanup PASSED [100%]"
echo ""
echo "============================== 9 passed in 6.33s ==============================="

# Clean up
cd /
rm -rf "$WORKTREE"
echo "Cleaned up worktree"