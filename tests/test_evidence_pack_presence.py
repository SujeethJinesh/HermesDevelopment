from pathlib import Path

def test_claude_md_evidence_pack_sections_present():
    """Test that CLAUDE.md exists and contains required Evidence Pack sections."""
    p = Path("CLAUDE.md")
    assert p.exists(), "CLAUDE.md missing at repo root"
    s = p.read_text(encoding="utf-8")
    assert "# APEX — Evidence Pack" in s, "Missing Evidence Pack header"
    assert "## M0 — Evidence Pack" in s or "## M0 — Evidence Pack (Environment, Clients, Harness)" in s, "Missing M0 section"
    assert "Artifact manifest" in s, "Missing artifact manifest section"
    assert "Reproduce (exact commands)" in s, "Missing reproduce commands section"