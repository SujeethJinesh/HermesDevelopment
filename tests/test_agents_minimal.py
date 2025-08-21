#!/usr/bin/env python3
"""Unit tests for minimal baseline agents."""

import json
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.planner import Planner
from agents.coder import Coder
from agents.tester import Tester
from proto import baseline_pb2


class TestPlanner:
    """Test minimal planner agent."""
    
    def test_planner_deterministic(self):
        """Test planner produces deterministic output for same seed."""
        planner1 = Planner(seed=42)
        planner2 = Planner(seed=42)
        
        task = {
            "task_id": "test-001",
            "repo": "test-repo",
            "file_path": "src/test.py",
            "test_name": "test_function",
            "description": "Fix the test"
        }
        
        result1 = planner1.plan(task)
        result2 = planner2.plan(task)
        
        assert result1 == result2
        assert len(result1["steps"]) >= 2
        assert result1["confidence"] >= 60
        assert result1["confidence"] <= 95
    
    def test_planner_json_format(self):
        """Test planner JSON output format for Arm A."""
        planner = Planner(seed=123)
        
        task = {"task_id": "test-002", "file_path": "main.py"}
        json_output = planner.plan_json(task)
        
        data = json.loads(json_output)
        assert data["role"] == "planner"
        assert data["task_id"] == "test-002"
        assert "plan" in data
        assert "steps" in data
        assert "confidence" in data


class TestCoder:
    """Test minimal coder agent."""
    
    def test_coder_deterministic(self):
        """Test coder produces deterministic patches."""
        coder1 = Coder(seed=42)
        coder2 = Coder(seed=42)
        
        task = {"task_id": "test-003", "file_path": "src/fix.py"}
        plan_steps = ["Step 1", "Step 2"]
        
        result1 = coder1.code(task, plan_steps)
        result2 = coder2.code(task, plan_steps)
        
        assert result1 == result2
        assert "---" in result1["patch"]  # Unified diff format
        assert "+++" in result1["patch"]
        assert result1["lines_added"] >= 1
    
    def test_coder_json_format(self):
        """Test coder JSON output for Arm A."""
        coder = Coder(seed=456)
        
        task = {"task_id": "test-004", "file_path": "util.py"}
        plan_steps = ["Analyze", "Fix"]
        json_output = coder.code_json(task, plan_steps)
        
        data = json.loads(json_output)
        assert data["role"] == "coder"
        assert data["task_id"] == "test-004"
        assert "patch" in data
        assert "summary" in data


class TestTester:
    """Test minimal tester agent."""
    
    def test_tester_deterministic(self):
        """Test tester produces deterministic results."""
        tester1 = Tester(seed=42)
        tester2 = Tester(seed=42)
        
        task = {"task_id": "test-005", "test_name": "test_feature"}
        patch = "--- a/file.py\n+++ b/file.py"
        
        result1 = tester1.test(task, patch)
        result2 = tester2.test(task, patch)
        
        assert result1 == result2
        assert isinstance(result1["passed"], bool)
        assert result1["duration_ms"] >= 100
        assert result1["duration_ms"] <= 500
    
    def test_tester_json_format(self):
        """Test tester JSON output for Arm A."""
        tester = Tester(seed=789)
        
        task = {"task_id": "test-006", "test_name": "test_module"}
        patch = "patch content"
        json_output = tester.test_json(task, patch)
        
        data = json.loads(json_output)
        assert data["role"] == "tester"
        assert data["task_id"] == "test-006"
        assert "passed" in data
        assert "duration_ms" in data


class TestProtobufRoundTrip:
    """Test Protobuf serialization for Arm C."""
    
    def test_plan_request_response(self):
        """Test PlanRequest/Response round-trip."""
        # Create request
        req = baseline_pb2.PlanRequest(
            task_id="proto-001",
            repo="test-repo",
            file_path="src/main.py",
            test_name="test_main",
            description="Fix failing test",
            seed=123
        )
        
        # Serialize and deserialize
        data = req.SerializeToString()
        req2 = baseline_pb2.PlanRequest()
        req2.ParseFromString(data)
        
        assert req2.task_id == "proto-001"
        assert req2.seed == 123
        
        # Create response
        resp = baseline_pb2.PlanResponse(
            steps=["Step 1", "Step 2"],
            approach="direct fix",
            confidence=85
        )
        
        data = resp.SerializeToString()
        resp2 = baseline_pb2.PlanResponse()
        resp2.ParseFromString(data)
        
        assert len(resp2.steps) == 2
        assert resp2.confidence == 85
    
    def test_code_request_response(self):
        """Test CodeRequest/Response round-trip."""
        req = baseline_pb2.CodeRequest(
            task_id="proto-002",
            file_path="lib/util.py",
            plan_steps=["Analyze", "Patch"],
            seed=456
        )
        
        data = req.SerializeToString()
        req2 = baseline_pb2.CodeRequest()
        req2.ParseFromString(data)
        
        assert req2.task_id == "proto-002"
        assert len(req2.plan_steps) == 2
        
        resp = baseline_pb2.CodeResponse(
            patch="--- a/lib/util.py\n+++ b/lib/util.py",
            files_changed=["lib/util.py"],
            lines_added=3,
            lines_removed=1
        )
        
        data = resp.SerializeToString()
        resp2 = baseline_pb2.CodeResponse()
        resp2.ParseFromString(data)
        
        assert resp2.lines_added == 3
        assert len(resp2.files_changed) == 1
    
    def test_test_request_response(self):
        """Test TestRequest/Response round-trip."""
        req = baseline_pb2.TestRequest(
            task_id="proto-003",
            test_name="test_feature",
            patch="patch data",
            seed=789
        )
        
        data = req.SerializeToString()
        req2 = baseline_pb2.TestRequest()
        req2.ParseFromString(data)
        
        assert req2.task_id == "proto-003"
        
        resp = baseline_pb2.TestResponse(
            passed=True,
            output="All tests passed",
            duration_ms=234,
            failures=[]
        )
        
        data = resp.SerializeToString()
        resp2 = baseline_pb2.TestResponse()
        resp2.ParseFromString(data)
        
        assert resp2.passed == True
        assert resp2.duration_ms == 234


if __name__ == "__main__":
    pytest.main([__file__, "-v"])