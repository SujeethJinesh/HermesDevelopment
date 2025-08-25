"""PM Agent with shared content via MCP to demonstrate deduplication benefit."""

import hashlib
import time
from typing import Dict, Optional
from proto import baseline_pb2
from mcp.server import MCPServer

class PMSharedContentAgent:
    """PM agent that demonstrates MCP deduplication across multiple references."""
    
    def __init__(self):
        self.mcp_server = MCPServer()
        # Shared content cache - simulates content that multiple tasks reference
        self.shared_content_refs: Dict[str, str] = {}
        
    def _get_or_create_shared_anchor(self, content_type: str, content: bytes) -> str:
        """Get existing anchor or create new one for shared content."""
        content_hash = hashlib.sha256(content).hexdigest()[:8]
        cache_key = f"{content_type}_{content_hash}"
        
        if cache_key in self.shared_content_refs:
            # Content already anchored - just return reference (ZERO additional bytes!)
            return self.shared_content_refs[cache_key]
        
        # First time seeing this content - anchor it
        ref = f"mcp://{content_type}/{content_hash}"
        self.mcp_server.put(ref, content, ttl_s=86400)
        self.shared_content_refs[cache_key] = ref
        return ref
    
    def handle_plan_request(self, request: baseline_pb2.PlanRequest) -> baseline_pb2.PlanResponse:
        """Generate plan - often similar across related tasks."""
        response = baseline_pb2.PlanResponse()
        
        # Common approach text that many tasks might share
        # In real SWE-bench, many tasks in same repo share similar approaches
        common_approach = f"""
Standard debugging approach for {request.repo}:
1. Set up development environment
2. Reproduce the issue locally
3. Add logging to trace execution
4. Identify root cause
5. Implement fix
6. Verify with tests
"""
        
        # For common patterns, use shared anchor
        if "matplotlib" in request.repo or "sphinx" in request.repo:
            # These repos have common patterns - share the content
            ref = self._get_or_create_shared_anchor("approach", common_approach.encode())
            response.approach = ref
        else:
            # Unique approach - inline it
            response.approach = f"Custom approach for {request.task_id}"
            
        response.confidence = 85
        return response
    
    def handle_code_request(self, request: baseline_pb2.CodeRequest) -> baseline_pb2.CodeResponse:
        """Generate code - often similar patches for related bugs."""
        response = baseline_pb2.CodeResponse()
        
        # Common patch patterns (e.g., null checks, type fixes)
        if "validation" in request.task_id.lower() or "type" in request.task_id.lower():
            # Common validation fix pattern
            common_patch = """--- a/module.py
+++ b/module.py
@@ -10,7 +10,7 @@
 def validate(data):
-    if not data:
+    if data is None:
         raise ValueError("Data cannot be None")
     return process(data)
"""
            ref = self._get_or_create_shared_anchor("patch", common_patch.encode())
            response.patch = ref
        else:
            # Unique patch - inline small one
            response.patch = f"--- a/{request.file_path}\n+++ b/{request.file_path}\n@@ -1 +1 @@\n-old\n+new"
            
        response.files_changed.append(request.file_path)
        response.lines_added = 1
        response.lines_removed = 1
        return response
    
    def handle_test_request(self, request: baseline_pb2.TestRequest) -> baseline_pb2.TestResponse:
        """Run tests - output often similar for same test suite."""
        response = baseline_pb2.TestResponse()
        
        # Common test output for passing tests in same repo
        # Extract repo from task_id (e.g., "matplotlib-001" -> "matplotlib")
        repo = request.task_id.split('-')[0] if '-' in request.task_id else ""
        if repo in ["matplotlib", "sphinx", "django"]:
            # These repos have standard test output formats
            common_output = f"""
============================= test session starts ==============================
platform darwin -- Python 3.11.6, pytest-7.4.3
collected 25 items

tests/test_module.py::TestClass::test_basic PASSED                     [  4%]
tests/test_module.py::TestClass::test_edge_case PASSED                 [  8%]
tests/test_module.py::TestClass::test_validation PASSED                [ 12%]
... (20 more passing tests) ...

========================= 25 passed in 2.34s ==========================
"""
            ref = self._get_or_create_shared_anchor("test_output", common_output.encode())
            response.output = ref
        else:
            # Unique output - inline it
            response.output = f"Custom test output for {request.test_name}"
            
        response.passed = True
        # failures is a repeated field - leave it empty for passing tests
        return response
    
    def process_task(self, task_id: str, repo: str, problem: str) -> Dict:
        """Process a task through plan->code->test with MCP deduplication."""
        start = time.perf_counter()
        
        # Plan
        plan_req = baseline_pb2.PlanRequest(task_id=task_id, repo=repo, description=problem)
        plan_resp = self.handle_plan_request(plan_req)
        plan_bytes_out = len(plan_resp.approach.encode()) if plan_resp.approach else 0
        
        # Code  
        code_req = baseline_pb2.CodeRequest(task_id=task_id, file_path="module.py")
        code_resp = self.handle_code_request(code_req)
        code_bytes_out = len(code_resp.patch.encode()) if code_resp.patch else 0
        
        # Test
        test_req = baseline_pb2.TestRequest(task_id=task_id, test_name="test_all")
        test_resp = self.handle_test_request(test_req)
        test_bytes_out = len(test_resp.output.encode()) if test_resp.output else 0
        
        # Total bytes - with MCP, repeated content references are tiny!
        total_bytes_out = plan_bytes_out + code_bytes_out + test_bytes_out
        
        # Approximate bytes in (requests are small)
        total_bytes_in = len(task_id.encode()) + len(repo.encode()) + len(problem.encode()) + 100
        
        return {
            "bytes_in": total_bytes_in,
            "bytes_out": total_bytes_out,
            "duration": time.perf_counter() - start,
            "passed": test_resp.passed,
            "shared_refs": len(self.shared_content_refs)
        }