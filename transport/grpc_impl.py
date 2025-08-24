#!/usr/bin/env python3
"""gRPC transport implementation over UNIX domain sockets for hermetic execution."""

import concurrent.futures
import grpc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add parent to path for proto imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proto import baseline_pb2, baseline_pb2_grpc
from agents.planner import Planner
from agents.coder import Coder
from agents.tester import Tester
from agents.pm_arm import PMAgent
from mcp.server import MCPServer

logger = logging.getLogger(__name__)


class ArmServiceImpl(baseline_pb2_grpc.ArmServiceServicer):
    """gRPC service implementation for baseline agents."""
    
    def __init__(self, arm: str, seed: int = 0, config: Optional[Dict] = None):
        """Initialize service with agents.
        
        Args:
            arm: "A" for JSON, "C" for Protobuf, "PM" for Protobuf+MCP
            seed: Random seed for determinism
            config: Configuration dict (for PM arm)
        """
        self.arm = arm
        self.seed = seed
        self.config = config or {}
        
        # Initialize agents based on arm
        if arm == "PM":
            # For PM, create MCP server and PM agent
            self.mcp_server = MCPServer()
            self.pm_agent = PMAgent(self.mcp_server, config)
            # PM still uses regular agents for now (could be PM-specific later)
            self.planner = Planner(seed)
            self.coder = Coder(seed)
            self.tester = Tester(seed)
        else:
            self.planner = Planner(seed)
            self.coder = Coder(seed)
            self.tester = Tester(seed)
        
        # Track state for multi-turn conversations
        self.task_state: Dict[str, Dict[str, Any]] = {}
    
    def Handle(self, request: baseline_pb2.AgentEnvelope, context) -> baseline_pb2.AgentResult:
        """Handle agent request based on role and content type."""
        start_ns = time.perf_counter_ns()
        
        # Initialize task state if needed
        if request.task_id not in self.task_state:
            self.task_state[request.task_id] = {}
        
        state = self.task_state[request.task_id]
        
        try:
            if request.content_type == "application/json":
                # Arm A: JSON handling
                result_payload = self._handle_json(request, state)
                content_type = "application/json"
            elif request.content_type == "application/x-protobuf":
                # Arm C: Protobuf handling
                result_payload = self._handle_protobuf(request, state)
                content_type = "application/x-protobuf"
            else:
                raise ValueError(f"Unsupported content_type: {request.content_type}")
            
            # Calculate message path time
            end_ns = time.perf_counter_ns()
            message_path_ms = (end_ns - start_ns) // 1_000_000
            
            return baseline_pb2.AgentResult(
                ok=True,
                content_type=content_type,
                payload=result_payload,
                bytes_in=len(request.payload),
                bytes_out=len(result_payload),
                message_path_ms=message_path_ms
            )
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return baseline_pb2.AgentResult(
                ok=False,
                error=str(e),
                bytes_in=len(request.payload),
                bytes_out=0
            )
    
    def _handle_json(self, request: baseline_pb2.AgentEnvelope, state: Dict) -> bytes:
        """Handle JSON payload for Arm A."""
        # Parse input
        input_data = json.loads(request.payload) if request.payload else {}
        
        # Merge with task_id from envelope
        task = input_data.copy()
        task["task_id"] = request.task_id
        
        if request.role == "planner":
            result = self.planner.plan_json(task)
            # Store plan in state
            plan_data = json.loads(result)
            state["plan_steps"] = plan_data.get("steps", [])
        elif request.role == "coder":
            plan_steps = state.get("plan_steps", [])
            result = self.coder.code_json(task, plan_steps)
            # Store patch in state
            code_data = json.loads(result)
            state["patch"] = code_data.get("patch", "")
        elif request.role == "tester":
            patch = state.get("patch", "")
            result = self.tester.test_json(task, patch)
        else:
            raise ValueError(f"Unknown role: {request.role}")
        
        return result.encode("utf-8")
    
    def _handle_protobuf(self, request: baseline_pb2.AgentEnvelope, state: Dict) -> bytes:
        """Handle Protobuf payload for Arm C and PM."""
        # For PM arm, delegate to PM agent
        if self.arm == "PM":
            return self._handle_pm(request, state)
            
        # Otherwise, handle as regular Arm C
        if request.role == "planner":
            # Parse PlanRequest
            plan_req = baseline_pb2.PlanRequest()
            plan_req.ParseFromString(request.payload)
            
            # Generate plan
            task = {
                "task_id": plan_req.task_id,
                "repo": plan_req.repo,
                "file_path": plan_req.file_path,
                "test_name": plan_req.test_name,
                "description": plan_req.description
            }
            plan_result = self.planner.plan(task)
            
            # Store in state
            state["plan_steps"] = plan_result["steps"]
            
            # Create PlanResponse
            plan_resp = baseline_pb2.PlanResponse(
                steps=plan_result["steps"],
                approach=plan_result["approach"],
                confidence=plan_result["confidence"]
            )
            return plan_resp.SerializeToString()
            
        elif request.role == "coder":
            # Parse CodeRequest
            code_req = baseline_pb2.CodeRequest()
            code_req.ParseFromString(request.payload)
            
            # Generate code
            task = {
                "task_id": code_req.task_id,
                "file_path": code_req.file_path
            }
            code_result = self.coder.code(task, list(code_req.plan_steps))
            
            # Store in state
            state["patch"] = code_result["patch"]
            
            # Create CodeResponse
            code_resp = baseline_pb2.CodeResponse(
                patch=code_result["patch"],
                files_changed=code_result["files_changed"],
                lines_added=code_result["lines_added"],
                lines_removed=code_result["lines_removed"]
            )
            return code_resp.SerializeToString()
            
        elif request.role == "tester":
            # Parse TestRequest
            test_req = baseline_pb2.TestRequest()
            test_req.ParseFromString(request.payload)
            
            # Run test
            task = {
                "task_id": test_req.task_id,
                "test_name": test_req.test_name
            }
            test_result = self.tester.test(task, test_req.patch)
            
            # Create TestResponse
            test_resp = baseline_pb2.TestResponse(
                passed=test_result["passed"],
                output=test_result["output"],
                duration_ms=test_result["duration_ms"],
                failures=test_result["failures"]
            )
            return test_resp.SerializeToString()
        
        else:
            raise ValueError(f"Unknown role: {request.role}")
    
    def _handle_pm(self, request: baseline_pb2.AgentEnvelope, state: Dict) -> bytes:
        """Handle requests for Arm PM using MCP anchors."""
        if request.role == "planner":
            # Parse PlanRequest
            plan_req = baseline_pb2.PlanRequest()
            plan_req.ParseFromString(request.payload)
            
            # Use PM agent to handle with MCP anchors
            plan_resp = self.pm_agent.handle_plan_request(plan_req)
            
            # Store in state (may contain MCP refs)
            state["plan_steps"] = list(plan_resp.steps)
            state["approach"] = plan_resp.approach
            
            return plan_resp.SerializeToString()
            
        elif request.role == "coder":
            # Parse CodeRequest
            code_req = baseline_pb2.CodeRequest()
            code_req.ParseFromString(request.payload)
            
            # Use PM agent to handle with MCP anchors
            code_resp = self.pm_agent.handle_code_request(code_req)
            
            # Store in state (may contain MCP ref)
            state["patch"] = code_resp.patch
            
            return code_resp.SerializeToString()
            
        elif request.role == "tester":
            # Parse TestRequest
            test_req = baseline_pb2.TestRequest()
            test_req.ParseFromString(request.payload)
            
            # Use PM agent to handle with MCP anchors
            test_resp = self.pm_agent.handle_test_request(test_req)
            
            return test_resp.SerializeToString()
        
        else:
            raise ValueError(f"Unknown role: {request.role}")


class GrpcTransport:
    """Manages gRPC server and client over UNIX domain sockets."""
    
    def __init__(self, socket_path: str, arm: str = "A", seed: int = 0, config: Optional[Dict] = None):
        """Initialize transport.
        
        Args:
            socket_path: Path to UNIX domain socket
            arm: "A" for JSON, "C" for Protobuf, "PM" for Protobuf+MCP
            seed: Random seed for determinism
            config: Configuration dict (needed for PM arm)
        """
        self.socket_path = socket_path
        self.arm = arm
        self.seed = seed
        self.config = config
        self.server = None
        self.channel = None
        self.stub = None
        
        # Ensure socket directory exists
        socket_dir = Path(socket_path).parent
        socket_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up any existing socket
        if os.path.exists(socket_path):
            os.unlink(socket_path)
    
    def start_server(self) -> None:
        """Start gRPC server on UNIX domain socket."""
        self.server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(max_workers=4)
        )
        service = ArmServiceImpl(self.arm, self.seed, self.config)
        baseline_pb2_grpc.add_ArmServiceServicer_to_server(service, self.server)
        
        # Bind to UNIX domain socket (not TCP!)
        # Python gRPC requires unix: with single slash
        self.server.add_insecure_port(f"unix:{self.socket_path}")
        self.server.start()
        logger.info(f"gRPC server started on unix:{self.socket_path}")
    
    def connect_client(self) -> None:
        """Connect client to UNIX domain socket."""
        # Python gRPC requires unix: with single slash
        self.channel = grpc.insecure_channel(f"unix:{self.socket_path}")
        self.stub = baseline_pb2_grpc.ArmServiceStub(self.channel)
        logger.info(f"gRPC client connected to unix:{self.socket_path}")
    
    def call_agent(
        self,
        task_id: str,
        role: str,
        payload: bytes,
        content_type: str,
        trace_id: str = ""
    ) -> Tuple[baseline_pb2.AgentResult, float]:
        """Call agent via gRPC and measure RTT.
        
        Returns:
            (result, rtt_ms)
        """
        if not self.stub:
            raise RuntimeError("Client not connected")
        
        # Create request
        request = baseline_pb2.AgentEnvelope(
            task_id=task_id,
            role=role,
            content_type=content_type,
            payload=payload,
            trace_id=trace_id,
            span_id=f"{role}_{task_id}",
            timestamp_ns=time.time_ns()
        )
        
        # Measure RTT
        start_ns = time.perf_counter_ns()
        result = self.stub.Handle(request)
        end_ns = time.perf_counter_ns()
        rtt_ms = (end_ns - start_ns) / 1_000_000
        
        return result, rtt_ms
    
    def stop(self) -> None:
        """Stop server and close connections."""
        if self.channel:
            self.channel.close()
        if self.server:
            self.server.stop(grace=1.0)
        # Clean up socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)