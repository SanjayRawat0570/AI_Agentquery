"""
PHASE 1.5: Logging & Monitoring System
Track agent performance, system health, and metrics
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgentMetrics:
    """
    Tracks and manages agent performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "agent_usage": defaultdict(int),
            "agent_success_rate": defaultdict(lambda: {"success": 0, "total": 0}),
            "tool_usage": defaultdict(int),
            "errors": [],
            "start_time": datetime.utcnow().isoformat() + "Z"
        }
        logger.info("AgentMetrics initialized")
    
    def log_query(self, agent: str, success: bool, response_time: float, error: Optional[str] = None):
        """
        Log a query execution
        
        Args:
            agent: Name of the agent that handled the query
            success: Whether the query was successful
            response_time: Time taken to process the query in seconds
            error: Error message if the query failed
        """
        self.metrics["total_queries"] += 1
        
        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
            if error:
                self.metrics["errors"].append({
                    "agent": agent,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                # Keep only last 100 errors
                if len(self.metrics["errors"]) > 100:
                    self.metrics["errors"] = self.metrics["errors"][-100:]
        
        # Update agent-specific metrics
        self.metrics["agent_usage"][agent] += 1
        self.metrics["agent_success_rate"][agent]["total"] += 1
        if success:
            self.metrics["agent_success_rate"][agent]["success"] += 1
        
        # Update response time
        self.metrics["total_response_time"] += response_time
        self.metrics["avg_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["total_queries"]
        )
        
        logger.info(
            f"Query logged - Agent: {agent}, Success: {success}, "
            f"Response Time: {response_time:.2f}s"
        )
    
    def log_tool_usage(self, tool_name: str, success: bool):
        """Log tool usage"""
        key = f"{tool_name}_{'success' if success else 'failure'}"
        self.metrics["tool_usage"][key] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        # Calculate success rates
        agent_success_rates = {}
        for agent, data in self.metrics["agent_success_rate"].items():
            if data["total"] > 0:
                agent_success_rates[agent] = {
                    "success_rate": data["success"] / data["total"],
                    "total_queries": data["total"],
                    "successful_queries": data["success"]
                }
        
        return {
            "total_queries": self.metrics["total_queries"],
            "successful_queries": self.metrics["successful_queries"],
            "failed_queries": self.metrics["failed_queries"],
            "success_rate": (
                self.metrics["successful_queries"] / self.metrics["total_queries"]
                if self.metrics["total_queries"] > 0 else 0
            ),
            "avg_response_time": round(self.metrics["avg_response_time"], 3),
            "agent_usage": dict(self.metrics["agent_usage"]),
            "agent_success_rates": agent_success_rates,
            "tool_usage": dict(self.metrics["tool_usage"]),
            "recent_errors": self.metrics["errors"][-10:],
            "uptime_since": self.metrics["start_time"]
        }
    
    def reset_metrics(self):
        """Reset all metrics (use with caution)"""
        self.__init__()
        logger.warning("Metrics have been reset")


class PerformanceMonitor:
    """
    Monitor for tracking individual operation performance
    """
    
    def __init__(self):
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.completed_operations: List[Dict[str, Any]] = []
        self.max_history = 1000
        logger.info("PerformanceMonitor initialized")
    
    def start_operation(self, operation_id: str, operation_type: str, metadata: Optional[Dict] = None):
        """Start tracking an operation"""
        self.active_operations[operation_id] = {
            "operation_type": operation_type,
            "start_time": time.time(),
            "start_timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata or {}
        }
        logger.debug(f"Started tracking operation: {operation_id} ({operation_type})")
    
    def end_operation(self, operation_id: str, success: bool = True, result: Optional[Any] = None):
        """End tracking an operation"""
        if operation_id not in self.active_operations:
            logger.warning(f"Attempted to end unknown operation: {operation_id}")
            return
        
        operation = self.active_operations.pop(operation_id)
        duration = time.time() - operation["start_time"]
        
        completed_op = {
            "operation_id": operation_id,
            "operation_type": operation["operation_type"],
            "duration": duration,
            "success": success,
            "start_timestamp": operation["start_timestamp"],
            "end_timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": operation["metadata"]
        }
        
        if result is not None:
            completed_op["result"] = result
        
        self.completed_operations.append(completed_op)
        
        # Keep only recent operations
        if len(self.completed_operations) > self.max_history:
            self.completed_operations = self.completed_operations[-self.max_history:]
        
        logger.debug(f"Completed operation: {operation_id} in {duration:.2f}s")
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get list of currently active operations"""
        return [
            {
                "operation_id": op_id,
                "operation_type": op_data["operation_type"],
                "elapsed_time": time.time() - op_data["start_time"],
                "start_timestamp": op_data["start_timestamp"]
            }
            for op_id, op_data in self.active_operations.items()
        ]
    
    def get_recent_operations(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed operations"""
        return self.completed_operations[-count:]


def track_performance(operation_type: str = None):
    """
    Decorator to track function performance
    
    Usage:
        @track_performance(operation_type="agent_query")
        async def process_query(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = operation_type or func.__name__
            success = True
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                logger.error(f"{func_name} failed: {error}")
                raise
            finally:
                response_time = time.time() - start_time
                logger.info(
                    f"{func_name} completed in {response_time:.2f}s "
                    f"(success={success})"
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = operation_type or func.__name__
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                logger.error(f"{func_name} failed: {error}")
                raise
            finally:
                response_time = time.time() - start_time
                logger.info(
                    f"{func_name} completed in {response_time:.2f}s "
                    f"(success={success})"
                )
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class HealthCheck:
    """
    System health checker
    """
    
    def __init__(self):
        self.components = {}
        logger.info("HealthCheck initialized")
    
    def register_component(self, name: str, check_func: callable):
        """Register a component health check function"""
        self.components[name] = check_func
        logger.info(f"Registered health check for: {name}")
    
    def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks and return status
        
        Returns:
            Dictionary with overall status and component statuses
        """
        results = {}
        all_healthy = True
        
        for name, check_func in self.components.items():
            try:
                is_healthy = check_func()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "checked_at": datetime.utcnow().isoformat() + "Z"
                }
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "checked_at": datetime.utcnow().isoformat() + "Z"
                }
                all_healthy = False
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global instances
agent_metrics = AgentMetrics()
performance_monitor = PerformanceMonitor()
health_check = HealthCheck()
