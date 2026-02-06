"""
PHASE 2.4: Admin Dashboard Backend
Provides analytics and admin endpoints for system monitoring
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AdminAnalytics:
    """
    Analytics engine for admin dashboard
    Tracks performance, usage patterns, and system health
    """
    
    def __init__(self):
        self.data = {
            "queries_by_hour": defaultdict(int),
            "queries_by_agent": defaultdict(int),
            "queries_by_type": defaultdict(int),
            "customer_satisfaction": [],
            "error_rate_timeline": [],
            "top_issues": defaultdict(int),
            "agent_performance": {},
            "system_events": []
        }
        logger.info("AdminAnalytics initialized")
    
    def record_query(self, agent: str, query_type: str, success: bool,
                    response_time: float, customer_id: Optional[str] = None):
        """Record a query for analytics"""
        now = datetime.utcnow()
        hour_key = now.strftime("%Y-%m-%d %H:00")
        
        self.data["queries_by_hour"][hour_key] += 1
        self.data["queries_by_agent"][agent] += 1
        self.data["queries_by_type"][query_type] += 1
        
        # Track agent performance
        if agent not in self.data["agent_performance"]:
            self.data["agent_performance"][agent] = {
                "total": 0,
                "successful": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        perf = self.data["agent_performance"][agent]
        perf["total"] += 1
        if success:
            perf["successful"] += 1
        perf["total_time"] += response_time
        perf["avg_time"] = perf["total_time"] / perf["total"]
        
        logger.debug(f"Analytics recorded: {agent} - {query_type} - {success}")
    
    def record_satisfaction(self, rating: int, feedback: Optional[str] = None,
                           agent: Optional[str] = None):
        """Record customer satisfaction feedback"""
        self.data["customer_satisfaction"].append({
            "rating": rating,
            "feedback": feedback,
            "agent": agent,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        # Keep last 1000 records
        if len(self.data["customer_satisfaction"]) > 1000:
            self.data["customer_satisfaction"] = self.data["customer_satisfaction"][-1000:]
    
    def record_error(self, error_type: str, agent: str, details: Optional[str] = None):
        """Record system error"""
        self.data["top_issues"][f"{agent}:{error_type}"] += 1
        
        self.data["system_events"].append({
            "type": "error",
            "error_type": error_type,
            "agent": agent,
            "details": details,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        # Keep last 500 events
        if len(self.data["system_events"]) > 500:
            self.data["system_events"] = self.data["system_events"][-500:]
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get high-level dashboard summary"""
        total_queries = sum(self.data["queries_by_hour"].values())
        
        avg_satisfaction = None
        if self.data["customer_satisfaction"]:
            ratings = [r["rating"] for r in self.data["customer_satisfaction"]]
            avg_satisfaction = sum(ratings) / len(ratings)
        
        top_agent = max(
            self.data["queries_by_agent"].items(),
            key=lambda x: x[1]
        ) if self.data["queries_by_agent"] else (None, 0)
        
        # Calculate success rate
        total_successful = sum(
            perf["successful"] for perf in self.data["agent_performance"].values()
        )
        total_recorded = sum(
            perf["total"] for perf in self.data["agent_performance"].values()
        )
        success_rate = (total_successful / total_recorded * 100) if total_recorded > 0 else 0
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_queries": total_queries,
                "unique_agents": len(self.data["queries_by_agent"]),
                "success_rate": round(success_rate, 2),
                "avg_satisfaction": round(avg_satisfaction, 2) if avg_satisfaction else None,
                "top_agent": {"name": top_agent[0], "queries": top_agent[1]} if top_agent[0] else None
            },
            "queries_by_type": dict(self.data["queries_by_type"]),
            "recent_events": self.data["system_events"][-10:]
        }
    
    def get_agent_analytics(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed analytics for specific agent or all agents"""
        if agent:
            if agent in self.data["agent_performance"]:
                perf = self.data["agent_performance"][agent]
                success_rate = (perf["successful"] / perf["total"] * 100) if perf["total"] > 0 else 0
                return {
                    "agent": agent,
                    "total_queries": perf["total"],
                    "successful_queries": perf["successful"],
                    "success_rate": round(success_rate, 2),
                    "avg_response_time": round(perf["avg_time"], 3),
                    "query_count": self.data["queries_by_agent"].get(agent, 0)
                }
            else:
                return {"error": f"Agent {agent} not found"}
        else:
            # Return analytics for all agents
            all_agents = {}
            for agent_name, perf in self.data["agent_performance"].items():
                success_rate = (perf["successful"] / perf["total"] * 100) if perf["total"] > 0 else 0
                all_agents[agent_name] = {
                    "total_queries": perf["total"],
                    "successful_queries": perf["successful"],
                    "success_rate": round(success_rate, 2),
                    "avg_response_time": round(perf["avg_time"], 3)
                }
            return {"agents": all_agents}
    
    def get_satisfaction_report(self, days: int = 7) -> Dict[str, Any]:
        """Get customer satisfaction report"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_feedback = [
            r for r in self.data["customer_satisfaction"]
            if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).date() >= cutoff_date.date()
        ]
        
        if not recent_feedback:
            return {
                "period_days": days,
                "total_responses": 0,
                "avg_rating": None,
                "feedback": []
            }
        
        ratings = [r["rating"] for r in recent_feedback]
        avg_rating = sum(ratings) / len(ratings)
        
        # Group by rating
        rating_distribution = defaultdict(int)
        for r in recent_feedback:
            rating_distribution[r["rating"]] += 1
        
        return {
            "period_days": days,
            "total_responses": len(recent_feedback),
            "avg_rating": round(avg_rating, 2),
            "rating_distribution": dict(rating_distribution),
            "recent_feedback": [
                {
                    "rating": r["rating"],
                    "feedback": r["feedback"],
                    "agent": r["agent"],
                    "timestamp": r["timestamp"]
                }
                for r in recent_feedback[-20:]  # Last 20
            ]
        }
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get error summary and trending issues"""
        if not self.data["top_issues"]:
            return {"total_errors": 0, "top_issues": []}
        
        sorted_issues = sorted(
            self.data["top_issues"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recent_errors = [
            e for e in self.data["system_events"]
            if e["type"] == "error"
        ]
        
        return {
            "total_errors": sum(self.data["top_issues"].values()),
            "unique_error_types": len(self.data["top_issues"]),
            "top_issues": [
                {"issue": issue, "count": count}
                for issue, count in sorted_issues[:10]
            ],
            "recent_errors": recent_errors[-20:]
        }
    
    def get_hourly_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get query trend data for time series charts"""
        now = datetime.utcnow()
        hours_data = []
        
        for i in range(hours, 0, -1):
            hour_time = now - timedelta(hours=i)
            hour_key = hour_time.strftime("%Y-%m-%d %H:00")
            count = self.data["queries_by_hour"].get(hour_key, 0)
            hours_data.append({
                "hour": hour_key,
                "queries": count
            })
        
        return {
            "period_hours": hours,
            "data": hours_data,
            "total": sum(d["queries"] for d in hours_data)
        }
    
    def export_analytics(self, format: str = "json") -> Dict[str, Any] | str:
        """Export full analytics data"""
        export_data = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "agent_performance": dict(self.data["agent_performance"]),
            "queries_by_type": dict(self.data["queries_by_type"]),
            "queries_by_agent": dict(self.data["queries_by_agent"]),
            "total_issues": dict(self.data["top_issues"]),
            "satisfaction_data": self.data["customer_satisfaction"][-100:],
            "recent_events": self.data["system_events"][-100:]
        }
        
        if format == "json":
            import json
            return json.dumps(export_data, indent=2)
        
        return export_data


# Global analytics instance
admin_analytics = AdminAnalytics()
