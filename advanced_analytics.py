"""
PHASE 3.3: Advanced Analytics Engine
=====================================
User satisfaction tracking, agent effectiveness metrics,
revenue impact analysis, and churn prediction.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class UserSatisfactionMetric:
    """User satisfaction data point"""
    user_id: str
    agent_id: str
    rating: int  # 1-5
    comment: str
    resolution_time_seconds: int
    issue_complexity: str  # low, medium, high
    category: str
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AgentEffectivenessMetric:
    """Agent performance metric"""
    agent_id: str
    metric_date: str
    total_interactions: int
    successful_resolutions: int
    average_resolution_time: float
    customer_satisfaction_avg: float
    escalation_rate: float
    repeat_issue_rate: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RevenueImpact:
    """Revenue impact data"""
    agent_id: str
    period: str
    revenue_generated: float
    orders_facilitated: int
    average_order_value: float
    upsells_completed: int
    cost_of_operation: float
    roi: float
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ChurnPrediction:
    """Customer churn prediction"""
    customer_id: str
    churn_probability: float  # 0.0 to 1.0
    risk_level: str  # low, medium, high, critical
    primary_factors: List[str]
    recommended_actions: List[str]
    prediction_date: str
    confidence: float


class AdvancedAnalyticsEngine:
    """
    Enterprise-grade analytics for:
    - Customer satisfaction tracking
    - Agent effectiveness metrics
    - Revenue impact analysis
    - Churn prediction
    """
    
    def __init__(self):
        self.satisfaction_metrics: List[UserSatisfactionMetric] = []
        self.agent_effectiveness: Dict[str, List[AgentEffectivenessMetric]] = defaultdict(list)
        self.revenue_impacts: List[RevenueImpact] = []
        self.churn_predictions: Dict[str, ChurnPrediction] = {}
        self.customer_history: Dict[str, Dict] = {}
        
        logger.info("✅ Advanced Analytics Engine initialized")
    
    async def record_satisfaction(self, 
                                 user_id: str,
                                 agent_id: str,
                                 rating: int,
                                 comment: str,
                                 resolution_time_seconds: int,
                                 issue_complexity: str = "medium",
                                 category: str = "general") -> UserSatisfactionMetric:
        """
        Record user satisfaction metric
        
        Args:
            user_id: User identifier
            agent_id: Agent who handled the interaction
            rating: Satisfaction rating (1-5)
            comment: User feedback
            resolution_time_seconds: Time to resolve issue
            issue_complexity: Complexity level
            category: Issue category
        """
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        metric = UserSatisfactionMetric(
            user_id=user_id,
            agent_id=agent_id,
            rating=rating,
            comment=comment,
            resolution_time_seconds=resolution_time_seconds,
            issue_complexity=issue_complexity,
            category=category,
            timestamp=datetime.now().isoformat()
        )
        
        self.satisfaction_metrics.append(metric)
        
        # Update customer history
        if user_id not in self.customer_history:
            self.customer_history[user_id] = {
                "total_interactions": 0,
                "ratings": [],
                "agents_interacted": set(),
                "first_contact": datetime.now().isoformat()
            }
        
        history = self.customer_history[user_id]
        history["total_interactions"] += 1
        history["ratings"].append(rating)
        history["agents_interacted"].add(agent_id)
        
        logger.info(f"📊 Satisfaction recorded: User {user_id} → Agent {agent_id}, Rating: {rating}/5")
        
        return metric
    
    async def calculate_agent_effectiveness(self, agent_id: str,
                                           start_date: Optional[str] = None,
                                           end_date: Optional[str] = None) -> AgentEffectivenessMetric:
        """
        Calculate overall effectiveness metric for an agent
        
        Args:
            agent_id: Agent to analyze
            start_date: Start of analysis period (ISO format)
            end_date: End of analysis period (ISO format)
        
        Returns:
            AgentEffectivenessMetric with calculated scores
        """
        # Filter metrics for agent in date range
        agent_metrics = [m for m in self.satisfaction_metrics if m.agent_id == agent_id]
        
        if start_date:
            agent_metrics = [m for m in agent_metrics if m.timestamp >= start_date]
        if end_date:
            agent_metrics = [m for m in agent_metrics if m.timestamp <= end_date]
        
        if not agent_metrics:
            logger.warning(f"No metrics found for agent {agent_id}")
            return None
        
        total_interactions = len(agent_metrics)
        
        # Calculate successful resolutions (rating >= 4)
        successful = sum(1 for m in agent_metrics if m.rating >= 4)
        success_rate = successful / total_interactions if total_interactions > 0 else 0
        
        # Calculate average satisfaction
        avg_satisfaction = sum(m.rating for m in agent_metrics) / total_interactions
        
        # Calculate average resolution time
        avg_resolution_time = sum(m.resolution_time_seconds for m in agent_metrics) / total_interactions
        
        # Calculate escalation rate (low ratings = escalations)
        escalations = sum(1 for m in agent_metrics if m.rating < 3)
        escalation_rate = escalations / total_interactions if total_interactions > 0 else 0
        
        # Estimate repeat issue rate (customers contacting again)
        repeat_customers = sum(1 for customer_id, history in self.customer_history.items()
                              if history["total_interactions"] > 1 and agent_id in history["agents_interacted"])
        repeat_rate = repeat_customers / total_interactions if total_interactions > 0 else 0
        
        effectiveness = AgentEffectivenessMetric(
            agent_id=agent_id,
            metric_date=datetime.now().isoformat(),
            total_interactions=total_interactions,
            successful_resolutions=successful,
            average_resolution_time=avg_resolution_time,
            customer_satisfaction_avg=avg_satisfaction,
            escalation_rate=escalation_rate,
            repeat_issue_rate=repeat_rate
        )
        
        self.agent_effectiveness[agent_id].append(effectiveness)
        
        logger.info(f"✅ Agent effectiveness calculated for {agent_id}")
        logger.info(f"   Total interactions: {total_interactions}")
        logger.info(f"   Success rate: {success_rate:.2%}")
        logger.info(f"   Avg satisfaction: {avg_satisfaction:.2f}/5")
        
        return effectiveness
    
    async def track_revenue_impact(self,
                                  agent_id: str,
                                  revenue_generated: float,
                                  orders_facilitated: int,
                                  upsells_completed: int = 0,
                                  cost_of_operation: float = 10.0,
                                  period: str = "daily") -> RevenueImpact:
        """
        Track revenue impact of an agent
        
        Args:
            agent_id: Agent to track
            revenue_generated: Total revenue attributed to agent
            orders_facilitated: Number of orders completed
            upsells_completed: Number of successful upsells
            cost_of_operation: Cost to run the agent
            period: Analysis period (daily, weekly, monthly)
        
        Returns:
            RevenueImpact metrics
        """
        avg_order_value = revenue_generated / orders_facilitated if orders_facilitated > 0 else 0
        
        # Calculate ROI
        net_revenue = revenue_generated - cost_of_operation
        roi = (net_revenue / cost_of_operation * 100) if cost_of_operation > 0 else 0
        
        impact = RevenueImpact(
            agent_id=agent_id,
            period=period,
            revenue_generated=revenue_generated,
            orders_facilitated=orders_facilitated,
            average_order_value=avg_order_value,
            upsells_completed=upsells_completed,
            cost_of_operation=cost_of_operation,
            roi=roi,
            timestamp=datetime.now().isoformat()
        )
        
        self.revenue_impacts.append(impact)
        
        logger.info(f"💰 Revenue impact tracked for {agent_id}")
        logger.info(f"   Revenue: ${revenue_generated:.2f}")
        logger.info(f"   Orders: {orders_facilitated}")
        logger.info(f"   ROI: {roi:.2f}%")
        
        return impact
    
    async def predict_churn(self, customer_id: str) -> ChurnPrediction:
        """
        Predict likelihood of customer churn
        
        Args:
            customer_id: Customer to analyze
        
        Returns:
            ChurnPrediction with probability and recommendations
        """
        if customer_id not in self.customer_history:
            logger.warning(f"No history for customer {customer_id}")
            return None
        
        history = self.customer_history[customer_id]
        
        # Calculate churn risk factors
        risk_factors = []
        churn_probability = 0.0
        
        # Factor 1: Low satisfaction trend
        recent_ratings = history["ratings"][-5:] if history["ratings"] else []
        if recent_ratings:
            avg_rating = sum(recent_ratings) / len(recent_ratings)
            if avg_rating < 2.5:
                risk_factors.append("Low satisfaction ratings")
                churn_probability += 0.3
            elif avg_rating < 3.0:
                risk_factors.append("Declining satisfaction trend")
                churn_probability += 0.15
        
        # Factor 2: Decreasing engagement
        interaction_frequency = history["total_interactions"]
        if interaction_frequency == 1:
            risk_factors.append("Single interaction (one-time user)")
            churn_probability += 0.2
        elif interaction_frequency < 3:
            risk_factors.append("Low engagement frequency")
            churn_probability += 0.1
        
        # Factor 3: Multiple agent interactions without resolution
        num_agents = len(history["agents_interacted"])
        if num_agents > 3 and interaction_frequency < 10:
            risk_factors.append("Multiple agents (poor resolution)")
            churn_probability += 0.2
        
        # Factor 4: Negative sentiment pattern
        if recent_ratings and any(r < 2 for r in recent_ratings):
            risk_factors.append("Recent negative feedback")
            churn_probability += 0.15
        
        # Cap probability at 1.0
        churn_probability = min(churn_probability, 0.99)
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "low"
        elif churn_probability < 0.6:
            risk_level = "medium"
        elif churn_probability < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Generate recommended actions
        recommended_actions = []
        if risk_level in ["high", "critical"]:
            recommended_actions.append("Proactive outreach from support team")
            recommended_actions.append("Personalized retention offer")
        
        if "Low satisfaction" in risk_factors or "Declining satisfaction" in risk_factors:
            recommended_actions.append("Quality improvement focus")
            recommended_actions.append("Dedicated support agent assignment")
        
        if "Multiple agents" in risk_factors:
            recommended_actions.append("Streamline support process")
            recommended_actions.append("Quick resolution commitment")
        
        if not recommended_actions:
            recommended_actions.append("Continue monitoring")
        
        prediction = ChurnPrediction(
            customer_id=customer_id,
            churn_probability=churn_probability,
            risk_level=risk_level,
            primary_factors=risk_factors,
            recommended_actions=recommended_actions,
            prediction_date=datetime.now().isoformat(),
            confidence=min(interaction_frequency / 10, 1.0)  # Confidence increases with data
        )
        
        self.churn_predictions[customer_id] = prediction
        
        logger.info(f"🎯 Churn prediction for {customer_id}: {risk_level.upper()}")
        if risk_factors:
            for factor in risk_factors:
                logger.info(f"   • {factor}")
        
        return prediction
    
    async def get_cohort_analysis(self, start_date: str, end_date: str) -> Dict:
        """
        Analyze customer cohorts by satisfaction
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
        
        Returns:
            Cohort analysis data
        """
        metrics_in_period = [m for m in self.satisfaction_metrics
                            if start_date <= m.timestamp <= end_date]
        
        cohorts = defaultdict(list)
        for metric in metrics_in_period:
            cohorts[metric.issue_complexity].append(metric.rating)
        
        cohort_stats = {}
        for complexity, ratings in cohorts.items():
            cohort_stats[complexity] = {
                "count": len(ratings),
                "avg_rating": sum(ratings) / len(ratings),
                "satisfaction_rate": sum(1 for r in ratings if r >= 4) / len(ratings)
            }
        
        return {
            "period": f"{start_date} to {end_date}",
            "total_interactions": len(metrics_in_period),
            "cohorts": cohort_stats
        }
    
    async def get_trend_analysis(self, days: int = 30) -> Dict:
        """
        Analyze satisfaction trends over time
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Trend data
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_metrics = [m for m in self.satisfaction_metrics 
                         if m.timestamp >= cutoff_date]
        
        if not recent_metrics:
            return {"trend": "insufficient_data", "data_points": 0}
        
        # Group by date
        daily_data = defaultdict(list)
        for metric in recent_metrics:
            date = metric.timestamp.split("T")[0]
            daily_data[date].append(metric.rating)
        
        # Calculate daily averages
        daily_avg = {}
        for date in sorted(daily_data.keys()):
            ratings = daily_data[date]
            daily_avg[date] = sum(ratings) / len(ratings)
        
        # Determine trend direction
        if len(daily_avg) >= 2:
            values = list(daily_avg.values())
            first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
            second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            
            if second_half_avg > first_half_avg:
                trend = "improving"
            elif second_half_avg < first_half_avg:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "days_analyzed": days,
            "data_points": len(recent_metrics),
            "daily_averages": daily_avg,
            "overall_avg": sum(recent_metrics[m].rating for m in range(len(recent_metrics))) / len(recent_metrics)
        }
    
    def get_analytics_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        if not self.satisfaction_metrics:
            return {"status": "no_data"}
        
        all_ratings = [m.rating for m in self.satisfaction_metrics]
        
        high_risk_customers = sum(1 for p in self.churn_predictions.values() 
                                 if p.risk_level in ["high", "critical"])
        
        return {
            "total_interactions": len(self.satisfaction_metrics),
            "avg_satisfaction": sum(all_ratings) / len(all_ratings),
            "satisfaction_distribution": {
                "5_stars": sum(1 for r in all_ratings if r == 5),
                "4_stars": sum(1 for r in all_ratings if r == 4),
                "3_stars": sum(1 for r in all_ratings if r == 3),
                "2_stars": sum(1 for r in all_ratings if r == 2),
                "1_stars": sum(1 for r in all_ratings if r == 1),
            },
            "unique_customers": len(self.customer_history),
            "customers_at_risk": high_risk_customers,
            "total_revenue_tracked": sum(r.revenue_generated for r in self.revenue_impacts),
            "average_roi": sum(r.roi for r in self.revenue_impacts) / len(self.revenue_impacts) if self.revenue_impacts else 0
        }


# Global instance
advanced_analytics_engine = AdvancedAnalyticsEngine()
