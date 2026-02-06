"""
PHASE 3.1: Agent Collaboration System
======================================
Multiple agents reasoning together, reaching consensus on complex issues,
and managing escalation workflows.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConsensusStrategy(Enum):
    """Consensus decision strategies for agent collaboration"""
    MAJORITY = "majority"  # More than 50% agreement
    UNANIMOUS = "unanimous"  # All agents must agree
    EXPERT_WEIGHTED = "expert_weighted"  # Weight by agent expertise score
    PRIORITY_BASED = "priority_based"  # Highest priority agent decides


@dataclass
class AgentOpinion:
    """Individual agent's opinion on a topic"""
    agent_id: str
    agent_name: str
    opinion: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    expertise_score: float  # 0.0 to 1.0
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus"""
    topic: str
    final_decision: str
    strategy: str
    agreement_level: float  # 0.0 to 1.0
    individual_opinions: List[Dict]
    dissenting_opinions: List[Dict]
    confidence: float
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


class AgentCollaborationHub:
    """
    Central hub for multi-agent collaboration and consensus
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.collaboration_sessions: Dict[str, Dict] = {}
        self.consensus_history: List[ConsensusResult] = []
        self.escalation_queue: List[Dict] = []
        logger.info("✅ Agent Collaboration Hub initialized")
    
    def register_agent(self, agent_id: str, agent_name: str, 
                       specialty: str, expertise_score: float = 0.7):
        """
        Register an agent for collaboration
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            specialty: Agent's area of expertise
            expertise_score: Agent's expertise level (0.0-1.0)
        """
        self.agents[agent_id] = {
            "id": agent_id,
            "name": agent_name,
            "specialty": specialty,
            "expertise_score": expertise_score,
            "total_collaborations": 0,
            "successful_consensus": 0,
            "registered_at": datetime.now().isoformat()
        }
        logger.info(f"✅ Agent registered: {agent_name} ({specialty})")
        return self.agents[agent_id]
    
    def get_agent_pool(self, specialty: Optional[str] = None) -> List[Dict]:
        """Get available agents, optionally filtered by specialty"""
        if specialty:
            return [a for a in self.agents.values() if a["specialty"] == specialty]
        return list(self.agents.values())
    
    async def initiate_collaboration(self, 
                                     session_id: str,
                                     topic: str,
                                     agent_ids: Optional[List[str]] = None,
                                     strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY):
        """
        Start a multi-agent collaboration session
        
        Args:
            session_id: Unique session identifier
            topic: Topic to discuss and reach consensus on
            agent_ids: Specific agents to involve (all if None)
            strategy: Consensus strategy to use
        """
        selected_agents = agent_ids or list(self.agents.keys())
        
        self.collaboration_sessions[session_id] = {
            "id": session_id,
            "topic": topic,
            "agents": selected_agents,
            "strategy": strategy.value,
            "opinions": [],
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"🤝 Collaboration session {session_id} initiated")
        logger.info(f"   Topic: {topic}")
        logger.info(f"   Agents involved: {len(selected_agents)}")
        logger.info(f"   Strategy: {strategy.value}")
        
        return self.collaboration_sessions[session_id]
    
    async def add_agent_opinion(self,
                               session_id: str,
                               agent_id: str,
                               opinion: str,
                               confidence: float,
                               reasoning: str) -> AgentOpinion:
        """
        Add an agent's opinion to the collaboration
        
        Args:
            session_id: Collaboration session ID
            agent_id: Agent providing the opinion
            opinion: The agent's position/decision
            confidence: How confident the agent is (0.0-1.0)
            reasoning: Explanation for the opinion
        """
        if session_id not in self.collaboration_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent = self.agents[agent_id]
        
        agent_opinion = AgentOpinion(
            agent_id=agent_id,
            agent_name=agent["name"],
            opinion=opinion,
            confidence=confidence,
            reasoning=reasoning,
            expertise_score=agent["expertise_score"],
            timestamp=datetime.now().isoformat()
        )
        
        self.collaboration_sessions[session_id]["opinions"].append(agent_opinion.to_dict())
        
        logger.info(f"💬 Opinion added from {agent['name']}: {opinion} (confidence: {confidence:.2f})")
        
        return agent_opinion
    
    async def reach_consensus(self, session_id: str) -> ConsensusResult:
        """
        Analyze all agent opinions and reach consensus
        
        Args:
            session_id: Collaboration session ID
        
        Returns:
            ConsensusResult with final decision and reasoning
        """
        if session_id not in self.collaboration_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.collaboration_sessions[session_id]
        opinions = session["opinions"]
        strategy = ConsensusStrategy(session["strategy"])
        topic = session["topic"]
        
        if not opinions:
            raise ValueError("No agent opinions collected yet")
        
        # Calculate consensus based on strategy
        final_decision, agreement_level = self._calculate_consensus(opinions, strategy)
        
        # Identify dissenting opinions
        majority_opinion = final_decision
        dissenting = [op for op in opinions if op["opinion"] != majority_opinion]
        
        # Calculate overall confidence
        confidence = sum(op["confidence"] for op in opinions) / len(opinions)
        
        consensus_result = ConsensusResult(
            topic=topic,
            final_decision=final_decision,
            strategy=strategy.value,
            agreement_level=agreement_level,
            individual_opinions=opinions,
            dissenting_opinions=dissenting,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        self.consensus_history.append(consensus_result)
        session["status"] = "completed"
        
        logger.info(f"✅ Consensus reached on '{topic}'")
        logger.info(f"   Decision: {final_decision}")
        logger.info(f"   Agreement level: {agreement_level:.2%}")
        logger.info(f"   Confidence: {confidence:.2f}")
        
        return consensus_result
    
    def _calculate_consensus(self, opinions: List[Dict], 
                            strategy: ConsensusStrategy) -> tuple[str, float]:
        """
        Calculate consensus based on selected strategy
        
        Returns:
            (final_decision, agreement_level)
        """
        if strategy == ConsensusStrategy.MAJORITY:
            # Most common opinion wins
            opinion_counts = {}
            for op in opinions:
                opinion = op["opinion"]
                opinion_counts[opinion] = opinion_counts.get(opinion, 0) + 1
            
            majority_opinion = max(opinion_counts, key=opinion_counts.get)
            agreement_level = opinion_counts[majority_opinion] / len(opinions)
            return majority_opinion, agreement_level
        
        elif strategy == ConsensusStrategy.UNANIMOUS:
            # All opinions must be the same
            unique_opinions = set(op["opinion"] for op in opinions)
            if len(unique_opinions) == 1:
                return opinions[0]["opinion"], 1.0
            else:
                # If not unanimous, use most confident opinion
                best_op = max(opinions, key=lambda x: x["confidence"])
                agreement_level = 1 / len(unique_opinions)
                return best_op["opinion"], agreement_level
        
        elif strategy == ConsensusStrategy.EXPERT_WEIGHTED:
            # Weight opinions by expertise
            weighted_scores = {}
            total_weight = 0
            
            for op in opinions:
                opinion = op["opinion"]
                weight = op["confidence"] * op["expertise_score"]
                weighted_scores[opinion] = weighted_scores.get(opinion, 0) + weight
                total_weight += weight
            
            best_opinion = max(weighted_scores, key=weighted_scores.get)
            agreement_level = weighted_scores[best_opinion] / total_weight if total_weight > 0 else 0
            return best_opinion, agreement_level
        
        elif strategy == ConsensusStrategy.PRIORITY_BASED:
            # Highest priority (expertise) agent decides
            best_agent_op = max(opinions, key=lambda x: x["expertise_score"])
            return best_agent_op["opinion"], best_agent_op["expertise_score"]
        
        return opinions[0]["opinion"], 1 / len(opinions)
    
    async def escalate_issue(self, issue: str, severity: str = "medium",
                            specialist_required: Optional[str] = None) -> Dict:
        """
        Escalate an issue when agents can't reach consensus or decision is critical
        
        Args:
            issue: Description of the issue
            severity: Issue severity (low, medium, high, critical)
            specialist_required: Specific specialist needed
        """
        escalation = {
            "id": f"esc_{datetime.now().timestamp()}",
            "issue": issue,
            "severity": severity,
            "specialist_required": specialist_required,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "assigned_to": None,
            "resolution": None
        }
        
        self.escalation_queue.append(escalation)
        
        logger.warning(f"⚠️  ESCALATION: {severity.upper()} - {issue}")
        if specialist_required:
            logger.warning(f"   Specialist required: {specialist_required}")
        
        return escalation
    
    async def resolve_escalation(self, escalation_id: str, 
                                resolution: str, assigned_to: str) -> Dict:
        """
        Resolve an escalated issue
        
        Args:
            escalation_id: ID of escalation to resolve
            resolution: How the issue was resolved
            assigned_to: Person/agent who resolved it
        """
        for esc in self.escalation_queue:
            if esc["id"] == escalation_id:
                esc["status"] = "resolved"
                esc["resolution"] = resolution
                esc["assigned_to"] = assigned_to
                esc["resolved_at"] = datetime.now().isoformat()
                
                logger.info(f"✅ Escalation {escalation_id} resolved by {assigned_to}")
                return esc
        
        raise ValueError(f"Escalation {escalation_id} not found")
    
    def get_collaboration_metrics(self) -> Dict:
        """Get metrics about agent collaborations"""
        total_collaborations = len(self.collaboration_sessions)
        completed_collaborations = sum(1 for s in self.collaboration_sessions.values() 
                                       if s["status"] == "completed")
        
        avg_agreement = 0
        if self.consensus_history:
            avg_agreement = sum(c.agreement_level for c in self.consensus_history) / len(self.consensus_history)
        
        return {
            "total_agents": len(self.agents),
            "total_collaboration_sessions": total_collaborations,
            "completed_sessions": completed_collaborations,
            "active_sessions": total_collaborations - completed_collaborations,
            "average_agreement_level": avg_agreement,
            "escalations_pending": sum(1 for e in self.escalation_queue if e["status"] == "pending"),
            "escalations_resolved": sum(1 for e in self.escalation_queue if e["status"] == "resolved"),
            "consensus_history_count": len(self.consensus_history)
        }


# Global instance
agent_collaboration_hub = AgentCollaborationHub()
