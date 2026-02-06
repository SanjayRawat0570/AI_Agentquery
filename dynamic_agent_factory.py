"""
PHASE 3.2: Dynamic Agent Creation System
=========================================
Create specialized agents at runtime with custom tool binding
and configuration for specific use cases.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents that can be dynamically created"""
    SUPPORT = "support"  # Customer support
    SALES = "sales"  # Sales assistance
    TECHNICAL = "technical"  # Technical support
    BILLING = "billing"  # Billing inquiries
    FEEDBACK = "feedback"  # Feedback collection
    CUSTOM = "custom"  # Custom specialized agent


@dataclass
class ToolBinding:
    """Binding of a tool to an agent"""
    tool_name: str
    tool_func: Callable
    description: str
    required_params: List[str]
    enabled: bool = True
    
    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "required_params": self.required_params,
            "enabled": self.enabled
        }


@dataclass
class AgentConfiguration:
    """Configuration for a dynamically created agent"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    personality: str  # e.g., "professional", "friendly", "technical"
    expertise_level: float  # 0.0 to 1.0
    tools: List[ToolBinding]
    system_prompt: str
    response_style: str  # e.g., "concise", "detailed", "conversational"
    temperature: float  # 0.0 to 1.0 for LLM randomness
    max_turns: int  # Max conversation turns before escalation
    created_at: str
    created_by: str
    is_active: bool = True
    
    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "description": self.description,
            "personality": self.personality,
            "expertise_level": self.expertise_level,
            "tools": [t.to_dict() for t in self.tools],
            "system_prompt": self.system_prompt,
            "response_style": self.response_style,
            "temperature": self.temperature,
            "max_turns": self.max_turns,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "is_active": self.is_active
        }


class DynamicAgentFactory:
    """
    Factory for creating specialized agents at runtime with
    custom configurations and tool bindings.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentConfiguration] = {}
        self.agent_templates: Dict[str, Dict] = self._init_templates()
        self.available_tools: Dict[str, Callable] = {}
        self.agent_sessions: Dict[str, Dict] = {}
        logger.info("✅ Dynamic Agent Factory initialized")
    
    def _init_templates(self) -> Dict[str, Dict]:
        """Initialize predefined agent templates"""
        return {
            "support": {
                "type": AgentType.SUPPORT,
                "description": "Customer support agent",
                "personality": "friendly",
                "response_style": "helpful",
                "temperature": 0.5,
                "max_turns": 8,
                "base_tools": ["get_customer_info", "check_status", "create_ticket"]
            },
            "sales": {
                "type": AgentType.SALES,
                "description": "Sales assistant",
                "personality": "professional",
                "response_style": "concise",
                "temperature": 0.4,
                "max_turns": 6,
                "base_tools": ["check_inventory", "get_pricing", "create_order"]
            },
            "technical": {
                "type": AgentType.TECHNICAL,
                "description": "Technical support specialist",
                "personality": "technical",
                "response_style": "detailed",
                "temperature": 0.3,
                "max_turns": 10,
                "base_tools": ["diagnose_issue", "get_logs", "restart_service"]
            },
            "billing": {
                "type": AgentType.BILLING,
                "description": "Billing specialist",
                "personality": "professional",
                "response_style": "formal",
                "temperature": 0.2,
                "max_turns": 5,
                "base_tools": ["get_invoice", "process_payment", "check_subscription"]
            }
        }
    
    def register_tool(self, tool_name: str, tool_func: Callable, 
                     description: str, required_params: List[str]):
        """
        Register a tool available for agent binding
        
        Args:
            tool_name: Name of the tool
            tool_func: Callable function for the tool
            description: Tool description
            required_params: List of required parameters
        """
        self.available_tools[tool_name] = {
            "func": tool_func,
            "description": description,
            "required_params": required_params
        }
        logger.info(f"✅ Tool registered: {tool_name}")
    
    async def create_agent_from_template(self, 
                                        agent_id: str,
                                        name: str,
                                        template_name: str,
                                        created_by: str = "system",
                                        custom_tools: Optional[List[str]] = None) -> AgentConfiguration:
        """
        Create an agent from a predefined template
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            template_name: Name of template to use
            created_by: Who created this agent
            custom_tools: Additional tools to bind
        
        Returns:
            AgentConfiguration for the created agent
        """
        if template_name not in self.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.agent_templates[template_name]
        
        # Gather tools
        tool_names = template["base_tools"].copy()
        if custom_tools:
            tool_names.extend(custom_tools)
        
        tools = self._bind_tools(tool_names)
        
        # Build system prompt
        system_prompt = self._generate_system_prompt(
            name,
            template["description"],
            template["personality"],
            template["response_style"]
        )
        
        agent_config = AgentConfiguration(
            agent_id=agent_id,
            agent_type=template["type"].value,
            name=name,
            description=template["description"],
            personality=template["personality"],
            expertise_level=0.7,
            tools=tools,
            system_prompt=system_prompt,
            response_style=template["response_style"],
            temperature=template["temperature"],
            max_turns=template["max_turns"],
            created_at=datetime.now().isoformat(),
            created_by=created_by
        )
        
        self.agents[agent_id] = agent_config
        
        logger.info(f"🤖 Agent created from template: {name} ({template_name})")
        logger.info(f"   Agent ID: {agent_id}")
        logger.info(f"   Tools: {', '.join([t.tool_name for t in tools])}")
        
        return agent_config
    
    async def create_custom_agent(self,
                                 agent_id: str,
                                 name: str,
                                 description: str,
                                 personality: str,
                                 expertise_level: float,
                                 tools: List[str],
                                 system_prompt: str,
                                 response_style: str = "helpful",
                                 temperature: float = 0.5,
                                 max_turns: int = 10,
                                 created_by: str = "system") -> AgentConfiguration:
        """
        Create a fully custom agent with specific configuration
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            description: Agent description
            personality: Agent personality type
            expertise_level: Expertise level (0.0-1.0)
            tools: List of tool names to bind
            system_prompt: Custom system prompt
            response_style: How agent responds
            temperature: LLM temperature (0.0-1.0)
            max_turns: Max conversation turns
            created_by: Who created this agent
        
        Returns:
            AgentConfiguration for the created agent
        """
        tool_bindings = self._bind_tools(tools)
        
        agent_config = AgentConfiguration(
            agent_id=agent_id,
            agent_type=AgentType.CUSTOM.value,
            name=name,
            description=description,
            personality=personality,
            expertise_level=expertise_level,
            tools=tool_bindings,
            system_prompt=system_prompt,
            response_style=response_style,
            temperature=temperature,
            max_turns=max_turns,
            created_at=datetime.now().isoformat(),
            created_by=created_by
        )
        
        self.agents[agent_id] = agent_config
        
        logger.info(f"🤖 Custom agent created: {name}")
        logger.info(f"   Agent ID: {agent_id}")
        logger.info(f"   Personality: {personality}")
        logger.info(f"   Tools: {', '.join(tools)}")
        logger.info(f"   Expertise: {expertise_level:.2f}")
        
        return agent_config
    
    def _bind_tools(self, tool_names: List[str]) -> List[ToolBinding]:
        """Bind tools to agent"""
        bindings = []
        for tool_name in tool_names:
            if tool_name not in self.available_tools:
                logger.warning(f"⚠️  Tool '{tool_name}' not registered, skipping")
                continue
            
            tool_info = self.available_tools[tool_name]
            binding = ToolBinding(
                tool_name=tool_name,
                tool_func=tool_info["func"],
                description=tool_info["description"],
                required_params=tool_info["required_params"],
                enabled=True
            )
            bindings.append(binding)
        
        return bindings
    
    def _generate_system_prompt(self, name: str, description: str,
                               personality: str, response_style: str) -> str:
        """Generate system prompt based on agent configuration"""
        prompt = f"""You are {name}. {description}

Personality: {personality}
Response style: {response_style}

Guidelines:
- Be helpful and professional
- Stay focused on your area of expertise
- If unsure, ask clarifying questions
- Escalate complex issues when needed
- Maintain context of the conversation"""
        
        if personality == "friendly":
            prompt += "\n- Use a warm and approachable tone"
        elif personality == "technical":
            prompt += "\n- Provide detailed technical explanations"
        elif personality == "professional":
            prompt += "\n- Maintain a professional and formal tone"
        
        return prompt
    
    async def update_agent(self, agent_id: str, **updates) -> AgentConfiguration:
        """
        Update agent configuration
        
        Args:
            agent_id: Agent to update
            **updates: Fields to update (name, description, tools, etc.)
        
        Returns:
            Updated AgentConfiguration
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # Update allowed fields
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        
        logger.info(f"✅ Agent {agent_id} updated: {', '.join(updates.keys())}")
        return agent
    
    async def activate_agent(self, agent_id: str) -> AgentConfiguration:
        """Activate a dormant agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.agents[agent_id].is_active = True
        logger.info(f"✅ Agent {agent_id} activated")
        return self.agents[agent_id]
    
    async def deactivate_agent(self, agent_id: str) -> AgentConfiguration:
        """Deactivate an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.agents[agent_id].is_active = False
        logger.info(f"⏸️  Agent {agent_id} deactivated")
        return self.agents[agent_id]
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        del self.agents[agent_id]
        logger.info(f"🗑️  Agent {agent_id} deleted")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfiguration]:
        """Retrieve agent configuration"""
        return self.agents.get(agent_id)
    
    def list_agents(self, agent_type: Optional[str] = None) -> List[AgentConfiguration]:
        """List all agents or filter by type"""
        agents = list(self.agents.values())
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        return agents
    
    def list_templates(self) -> List[Dict]:
        """List available agent templates"""
        return [
            {
                "name": name,
                "description": template["description"],
                "type": template["type"].value,
                "tools": template["base_tools"]
            }
            for name, template in self.agent_templates.items()
        ]
    
    def list_available_tools(self) -> List[Dict]:
        """List all available tools for binding"""
        return [
            {
                "name": name,
                "description": info["description"],
                "required_params": info["required_params"]
            }
            for name, info in self.available_tools.items()
        ]
    
    async def start_agent_session(self, agent_id: str, 
                                 conversation_id: str) -> Dict:
        """
        Start a session with an agent
        
        Args:
            agent_id: Agent to use
            conversation_id: Conversation context
        
        Returns:
            Session info
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        if not agent.is_active:
            raise ValueError(f"Agent {agent_id} is not active")
        
        session_id = f"session_{agent_id}_{conversation_id}_{datetime.now().timestamp()}"
        
        session = {
            "session_id": session_id,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "conversation_id": conversation_id,
            "turns": 0,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.agent_sessions[session_id] = session
        
        logger.info(f"🟢 Agent session started: {session_id}")
        logger.info(f"   Agent: {agent.name}")
        logger.info(f"   Conversation: {conversation_id}")
        
        return session
    
    async def end_agent_session(self, session_id: str) -> Dict:
        """End an agent session"""
        if session_id not in self.agent_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.agent_sessions[session_id]
        session["status"] = "closed"
        session["closed_at"] = datetime.now().isoformat()
        
        logger.info(f"🔴 Agent session closed: {session_id}")
        logger.info(f"   Total turns: {session['turns']}")
        
        return session
    
    def get_factory_metrics(self) -> Dict:
        """Get metrics about agent factory"""
        active_agents = sum(1 for a in self.agents.values() if a.is_active)
        active_sessions = sum(1 for s in self.agent_sessions.values() 
                             if s["status"] == "active")
        
        agent_types = {}
        for agent in self.agents.values():
            agent_types[agent.agent_type] = agent_types.get(agent.agent_type, 0) + 1
        
        return {
            "total_agents_created": len(self.agents),
            "active_agents": active_agents,
            "inactive_agents": len(self.agents) - active_agents,
            "agents_by_type": agent_types,
            "active_sessions": active_sessions,
            "total_sessions": len(self.agent_sessions),
            "available_templates": len(self.agent_templates),
            "available_tools": len(self.available_tools)
        }


# Global instance
dynamic_agent_factory = DynamicAgentFactory()
