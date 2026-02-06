"""
PHASE 2.1: ReAct Pattern Implementation
Reasoning + Acting + Observing framework for advanced agent capabilities
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions an agent can take"""
    REASON = "reason"
    ACT = "act"
    OBSERVE = "observe"
    CONCLUDE = "conclude"


class ThoughtAction:
    """Represents a single thought or action in the ReAct process"""
    
    def __init__(self, action_type: ActionType, content: str, metadata: Optional[Dict] = None):
        self.action_type = action_type
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ReActAgent:
    """
    Agent implementing the ReAct pattern:
    - Reason: Think about the problem
    - Act: Take action (call a tool)
    - Observe: See the result
    - Repeat until problem is solved
    """
    
    def __init__(self, name: str, llm_utils, tool_registry, max_iterations: int = 5):
        self.name = name
        self.llm_utils = llm_utils
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.thought_action_history: List[ThoughtAction] = []
        logger.info(f"ReActAgent '{name}' initialized")
    
    async def reason(self, query: str, observations: List[str] = None) -> str:
        """
        Step 1: Reason about the problem
        
        Args:
            query: The user's question
            observations: Previous observations from tool calls
            
        Returns:
            Reasoning about what to do next
        """
        observation_context = ""
        if observations:
            observation_context = "\n\nPrevious observations:\n" + "\n".join(observations)
        
        reasoning_prompt = f"""
Given the user query and previous observations, reason about what action to take next.

User Query: {query}
{observation_context}

Think step-by-step about:
1. What information do we have?
2. What information do we still need?
3. What tool should we use or what conclusion can we draw?

Provide clear reasoning (2-3 sentences).
"""
        
        reasoning_system = """You are a strategic reasoning assistant. Analyze problems carefully and recommend the best course of action."""
        
        reasoning = self.llm_utils.generate_response(reasoning_prompt, reasoning_system)
        thought = ThoughtAction(ActionType.REASON, reasoning)
        self.thought_action_history.append(thought)
        
        logger.info(f"[{self.name}] Reasoning: {reasoning[:100]}...")
        return reasoning
    
    async def select_action(self, reasoning: str, available_tools: List[str] = None) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Step 2: Decide which tool to use based on reasoning
        
        Args:
            reasoning: The reasoning from the previous step
            available_tools: List of available tool names
            
        Returns:
            Tuple of (tool_name, tool_params) or (None, None) if no tool needed
        """
        if not available_tools:
            available_tools = [t["name"] for t in self.tool_registry.list_tools()]
        
        tools_description = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tool_registry.list_tools()]
        )
        
        action_prompt = f"""
Based on this reasoning:
{reasoning}

Available tools:
{tools_description}

Respond with JSON in this exact format:
{{
    "should_use_tool": true/false,
    "tool_name": "tool_name_or_null",
    "tool_params": {{"param": "value"}} or null,
    "reasoning": "why this tool or why no tool"
}}

Only use tools if they directly help answer the user's question.
"""
        
        action_system = """You are an action selector. Determine if a tool should be called and which one."""
        
        action_json = self.llm_utils.generate_response(action_prompt, action_system)
        
        try:
            # Try to extract JSON from response
            if "{" in action_json and "}" in action_json:
                start = action_json.find("{")
                end = action_json.rfind("}") + 1
                action_json = action_json[start:end]
            
            action_data = json.loads(action_json)
            
            if action_data.get("should_use_tool"):
                tool_name = action_data.get("tool_name")
                tool_params = action_data.get("tool_params", {})
                logger.info(f"[{self.name}] Selected tool: {tool_name}")
                return tool_name, tool_params
            else:
                logger.info(f"[{self.name}] No tool needed")
                return None, None
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse action JSON: {e}")
            return None, None
    
    async def act(self, tool_name: str, tool_params: Dict) -> Any:
        """
        Step 3: Execute the selected tool
        
        Args:
            tool_name: Name of the tool to execute
            tool_params: Parameters for the tool
            
        Returns:
            The result from the tool
        """
        logger.info(f"[{self.name}] Executing tool: {tool_name} with params: {tool_params}")
        
        try:
            result = self.tool_registry.execute(tool_name, **tool_params)
            
            action = ThoughtAction(
                ActionType.ACT,
                f"Called tool: {tool_name}",
                {"tool": tool_name, "params": tool_params, "result": result}
            )
            self.thought_action_history.append(action)
            
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def observe(self, action_result: Any, query: str) -> str:
        """
        Step 4: Observe and interpret the result
        
        Args:
            action_result: The result from tool execution
            query: The original user query
            
        Returns:
            Observation/interpretation of the result
        """
        observation_prompt = f"""
User Query: {query}

Tool Result:
{json.dumps(action_result, indent=2) if isinstance(action_result, dict) else str(action_result)}

Interpret this result. What does it tell us about answering the user's query?
If successful, what information did we gain? If failed, what should we try next?
"""
        
        observation_system = """You are an observer. Interpret tool results and determine if they help answer the user's question."""
        
        observation = self.llm_utils.generate_response(observation_prompt, observation_system)
        
        action = ThoughtAction(ActionType.OBSERVE, observation, {"action_result": action_result})
        self.thought_action_history.append(action)
        
        logger.info(f"[{self.name}] Observation: {observation[:100]}...")
        return observation
    
    async def conclude(self, query: str, observations: List[str]) -> str:
        """
        Step 5: Generate final answer based on all observations
        
        Args:
            query: The original user query
            observations: All observations gathered
            
        Returns:
            Final answer to the user
        """
        observations_text = "\n".join([f"- {obs}" for obs in observations])
        
        conclusion_prompt = f"""
User Query: {query}

All Observations:
{observations_text}

Based on the observations above, provide a comprehensive answer to the user's query.
Be specific, helpful, and reference the information from observations.
"""
        
        conclusion_system = """You are a conclusion generator. Synthesize all observations into a clear, helpful answer."""
        
        conclusion = self.llm_utils.generate_response(conclusion_prompt, conclusion_system)
        
        action = ThoughtAction(ActionType.CONCLUDE, conclusion)
        self.thought_action_history.append(action)
        
        logger.info(f"[{self.name}] Conclusion: {conclusion[:100]}...")
        return conclusion
    
    async def process_query_with_react(self, query: str) -> Dict[str, Any]:
        """
        Main entry point: Process query using full ReAct pattern
        
        Returns:
            Dict with final_answer, reasoning_process, and status
        """
        logger.info(f"[{self.name}] Starting ReAct process for query: {query}")
        
        self.thought_action_history = []  # Reset history
        observations = []
        iteration = 0
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"[{self.name}] ReAct Iteration {iteration}/{self.max_iterations}")
                
                # Step 1: Reason
                reasoning = await self.reason(query, observations)
                
                # Step 2: Select action
                tool_name, tool_params = await self.select_action(reasoning)
                
                # If no tool selected, we're ready to conclude
                if tool_name is None:
                    logger.info(f"[{self.name}] Ready to conclude")
                    break
                
                # Step 3: Act
                action_result = await self.act(tool_name, tool_params)
                
                # Step 4: Observe
                observation = await self.observe(action_result, query)
                observations.append(observation)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            # Step 5: Conclude
            if observations or iteration >= self.max_iterations:
                final_answer = await self.conclude(query, observations)
            else:
                final_answer = "Unable to generate answer after reasoning."
            
            return {
                "success": True,
                "final_answer": final_answer,
                "reasoning_process": [action.to_dict() for action in self.thought_action_history],
                "iterations": iteration,
                "observations_count": len(observations),
                "agent": self.name
            }
        
        except Exception as e:
            logger.error(f"ReAct process failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "reasoning_process": [action.to_dict() for action in self.thought_action_history],
                "agent": self.name
            }
    
    def get_reasoning_trace(self) -> List[Dict]:
        """Get the full reasoning trace for debugging"""
        return [action.to_dict() for action in self.thought_action_history]
    
    def reset_history(self):
        """Clear thought-action history"""
        self.thought_action_history = []
