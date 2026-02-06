"""
PHASE 1.2: Conversation Memory Management
Handles conversation context storage, retrieval, and management
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history and context for a single conversation
    """
    
    def __init__(self, conversation_id: str, max_turns: int = 10):
        self.conversation_id = conversation_id
        self.turns: List[Dict[str, Any]] = []
        self.max_turns = max_turns
        self.metadata = {
            "customer_id": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "context": {},
            "agent_history": []
        }
        logger.info(f"ConversationMemory initialized for conversation: {conversation_id}")
    
    def add_turn(self, role: str, content: str, agent: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Add a conversation turn (user/assistant)
        
        Args:
            role: 'user' or 'assistant'
            content: The message content
            agent: Name of the agent that handled this turn (if assistant)
            metadata: Additional metadata for this turn
        """
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        if agent:
            turn["agent"] = agent
            # Track which agents have been used
            if agent not in self.metadata["agent_history"]:
                self.metadata["agent_history"].append(agent)
        
        if metadata:
            turn["metadata"] = metadata
        
        self.turns.append(turn)
        
        # Keep last N turns (user + assistant pairs)
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]
        
        logger.debug(f"Added {role} turn to conversation {self.conversation_id}")
    
    def get_context(self, last_n_turns: int = 5) -> str:
        """
        Get formatted context for LLM
        
        Args:
            last_n_turns: Number of recent conversation pairs to include
            
        Returns:
            Formatted conversation history string
        """
        context_turns = self.turns[-last_n_turns * 2:] if self.turns else []
        context_lines = []
        
        for turn in context_turns:
            role_label = "Customer" if turn["role"] == "user" else "Assistant"
            agent_info = f" ({turn.get('agent', 'unknown')})" if turn["role"] == "assistant" else ""
            context_lines.append(f"{role_label}{agent_info}: {turn['content']}")
        
        return "\n".join(context_lines) if context_lines else "No previous conversation"
    
    def get_turns(self) -> List[Dict[str, Any]]:
        """Get all conversation turns"""
        return self.turns.copy()
    
    def get_last_turn(self) -> Optional[Dict[str, Any]]:
        """Get the most recent turn"""
        return self.turns[-1] if self.turns else None
    
    def set_customer_id(self, customer_id: str):
        """Associate this conversation with a customer"""
        self.metadata["customer_id"] = customer_id
        logger.info(f"Conversation {self.conversation_id} linked to customer {customer_id}")
    
    def add_context(self, key: str, value: Any):
        """Add arbitrary context data"""
        self.metadata["context"][key] = value
    
    def get_context_value(self, key: str) -> Any:
        """Retrieve context value by key"""
        return self.metadata["context"].get(key)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            "conversation_id": self.conversation_id,
            "turn_count": len(self.turns),
            "agents_used": self.metadata["agent_history"],
            "customer_id": self.metadata["customer_id"],
            "created_at": self.metadata["created_at"],
            "last_activity": self.turns[-1]["timestamp"] if self.turns else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "turns": self.turns,
            "metadata": self.metadata,
            "max_turns": self.max_turns
        }
    
    def to_json(self) -> str:
        """Serialize conversation to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """Deserialize conversation from dictionary"""
        conversation = cls(
            conversation_id=data["conversation_id"],
            max_turns=data.get("max_turns", 10)
        )
        conversation.turns = data.get("turns", [])
        conversation.metadata = data.get("metadata", {})
        return conversation
    
    def save_to_file(self, filepath: str):
        """Save conversation to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
            logger.info(f"Conversation saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save conversation to file: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationMemory':
        """Load conversation from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Conversation loaded from {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load conversation from file: {e}")
            raise


class ConversationManager:
    """
    Manages multiple conversations in memory
    """
    
    def __init__(self, max_conversations: int = 100):
        self.conversations: Dict[str, ConversationMemory] = {}
        self.max_conversations = max_conversations
        logger.info("ConversationManager initialized")
    
    def get_or_create_conversation(self, conversation_id: str, max_turns: int = 10) -> ConversationMemory:
        """Get existing conversation or create a new one"""
        if conversation_id not in self.conversations:
            # Check if we need to evict old conversations
            if len(self.conversations) >= self.max_conversations:
                self._evict_oldest_conversation()
            
            self.conversations[conversation_id] = ConversationMemory(conversation_id, max_turns)
            logger.info(f"Created new conversation: {conversation_id}")
        
        return self.conversations[conversation_id]
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Get existing conversation or None"""
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False
    
    def _evict_oldest_conversation(self):
        """Remove the oldest conversation based on creation time"""
        if not self.conversations:
            return
        
        oldest_id = min(
            self.conversations.keys(),
            key=lambda k: self.conversations[k].metadata["created_at"]
        )
        logger.info(f"Evicting oldest conversation: {oldest_id}")
        del self.conversations[oldest_id]
    
    def get_all_conversations(self) -> List[str]:
        """Get list of all conversation IDs"""
        return list(self.conversations.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about managed conversations"""
        return {
            "total_conversations": len(self.conversations),
            "max_conversations": self.max_conversations,
            "conversation_ids": list(self.conversations.keys())
        }


# Global conversation manager instance
conversation_manager = ConversationManager()
