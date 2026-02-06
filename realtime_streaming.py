"""
PHASE 3.4: Real-time Streaming Support
=======================================
Server-Sent Events (SSE) and WebSocket support for progressive
response generation and improved user experience.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events"""
    THINKING = "thinking"  # Agent is thinking
    ACTION = "action"  # Agent is taking an action
    RESULT = "result"  # Action result
    RESPONSE = "response"  # Final response
    ERROR = "error"  # Error occurred
    METRICS = "metrics"  # Performance metrics
    STATUS = "status"  # Status update
    COMPLETE = "complete"  # Stream complete


@dataclass
class StreamEvent:
    """Single event in a stream"""
    event_type: str
    content: Any
    timestamp: str
    sequence: int
    stream_id: str
    metadata: Optional[Dict] = None
    
    def to_json(self):
        """Serialize to JSON for streaming"""
        return {
            "event": self.event_type,
            "data": self.content,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "stream_id": self.stream_id,
            "metadata": self.metadata or {}
        }


class RealtimeStreamingEngine:
    """
    Handles real-time streaming of agent responses via
    Server-Sent Events (SSE) and WebSocket.
    """
    
    def __init__(self):
        self.active_streams: Dict[str, Dict] = {}
        self.stream_subscribers: Dict[str, List[Callable]] = {}
        self.stream_history: Dict[str, List[StreamEvent]] = {}
        logger.info("✅ Real-time Streaming Engine initialized")
    
    async def create_stream(self, query: str, agent_id: str,
                          conversation_id: str) -> str:
        """
        Create a new streaming session
        
        Args:
            query: User query to process
            agent_id: Agent handling the query
            conversation_id: Conversation context
        
        Returns:
            Stream ID for reference
        """
        stream_id = str(uuid.uuid4())
        
        self.active_streams[stream_id] = {
            "id": stream_id,
            "query": query,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "event_count": 0,
            "start_time": datetime.now()
        }
        
        self.stream_subscribers[stream_id] = []
        self.stream_history[stream_id] = []
        
        logger.info(f"🟢 Stream created: {stream_id}")
        logger.info(f"   Query: {query[:50]}...")
        logger.info(f"   Agent: {agent_id}")
        
        return stream_id
    
    async def emit_event(self, stream_id: str, event_type: StreamEventType,
                        content: Any, metadata: Optional[Dict] = None) -> StreamEvent:
        """
        Emit an event to a stream
        
        Args:
            stream_id: Target stream ID
            event_type: Type of event
            content: Event content/data
            metadata: Additional metadata
        
        Returns:
            The emitted StreamEvent
        """
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self.active_streams[stream_id]
        sequence = stream["event_count"]
        
        event = StreamEvent(
            event_type=event_type.value,
            content=content,
            timestamp=datetime.now().isoformat(),
            sequence=sequence,
            stream_id=stream_id,
            metadata=metadata
        )
        
        stream["event_count"] += 1
        self.stream_history[stream_id].append(event)
        
        # Notify subscribers
        await self._notify_subscribers(stream_id, event)
        
        logger.debug(f"📤 Event {sequence}: {event_type.value}")
        
        return event
    
    async def emit_thinking(self, stream_id: str, thought: str,
                           reasoning: Optional[str] = None) -> StreamEvent:
        """
        Emit that agent is thinking/reasoning
        
        Args:
            stream_id: Stream ID
            thought: Current thought
            reasoning: Detailed reasoning (optional)
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.THINKING,
            {
                "thought": thought,
                "reasoning": reasoning
            }
        )
    
    async def emit_action(self, stream_id: str, action_type: str,
                         action_data: Dict) -> StreamEvent:
        """
        Emit that agent is taking an action
        
        Args:
            stream_id: Stream ID
            action_type: Type of action (call_tool, query_db, etc.)
            action_data: Action parameters
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.ACTION,
            {
                "action_type": action_type,
                "action_data": action_data
            }
        )
    
    async def emit_result(self, stream_id: str, action_type: str,
                         result: Any, success: bool = True) -> StreamEvent:
        """
        Emit the result of an action
        
        Args:
            stream_id: Stream ID
            action_type: Type of action
            result: Result data
            success: Whether action succeeded
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.RESULT,
            {
                "action_type": action_type,
                "result": result,
                "success": success
            }
        )
    
    async def emit_response_chunk(self, stream_id: str, chunk: str,
                                 is_final: bool = False) -> StreamEvent:
        """
        Emit a chunk of the response (progressive generation)
        
        Args:
            stream_id: Stream ID
            chunk: Response chunk
            is_final: Whether this is the final chunk
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.RESPONSE,
            {
                "chunk": chunk,
                "is_final": is_final
            }
        )
    
    async def emit_error(self, stream_id: str, error_message: str,
                        error_type: str = "unknown") -> StreamEvent:
        """
        Emit an error event
        
        Args:
            stream_id: Stream ID
            error_message: Error description
            error_type: Type of error
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.ERROR,
            {
                "error_message": error_message,
                "error_type": error_type
            }
        )
    
    async def emit_metrics(self, stream_id: str, metrics: Dict) -> StreamEvent:
        """
        Emit performance metrics
        
        Args:
            stream_id: Stream ID
            metrics: Performance metrics
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.METRICS,
            metrics
        )
    
    async def emit_status(self, stream_id: str, status: str,
                         details: Optional[Dict] = None) -> StreamEvent:
        """
        Emit status update
        
        Args:
            stream_id: Stream ID
            status: Current status
            details: Status details
        """
        return await self.emit_event(
            stream_id,
            StreamEventType.STATUS,
            {
                "status": status,
                "details": details or {}
            }
        )
    
    async def complete_stream(self, stream_id: str,
                             final_response: str) -> StreamEvent:
        """
        Mark stream as complete
        
        Args:
            stream_id: Stream ID
            final_response: Final response content
        
        Returns:
            Completion event
        """
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self.active_streams[stream_id]
        duration = (datetime.now() - stream["start_time"]).total_seconds()
        
        event = await self.emit_event(
            stream_id,
            StreamEventType.COMPLETE,
            {
                "final_response": final_response,
                "total_events": stream["event_count"],
                "duration_seconds": duration
            }
        )
        
        stream["status"] = "completed"
        stream["duration"] = duration
        
        logger.info(f"🔴 Stream completed: {stream_id}")
        logger.info(f"   Events: {stream['event_count']}")
        logger.info(f"   Duration: {duration:.2f}s")
        
        return event
    
    async def subscribe(self, stream_id: str, callback: Callable):
        """
        Subscribe to stream events
        
        Args:
            stream_id: Stream ID to subscribe to
            callback: Async function to call on each event
        """
        if stream_id not in self.stream_subscribers:
            self.stream_subscribers[stream_id] = []
        
        self.stream_subscribers[stream_id].append(callback)
        logger.debug(f"📌 Subscriber added to stream {stream_id}")
    
    async def _notify_subscribers(self, stream_id: str, event: StreamEvent):
        """Notify all subscribers of an event"""
        if stream_id in self.stream_subscribers:
            for callback in self.stream_subscribers[stream_id]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {str(e)}")
    
    async def get_stream_generator(self, stream_id: str) -> AsyncGenerator[str, None]:
        """
        Get an async generator for SSE format streaming
        
        Args:
            stream_id: Stream ID
        
        Yields:
            SSE formatted event strings
        """
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        # Send all historical events
        for event in self.stream_history[stream_id]:
            yield self._format_sse(event)
        
        # Subscribe to new events
        last_sent = len(self.stream_history[stream_id])
        
        while self.active_streams[stream_id]["status"] == "active":
            if len(self.stream_history[stream_id]) > last_sent:
                event = self.stream_history[stream_id][last_sent]
                yield self._format_sse(event)
                last_sent += 1
            
            await asyncio.sleep(0.1)
    
    def _format_sse(self, event: StreamEvent) -> str:
        """Format event as Server-Sent Event"""
        data = json.dumps(event.to_json())
        return f"data: {data}\n\n"
    
    def get_stream_events(self, stream_id: str) -> List[StreamEvent]:
        """Get all events for a stream"""
        return self.stream_history.get(stream_id, [])
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict]:
        """Get information about a stream"""
        return self.active_streams.get(stream_id)
    
    async def cleanup_stream(self, stream_id: str):
        """Clean up a completed stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        if stream_id in self.stream_subscribers:
            del self.stream_subscribers[stream_id]
        
        logger.debug(f"🗑️  Stream {stream_id} cleaned up")
    
    def get_streaming_metrics(self) -> Dict:
        """Get metrics about streaming usage"""
        total_streams = len(self.stream_history)
        active_streams = len([s for s in self.active_streams.values() 
                            if s["status"] == "active"])
        
        total_events = sum(len(events) for events in self.stream_history.values())
        
        return {
            "total_streams": total_streams,
            "active_streams": active_streams,
            "completed_streams": total_streams - active_streams,
            "total_events_emitted": total_events,
            "avg_events_per_stream": total_events / total_streams if total_streams > 0 else 0
        }


class WebSocketStreamManager:
    """
    Manage WebSocket connections for bidirectional streaming
    """
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}  # stream_id -> connection
        self.connection_metadata: Dict[str, Dict] = {}
        logger.info("✅ WebSocket Stream Manager initialized")
    
    async def register_connection(self, stream_id: str, websocket: Any) -> bool:
        """Register a WebSocket connection"""
        self.connections[stream_id] = websocket
        self.connection_metadata[stream_id] = {
            "connected_at": datetime.now().isoformat(),
            "messages_sent": 0,
            "messages_received": 0
        }
        logger.info(f"🔗 WebSocket connected: {stream_id}")
        return True
    
    async def send_message(self, stream_id: str, message: Dict):
        """Send a message through WebSocket"""
        if stream_id not in self.connections:
            raise ValueError(f"Connection {stream_id} not found")
        
        websocket = self.connections[stream_id]
        await websocket.send_json(message)
        self.connection_metadata[stream_id]["messages_sent"] += 1
    
    async def receive_message(self, stream_id: str) -> Dict:
        """Receive a message from WebSocket"""
        if stream_id not in self.connections:
            raise ValueError(f"Connection {stream_id} not found")
        
        websocket = self.connections[stream_id]
        message = await websocket.receive_json()
        self.connection_metadata[stream_id]["messages_received"] += 1
        return message
    
    async def close_connection(self, stream_id: str, code: int = 1000):
        """Close a WebSocket connection"""
        if stream_id in self.connections:
            await self.connections[stream_id].close(code=code)
            del self.connections[stream_id]
            logger.info(f"❌ WebSocket closed: {stream_id}")
    
    def get_active_connections(self) -> int:
        """Get number of active WebSocket connections"""
        return len(self.connections)


# Global instances
realtime_streaming_engine = RealtimeStreamingEngine()
websocket_stream_manager = WebSocketStreamManager()
