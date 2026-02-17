"""
Executive Agent for Agentic UX System
Coordinates multi-agent orchestration and high-level decision making.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    ADAPTING = "adapting"
    LEARNING = "learning"
    ERROR = "error"


class MessageType(Enum):
    """Message types for inter-agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: MessageType = MessageType.REQUEST
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: int = 5  # 1-10, higher is more urgent

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)


@dataclass
class UserContext:
    """User behavioral and cognitive context"""
    user_id: str
    cognitive_load: float = 0.0  # 0-100
    task_type: str = ""
    page_url: str = ""
    time_on_page: float = 0.0
    viewport_size: Tuple[int, int] = (1920, 1080)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    device_type: str = "desktop"  # desktop, mobile, tablet


class ExecutiveAgent:
    """
    High-level coordinator that manages agent communication and orchestration.
    Uses publish-subscribe pattern for efficient message passing.
    """

    def __init__(self, agent_id: str = "executive_agent"):
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[str, List[callable]] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, UserContext] = {}
        self.coordination_strategies = {
            "cognitive_load_reduction": self._handle_cognitive_load_reduction,
            "interface_optimization": self._handle_interface_optimization,
            "workflow_acceleration": self._handle_workflow_acceleration,
            "personalization": self._handle_personalization,
        }
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 10000

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        callback: Optional[callable] = None
    ) -> bool:
        """Register an agent in the system"""
        self.agent_registry[agent_id] = {
            "type": agent_type,
            "capabilities": capabilities,
            "status": "registered",
            "registered_at": datetime.utcnow().isoformat(),
            "callback": callback
        }
        logger.info(f"Registered agent: {agent_id} ({agent_type})")
        return True

    def subscribe(self, topic: str, callback: callable) -> str:
        """Subscribe to topic with callback"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.info(f"New subscriber for topic: {topic}")
        return topic

    async def publish(self, topic: str, message: AgentMessage) -> None:
        """Publish message to all subscribers of topic"""
        if topic in self.subscribers:
            tasks = [callback(message) for callback in self.subscribers[topic]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        self._store_message(message)

    async def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 5
    ) -> Optional[AgentMessage]:
        """Send direct message to agent"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            priority=priority
        )

        if recipient in self.agent_registry:
            callback = self.agent_registry[recipient].get("callback")
            if callback:
                try:
                    response = await callback(message) if asyncio.iscoroutinefunction(callback) else callback(message)
                    self._store_message(message)
                    return response
                except Exception as e:
                    logger.error(f"Error sending message to {recipient}: {e}")
                    return None
        return None

    async def orchestrate_adaptation(
        self,
        user_id: str,
        user_context: UserContext,
        trigger_event: str
    ) -> Dict[str, Any]:
        """
        Orchestrate multi-agent adaptation based on user context.
        This is the core coordination logic.
        """
        self.state = AgentState.ANALYZING
        logger.info(f"Orchestrating adaptation for user {user_id}: {trigger_event}")

        self.active_sessions[user_id] = user_context

        # Determine adaptation strategy based on cognitive load
        adaptation_strategy = self._determine_strategy(user_context)

        # Dispatch to appropriate coordination handler
        if adaptation_strategy in self.coordination_strategies:
            result = await self.coordination_strategies[adaptation_strategy](
                user_id, user_context
            )
        else:
            result = {"status": "no_adaptation_needed"}

        self.state = AgentState.IDLE
        return result

    def _determine_strategy(self, context: UserContext) -> str:
        """Determine which coordination strategy to use"""
        if context.cognitive_load > 75:
            return "cognitive_load_reduction"
        elif context.cognitive_load > 50:
            return "interface_optimization"
        elif context.task_type in ["multi_step", "complex"]:
            return "workflow_acceleration"
        else:
            return "personalization"

    async def _handle_cognitive_load_reduction(
        self,
        user_id: str,
        context: UserContext
    ) -> Dict[str, Any]:
        """Handle high cognitive load with load reduction strategies"""
        logger.info(f"Executing cognitive load reduction for {user_id}")

        # Send request to behavior analysis agent
        behavior_payload = {
            "user_id": user_id,
            "context": {
                "cognitive_load": context.cognitive_load,
                "task_type": context.task_type,
                "page_url": context.page_url
            }
        }

        # Analyze current behavior patterns
        behavior_analysis = await self.send_message(
            "behavior_analysis_agent",
            MessageType.REQUEST,
            behavior_payload,
            priority=9
        )

        # Request interface adaptations
        interface_payload = {
            "user_id": user_id,
            "adaptations": ["simplify_layout", "reduce_options", "highlight_primary_action"],
            "cognitive_load": context.cognitive_load
        }

        interface_response = await self.send_message(
            "interface_agent",
            MessageType.REQUEST,
            interface_payload,
            priority=9
        )

        return {
            "strategy": "cognitive_load_reduction",
            "behavior_analysis": behavior_analysis,
            "interface_changes": interface_response,
            "estimated_load_reduction": 20  # percentage
        }

    async def _handle_interface_optimization(
        self,
        user_id: str,
        context: UserContext
    ) -> Dict[str, Any]:
        """Handle moderate cognitive load with interface optimization"""
        logger.info(f"Executing interface optimization for {user_id}")

        optimizations = [
            "reorganize_elements",
            "improve_visual_hierarchy",
            "optimize_spacing"
        ]

        payload = {
            "user_id": user_id,
            "optimizations": optimizations,
            "context": {
                "task_type": context.task_type,
                "device_type": context.device_type
            }
        }

        response = await self.send_message(
            "interface_agent",
            MessageType.REQUEST,
            payload,
            priority=7
        )

        return {
            "strategy": "interface_optimization",
            "optimizations_applied": optimizations,
            "response": response
        }

    async def _handle_workflow_acceleration(
        self,
        user_id: str,
        context: UserContext
    ) -> Dict[str, Any]:
        """Handle complex workflows with acceleration strategies"""
        logger.info(f"Executing workflow acceleration for {user_id}")

        payload = {
            "user_id": user_id,
            "task_type": context.task_type,
            "page_url": context.page_url,
            "workflow_optimizations": [
                "auto_fill_forms",
                "predictive_suggestions",
                "streamlined_navigation"
            ]
        }

        response = await self.send_message(
            "workflow_agent",
            MessageType.REQUEST,
            payload,
            priority=8
        )

        return {
            "strategy": "workflow_acceleration",
            "optimizations": payload["workflow_optimizations"],
            "response": response
        }

    async def _handle_personalization(
        self,
        user_id: str,
        context: UserContext
    ) -> Dict[str, Any]:
        """Handle general personalization"""
        logger.info(f"Executing personalization for {user_id}")

        payload = {
            "user_id": user_id,
            "context": {
                "task_type": context.task_type,
                "device_type": context.device_type,
                "interaction_history": context.interaction_history[-10:]
            }
        }

        # Request learning from learning module
        response = await self.send_message(
            "learning_module",
            MessageType.REQUEST,
            payload,
            priority=6
        )

        return {
            "strategy": "personalization",
            "learning_response": response
        }

    def _store_message(self, message: AgentMessage) -> None:
        """Store message in history with size limit"""
        self.message_history.append(message)
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]

    def get_session_context(self, user_id: str) -> Optional[UserContext]:
        """Get current session context for user"""
        return self.active_sessions.get(user_id)

    def update_context(self, user_id: str, **kwargs) -> None:
        """Update user context with new data"""
        if user_id in self.active_sessions:
            context = self.active_sessions[user_id]
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)

    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of registered agents"""
        if agent_id:
            return self.agent_registry.get(agent_id, {})
        return self.agent_registry

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics and statistics"""
        return {
            "agent_count": len(self.agent_registry),
            "active_sessions": len(self.active_sessions),
            "message_count": len(self.message_history),
            "state": self.state.value,
            "registered_agents": list(self.agent_registry.keys()),
            "subscribers": {k: len(v) for k, v in self.subscribers.items()}
        }

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down Executive Agent")
        self.state = AgentState.IDLE
        self.active_sessions.clear()
        logger.info("Executive Agent shutdown complete")


# Example usage and testing
if __name__ == "__main__":

    async def test_executive_agent():
        """Test executive agent functionality"""
        agent = ExecutiveAgent()

        # Register mock agents
        async def mock_behavior_callback(msg: AgentMessage):
            return {"analysis": "normal", "patterns": []}

        async def mock_interface_callback(msg: AgentMessage):
            return {"changes_applied": True, "count": 3}

        agent.register_agent(
            "behavior_analysis_agent",
            "behavior_analyzer",
            ["analyze_patterns", "detect_anomalies"],
            mock_behavior_callback
        )

        agent.register_agent(
            "interface_agent",
            "interface_adapter",
            ["adapt_layout", "optimize_ui"],
            mock_interface_callback
        )

        # Create test context
        user_id = "test_user_001"
        context = UserContext(
            user_id=user_id,
            cognitive_load=80.0,
            task_type="form_completion",
            page_url="https://example.com/form",
            device_type="desktop"
        )

        # Test orchestration
        result = await agent.orchestrate_adaptation(
            user_id,
            context,
            "high_cognitive_load"
        )

        print(f"Orchestration result: {json.dumps(result, indent=2, default=str)}")
        print(f"Metrics: {agent.get_metrics()}")

    asyncio.run(test_executive_agent())
