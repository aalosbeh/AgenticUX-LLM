"""
Multi-agent Coordination Protocols
Efficient message passing and agent orchestration.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Multi-agent communication protocols"""
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    BROADCAST = "broadcast"
    PIPELINE = "pipeline"


@dataclass
class ProtocolMessage:
    """Standard message for agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_ids: List[str] = field(default_factory=list)
    message_type: str = "standard"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: int = 5  # 1-10
    requires_response: bool = False
    response_timeout: float = 5.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolResponse:
    """Response from message processing"""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_message_id: str = ""
    sender_id: str = ""
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentRegistry:
    """Registry of all agents in the system"""

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.status: Dict[str, str] = {}

    def register(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        handler: Optional[Callable] = None
    ) -> None:
        """Register an agent"""
        self.agents[agent_id] = {
            "type": agent_type,
            "handler": handler,
            "registered_at": datetime.utcnow().isoformat(),
        }
        self.capabilities[agent_id] = set(capabilities)
        self.status[agent_id] = "idle"
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        return self.agents.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a capability"""
        return [agent_id for agent_id, caps in self.capabilities.items() if capability in caps]

    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Update agent status"""
        if agent_id in self.status:
            self.status[agent_id] = status

    def get_agent_status(self, agent_id: str) -> str:
        """Get agent status"""
        return self.status.get(agent_id, "unknown")


class MessageQueue:
    """Priority message queue for agents"""

    def __init__(self, max_size: int = 10000):
        self.queue: asyncio.PriorityQueue = None
        self.max_size = max_size
        self.message_count = 0
        self.dispatched_count = 0

    async def initialize(self) -> None:
        """Initialize async queue"""
        self.queue = asyncio.PriorityQueue(maxsize=self.max_size)

    async def put(self, message: ProtocolMessage) -> None:
        """Add message to queue"""
        if self.queue is None:
            await self.initialize()

        # Priority is negative so higher priority goes first
        priority = -message.priority
        await self.queue.put((priority, message.message_id, message))
        self.message_count += 1

    async def get(self, timeout: float = 1.0) -> Optional[ProtocolMessage]:
        """Get message from queue"""
        if self.queue is None:
            await self.initialize()

        try:
            _, _, message = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            self.dispatched_count += 1
            return message
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        """Get queue size"""
        if self.queue is None:
            return 0
        return self.queue.qsize()


class CoordinationProtocol:
    """Base coordination protocol"""

    def __init__(self, protocol_type: ProtocolType):
        self.protocol_type = protocol_type
        self.message_queue = MessageQueue()
        self.pending_responses: Dict[str, asyncio.Future] = {}

    async def initialize(self) -> None:
        """Initialize protocol"""
        await self.message_queue.initialize()

    async def send_message(self, message: ProtocolMessage) -> Optional[ProtocolResponse]:
        """Send message and optionally wait for response"""
        await self.message_queue.put(message)

        if message.requires_response:
            future = asyncio.Future()
            self.pending_responses[message.message_id] = future

            try:
                response = await asyncio.wait_for(future, timeout=message.response_timeout)
                del self.pending_responses[message.message_id]
                return response
            except asyncio.TimeoutError:
                logger.error(f"No response to message {message.message_id} after {message.response_timeout}s")
                del self.pending_responses[message.message_id]
                return None

        return None

    async def receive_message(self, timeout: float = 1.0) -> Optional[ProtocolMessage]:
        """Receive message from queue"""
        return await self.message_queue.get(timeout=timeout)

    def submit_response(self, response: ProtocolResponse) -> None:
        """Submit response to pending request"""
        if response.request_message_id in self.pending_responses:
            future = self.pending_responses[response.request_message_id]
            if not future.done():
                future.set_result(response)


class RequestResponseProtocol(CoordinationProtocol):
    """Request-response coordination protocol"""

    def __init__(self):
        super().__init__(ProtocolType.REQUEST_RESPONSE)
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register handler for message type"""
        self.handlers[message_type] = handler

    async def dispatch(self) -> None:
        """Dispatch messages to handlers"""
        while True:
            message = await self.receive_message()
            if message is None:
                continue

            if message.message_type in self.handlers:
                handler = self.handlers[message.message_type]
                try:
                    result = await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)

                    response = ProtocolResponse(
                        request_message_id=message.message_id,
                        sender_id=message.sender_id,
                        success=True,
                        data=result or {}
                    )
                except Exception as e:
                    logger.error(f"Handler error: {e}")
                    response = ProtocolResponse(
                        request_message_id=message.message_id,
                        sender_id=message.sender_id,
                        success=False,
                        error=str(e)
                    )

                self.submit_response(response)


class PublishSubscribeProtocol(CoordinationProtocol):
    """Publish-subscribe coordination protocol"""

    def __init__(self):
        super().__init__(ProtocolType.PUBLISH_SUBSCRIBE)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, subscriber: Callable) -> str:
        """Subscribe to topic"""
        self.subscribers[topic].append(subscriber)
        return f"subscription_{topic}_{len(self.subscribers[topic])}"

    async def publish(self, topic: str, message: ProtocolMessage) -> None:
        """Publish message to all subscribers"""
        message.metadata["topic"] = topic

        for subscriber in self.subscribers[topic]:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(message)
                else:
                    subscriber(message)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")


class BroadcastProtocol(CoordinationProtocol):
    """Broadcast coordination protocol"""

    def __init__(self, registry: AgentRegistry):
        super().__init__(ProtocolType.BROADCAST)
        self.registry = registry

    async def broadcast(self, message: ProtocolMessage) -> List[Optional[ProtocolResponse]]:
        """Broadcast message to all agents"""
        responses = []
        agent_ids = list(self.registry.agents.keys())
        message.receiver_ids = agent_ids

        await self.send_message(message)

        # Collect responses from all agents
        for _ in range(len(agent_ids)):
            # Wait for responses with timeout
            await asyncio.sleep(0.1)

        return responses


class PipelineProtocol(CoordinationProtocol):
    """Pipeline coordination protocol - sequential agent processing"""

    def __init__(self, registry: AgentRegistry):
        super().__init__(ProtocolType.PIPELINE)
        self.registry = registry

    async def execute_pipeline(
        self,
        agent_sequence: List[str],
        initial_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute pipeline of agents in sequence"""
        current_data = initial_data

        for agent_id in agent_sequence:
            agent = self.registry.get_agent(agent_id)
            if not agent or agent["handler"] is None:
                logger.error(f"Agent {agent_id} not found or has no handler")
                return None

            # Create message
            message = ProtocolMessage(
                sender_id="pipeline",
                receiver_ids=[agent_id],
                message_type="pipeline_data",
                payload=current_data,
                requires_response=True,
                response_timeout=10.0
            )

            # Send to agent
            response = await self.send_message(message)

            if response and response.success:
                current_data = response.data
            else:
                logger.error(f"Pipeline failed at agent {agent_id}")
                return None

        return current_data


class AgentCoordinator:
    """
    Master coordinator for multi-agent system.
    Manages protocols, message routing, and agent lifecycle.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self.rr_protocol = RequestResponseProtocol()
        self.ps_protocol = PublishSubscribeProtocol()
        self.broadcast_protocol = BroadcastProtocol(self.registry)
        self.pipeline_protocol = PipelineProtocol(self.registry)
        self.is_running = False

    async def initialize(self) -> None:
        """Initialize coordinator"""
        await self.rr_protocol.initialize()
        await self.ps_protocol.initialize()
        await self.broadcast_protocol.initialize()
        await self.pipeline_protocol.initialize()
        logger.info("Agent coordinator initialized")

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        handler: Optional[Callable] = None
    ) -> None:
        """Register agent in system"""
        self.registry.register(agent_id, agent_type, capabilities, handler)

    async def send_request(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        payload: Dict[str, Any],
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """Send request-response message"""
        message = ProtocolMessage(
            sender_id=sender_id,
            receiver_ids=[receiver_id],
            message_type=message_type,
            payload=payload,
            requires_response=True,
            response_timeout=timeout
        )

        response = await self.rr_protocol.send_message(message)
        return response.data if response and response.success else None

    async def publish_event(
        self,
        sender_id: str,
        topic: str,
        data: Dict[str, Any]
    ) -> None:
        """Publish event to subscribers"""
        message = ProtocolMessage(
            sender_id=sender_id,
            message_type="event",
            payload=data
        )

        await self.ps_protocol.publish(topic, message)

    def subscribe_to_topic(self, topic: str, callback: Callable) -> str:
        """Subscribe to topic"""
        return self.ps_protocol.subscribe(topic, callback)

    async def execute_pipeline(
        self,
        agent_sequence: List[str],
        initial_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute agent pipeline"""
        return await self.pipeline_protocol.execute_pipeline(agent_sequence, initial_data)

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": agent_id,
            "status": self.registry.get_agent_status(agent_id),
            "agent_info": self.registry.get_agent(agent_id)
        }

    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "is_running": self.is_running,
            "agents_registered": len(self.registry.agents),
            "message_queue_size": self.rr_protocol.message_queue.size(),
            "pending_responses": len(self.rr_protocol.pending_responses),
            "subscribers": sum(len(subs) for subs in self.ps_protocol.subscribers.values()),
            "total_messages": self.rr_protocol.message_queue.message_count,
            "dispatched_messages": self.rr_protocol.message_queue.dispatched_count
        }

    async def shutdown(self) -> None:
        """Shutdown coordinator"""
        logger.info("Shutting down agent coordinator")
        self.is_running = False


if __name__ == "__main__":

    async def test_coordinator():
        coordinator = AgentCoordinator()
        await coordinator.initialize()

        # Register mock agents
        async def agent_handler(message: ProtocolMessage) -> Dict[str, Any]:
            return {"processed": True, "input": message.payload}

        coordinator.register_agent(
            "agent_1",
            "analyzer",
            ["analyze", "report"],
            agent_handler
        )

        coordinator.register_agent(
            "agent_2",
            "transformer",
            ["transform", "optimize"],
            agent_handler
        )

        # Test request-response
        print("Testing request-response protocol...")
        result = await coordinator.send_request(
            sender_id="test",
            receiver_id="agent_1",
            message_type="test",
            payload={"data": "test"}
        )
        print(f"Result: {result}")

        # Test publish-subscribe
        print("\nTesting publish-subscribe protocol...")

        def subscriber(msg: ProtocolMessage):
            print(f"  Subscriber received: {msg.payload}")

        coordinator.subscribe_to_topic("events", subscriber)
        await coordinator.publish_event("test", "events", {"event": "test"})

        # Get stats
        stats = coordinator.get_coordinator_stats()
        print(f"\nCoordinator stats:")
        print(f"  Agents: {stats['agents_registered']}")
        print(f"  Messages: {stats['total_messages']}")

    asyncio.run(test_coordinator())
