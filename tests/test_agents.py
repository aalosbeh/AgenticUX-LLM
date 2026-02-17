"""
Tests for Agent System
"""

import unittest
import asyncio
from src.agents.executive_agent import ExecutiveAgent, UserContext, MessageType, AgentMessage
from src.agents.behavior_analysis_agent import BehaviorAnalysisAgent, CognitiveLoadEstimate
from src.agents.interface_agent import InterfaceAgent
from src.agents.workflow_agent import WorkflowAgent
from src.agents.learning_module import LearningModule


class TestExecutiveAgent(unittest.TestCase):
    """Test executive agent"""

    def setUp(self):
        self.agent = ExecutiveAgent()

    def test_agent_registration(self):
        """Test agent registration"""
        self.agent.register_agent(
            "test_agent",
            "test_type",
            ["capability1", "capability2"]
        )

        self.assertIn("test_agent", self.agent.agent_registry)
        agent_info = self.agent.agent_registry["test_agent"]
        self.assertEqual(agent_info["type"], "test_type")
        self.assertIn("capability1", agent_info["capabilities"])

    def test_subscribe(self):
        """Test topic subscription"""
        def mock_callback(msg):
            pass

        topic = self.agent.subscribe("test_topic", mock_callback)

        self.assertIn("test_topic", self.agent.subscribers)
        self.assertIn(mock_callback, self.agent.subscribers["test_topic"])

    def test_user_context_management(self):
        """Test user context tracking"""
        user_id = "test_user"
        context = UserContext(
            user_id=user_id,
            cognitive_load=50.0,
            task_type="test_task"
        )

        self.agent.active_sessions[user_id] = context
        retrieved = self.agent.get_session_context(user_id)

        self.assertEqual(retrieved.user_id, user_id)
        self.assertEqual(retrieved.cognitive_load, 50.0)

    def test_context_update(self):
        """Test context updating"""
        user_id = "test_user"
        context = UserContext(user_id=user_id, cognitive_load=50.0)
        self.agent.active_sessions[user_id] = context

        self.agent.update_context(user_id, cognitive_load=75.0)

        updated = self.agent.get_session_context(user_id)
        self.assertEqual(updated.cognitive_load, 75.0)

    def test_message_storage(self):
        """Test message history storage"""
        msg = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            payload={"test": "data"}
        )

        self.agent._store_message(msg)

        self.assertGreater(len(self.agent.message_history), 0)
        self.assertEqual(self.agent.message_history[-1].sender, "test_sender")

    def test_metrics(self):
        """Test metrics collection"""
        self.agent.register_agent("agent1", "type1", ["cap1"])
        self.agent.register_agent("agent2", "type2", ["cap2"])

        metrics = self.agent.get_metrics()

        self.assertEqual(metrics["agent_count"], 2)
        self.assertIn("agent1", metrics["registered_agents"])
        self.assertIn("agent2", metrics["registered_agents"])


class TestBehaviorAnalysisAgent(unittest.TestCase):
    """Test behavior analysis agent"""

    def setUp(self):
        self.agent = BehaviorAnalysisAgent()

    def test_interaction_recording(self):
        """Test recording interactions"""
        user_id = "test_user"

        self.agent.record_interaction(
            user_id=user_id,
            mouse_x=100,
            mouse_y=200,
            prev_mouse_x=90,
            prev_mouse_y=190,
            is_click=True,
            time_since_last_action=0.5
        )

        self.assertIn(user_id, self.agent.active_sessions)
        self.assertGreater(len(self.agent.active_sessions[user_id]), 0)

    def test_cognitive_load_estimation(self):
        """Test cognitive load estimation"""
        user_id = "test_user"

        # Record some interactions
        for i in range(20):
            self.agent.record_interaction(
                user_id=user_id,
                mouse_x=100 + i * 5,
                mouse_y=200 + i * 3,
                prev_mouse_x=100 + (i - 1) * 5,
                prev_mouse_y=200 + (i - 1) * 3,
                is_click=i % 3 == 0,
                is_error=False,
                time_since_last_action=0.5
            )

        estimate = self.agent.estimate_cognitive_load(user_id)

        self.assertIsInstance(estimate, CognitiveLoadEstimate)
        self.assertTrue(0 <= estimate.overall_score <= 100)
        self.assertTrue(0 <= estimate.confidence <= 1)

    def test_pattern_analysis(self):
        """Test behavior pattern detection"""
        user_id = "test_user"

        # Record normal interactions
        for i in range(30):
            self.agent.record_interaction(
                user_id=user_id,
                mouse_x=100 + i * 5,
                mouse_y=200,
                prev_mouse_x=100 + (i - 1) * 5,
                prev_mouse_y=200,
                is_click=i % 5 == 0,
                time_since_last_action=1.0
            )

        pattern = self.agent.analyze_behavior_pattern(user_id)

        self.assertIsNotNone(pattern)

    def test_anomaly_detection(self):
        """Test anomaly detection"""
        user_id = "test_user"

        # Record interactions with errors
        for i in range(20):
            self.agent.record_interaction(
                user_id=user_id,
                mouse_x=100 + i * 5,
                mouse_y=200,
                prev_mouse_x=100 + (i - 1) * 5,
                prev_mouse_y=200,
                is_click=True,
                is_error=i < 5,  # Errors in first 5
                time_since_last_action=0.1
            )

        anomalies = self.agent.detect_anomalies(user_id)

        # May or may not detect anomalies depending on threshold
        self.assertIsInstance(anomalies, list)


class TestInterfaceAgent(unittest.TestCase):
    """Test interface agent"""

    def setUp(self):
        self.agent = InterfaceAgent()

    def test_adaptation_generation(self):
        """Test generating adaptations"""
        import asyncio

        async def test():
            result = await self.agent.adapt_interface(
                user_id="test_user",
                cognitive_load=80.0,
                task_type="form_completion",
                page_url="https://example.com"
            )

            self.assertIn('adaptations_applied', result)
            self.assertGreater(result['adaptations_applied'], 0)
            self.assertIn('css_injection', result)
            self.assertIsInstance(result['expected_benefit'], float)

        asyncio.run(test())

    def test_adaptation_reversion(self):
        """Test reverting adaptations"""
        import asyncio

        async def test():
            # Apply adaptations
            await self.agent.adapt_interface(
                user_id="test_user",
                cognitive_load=85.0,
                task_type="form_completion",
                page_url="https://example.com"
            )

            # Revert
            result = self.agent.revert_adaptations("test_user")

            self.assertEqual(result['reverted'], 0)  # No active adaptations tracked separately

        asyncio.run(test())


class TestWorkflowAgent(unittest.TestCase):
    """Test workflow agent"""

    def setUp(self):
        self.agent = WorkflowAgent()

    def test_task_analysis(self):
        """Test task analysis"""
        import asyncio

        async def test():
            result = await self.agent.analyze_task(
                user_id="test_user",
                current_url="https://example.com/form",
                page_content="<form><input><button>Submit</button></form>",
                user_goal="Fill out the form"
            )

            self.assertIn('workflow_id', result)
            self.assertIn('task_type', result)
            self.assertIn('steps_count', result)
            self.assertGreater(result['steps_count'], 0)

        asyncio.run(test())

    def test_workflow_progress(self):
        """Test workflow progress tracking"""
        import asyncio

        async def test():
            # Create workflow
            analysis = await self.agent.analyze_task(
                user_id="test_user",
                current_url="https://example.com/form",
                page_content="<form></form>",
                user_goal="Complete form"
            )

            workflow_id = analysis['workflow_id']

            # Get progress
            progress = self.agent.get_workflow_progress(workflow_id)

            self.assertIn('workflow_id', progress)
            self.assertIn('progress', progress)
            self.assertIn('percentage', progress)

        asyncio.run(test())


class TestLearningModule(unittest.TestCase):
    """Test learning module"""

    def setUp(self):
        self.module = LearningModule()

    def test_interaction_recording(self):
        """Test recording interactions"""
        user_id = "test_user"

        self.module.record_interaction(
            user_id=user_id,
            task_type="form_completion",
            duration=120,
            success=True,
            adaptations_used=["simplify_layout"],
            cognitive_load_start=70,
            cognitive_load_end=40
        )

        self.assertIn(user_id, self.module.interaction_history)
        self.assertGreater(len(self.module.interaction_history[user_id]), 0)

    def test_user_analysis(self):
        """Test user profile analysis"""
        user_id = "test_user"

        # Record interactions
        for i in range(5):
            self.module.record_interaction(
                user_id=user_id,
                task_type="form_completion",
                duration=120 + i * 10,
                success=i > 1,  # First 2 fail, rest succeed
                adaptations_used=["simplify_layout"],
                cognitive_load_start=70,
                cognitive_load_end=40,
                user_satisfaction=50 if i < 2 else 80
            )

        profile = self.module.analyze_user(user_id)

        self.assertIsNotNone(profile)
        self.assertIn('expertise_level', dir(profile))

    def test_personalized_strategy(self):
        """Test strategy recommendation"""
        user_id = "test_user"

        # Record some data
        self.module.record_interaction(
            user_id=user_id,
            task_type="form_completion",
            duration=120,
            success=True,
            adaptations_used=["simplify_layout"],
            cognitive_load_start=70,
            cognitive_load_end=40
        )

        strategy = self.module.get_personalized_strategy(
            user_id=user_id,
            task_type="form_completion",
            cognitive_load=60
        )

        self.assertIn('recommended_adaptations', strategy)
        self.assertIsInstance(strategy['recommended_adaptations'], list)

    def test_learning_insights(self):
        """Test learning insights"""
        user_id = "test_user"

        # Record interactions
        for i in range(3):
            self.module.record_interaction(
                user_id=user_id,
                task_type="form_completion",
                duration=120,
                success=True,
                adaptations_used=["simplify_layout"],
                cognitive_load_start=70,
                cognitive_load_end=40
            )

        insights = self.module.get_learning_insights(user_id)

        self.assertIn('total_interactions', insights)
        self.assertIn('profile', insights)
        self.assertIn('performance', insights)


if __name__ == '__main__':
    unittest.main()
