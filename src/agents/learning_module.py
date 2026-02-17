"""
Learning Module for Agentic UX System
Continuous learning and personalization through user interaction history.
"""

import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User's learned profile and preferences"""
    user_id: str
    expertise_level: str = "intermediate"  # novice, intermediate, expert
    preferred_task_pace: str = "moderate"  # slow, moderate, fast
    learning_curve_stage: str = "practicing"  # learning, practicing, mastered
    favorite_adaptations: List[str] = field(default_factory=list)
    disliked_adaptations: List[str] = field(default_factory=list)
    interaction_style: str = "exploratory"  # linear, exploratory, deliberate
    optimal_load_range: Tuple[float, float] = (30.0, 60.0)
    task_completion_rates: Dict[str, float] = field(default_factory=dict)
    average_task_time: Dict[str, float] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class InteractionEvent:
    """Single user interaction event"""
    event_id: str
    user_id: str
    task_type: str
    timestamp: str
    duration: float
    success: bool
    adaptations_used: List[str]
    cognitive_load_start: float
    cognitive_load_end: float
    user_satisfaction: Optional[float] = None  # 0-100
    feedback: Optional[str] = None


class LearningModule:
    """
    Learns from user interactions to continuously improve personalization.
    Tracks preferences, learning progress, and optimal adaptation strategies.
    """

    def __init__(self, agent_id: str = "learning_module"):
        self.agent_id = agent_id
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.adaptation_effectiveness: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.learning_models = self._initialize_learning_models()
        self.personalization_rules = {}

    def _initialize_learning_models(self) -> Dict[str, Any]:
        """Initialize learning models"""
        return {
            "task_success_rate": {},
            "adaptation_effectiveness": {},
            "user_preferences": {},
            "optimal_load_ranges": {},
        }

    def record_interaction(
        self,
        user_id: str,
        task_type: str,
        duration: float,
        success: bool,
        adaptations_used: List[str],
        cognitive_load_start: float,
        cognitive_load_end: float,
        user_satisfaction: Optional[float] = None
    ) -> None:
        """Record a user interaction event"""
        event_id = f"evt_{user_id}_{int(datetime.utcnow().timestamp() * 1000)}"

        event = InteractionEvent(
            event_id=event_id,
            user_id=user_id,
            task_type=task_type,
            timestamp=datetime.utcnow().isoformat(),
            duration=duration,
            success=success,
            adaptations_used=adaptations_used,
            cognitive_load_start=cognitive_load_start,
            cognitive_load_end=cognitive_load_end,
            user_satisfaction=user_satisfaction
        )

        self.interaction_history[user_id].append(event)

        # Update adaptation effectiveness
        for adaptation in adaptations_used:
            load_reduction = cognitive_load_start - cognitive_load_end
            self.adaptation_effectiveness[user_id][adaptation].append(load_reduction)

        logger.info(f"Recorded interaction for {user_id}: {task_type} (success={success})")

    def analyze_user(self, user_id: str) -> UserProfile:
        """Analyze user profile from interaction history"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        if user_id not in self.interaction_history or len(self.interaction_history[user_id]) == 0:
            return self.user_profiles[user_id]

        events = list(self.interaction_history[user_id])
        profile = self.user_profiles[user_id]

        # Calculate success rate
        success_rate = np.mean([e.success for e in events])

        # Determine expertise level
        if success_rate > 0.9:
            profile.expertise_level = "expert"
            profile.learning_curve_stage = "mastered"
        elif success_rate > 0.7:
            profile.expertise_level = "intermediate"
            profile.learning_curve_stage = "practicing"
        else:
            profile.expertise_level = "novice"
            profile.learning_curve_stage = "learning"

        # Analyze task completion rates and times
        task_stats = defaultdict(lambda: {"success": [], "duration": []})
        for event in events:
            task_stats[event.task_type]["success"].append(event.success)
            task_stats[event.task_type]["duration"].append(event.duration)

        for task_type, stats in task_stats.items():
            profile.task_completion_rates[task_type] = np.mean(stats["success"])
            profile.average_task_time[task_type] = np.mean(stats["duration"])

        # Determine interaction style
        profile.interaction_style = self._analyze_interaction_style(events)

        # Determine optimal load range
        profile.optimal_load_range = self._calculate_optimal_load_range(events)

        # Determine favorite adaptations
        profile.favorite_adaptations = self._rank_adaptations(user_id, "effectiveness")
        profile.disliked_adaptations = self._rank_adaptations(user_id, "ineffective")

        # Determine task pace preference
        avg_duration = np.mean([e.duration for e in events])
        if avg_duration < 120:
            profile.preferred_task_pace = "fast"
        elif avg_duration < 300:
            profile.preferred_task_pace = "moderate"
        else:
            profile.preferred_task_pace = "slow"

        profile.last_updated = datetime.utcnow().isoformat()

        return profile

    def _analyze_interaction_style(self, events: List[InteractionEvent]) -> str:
        """Analyze user's interaction style"""
        # Analyze action sequences and patterns
        action_sequences = []
        for event in events[-20:]:
            action_sequences.append(len(event.adaptations_used))

        if len(action_sequences) > 0:
            avg_actions = np.mean(action_sequences)
            if avg_actions < 1.5:
                return "linear"
            elif avg_actions < 3.0:
                return "exploratory"
            else:
                return "deliberate"

        return "exploratory"

    def _calculate_optimal_load_range(self, events: List[InteractionEvent]) -> Tuple[float, float]:
        """Calculate optimal cognitive load range for user"""
        if not events:
            return (30.0, 60.0)

        # Find load ranges where user succeeded
        successful_loads = []
        for event in events:
            if event.success:
                successful_loads.append((event.cognitive_load_start + event.cognitive_load_end) / 2)

        if not successful_loads:
            return (30.0, 60.0)

        loads = np.array(successful_loads)
        mean_load = np.mean(loads)
        std_load = np.std(loads)

        optimal_min = max(0, mean_load - std_load)
        optimal_max = min(100, mean_load + std_load)

        return (optimal_min, optimal_max)

    def _rank_adaptations(self, user_id: str, criteria: str) -> List[str]:
        """Rank adaptations by effectiveness or user feedback"""
        if user_id not in self.adaptation_effectiveness:
            return []

        effectiveness_scores = {}
        for adaptation, improvements in self.adaptation_effectiveness[user_id].items():
            if improvements:
                effectiveness_scores[adaptation] = np.mean(improvements)

        if criteria == "effectiveness":
            ranked = sorted(
                effectiveness_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:  # ineffective
            ranked = sorted(
                effectiveness_scores.items(),
                key=lambda x: x[1]
            )

        return [a for a, _ in ranked[:5]]

    def get_personalized_strategy(self, user_id: str, task_type: str, cognitive_load: float) -> Dict[str, Any]:
        """
        Get personalized adaptation strategy based on learned user profile.
        """
        profile = self.analyze_user(user_id)

        strategy = {
            "user_id": user_id,
            "profile": {
                "expertise": profile.expertise_level,
                "pace": profile.preferred_task_pace,
                "interaction_style": profile.interaction_style,
            },
            "recommended_adaptations": [],
            "expected_effectiveness": 0.0,
        }

        # Base adaptations on expertise level
        if profile.expertise_level == "novice":
            strategy["recommended_adaptations"] = [
                "show_detailed_guidance",
                "add_helpful_hints",
                "simplify_layout",
                "highlight_primary_action"
            ]
            strategy["expected_effectiveness"] = 0.7

        elif profile.expertise_level == "intermediate":
            strategy["recommended_adaptations"] = [
                "show_progress",
                "enable_shortcuts",
                "organize_by_importance",
                "suggest_next_step"
            ]
            strategy["expected_effectiveness"] = 0.6

        else:  # expert
            strategy["recommended_adaptations"] = [
                "quick_access",
                "advanced_options",
                "keyboard_shortcuts",
                "power_user_features"
            ]
            strategy["expected_effectiveness"] = 0.5

        # Adjust based on cognitive load
        if cognitive_load > profile.optimal_load_range[1]:
            strategy["recommended_adaptations"].insert(0, "reduce_complexity")
            strategy["load_adjustment"] = "reduce"
        elif cognitive_load < profile.optimal_load_range[0]:
            strategy["recommended_adaptations"].append("show_advanced_options")
            strategy["load_adjustment"] = "increase"
        else:
            strategy["load_adjustment"] = "maintain"

        # Use user's favorite adaptations
        for fav in profile.favorite_adaptations[:2]:
            if fav not in strategy["recommended_adaptations"]:
                strategy["recommended_adaptations"].insert(0, fav)

        # Remove disliked adaptations
        strategy["recommended_adaptations"] = [
            a for a in strategy["recommended_adaptations"]
            if a not in profile.disliked_adaptations
        ]

        # Task-specific adjustments
        if task_type in profile.task_completion_rates:
            task_success_rate = profile.task_completion_rates[task_type]
            if task_success_rate < 0.6:
                strategy["recommended_adaptations"].insert(0, "provide_assistance")

        return strategy

    def collect_feedback(
        self,
        user_id: str,
        interaction_id: str,
        satisfaction_score: float,
        feedback_text: Optional[str] = None
    ) -> None:
        """Collect user feedback on adaptations"""
        events = list(self.interaction_history[user_id])
        for event in events:
            if event.event_id == interaction_id:
                event.user_satisfaction = satisfaction_score
                event.feedback = feedback_text
                logger.info(f"Collected feedback for {user_id}: {satisfaction_score}/100")
                return

    def get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get learning insights about the user"""
        profile = self.analyze_user(user_id)
        events = list(self.interaction_history[user_id])

        if not events:
            return {"error": "No interaction history"}

        insights = {
            "user_id": user_id,
            "total_interactions": len(events),
            "profile": {
                "expertise": profile.expertise_level,
                "learning_stage": profile.learning_curve_stage,
                "interaction_style": profile.interaction_style,
                "preferred_pace": profile.preferred_task_pace,
            },
            "performance": {
                "overall_success_rate": np.mean([e.success for e in events]),
                "task_completion_rates": profile.task_completion_rates,
                "average_task_times": profile.average_task_time,
            },
            "adaptations": {
                "most_effective": profile.favorite_adaptations,
                "least_effective": profile.disliked_adaptations,
            },
            "progress": {
                "tasks_completed": sum(1 for e in events if e.success),
                "tasks_failed": sum(1 for e in events if not e.success),
                "improvement_trend": self._calculate_improvement_trend(events)
            }
        }

        return insights

    def _calculate_improvement_trend(self, events: List[InteractionEvent]) -> str:
        """Calculate if user is improving over time"""
        if len(events) < 5:
            return "insufficient_data"

        first_half = [e.success for e in events[:len(events)//2]]
        second_half = [e.success for e in events[len(events)//2:]]

        first_rate = np.mean(first_half) if first_half else 0.5
        second_rate = np.mean(second_half) if second_half else 0.5

        improvement = second_rate - first_rate
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"

    def predict_task_performance(self, user_id: str, task_type: str) -> Dict[str, float]:
        """Predict user's performance on a task type"""
        profile = self.analyze_user(user_id)

        predicted_success_rate = profile.task_completion_rates.get(task_type, 0.5)
        predicted_duration = profile.average_task_time.get(task_type, 180)

        # Adjust based on expertise
        expertise_multipliers = {
            "novice": 1.5,
            "intermediate": 1.0,
            "expert": 0.7
        }
        predicted_duration *= expertise_multipliers.get(profile.expertise_level, 1.0)

        return {
            "predicted_success_rate": predicted_success_rate,
            "predicted_duration_seconds": predicted_duration,
            "confidence": min(len(self.interaction_history[user_id]) / 50, 1.0)
        }

    def recommend_adaptation_sequence(
        self,
        user_id: str,
        task_type: str,
        cognitive_load: float,
        max_adaptations: int = 3
    ) -> List[str]:
        """Recommend best sequence of adaptations for user"""
        strategy = self.get_personalized_strategy(user_id, task_type, cognitive_load)
        return strategy["recommended_adaptations"][:max_adaptations]

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            "users_tracked": len(self.user_profiles),
            "total_interactions": sum(len(h) for h in self.interaction_history.values()),
            "adaptations_learned": len(set(
                a for effects in self.adaptation_effectiveness.values()
                for a in effects.keys()
            ))
        }


if __name__ == "__main__":
    module = LearningModule()

    # Simulate user interactions
    user_id = "user_001"

    # Novice starting, making errors
    for i in range(5):
        module.record_interaction(
            user_id=user_id,
            task_type="form_completion",
            duration=300 + i * 50,  # Getting faster
            success=i > 2,  # First 2 fail, last 3 succeed
            adaptations_used=["simplify_layout", "highlight_action"],
            cognitive_load_start=75,
            cognitive_load_end=45,
            user_satisfaction=70 if i > 2 else 40
        )

    # Analyze user
    profile = module.analyze_user(user_id)
    print(f"User expertise: {profile.expertise_level}")
    print(f"Task completion rates: {profile.task_completion_rates}")

    # Get personalized strategy
    strategy = module.get_personalized_strategy(user_id, "form_completion", 50)
    print(f"\nPersonalized strategy:")
    print(f"  Recommended: {strategy['recommended_adaptations']}")
    print(f"  Expected effectiveness: {strategy['expected_effectiveness']:.1%}")

    # Get insights
    insights = module.get_learning_insights(user_id)
    print(f"\nLearning insights:")
    print(f"  Total interactions: {insights['total_interactions']}")
    print(f"  Success rate: {insights['performance']['overall_success_rate']:.1%}")
    print(f"  Progress: {insights['progress']['improvement_trend']}")
