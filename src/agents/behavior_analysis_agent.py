"""
Behavior Analysis Agent
Analyzes user interaction patterns and detects cognitive load in real-time.
"""

import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorPattern(Enum):
    """Classification of user behavior patterns"""
    FOCUSED = "focused"  # Linear, efficient navigation
    EXPLORATORY = "exploratory"  # Browsing, discovering
    STRUGGLING = "struggling"  # Repeated actions, backtracking
    LEARNING = "learning"  # Initial phase, building mental model
    FRUSTRATED = "frustrated"  # Excessive errors, rapid clicking


class CognitiveLoadLevel(Enum):
    """Cognitive load severity levels"""
    VERY_LOW = (0, 20)
    LOW = (20, 40)
    MODERATE = (40, 60)
    HIGH = (60, 80)
    VERY_HIGH = (80, 100)


@dataclass
class InteractionMetrics:
    """Metrics for a single interaction session"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    mouse_movement_distance: float = 0.0
    mouse_velocity: float = 0.0
    click_frequency: float = 0.0  # clicks per second
    time_between_actions: float = 0.0  # seconds
    error_count: int = 0
    correction_count: int = 0
    page_visits: int = 0
    scroll_intensity: float = 0.0
    attention_switches: int = 0


@dataclass
class CognitiveLoadEstimate:
    """Cognitive load estimation result"""
    overall_score: float  # 0-100
    mental_demand: float
    physical_demand: float
    temporal_demand: float
    performance_effort: float
    frustration_level: float
    confidence: float  # 0-1, confidence in estimate
    contributing_factors: List[str] = field(default_factory=list)
    recommended_adaptations: List[str] = field(default_factory=list)


class BehaviorAnalysisAgent:
    """
    Analyzes user behavior patterns and estimates cognitive load using
    multi-modal behavioral signals and machine learning.
    """

    def __init__(self, agent_id: str = "behavior_analysis_agent", window_size: int = 100):
        self.agent_id = agent_id
        self.window_size = window_size
        self.active_sessions: Dict[str, List[InteractionMetrics]] = {}
        self.cognitive_load_cache: Dict[str, deque] = {}
        self.pattern_models = self._initialize_pattern_models()
        self.load_thresholds = {
            "critical_action_time": 3.0,  # seconds
            "error_threshold": 5,  # per session
            "click_spam_threshold": 10,  # clicks per second
            "attention_switch_threshold": 5,
        }

    def _initialize_pattern_models(self) -> Dict[str, Any]:
        """Initialize behavior pattern detection models"""
        return {
            "focused": {
                "expected_click_frequency": 1.5,  # clicks/sec
                "expected_error_rate": 0.02,
                "page_visit_efficiency": 0.8,
            },
            "exploratory": {
                "expected_click_frequency": 2.0,
                "expected_error_rate": 0.05,
                "page_visit_efficiency": 0.5,
            },
            "struggling": {
                "expected_click_frequency": 3.5,
                "expected_error_rate": 0.15,
                "page_visit_efficiency": 0.3,
            },
            "frustrated": {
                "expected_click_frequency": 5.0,
                "expected_error_rate": 0.25,
                "page_visit_efficiency": 0.1,
            },
        }

    def record_interaction(
        self,
        user_id: str,
        mouse_x: float,
        mouse_y: float,
        prev_mouse_x: float,
        prev_mouse_y: float,
        is_click: bool = False,
        is_error: bool = False,
        page_url: str = "",
        time_since_last_action: float = 0.0
    ) -> None:
        """Record single user interaction"""
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = deque(maxlen=self.window_size)
            self.cognitive_load_cache[user_id] = deque(maxlen=20)

        # Calculate metrics for this interaction
        mouse_distance = np.sqrt((mouse_x - prev_mouse_x)**2 + (mouse_y - prev_mouse_y)**2)

        metrics = InteractionMetrics(
            mouse_movement_distance=mouse_distance,
            time_between_actions=time_since_last_action,
        )

        if is_click:
            metrics.click_frequency = 1.0 / max(time_since_last_action, 0.1)

        if is_error:
            metrics.error_count = 1

        self.active_sessions[user_id].append(metrics)

    def analyze_behavior_pattern(self, user_id: str) -> BehaviorPattern:
        """Classify current user behavior pattern"""
        if user_id not in self.active_sessions or len(self.active_sessions[user_id]) == 0:
            return BehaviorPattern.LEARNING

        metrics = list(self.active_sessions[user_id])
        avg_click_freq = np.mean([m.click_frequency for m in metrics if m.click_frequency > 0]) or 0
        error_rate = np.sum([m.error_count for m in metrics]) / len(metrics)
        avg_action_time = np.mean([m.time_between_actions for m in metrics if m.time_between_actions > 0]) or 1.0

        # Pattern detection logic
        if avg_click_freq > self.load_thresholds["click_spam_threshold"]:
            return BehaviorPattern.FRUSTRATED
        elif error_rate > 0.15:
            return BehaviorPattern.STRUGGLING
        elif avg_action_time < 0.5:
            return BehaviorPattern.FOCUSED
        elif avg_click_freq > 2.5:
            return BehaviorPattern.EXPLORATORY
        else:
            return BehaviorPattern.LEARNING

    def estimate_cognitive_load(self, user_id: str) -> CognitiveLoadEstimate:
        """
        Estimate cognitive load from behavioral metrics.
        Uses weighted combination of multiple behavioral signals.
        """
        if user_id not in self.active_sessions or len(self.active_sessions[user_id]) == 0:
            return CognitiveLoadEstimate(overall_score=50.0)

        metrics_list = list(self.active_sessions[user_id])

        # Extract behavioral features
        avg_click_freq = np.mean([m.click_frequency for m in metrics_list if m.click_frequency > 0]) or 0.5
        error_count = np.sum([m.error_count for m in metrics_list])
        correction_count = np.sum([m.correction_count for m in metrics_list])
        avg_action_time = np.mean([m.time_between_actions for m in metrics_list if m.time_between_actions > 0]) or 1.0
        mouse_velocity = np.mean([m.mouse_velocity for m in metrics_list]) or 100
        page_visits = np.sum([m.page_visits for m in metrics_list])

        # Compute component loads (0-100 scale)
        mental_demand = self._estimate_mental_demand(
            avg_action_time, error_count, page_visits
        )
        physical_demand = self._estimate_physical_demand(
            mouse_velocity, avg_click_freq
        )
        temporal_demand = self._estimate_temporal_demand(
            avg_action_time, error_count, correction_count
        )
        performance_effort = self._estimate_performance_effort(
            error_count, correction_count
        )
        frustration_level = self._estimate_frustration(
            avg_click_freq, error_count, avg_action_time
        )

        # Weighted overall score (NASA-TLX style)
        weights = {
            "mental_demand": 0.25,
            "physical_demand": 0.15,
            "temporal_demand": 0.20,
            "performance_effort": 0.20,
            "frustration_level": 0.20,
        }

        overall_score = (
            mental_demand * weights["mental_demand"] +
            physical_demand * weights["physical_demand"] +
            temporal_demand * weights["temporal_demand"] +
            performance_effort * weights["performance_effort"] +
            frustration_level * weights["frustration_level"]
        )

        # Identify contributing factors
        contributing_factors = []
        if mental_demand > 60:
            contributing_factors.append("high_mental_demand")
        if error_count > 3:
            contributing_factors.append("frequent_errors")
        if avg_click_freq > 3.0:
            contributing_factors.append("excessive_interactions")
        if avg_action_time > 2.0:
            contributing_factors.append("slow_task_progression")

        # Generate recommendations
        recommended_adaptations = self._generate_recommendations(
            overall_score, contributing_factors
        )

        # Confidence based on data quantity and consistency
        confidence = min(len(metrics_list) / self.window_size, 1.0)

        estimate = CognitiveLoadEstimate(
            overall_score=overall_score,
            mental_demand=mental_demand,
            physical_demand=physical_demand,
            temporal_demand=temporal_demand,
            performance_effort=performance_effort,
            frustration_level=frustration_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommended_adaptations=recommended_adaptations,
        )

        # Cache the estimate
        self.cognitive_load_cache[user_id].append(estimate)

        return estimate

    def _estimate_mental_demand(
        self,
        avg_action_time: float,
        error_count: int,
        page_visits: int
    ) -> float:
        """Estimate mental demand component"""
        base_score = 50.0

        # Slow action time indicates more thinking
        if avg_action_time > 2.0:
            base_score += 30.0
        elif avg_action_time > 1.0:
            base_score += 15.0

        # Errors indicate confusion/complexity
        base_score += min(error_count * 5, 20)

        # Page visits indicate navigation complexity
        if page_visits > 5:
            base_score += 15.0

        return min(base_score, 100.0)

    def _estimate_physical_demand(
        self,
        mouse_velocity: float,
        click_frequency: float
    ) -> float:
        """Estimate physical demand component"""
        base_score = 30.0

        # Higher click frequency indicates more physical activity
        if click_frequency > 2.0:
            base_score += 25.0
        elif click_frequency > 1.0:
            base_score += 10.0

        # Mouse movement speed
        if mouse_velocity > 300:
            base_score += 15.0

        return min(base_score, 100.0)

    def _estimate_temporal_demand(
        self,
        avg_action_time: float,
        error_count: int,
        correction_count: int
    ) -> float:
        """Estimate temporal demand component"""
        base_score = 40.0

        # Slow actions create time pressure
        if avg_action_time > 2.0:
            base_score += 20.0

        # Errors cause time consumption
        base_score += min(error_count * 3, 15)
        base_score += min(correction_count * 2, 10)

        return min(base_score, 100.0)

    def _estimate_performance_effort(
        self,
        error_count: int,
        correction_count: int
    ) -> float:
        """Estimate performance effort component"""
        base_score = 30.0

        # Errors require effort to correct
        base_score += min(error_count * 8, 30)
        base_score += min(correction_count * 5, 20)

        return min(base_score, 100.0)

    def _estimate_frustration(
        self,
        click_frequency: float,
        error_count: int,
        avg_action_time: float
    ) -> float:
        """Estimate frustration level"""
        base_score = 35.0

        # Rapid clicking indicates frustration
        if click_frequency > 4.0:
            base_score += 35.0
        elif click_frequency > 2.5:
            base_score += 20.0

        # Frequent errors increase frustration
        base_score += min(error_count * 6, 25)

        # Slow progress increases frustration
        if avg_action_time > 2.5:
            base_score += 15.0

        return min(base_score, 100.0)

    def _generate_recommendations(
        self,
        overall_score: float,
        contributing_factors: List[str]
    ) -> List[str]:
        """Generate UI adaptation recommendations"""
        recommendations = []

        if overall_score > 75:
            recommendations.extend([
                "simplify_layout",
                "reduce_options",
                "highlight_primary_action"
            ])

        if "frequent_errors" in contributing_factors:
            recommendations.extend([
                "add_inline_validation",
                "show_helpful_hints"
            ])

        if "excessive_interactions" in contributing_factors:
            recommendations.append("auto_advance_steps")

        if "high_mental_demand" in contributing_factors:
            recommendations.extend([
                "show_task_progress",
                "reduce_info_density"
            ])

        if "slow_task_progression" in contributing_factors:
            recommendations.extend([
                "suggest_shortcuts",
                "auto_fill_available_data"
            ])

        return recommendations[:3]  # Limit to top 3

    def get_load_trends(self, user_id: str, window: int = 10) -> Dict[str, List[float]]:
        """Get recent cognitive load trends"""
        if user_id not in self.cognitive_load_cache:
            return {"scores": [], "timestamps": []}

        cache = list(self.cognitive_load_cache[user_id])[-window:]
        return {
            "scores": [e.overall_score for e in cache],
            "mental_demand": [e.mental_demand for e in cache],
            "frustration": [e.frustration_level for e in cache],
            "confidence": [e.confidence for e in cache],
        }

    def detect_anomalies(self, user_id: str, sensitivity: float = 1.5) -> List[Dict[str, Any]]:
        """Detect unusual behavior patterns"""
        if user_id not in self.active_sessions or len(self.active_sessions[user_id]) == 0:
            return []

        metrics_list = list(self.active_sessions[user_id])
        anomalies = []

        # Check for sudden spike in errors
        recent_errors = [m.error_count for m in metrics_list[-10:]]
        if len(recent_errors) > 5 and np.mean(recent_errors) > 2:
            anomalies.append({
                "type": "error_spike",
                "severity": "high",
                "suggestion": "provide_assistance"
            })

        # Check for rapid clicking (frustration)
        recent_clicks = [m.click_frequency for m in metrics_list[-10:] if m.click_frequency > 0]
        if len(recent_clicks) > 0 and np.mean(recent_clicks) > 4.0:
            anomalies.append({
                "type": "rapid_clicking",
                "severity": "high",
                "suggestion": "pause_and_guide"
            })

        # Check for no movement (user stuck)
        recent_distances = [m.mouse_movement_distance for m in metrics_list[-10:]]
        if len(recent_distances) > 5 and np.mean(recent_distances) < 10:
            anomalies.append({
                "type": "no_progress",
                "severity": "medium",
                "suggestion": "offer_help"
            })

        return anomalies

    def reset_session(self, user_id: str) -> None:
        """Reset session data for user"""
        if user_id in self.active_sessions:
            self.active_sessions[user_id].clear()
        if user_id in self.cognitive_load_cache:
            self.cognitive_load_cache[user_id].clear()

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "active_sessions": len(self.active_sessions),
            "total_interactions": sum(
                len(s) for s in self.active_sessions.values()
            ),
            "cache_size": sum(
                len(c) for c in self.cognitive_load_cache.values()
            ),
        }


# Example usage
if __name__ == "__main__":
    agent = BehaviorAnalysisAgent()

    # Simulate user interactions
    user_id = "user_001"

    # Simulate normal behavior
    for i in range(50):
        agent.record_interaction(
            user_id,
            mouse_x=100 + i * 10,
            mouse_y=200 + i * 5,
            prev_mouse_x=100 + (i - 1) * 10,
            prev_mouse_y=200 + (i - 1) * 5,
            is_click=i % 5 == 0,
            is_error=False,
            time_since_last_action=0.5
        )

    # Analyze
    pattern = agent.analyze_behavior_pattern(user_id)
    load_estimate = agent.estimate_cognitive_load(user_id)

    print(f"Behavior pattern: {pattern.value}")
    print(f"Cognitive load: {load_estimate.overall_score:.1f}")
    print(f"Contributing factors: {load_estimate.contributing_factors}")
    print(f"Recommendations: {load_estimate.recommended_adaptations}")
