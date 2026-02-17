"""
Performance Metrics Calculation
Metrics for evaluating system and user performance.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    task_completion_time: float  # seconds
    error_count: int
    success: bool
    pages_visited: int
    adaptations_applied: int
    user_satisfaction: Optional[float] = None  # 0-100
    cognitive_load_start: float = 50.0
    cognitive_load_end: float = 50.0
    sus_score: Optional[float] = None  # System Usability Scale
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class MetricsCalculator:
    """Calculate various performance and usability metrics"""

    @staticmethod
    def calculate_task_efficiency(
        completion_time: float,
        pages_visited: int,
        optimal_time: float = 120.0,
        optimal_pages: int = 3
    ) -> float:
        """
        Calculate task efficiency score (0-100).
        Higher is better.
        """
        time_ratio = completion_time / optimal_time
        pages_ratio = pages_visited / optimal_pages

        # Penalize excessive time and pages
        efficiency = 100 * (1 / (time_ratio * pages_ratio))
        return min(100, max(0, efficiency))

    @staticmethod
    def calculate_error_efficiency(
        error_count: int,
        total_actions: int
    ) -> float:
        """
        Calculate error rate (0-100, higher is better).
        """
        if total_actions == 0:
            return 100.0

        error_rate = error_count / total_actions
        efficiency = max(0, 100 - (error_rate * 100))
        return efficiency

    @staticmethod
    def calculate_cognitive_load_reduction(
        start_load: float,
        end_load: float
    ) -> float:
        """
        Calculate percentage reduction in cognitive load.
        Returns percentage points reduced.
        """
        reduction = start_load - end_load
        if start_load == 0:
            return 0.0
        return (reduction / start_load) * 100

    @staticmethod
    def calculate_sus_score(
        responses: List[int]  # 5 questions, 1-5 scale
    ) -> float:
        """
        Calculate System Usability Scale (SUS) score.
        Input: list of 5 responses (1-5 scale)
        Output: 0-100 scale
        """
        if len(responses) != 5:
            raise ValueError("SUS requires exactly 5 responses")

        # Odd questions (0,2,4): subtract 1
        # Even questions (1,3): subtract 5 from response and take absolute
        score = 0
        for i, response in enumerate(responses):
            if i % 2 == 0:
                score += (response - 1)
            else:
                score += (5 - response)

        return score * 2.5

    @staticmethod
    def calculate_nasa_tlx_score(
        mental_demand: float,
        physical_demand: float,
        temporal_demand: float,
        performance: float,
        effort: float,
        frustration: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate NASA Task Load Index (TLX) score.
        Inputs: 0-100 scales
        Output: 0-100 scale (overall workload)
        """
        if weights is None:
            weights = {
                "mental_demand": 0.25,
                "physical_demand": 0.15,
                "temporal_demand": 0.20,
                "performance": 0.20,
                "effort": 0.15,
                "frustration": 0.05,
            }

        # Normalize inputs to 0-1
        normalized = {
            "mental_demand": mental_demand / 100,
            "physical_demand": physical_demand / 100,
            "temporal_demand": temporal_demand / 100,
            "performance": performance / 100,
            "effort": effort / 100,
            "frustration": frustration / 100,
        }

        # Calculate weighted score
        score = sum(
            normalized[key] * weights[key]
            for key in normalized.keys()
        )

        return score * 100

    @staticmethod
    def calculate_learnability(
        early_performance: float,
        late_performance: float
    ) -> float:
        """
        Calculate learning improvement (Nielsen's learnability).
        Positive = improvement over time
        """
        if early_performance == 0:
            return 0.0
        improvement = ((late_performance - early_performance) / early_performance) * 100
        return max(-100, min(100, improvement))

    @staticmethod
    def calculate_memorability(
        retest_performance: float,
        final_performance: float
    ) -> float:
        """
        Calculate memorability (how well users remember after time).
        """
        if final_performance == 0:
            return 0.0
        memorability = (retest_performance / final_performance) * 100
        return min(100, memorability)

    @staticmethod
    def calculate_error_severity(
        error_count: int,
        critical_error_count: int,
        recovery_time: float
    ) -> float:
        """
        Calculate overall error severity score (0-100).
        Higher = worse.
        """
        if error_count == 0:
            return 0.0

        # Weight critical errors higher
        weighted_errors = (critical_error_count * 2) + (error_count - critical_error_count)
        severity_base = min(100, weighted_errors * 10)

        # Add recovery time penalty
        recovery_penalty = min(50, recovery_time / 2)

        return severity_base + recovery_penalty

    @staticmethod
    def calculate_adaptation_effectiveness(
        load_before: float,
        load_after: float,
        adaptation_count: int
    ) -> float:
        """
        Calculate effectiveness of adaptations.
        Returns load reduction per adaptation.
        """
        if adaptation_count == 0:
            return 0.0

        load_reduction = load_before - load_after
        return load_reduction / adaptation_count

    @staticmethod
    def aggregate_metrics(
        metrics_list: List[PerformanceMetrics]
    ) -> Dict[str, float]:
        """Aggregate metrics across multiple tasks"""
        if not metrics_list:
            return {}

        return {
            "avg_completion_time": np.mean([m.task_completion_time for m in metrics_list]),
            "std_completion_time": np.std([m.task_completion_time for m in metrics_list]),
            "total_errors": sum(m.error_count for m in metrics_list),
            "error_rate": np.mean([m.error_count for m in metrics_list]),
            "success_rate": sum(1 for m in metrics_list if m.success) / len(metrics_list),
            "avg_pages_visited": np.mean([m.pages_visited for m in metrics_list]),
            "avg_adaptations": np.mean([m.adaptations_applied for m in metrics_list]),
            "avg_satisfaction": np.mean([m.user_satisfaction for m in metrics_list if m.user_satisfaction]),
            "avg_cognitive_load_start": np.mean([m.cognitive_load_start for m in metrics_list]),
            "avg_cognitive_load_end": np.mean([m.cognitive_load_end for m in metrics_list]),
            "avg_cognitive_load_reduction": np.mean([
                MetricsCalculator.calculate_cognitive_load_reduction(m.cognitive_load_start, m.cognitive_load_end)
                for m in metrics_list
            ]),
        }

    @staticmethod
    def compare_metrics(
        group1: List[PerformanceMetrics],
        group2: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """
        Compare metrics between two groups (e.g., agentic vs control).
        Returns statistical comparison.
        """
        from scipy import stats

        agg1 = MetricsCalculator.aggregate_metrics(group1)
        agg2 = MetricsCalculator.aggregate_metrics(group2)

        # T-test on completion times
        times1 = [m.task_completion_time for m in group1]
        times2 = [m.task_completion_time for m in group2]

        if len(times1) > 1 and len(times2) > 1:
            t_stat, p_value = stats.ttest_ind(times1, times2)
        else:
            t_stat, p_value = 0, 1.0

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.std(times1) ** 2 + np.std(times2) ** 2) / 2
        )
        cohens_d = (np.mean(times2) - np.mean(times1)) / pooled_std if pooled_std > 0 else 0

        return {
            "group1": agg1,
            "group2": agg2,
            "completion_time_difference": agg2["avg_completion_time"] - agg1["avg_completion_time"],
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "error_rate_difference": agg2["error_rate"] - agg1["error_rate"],
            "success_rate_difference": agg2["success_rate"] - agg1["success_rate"],
            "satisfaction_difference": (agg2.get("avg_satisfaction", 0) or 0) - (agg1.get("avg_satisfaction", 0) or 0),
        }


# Example usage
if __name__ == "__main__":
    calc = MetricsCalculator()

    # Test task efficiency
    efficiency = calc.calculate_task_efficiency(
        completion_time=150,
        pages_visited=4,
        optimal_time=120,
        optimal_pages=3
    )
    print(f"Task efficiency: {efficiency:.1f}")

    # Test SUS score
    sus = calc.calculate_sus_score([4, 2, 4, 2, 4])
    print(f"SUS score: {sus:.1f}")

    # Test NASA-TLX
    tlx = calc.calculate_nasa_tlx_score(
        mental_demand=45,
        physical_demand=25,
        temporal_demand=35,
        performance=60,
        effort=40,
        frustration=30
    )
    print(f"NASA-TLX: {tlx:.1f}")

    # Test cognitive load reduction
    reduction = calc.calculate_cognitive_load_reduction(75, 45)
    print(f"Cognitive load reduction: {reduction:.1f}%")

    # Test aggregation
    metrics = [
        PerformanceMetrics(
            task_completion_time=120,
            error_count=1,
            success=True,
            pages_visited=3,
            adaptations_applied=2,
            user_satisfaction=85,
            cognitive_load_start=70,
            cognitive_load_end=40
        ),
        PerformanceMetrics(
            task_completion_time=140,
            error_count=0,
            success=True,
            pages_visited=4,
            adaptations_applied=3,
            user_satisfaction=90,
            cognitive_load_start=65,
            cognitive_load_end=35
        ),
    ]

    agg = calc.aggregate_metrics(metrics)
    print(f"\nAggregated metrics:")
    for key, value in agg.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
