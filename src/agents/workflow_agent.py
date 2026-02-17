"""
Workflow Agent for Multi-step Task Orchestration
Manages cross-site navigation and complex workflow optimization.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks the workflow agent handles"""
    FORM_COMPLETION = "form_completion"
    INFORMATION_RETRIEVAL = "information_retrieval"
    MULTI_STEP_PROCESS = "multi_step_process"
    ONLINE_TRANSACTION = "online_transaction"
    RESEARCH_TASK = "research_task"
    COMPARISON_TASK = "comparison_task"


@dataclass
class WorkflowStep:
    """Single step in a workflow"""
    step_id: str
    step_number: int
    url: str
    title: str
    required_actions: List[str]
    completion_criteria: str
    estimated_time: float  # seconds
    is_completed: bool = False
    actual_time: float = 0.0


@dataclass
class WorkflowPlan:
    """Complete workflow plan for a task"""
    workflow_id: str
    task_type: TaskType
    user_id: str
    start_url: str
    steps: List[WorkflowStep]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    current_step: int = 0
    total_time_estimate: float = 0.0
    optimizations_applied: List[str] = field(default_factory=list)


class WorkflowAgent:
    """
    Orchestrates multi-step tasks across different websites and applications.
    Provides intelligent guidance, suggestions, and automation opportunities.
    """

    def __init__(self, agent_id: str = "workflow_agent"):
        self.agent_id = agent_id
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        self.step_templates = self._initialize_step_templates()
        self.optimization_strategies = {
            "auto_fill": self._apply_auto_fill,
            "predictive_navigation": self._apply_predictive_navigation,
            "parallel_searches": self._apply_parallel_searches,
            "shortcut_suggestions": self._apply_shortcut_suggestions,
        }
        self.task_patterns = self._initialize_task_patterns()

    def _initialize_step_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize common step templates for different tasks"""
        return {
            "form_completion": [
                {"action": "locate_form", "importance": "critical"},
                {"action": "fill_personal_info", "importance": "high"},
                {"action": "fill_address", "importance": "high"},
                {"action": "fill_contact", "importance": "high"},
                {"action": "select_options", "importance": "medium"},
                {"action": "review_form", "importance": "high"},
                {"action": "submit", "importance": "critical"},
            ],
            "information_retrieval": [
                {"action": "go_to_search", "importance": "critical"},
                {"action": "enter_query", "importance": "critical"},
                {"action": "review_results", "importance": "high"},
                {"action": "select_source", "importance": "high"},
                {"action": "extract_info", "importance": "high"},
                {"action": "verify_accuracy", "importance": "medium"},
            ],
            "online_transaction": [
                {"action": "browse_products", "importance": "medium"},
                {"action": "select_item", "importance": "high"},
                {"action": "add_to_cart", "importance": "high"},
                {"action": "go_to_checkout", "importance": "high"},
                {"action": "enter_shipping", "importance": "critical"},
                {"action": "select_payment", "importance": "critical"},
                {"action": "review_order", "importance": "high"},
                {"action": "complete_purchase", "importance": "critical"},
            ],
        }

    def _initialize_task_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for detecting task types"""
        return {
            "form_completion": {
                "indicators": ["form", "input", "required", "submit"],
                "url_patterns": [r"form", r"signup", r"register", r"checkout"],
            },
            "information_retrieval": {
                "indicators": ["search", "query", "find", "information"],
                "url_patterns": [r"search", r"google", r"wikipedia", r"query"],
            },
            "online_transaction": {
                "indicators": ["product", "price", "cart", "checkout", "payment"],
                "url_patterns": [r"shop", r"store", r"buy", r"cart", r"checkout"],
            },
            "multi_step_process": {
                "indicators": ["step", "wizard", "process", "next", "continue"],
                "url_patterns": [r"step", r"wizard", r"process"],
            },
        }

    async def analyze_task(
        self,
        user_id: str,
        current_url: str,
        page_content: str,
        user_goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the current task and create an optimized workflow plan.
        """
        logger.info(f"Analyzing task for user {user_id}: {current_url}")

        # Detect task type
        task_type = self._detect_task_type(current_url, page_content, user_goal)

        # Create workflow plan
        workflow_id = f"wf_{user_id}_{int(datetime.utcnow().timestamp())}"
        steps = self._generate_workflow_steps(task_type, current_url, page_content)

        plan = WorkflowPlan(
            workflow_id=workflow_id,
            task_type=task_type,
            user_id=user_id,
            start_url=current_url,
            steps=steps,
            total_time_estimate=sum(s.estimated_time for s in steps)
        )

        self.active_workflows[workflow_id] = plan

        return {
            "workflow_id": workflow_id,
            "task_type": task_type.value,
            "steps_count": len(steps),
            "estimated_total_time": plan.total_time_estimate,
            "first_step": self._serialize_step(steps[0]) if steps else None,
            "optimization_opportunities": self._identify_optimizations(plan)
        }

    def _detect_task_type(
        self,
        url: str,
        content: str,
        goal: Optional[str]
    ) -> TaskType:
        """Detect the type of task user is performing"""
        # Check explicit goal
        if goal:
            goal_lower = goal.lower()
            if "form" in goal_lower or "fill" in goal_lower:
                return TaskType.FORM_COMPLETION
            if "find" in goal_lower or "search" in goal_lower or "lookup" in goal_lower:
                return TaskType.INFORMATION_RETRIEVAL
            if "buy" in goal_lower or "purchase" in goal_lower or "shop" in goal_lower:
                return TaskType.ONLINE_TRANSACTION
            if "step" in goal_lower or "process" in goal_lower:
                return TaskType.MULTI_STEP_PROCESS

        # Check URL patterns
        url_lower = url.lower()
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns["url_patterns"]:
                if re.search(pattern, url_lower):
                    return TaskType[task_type.upper().replace("_", "_")]

        # Check content indicators
        content_lower = content.lower()
        for task_type, patterns in self.task_patterns.items():
            indicator_count = sum(
                1 for ind in patterns["indicators"] if ind in content_lower
            )
            if indicator_count >= 2:
                return TaskType[task_type.upper().replace("_", "_")]

        # Default
        return TaskType.MULTI_STEP_PROCESS

    def _generate_workflow_steps(
        self,
        task_type: TaskType,
        start_url: str,
        page_content: str
    ) -> List[WorkflowStep]:
        """Generate workflow steps for detected task type"""
        steps = []
        template = self.step_templates.get(task_type.value, [])

        step_number = 1
        total_time = 0

        for i, action in enumerate(template):
            step_id = f"step_{step_number:02d}"

            # Estimate step duration based on importance
            importance_times = {"critical": 30, "high": 20, "medium": 10}
            step_time = importance_times.get(action["importance"], 15)

            step = WorkflowStep(
                step_id=step_id,
                step_number=step_number,
                url=start_url,
                title=self._get_step_title(action["action"]),
                required_actions=[action["action"]],
                completion_criteria=self._get_completion_criteria(action["action"]),
                estimated_time=step_time
            )

            steps.append(step)
            total_time += step_time
            step_number += 1

        return steps

    def _get_step_title(self, action: str) -> str:
        """Get human-readable title for step"""
        titles = {
            "locate_form": "Locate the form",
            "fill_personal_info": "Enter your personal information",
            "fill_address": "Enter your address",
            "fill_contact": "Enter contact information",
            "select_options": "Select options",
            "review_form": "Review the form",
            "submit": "Submit the form",
            "go_to_search": "Go to search",
            "enter_query": "Enter your search query",
            "review_results": "Review search results",
            "select_source": "Select a source",
            "extract_info": "Extract the information",
            "verify_accuracy": "Verify accuracy",
            "browse_products": "Browse products",
            "select_item": "Select an item",
            "add_to_cart": "Add to cart",
            "go_to_checkout": "Go to checkout",
            "enter_shipping": "Enter shipping address",
            "select_payment": "Select payment method",
            "review_order": "Review your order",
            "complete_purchase": "Complete purchase",
        }
        return titles.get(action, action.replace("_", " ").title())

    def _get_completion_criteria(self, action: str) -> str:
        """Get completion criteria for action"""
        criteria = {
            "submit": "Form successfully submitted",
            "complete_purchase": "Order confirmation received",
            "extract_info": "Required information extracted",
            "select_item": "Item selected and ready",
        }
        return criteria.get(action, f"{action.replace('_', ' ').title()} completed")

    def _identify_optimizations(self, plan: WorkflowPlan) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []

        # If task has many steps, suggest parallel processing
        if len(plan.steps) > 5:
            opportunities.append("Consider parallel searches")

        # If long estimated time, suggest auto-fill
        if plan.total_time_estimate > 120:
            opportunities.append("Enable auto-fill for faster completion")

        # Task-specific optimizations
        if plan.task_type == TaskType.FORM_COMPLETION:
            opportunities.append("Use saved form data")
            opportunities.append("Enable single-click submission")

        if plan.task_type == TaskType.INFORMATION_RETRIEVAL:
            opportunities.append("Compare multiple sources")
            opportunities.append("Extract and summarize information")

        if plan.task_type == TaskType.ONLINE_TRANSACTION:
            opportunities.append("Quick checkout with saved payment")
            opportunities.append("Price comparison across retailers")

        return opportunities[:3]

    async def get_next_step_guidance(
        self,
        workflow_id: str,
        current_page_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get guidance for the next step"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}

        plan = self.active_workflows[workflow_id]
        current_step_idx = plan.current_step

        if current_step_idx >= len(plan.steps):
            return {"completed": True, "message": "Workflow completed"}

        current_step = plan.steps[current_step_idx]

        guidance = {
            "step_number": current_step.step_number,
            "step_id": current_step.step_id,
            "title": current_step.title,
            "description": f"Complete: {current_step.completion_criteria}",
            "required_actions": current_step.required_actions,
            "tips": self._get_step_tips(current_step),
            "next_url": plan.steps[current_step_idx + 1].url if current_step_idx + 1 < len(plan.steps) else None,
            "estimated_remaining_time": self._calculate_remaining_time(plan, current_step_idx)
        }

        return guidance

    def _get_step_tips(self, step: WorkflowStep) -> List[str]:
        """Get helpful tips for completing a step"""
        tips_map = {
            "fill_personal_info": [
                "Start with your first name",
                "Use your full legal name if required",
                "Verify date format before entering"
            ],
            "enter_shipping": [
                "Double-check your address",
                "Include apartment/suite number if applicable",
                "Verify postal code format"
            ],
            "select_payment": [
                "Choose your preferred payment method",
                "Ensure billing address matches",
                "Save for future transactions"
            ],
        }

        for action in step.required_actions:
            if action in tips_map:
                return tips_map[action]

        return ["Take your time with this step", "Review your input carefully"]

    def _calculate_remaining_time(self, plan: WorkflowPlan, current_idx: int) -> float:
        """Calculate estimated remaining time"""
        remaining_steps = plan.steps[current_idx + 1:]
        return sum(s.estimated_time for s in remaining_steps)

    async def mark_step_complete(
        self,
        workflow_id: str,
        step_id: str,
        actual_time: float
    ) -> Dict[str, Any]:
        """Mark a step as completed"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}

        plan = self.active_workflows[workflow_id]

        for step in plan.steps:
            if step.step_id == step_id:
                step.is_completed = True
                step.actual_time = actual_time
                plan.current_step += 1

                return {
                    "step_completed": step_id,
                    "progress": f"{plan.current_step}/{len(plan.steps)}",
                    "time_saved": max(0, step.estimated_time - actual_time),
                    "next_step": self.active_workflows[workflow_id].steps[plan.current_step].step_id
                    if plan.current_step < len(plan.steps) else None
                }

        return {"error": "Step not found"}

    async def suggest_acceleration(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Suggest ways to accelerate workflow completion"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}

        plan = self.active_workflows[workflow_id]
        suggestions = []

        # Check for optimization opportunities
        for strategy_name, strategy_func in self.optimization_strategies.items():
            is_applicable = strategy_func(plan)
            if is_applicable:
                suggestions.append({
                    "strategy": strategy_name,
                    "description": self._get_strategy_description(strategy_name),
                    "estimated_time_saved": self._estimate_time_saved(plan, strategy_name)
                })

        return {
            "workflow_id": workflow_id,
            "current_step": plan.current_step,
            "suggestions": suggestions,
            "total_potential_savings": sum(s["estimated_time_saved"] for s in suggestions)
        }

    def _apply_auto_fill(self, plan: WorkflowPlan) -> bool:
        """Check if auto-fill is applicable"""
        return plan.task_type == TaskType.FORM_COMPLETION

    def _apply_predictive_navigation(self, plan: WorkflowPlan) -> bool:
        """Check if predictive navigation is applicable"""
        return plan.task_type in [TaskType.INFORMATION_RETRIEVAL, TaskType.ONLINE_TRANSACTION]

    def _apply_parallel_searches(self, plan: WorkflowPlan) -> bool:
        """Check if parallel searches are applicable"""
        return plan.task_type == TaskType.INFORMATION_RETRIEVAL and len(plan.steps) > 4

    def _apply_shortcut_suggestions(self, plan: WorkflowPlan) -> bool:
        """Check if shortcut suggestions are applicable"""
        return True

    def _get_strategy_description(self, strategy: str) -> str:
        """Get description of strategy"""
        descriptions = {
            "auto_fill": "Pre-fill form fields with saved data",
            "predictive_navigation": "Suggest next page before you click",
            "parallel_searches": "Search multiple sources simultaneously",
            "shortcut_suggestions": "Keyboard shortcuts for common actions",
        }
        return descriptions.get(strategy, strategy)

    def _estimate_time_saved(self, plan: WorkflowPlan, strategy: str) -> float:
        """Estimate time saved by strategy"""
        strategy_savings = {
            "auto_fill": 30.0,
            "predictive_navigation": 15.0,
            "parallel_searches": 20.0,
            "shortcut_suggestions": 10.0,
        }
        return strategy_savings.get(strategy, 10.0)

    def _serialize_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Serialize step for JSON output"""
        return {
            "step_id": step.step_id,
            "step_number": step.step_number,
            "title": step.title,
            "required_actions": step.required_actions,
            "estimated_time": step.estimated_time
        }

    def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow progress"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}

        plan = self.active_workflows[workflow_id]
        completed_time = sum(s.actual_time for s in plan.steps if s.is_completed)

        return {
            "workflow_id": workflow_id,
            "task_type": plan.task_type.value,
            "progress": f"{plan.current_step}/{len(plan.steps)}",
            "percentage": (plan.current_step / len(plan.steps)) * 100,
            "elapsed_time": completed_time,
            "estimated_remaining": plan.total_time_estimate - completed_time,
            "current_step": self._serialize_step(plan.steps[plan.current_step]) if plan.current_step < len(plan.steps) else None
        }

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            "active_workflows": len(self.active_workflows),
            "completed_workflows": 0,  # Track separately in production
            "total_steps_created": sum(len(p.steps) for p in self.active_workflows.values()),
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = WorkflowAgent()

        # Test task analysis
        result = await agent.analyze_task(
            user_id="user_001",
            current_url="https://example.com/form",
            page_content="<form><input name='email'><button>Submit</button></form>",
            user_goal="Fill out the registration form"
        )

        print(f"Task analysis result:")
        print(f"  Task type: {result['task_type']}")
        print(f"  Steps: {result['steps_count']}")
        print(f"  Estimated time: {result['estimated_total_time']:.0f}s")
        print(f"  Opportunities: {result['optimization_opportunities']}")

        # Test step guidance
        if result['workflow_id']:
            guidance = await agent.get_next_step_guidance(
                result['workflow_id'],
                {}
            )
            print(f"\nStep guidance:")
            print(f"  Step: {guidance['title']}")
            print(f"  Actions: {guidance['required_actions']}")

    asyncio.run(test())
