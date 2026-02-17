"""
Interface Agent for Dynamic UI Adaptation
Generates and applies CSS/DOM transformations based on cognitive load.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of interface adaptations"""
    HIDE_ELEMENTS = "hide_elements"
    SHOW_ELEMENTS = "show_elements"
    REORDER_ELEMENTS = "reorder_elements"
    CHANGE_STYLING = "change_styling"
    HIGHLIGHT_ELEMENT = "highlight_element"
    SIMPLIFY_FORM = "simplify_form"
    COMPRESS_LAYOUT = "compress_layout"
    EXPAND_LAYOUT = "expand_layout"
    CHANGE_TYPOGRAPHY = "change_typography"
    ADJUST_COLORS = "adjust_colors"


@dataclass
class StyleTransform:
    """CSS style transformation"""
    selector: str
    property: str
    value: str
    duration: float = 0.3  # animation duration in seconds


@dataclass
class DOMTransform:
    """DOM manipulation transformation"""
    element_id: str
    action: str  # hide, show, remove, append
    target_parent: Optional[str] = None
    data_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class AdaptationEffect:
    """Result of an adaptation"""
    adaptation_id: str
    adaptation_type: AdaptationType
    selector: str
    css_changes: Dict[str, str] = field(default_factory=dict)
    dom_changes: List[DOMTransform] = field(default_factory=list)
    expected_load_reduction: float = 0.0
    reversible: bool = True

    def to_css(self) -> str:
        """Generate CSS for this adaptation"""
        if not self.selector:
            return ""
        css_rules = [f"{self.selector} {{"]
        for prop, value in self.css_changes.items():
            css_rules.append(f"  {prop}: {value};")
        css_rules.append("}")
        return "\n".join(css_rules)


class InterfaceAgent:
    """
    Dynamically adapts web interfaces based on cognitive load and behavior patterns.
    Generates CSS/JavaScript transformations without modifying original HTML.
    """

    def __init__(self, agent_id: str = "interface_agent"):
        self.agent_id = agent_id
        self.active_adaptations: Dict[str, List[AdaptationEffect]] = {}
        self.style_cache: Dict[str, str] = {}
        self.dom_injection_counter = 0
        self.adaptation_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.css_prefix = "agentic-ux"

        # Predefined adaptation strategies
        self.adaptation_strategies = {
            "high_cognitive_load": self._strategy_high_load,
            "moderate_load": self._strategy_moderate_load,
            "low_load": self._strategy_low_load,
            "form_completion": self._strategy_form_completion,
            "information_retrieval": self._strategy_info_retrieval,
            "multi_step_task": self._strategy_multi_step,
        }

        # Color schemes for different load levels
        self.color_schemes = {
            "high_load": {
                "primary": "#2196F3",
                "secondary": "#E3F2FD",
                "highlight": "#1976D2",
                "background": "#FFFFFF"
            },
            "moderate_load": {
                "primary": "#1976D2",
                "secondary": "#E8EAF6",
                "highlight": "#1565C0",
                "background": "#FAFAFA"
            },
            "low_load": {
                "primary": "#0D47A1",
                "secondary": "#F3E5F5",
                "highlight": "#1565C0",
                "background": "#FFFFFF"
            }
        }

    async def adapt_interface(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str,
        page_url: str,
        custom_adaptations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate adaptations for the interface based on context.
        Returns CSS/JS code to inject and DOM operations.
        """
        logger.info(f"Adapting interface for {user_id}: load={cognitive_load}, task={task_type}")

        if user_id not in self.active_adaptations:
            self.active_adaptations[user_id] = []

        # Determine load level
        if cognitive_load > 75:
            load_level = "high_cognitive_load"
        elif cognitive_load > 50:
            load_level = "moderate_load"
        else:
            load_level = "low_load"

        # Get strategy
        strategy_key = custom_adaptations[0] if custom_adaptations else task_type.lower()
        if strategy_key not in self.adaptation_strategies:
            strategy_key = load_level

        strategy = self.adaptation_strategies.get(
            strategy_key,
            self.adaptation_strategies[load_level]
        )

        # Generate adaptations
        effects = await strategy(user_id, cognitive_load, task_type)

        # Store active adaptations
        self.active_adaptations[user_id].extend(effects)

        # Generate output
        output = {
            "user_id": user_id,
            "adaptations_applied": len(effects),
            "css_injection": self._generate_css_bundle(effects),
            "dom_operations": self._generate_dom_operations(effects),
            "animation_duration": 0.3,
            "load_level": load_level,
            "expected_benefit": self._calculate_expected_benefit(effects),
        }

        # Log adaptation
        self._log_adaptation(user_id, effects, output)

        return output

    async def _strategy_high_load(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str
    ) -> List[AdaptationEffect]:
        """Aggressive adaptations for high cognitive load"""
        effects = []

        # 1. Hide non-essential elements
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("hide-sidebar"),
            adaptation_type=AdaptationType.HIDE_ELEMENTS,
            selector=".sidebar, .related-products, .recommendations, .ads",
            css_changes={
                "display": "none !important"
            },
            expected_load_reduction=15.0,
            reversible=True
        ))

        # 2. Simplify forms - show only required fields
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("simplify-form"),
            adaptation_type=AdaptationType.SIMPLIFY_FORM,
            selector="form",
            css_changes={
                "max-width": "100%",
                "padding": "20px"
            },
            expected_load_reduction=10.0,
            reversible=True
        ))

        # 3. Increase element spacing and size
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("increase-spacing"),
            adaptation_type=AdaptationType.CHANGE_STYLING,
            selector="button, input, a.primary",
            css_changes={
                "padding": "12px 24px",
                "font-size": "16px",
                "margin": "12px 0",
                "min-height": "44px"
            },
            expected_load_reduction=8.0,
            reversible=True
        ))

        # 4. Highlight primary actions
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("highlight-primary"),
            adaptation_type=AdaptationType.HIGHLIGHT_ELEMENT,
            selector="button.primary, button[type='submit']",
            css_changes={
                "background-color": "#2196F3",
                "color": "white",
                "border": "2px solid #1976D2",
                "box-shadow": "0 2px 8px rgba(33, 150, 243, 0.3)",
                "font-weight": "bold"
            },
            expected_load_reduction=7.0,
            reversible=True
        ))

        # 5. Improve typography for readability
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("improve-typography"),
            adaptation_type=AdaptationType.CHANGE_TYPOGRAPHY,
            selector="body, p, label",
            css_changes={
                "font-size": "16px",
                "line-height": "1.6",
                "letter-spacing": "0.5px"
            },
            expected_load_reduction=5.0,
            reversible=True
        ))

        return effects

    async def _strategy_moderate_load(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str
    ) -> List[AdaptationEffect]:
        """Moderate adaptations for balanced load"""
        effects = []

        # 1. Organize layout into clear sections
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("section-layout"),
            adaptation_type=AdaptationType.COMPRESS_LAYOUT,
            selector=".content",
            css_changes={
                "max-width": "900px",
                "margin": "0 auto",
                "padding": "20px"
            },
            expected_load_reduction=5.0,
            reversible=True
        ))

        # 2. Improve visual hierarchy
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("visual-hierarchy"),
            adaptation_type=AdaptationType.CHANGE_STYLING,
            selector="h1, h2, h3",
            css_changes={
                "font-weight": "bold",
                "margin-top": "20px",
                "margin-bottom": "10px",
                "color": "#1976D2"
            },
            expected_load_reduction=4.0,
            reversible=True
        ))

        # 3. Add visual grouping to form fields
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("group-fields"),
            adaptation_type=AdaptationType.CHANGE_STYLING,
            selector="input, select, textarea",
            css_changes={
                "border": "1px solid #BDBDBD",
                "border-radius": "4px",
                "padding": "10px",
                "margin": "8px 0",
                "font-size": "14px"
            },
            expected_load_reduction=3.0,
            reversible=True
        ))

        return effects

    async def _strategy_low_load(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str
    ) -> List[AdaptationEffect]:
        """Minimal adaptations for low cognitive load"""
        effects = []

        # 1. Subtle enhancement
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("subtle-enhance"),
            adaptation_type=AdaptationType.CHANGE_STYLING,
            selector="button:hover",
            css_changes={
                "background-color": "#E3F2FD",
                "transition": "background-color 0.2s ease"
            },
            expected_load_reduction=1.0,
            reversible=True
        ))

        return effects

    async def _strategy_form_completion(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str
    ) -> List[AdaptationEffect]:
        """Specialized adaptations for form tasks"""
        effects = []

        # Show progress indicator
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("progress-indicator"),
            adaptation_type=AdaptationType.SHOW_ELEMENTS,
            selector=".form-progress",
            css_changes={
                "display": "block",
                "margin-bottom": "20px",
                "height": "4px",
                "background-color": "#E0E0E0",
                "border-radius": "2px"
            },
            expected_load_reduction=6.0,
            reversible=True
        ))

        # Inline validation
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("inline-validation"),
            adaptation_type=AdaptationType.CHANGE_STYLING,
            selector="input.error",
            css_changes={
                "border-color": "#F44336",
                "background-color": "#FFEBEE"
            },
            expected_load_reduction=4.0,
            reversible=True
        ))

        return effects

    async def _strategy_info_retrieval(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str
    ) -> List[AdaptationEffect]:
        """Specialized adaptations for information retrieval tasks"""
        effects = []

        # Highlight search results
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("highlight-results"),
            adaptation_type=AdaptationType.HIGHLIGHT_ELEMENT,
            selector=".search-result",
            css_changes={
                "border-left": "4px solid #2196F3",
                "padding-left": "16px",
                "margin": "12px 0"
            },
            expected_load_reduction=5.0,
            reversible=True
        ))

        return effects

    async def _strategy_multi_step(
        self,
        user_id: str,
        cognitive_load: float,
        task_type: str
    ) -> List[AdaptationEffect]:
        """Specialized adaptations for multi-step tasks"""
        effects = []

        # Show step indicator
        effects.append(AdaptationEffect(
            adaptation_id=self._gen_id("step-indicator"),
            adaptation_type=AdaptationType.SHOW_ELEMENTS,
            selector=".step-indicator",
            css_changes={
                "display": "flex",
                "justify-content": "space-between",
                "margin-bottom": "20px"
            },
            expected_load_reduction=8.0,
            reversible=True
        ))

        return effects

    def revert_adaptations(self, user_id: str) -> Dict[str, Any]:
        """Revert all adaptations for user"""
        if user_id in self.active_adaptations:
            count = len(self.active_adaptations[user_id])
            self.active_adaptations[user_id] = []
            return {"reverted": count}
        return {"reverted": 0}

    def _generate_css_bundle(self, effects: List[AdaptationEffect]) -> str:
        """Generate combined CSS for all effects"""
        css_parts = []

        for effect in effects:
            if effect.css_changes:
                css_parts.append(effect.to_css())

        # Add animations
        css_parts.append(self._get_animations_css())

        return "\n".join(css_parts)

    def _generate_dom_operations(self, effects: List[AdaptationEffect]) -> List[Dict[str, Any]]:
        """Generate DOM operations"""
        operations = []

        for effect in effects:
            for dom_change in effect.dom_changes:
                operations.append(asdict(dom_change))

        return operations

    def _get_animations_css(self) -> str:
        """Get CSS animations"""
        return """
@keyframes fadeInAdapt {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideInAdapt {
  from { transform: translateY(10px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.agentic-ux-animate-in {
  animation: fadeInAdapt 0.3s ease-in-out;
}
"""

    def _calculate_expected_benefit(self, effects: List[AdaptationEffect]) -> float:
        """Calculate total expected cognitive load reduction"""
        return sum(e.expected_load_reduction for e in effects)

    def _gen_id(self, prefix: str) -> str:
        """Generate unique ID"""
        self.dom_injection_counter += 1
        return f"{self.css_prefix}-{prefix}-{self.dom_injection_counter}"

    def _log_adaptation(
        self,
        user_id: str,
        effects: List[AdaptationEffect],
        output: Dict[str, Any]
    ) -> None:
        """Log adaptation for analytics"""
        log_entry = {
            "timestamp": str(__import__("datetime").datetime.utcnow()),
            "user_id": user_id,
            "effects_count": len(effects),
            "expected_benefit": output["expected_benefit"],
            "load_level": output["load_level"],
        }
        self.adaptation_history.append(log_entry)

        if len(self.adaptation_history) > self.max_history_size:
            self.adaptation_history = self.adaptation_history[-self.max_history_size:]

    def get_adaptations(self, user_id: str) -> List[AdaptationEffect]:
        """Get active adaptations for user"""
        return self.active_adaptations.get(user_id, [])

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        total_adaptations = sum(
            len(effects) for effects in self.active_adaptations.values()
        )

        return {
            "active_users": len(self.active_adaptations),
            "total_active_adaptations": total_adaptations,
            "history_size": len(self.adaptation_history),
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = InterfaceAgent()

        # Test high load adaptation
        result = await agent.adapt_interface(
            user_id="user_001",
            cognitive_load=85.0,
            task_type="form_completion",
            page_url="https://example.com/form"
        )

        print(f"Adaptations applied: {result['adaptations_applied']}")
        print(f"Expected benefit: {result['expected_benefit']:.1f}%")
        print(f"\nCSS (first 500 chars):\n{result['css_injection'][:500]}")

        # Test revert
        revert = agent.revert_adaptations("user_001")
        print(f"\nReverted {revert['reverted']} adaptations")

        # Metrics
        metrics = agent.get_agent_metrics()
        print(f"\nAgent metrics: {metrics}")

    asyncio.run(test())
