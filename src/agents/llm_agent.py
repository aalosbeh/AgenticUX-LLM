"""Optional LLM-backed agent with deterministic mock fallback."""

import json
import os
from typing import Dict, Any


class LLMAgent:
    VALID_ACTIONS = {
        "simplify_ui",
        "highlight_relevant",
        "no_change",
        "request_human_review",
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.mode = "mock"
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.mode = "openai"
            except Exception:
                self.mode = "mock"

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _mock_response(self, features: Dict[str, Any]) -> Dict[str, Any]:
        error_count = self._to_float(features.get("error_count", 0), 0.0)
        task_complexity = self._to_float(features.get("task_complexity", 0.5), 0.5)
        time_pressure = self._to_float(features.get("time_pressure", 0.5), 0.5)
        anomaly_score = self._to_float(features.get("anomaly_score", 0.0), 0.0)
        predicted_cognitive_load = self._to_float(
            features.get("predicted_cognitive_load", features.get("cognitive_load", 50.0)),
            50.0,
        )

        if anomaly_score >= 0.9 and error_count >= 2:
            action = "request_human_review"
            confidence = 0.9
            reasoning = "Deterministic mock: severe anomaly and error pattern."
        elif error_count >= 3 or task_complexity > 0.8 or predicted_cognitive_load >= 75:
            action = "simplify_ui"
            confidence = 0.75
            reasoning = "Deterministic mock: elevated predicted load or complexity."
        elif time_pressure > 0.7:
            action = "highlight_relevant"
            confidence = 0.7
            reasoning = "Deterministic mock: elevated time pressure."
        else:
            action = "no_change"
            confidence = 0.65
            reasoning = "Deterministic mock: no high-risk pattern detected."
        return {
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "mode": "mock",
        }

    def analyze_user_state(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode != "openai" or self.client is None:
            return self._mock_response(features)

        prompt = (
            "You are a UX adaptation policy module. "
            "Given user-state features, decide one action.\n"
            "Allowed actions: simplify_ui, highlight_relevant, no_change, request_human_review.\n"
            "Output STRICT JSON only with keys: action, reasoning, confidence.\n"
            "confidence must be numeric in [0,1]. No markdown, no extra text."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(features)},
            ],
        )
        try:
            parsed = json.loads(resp.choices[0].message.content)
        except Exception:
            return {
                "action": "request_human_review",
                "reasoning": "OpenAI response was not valid JSON.",
                "confidence": 0.0,
                "mode": "openai",
            }

        action = parsed.get("action", "request_human_review")
        if action not in self.VALID_ACTIONS:
            action = "request_human_review"
        confidence = self._to_float(parsed.get("confidence", 0.0), 0.0)
        confidence = min(1.0, max(0.0, confidence))
        return {
            "action": action,
            "reasoning": str(parsed.get("reasoning", "")),
            "confidence": confidence,
            "mode": "openai",
        }
