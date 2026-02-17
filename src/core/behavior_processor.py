"""
Real-time Behavior Processing Pipeline
Processes and aggregates behavioral data from multiple sources.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RawEvent:
    """Raw behavioral event from browser/sensors"""
    event_type: str  # 'click', 'scroll', 'keypress', 'error', etc.
    user_id: str
    timestamp: str
    data: Dict[str, Any]
    source: str = "browser"


@dataclass
class ProcessedMetrics:
    """Aggregated behavioral metrics"""
    timestamp: str
    user_id: str
    window_size: int  # seconds
    click_count: int = 0
    scroll_count: int = 0
    key_presses: int = 0
    errors: int = 0
    page_changes: int = 0
    idle_time: float = 0.0  # seconds
    avg_time_between_actions: float = 0.0
    mouse_distance: float = 0.0  # total pixels moved
    avg_mouse_speed: float = 0.0  # pixels/second


class BehaviorProcessor:
    """
    Real-time pipeline for processing behavioral data.
    Collects events, aggregates metrics, and streams results.
    """

    def __init__(self, window_size: int = 5, max_buffer_size: int = 10000):
        self.window_size = window_size  # seconds
        self.max_buffer_size = max_buffer_size
        self.event_buffer: Dict[str, deque] = {}
        self.metrics_buffer: Dict[str, deque] = {}
        self.subscribers: List[callable] = []
        self.lock = threading.RLock()
        self.processing_thread = None
        self.is_running = False

    def add_event(self, event: RawEvent) -> None:
        """Add event to processing pipeline"""
        with self.lock:
            if event.user_id not in self.event_buffer:
                self.event_buffer[event.user_id] = deque(maxlen=self.max_buffer_size)
                self.metrics_buffer[event.user_id] = deque(maxlen=100)

            self.event_buffer[event.user_id].append(event)

    def process_batch(self, user_id: str) -> Optional[ProcessedMetrics]:
        """Process events for user and generate metrics"""
        with self.lock:
            if user_id not in self.event_buffer or len(self.event_buffer[user_id]) == 0:
                return None

            events = list(self.event_buffer[user_id])

            # Filter events within window
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(seconds=self.window_size)

            window_events = [
                e for e in events
                if datetime.fromisoformat(e.timestamp) > cutoff_time
            ]

            if not window_events:
                return None

            # Process metrics
            metrics = self._calculate_metrics(user_id, window_events, self.window_size)
            self.metrics_buffer[user_id].append(metrics)

            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(metrics)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")

            return metrics

    def _calculate_metrics(
        self,
        user_id: str,
        events: List[RawEvent],
        window_size: int
    ) -> ProcessedMetrics:
        """Calculate aggregated metrics from events"""
        metrics = ProcessedMetrics(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            window_size=window_size
        )

        # Count event types
        for event in events:
            if event.event_type == "click":
                metrics.click_count += 1
            elif event.event_type == "scroll":
                metrics.scroll_count += 1
            elif event.event_type == "keypress":
                metrics.key_presses += 1
            elif event.event_type == "error":
                metrics.errors += 1
            elif event.event_type == "pagechange":
                metrics.page_changes += 1

        # Calculate timing metrics
        if len(events) > 1:
            timestamps = [datetime.fromisoformat(e.timestamp) for e in events]
            time_diffs = np.diff([ts.timestamp() for ts in timestamps])

            metrics.avg_time_between_actions = float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0.0

            # Idle time (gaps > 5 seconds)
            idle_gaps = [d for d in time_diffs if d > 5.0]
            metrics.idle_time = float(np.sum(idle_gaps)) if idle_gaps else 0.0

        # Calculate mouse metrics
        mouse_events = [e for e in events if e.event_type == "mousemove"]
        if len(mouse_events) > 1:
            distances = []
            for i in range(1, len(mouse_events)):
                prev_x = mouse_events[i - 1].data.get("x", 0)
                prev_y = mouse_events[i - 1].data.get("y", 0)
                curr_x = mouse_events[i].data.get("x", 0)
                curr_y = mouse_events[i].data.get("y", 0)

                distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                distances.append(distance)

            metrics.mouse_distance = float(np.sum(distances))
            metrics.avg_mouse_speed = metrics.mouse_distance / window_size if window_size > 0 else 0.0

        return metrics

    def subscribe(self, callback: callable) -> None:
        """Subscribe to processed metrics"""
        self.subscribers.append(callback)

    def get_recent_metrics(self, user_id: str, count: int = 10) -> List[ProcessedMetrics]:
        """Get recent processed metrics for user"""
        with self.lock:
            if user_id not in self.metrics_buffer:
                return []
            return list(self.metrics_buffer[user_id])[-count:]

    def get_aggregated_stats(self, user_id: str, window_count: int = 5) -> Dict[str, Any]:
        """Get aggregated statistics across recent windows"""
        recent_metrics = self.get_recent_metrics(user_id, window_count)

        if not recent_metrics:
            return {}

        return {
            "user_id": user_id,
            "windows_analyzed": len(recent_metrics),
            "total_clicks": sum(m.click_count for m in recent_metrics),
            "total_errors": sum(m.errors for m in recent_metrics),
            "avg_idle_time": np.mean([m.idle_time for m in recent_metrics]),
            "avg_mouse_speed": np.mean([m.avg_mouse_speed for m in recent_metrics]),
            "total_mouse_distance": sum(m.mouse_distance for m in recent_metrics),
            "error_rate": sum(m.errors for m in recent_metrics) / max(1, sum(m.click_count for m in recent_metrics))
        }

    def clear_user_data(self, user_id: str) -> None:
        """Clear data for user"""
        with self.lock:
            if user_id in self.event_buffer:
                del self.event_buffer[user_id]
            if user_id in self.metrics_buffer:
                del self.metrics_buffer[user_id]

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        with self.lock:
            total_events = sum(len(b) for b in self.event_buffer.values())
            total_metrics = sum(len(b) for b in self.metrics_buffer.values())

            return {
                "active_users": len(self.event_buffer),
                "total_buffered_events": total_events,
                "total_metrics": total_metrics,
                "window_size": self.window_size,
                "subscribers": len(self.subscribers)
            }


class StreamingAggregator:
    """
    Specialized aggregator for streaming behavioral analysis.
    Maintains running statistics without storing all events.
    """

    def __init__(self, user_id: str, buffer_size: int = 1000):
        self.user_id = user_id
        self.buffer_size = buffer_size
        self.event_queue: deque = deque(maxlen=buffer_size)
        self.stats = {
            "total_events": 0,
            "clicks": 0,
            "errors": 0,
            "last_action_time": datetime.utcnow(),
            "idle_duration": 0.0,
        }

    def add_event(self, event: RawEvent) -> Dict[str, Any]:
        """Add event and return updated stats"""
        self.event_queue.append(event)
        self.stats["total_events"] += 1

        if event.event_type == "click":
            self.stats["clicks"] += 1
        elif event.event_type == "error":
            self.stats["errors"] += 1

        # Calculate idle time
        event_time = datetime.fromisoformat(event.timestamp)
        time_since_last = (event_time - self.stats["last_action_time"]).total_seconds()
        if time_since_last > 5:
            self.stats["idle_duration"] += time_since_last

        self.stats["last_action_time"] = event_time

        return self.stats.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        events_in_buffer = len(self.event_queue)

        if events_in_buffer > 0:
            first_event_time = datetime.fromisoformat(list(self.event_queue)[0].timestamp)
            last_event_time = datetime.fromisoformat(list(self.event_queue)[-1].timestamp)
            duration = (last_event_time - first_event_time).total_seconds()
        else:
            duration = 0

        return {
            "user_id": self.user_id,
            "total_events": self.stats["total_events"],
            "events_in_buffer": events_in_buffer,
            "clicks": self.stats["clicks"],
            "errors": self.stats["errors"],
            "error_rate": self.stats["errors"] / max(1, self.stats["clicks"]),
            "idle_duration": self.stats["idle_duration"],
            "analysis_duration": duration,
            "events_per_second": self.stats["total_events"] / max(1, duration)
        }

    def reset(self) -> None:
        """Reset aggregator"""
        self.event_queue.clear()
        self.stats = {
            "total_events": 0,
            "clicks": 0,
            "errors": 0,
            "last_action_time": datetime.utcnow(),
            "idle_duration": 0.0,
        }


# Example usage
if __name__ == "__main__":
    processor = BehaviorProcessor(window_size=5)

    # Subscribe to metrics
    def on_metrics(metrics: ProcessedMetrics):
        print(f"Metrics for {metrics.user_id}: {metrics.click_count} clicks, {metrics.errors} errors")

    processor.subscribe(on_metrics)

    # Simulate events
    user_id = "user_001"

    # Create sample events
    for i in range(20):
        event = RawEvent(
            event_type="click" if i % 3 == 0 else "mousemove",
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            data={"x": 100 + i * 10, "y": 200 + i * 5} if i % 3 == 0 else {"x": 100 + i * 10, "y": 200 + i * 5},
        )
        processor.add_event(event)

    # Process
    metrics = processor.process_batch(user_id)
    if metrics:
        print(f"\nProcessed metrics:")
        print(f"  Clicks: {metrics.click_count}")
        print(f"  Avg time between actions: {metrics.avg_time_between_actions:.2f}s")
        print(f"  Avg mouse speed: {metrics.avg_mouse_speed:.1f} px/s")

    # Get aggregated stats
    stats = processor.get_aggregated_stats(user_id)
    print(f"\nAggregated stats: {stats}")

    # Test streaming aggregator
    print("\n" + "=" * 60)
    print("Testing StreamingAggregator:")

    aggregator = StreamingAggregator(user_id)

    for i in range(10):
        event = RawEvent(
            event_type="click" if i % 2 == 0 else "error" if i % 5 == 0 else "mousemove",
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            data={}
        )
        aggregator.add_event(event)

    stats = aggregator.get_statistics()
    print(f"\nStreaming stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
