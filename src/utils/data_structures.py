"""
Efficient Data Structures for Streaming Analysis
Optimized containers for real-time behavioral data processing.
"""

import logging
from typing import Dict, List, Any, Optional, Generic, TypeVar
from collections import deque
from dataclasses import dataclass
import numpy as np
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class RollingWindow(Generic[T]):
    """
    Efficient rolling window for streaming data analysis.
    Maintains running statistics without storing all data.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
        self.sum = 0.0
        self.sum_squares = 0.0
        self.lock = threading.RLock()

    def add(self, value: float) -> None:
        """Add value to window"""
        with self.lock:
            # Remove oldest value if full
            if len(self.buffer) == self.window_size:
                removed = self.buffer[0]
                self.sum -= removed
                self.sum_squares -= removed ** 2

            # Add new value
            self.buffer.append(value)
            self.sum += value
            self.sum_squares += value ** 2

    def get_mean(self) -> float:
        """Get running mean"""
        with self.lock:
            if len(self.buffer) == 0:
                return 0.0
            return self.sum / len(self.buffer)

    def get_variance(self) -> float:
        """Get running variance"""
        with self.lock:
            n = len(self.buffer)
            if n < 2:
                return 0.0

            mean = self.sum / n
            variance = (self.sum_squares / n) - (mean ** 2)
            return max(0, variance)

    def get_std(self) -> float:
        """Get running standard deviation"""
        return np.sqrt(self.get_variance())

    def get_min(self) -> Optional[float]:
        """Get minimum in window"""
        with self.lock:
            return min(self.buffer) if self.buffer else None

    def get_max(self) -> Optional[float]:
        """Get maximum in window"""
        with self.lock:
            return max(self.buffer) if self.buffer else None

    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get percentile value"""
        with self.lock:
            if not self.buffer:
                return None
            return np.percentile(self.buffer, percentile)

    def clear(self) -> None:
        """Clear window"""
        with self.lock:
            self.buffer.clear()
            self.sum = 0.0
            self.sum_squares = 0.0

    def get_values(self) -> List[float]:
        """Get all values in window"""
        with self.lock:
            return list(self.buffer)

    def size(self) -> int:
        """Get current window size"""
        return len(self.buffer)


class TimeSeriesBuffer:
    """
    Efficient time series buffer for behavioral data.
    Supports fast queries and aggregations.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.timestamps: deque = deque(maxlen=max_size)
        self.values: deque = deque(maxlen=max_size)
        self.metadata: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()

    def add(self, timestamp: float, value: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add time series point"""
        with self.lock:
            self.timestamps.append(timestamp)
            self.values.append(value)
            self.metadata.append(meta or {})

    def get_range(self, start_time: float, end_time: float) -> List[Tuple[float, float]]:
        """Get values in time range"""
        with self.lock:
            result = []
            for ts, val in zip(self.timestamps, self.values):
                if start_time <= ts <= end_time:
                    result.append((ts, val))
            return result

    def get_stats(self, start_time: float, end_time: float) -> Dict[str, float]:
        """Get statistics for time range"""
        values = [val for ts, val in self.get_range(start_time, end_time)]

        if not values:
            return {}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }

    def clear(self) -> None:
        """Clear buffer"""
        with self.lock:
            self.timestamps.clear()
            self.values.clear()
            self.metadata.clear()

    def size(self) -> int:
        """Get buffer size"""
        return len(self.values)


@dataclass
class StreamingSummary:
    """Streaming summary statistics"""
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    sum_val: float = 0.0
    sum_sq: float = 0.0


class OnlineStatistics:
    """
    Welford's online algorithm for computing statistics
    without storing all data points.
    """

    def __init__(self):
        self.summary = StreamingSummary()
        self.lock = threading.RLock()

    def add(self, value: float) -> None:
        """Add value using Welford's algorithm"""
        with self.lock:
            self.summary.count += 1
            self.summary.sum_val += value
            self.summary.sum_sq += value ** 2

            # Update min/max
            self.summary.min_val = min(self.summary.min_val, value)
            self.summary.max_val = max(self.summary.max_val, value)

            # Update mean and variance
            self.summary.mean = self.summary.sum_val / self.summary.count

            if self.summary.count > 1:
                self.summary.variance = (
                    (self.summary.sum_sq - self.summary.count * self.summary.mean ** 2) /
                    (self.summary.count - 1)
                )

    def get_mean(self) -> float:
        """Get current mean"""
        with self.lock:
            return self.summary.mean

    def get_std(self) -> float:
        """Get current standard deviation"""
        with self.lock:
            return np.sqrt(max(0, self.summary.variance))

    def get_variance(self) -> float:
        """Get current variance"""
        with self.lock:
            return self.summary.variance

    def get_summary(self) -> Dict[str, float]:
        """Get all statistics"""
        with self.lock:
            return {
                "count": self.summary.count,
                "mean": self.summary.mean,
                "std": self.get_std(),
                "variance": self.summary.variance,
                "min": self.summary.min_val if self.summary.min_val != float('inf') else None,
                "max": self.summary.max_val if self.summary.max_val != float('-inf') else None,
            }

    def reset(self) -> None:
        """Reset statistics"""
        with self.lock:
            self.summary = StreamingSummary()


class LRUCache:
    """
    Least Recently Used cache for caching computation results.
    Efficient for behavioral data lookups.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: deque = deque()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recent)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size
        }


class BloomFilter:
    """
    Bloom filter for efficient membership testing.
    Useful for duplicate event detection.
    """

    def __init__(self, size: int = 10000, hash_functions: int = 3):
        self.size = size
        self.hash_functions = hash_functions
        self.bits = [False] * size

    def add(self, item: str) -> None:
        """Add item to filter"""
        for i in range(self.hash_functions):
            hash_val = hash(f"{item}_{i}") % self.size
            self.bits[hash_val] = True

    def contains(self, item: str) -> bool:
        """Check if item might be in set"""
        for i in range(self.hash_functions):
            hash_val = hash(f"{item}_{i}") % self.size
            if not self.bits[hash_val]:
                return False
        return True

    def clear(self) -> None:
        """Clear filter"""
        self.bits = [False] * self.size


# Example usage
if __name__ == "__main__":
    # Test RollingWindow
    print("Testing RollingWindow:")
    window = RollingWindow(window_size=10)

    for i in range(15):
        window.add(float(i))

    print(f"  Mean: {window.get_mean():.2f}")
    print(f"  Std: {window.get_std():.2f}")
    print(f"  Min: {window.get_min():.2f}")
    print(f"  Max: {window.get_max():.2f}")

    # Test TimeSeriesBuffer
    print("\nTesting TimeSeriesBuffer:")
    ts_buffer = TimeSeriesBuffer()

    for i in range(20):
        ts_buffer.add(float(i), float(i * 2))

    stats = ts_buffer.get_stats(5.0, 15.0)
    print(f"  Stats for 5-15: {stats}")

    # Test OnlineStatistics
    print("\nTesting OnlineStatistics:")
    online = OnlineStatistics()

    values = [1, 2, 3, 4, 5, 10, 20, 30]
    for v in values:
        online.add(float(v))

    summary = online.get_summary()
    print(f"  Summary: {summary}")

    # Test LRUCache
    print("\nTesting LRUCache:")
    cache = LRUCache(max_size=3)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)  # Evicts "a"

    print(f"  Cache: {cache.cache}")
    print(f"  Stats: {cache.stats()}")

    # Test BloomFilter
    print("\nTesting BloomFilter:")
    bloom = BloomFilter(size=100)

    bloom.add("user_001")
    bloom.add("user_002")

    print(f"  Contains user_001: {bloom.contains('user_001')}")
    print(f"  Contains user_003: {bloom.contains('user_003')}")
