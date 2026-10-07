# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class TimingMonitor:
    """
    A utility class for monitoring and reporting execution times of different components.
    """

    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        self.cumulative_times = defaultdict(float)

    def start(self, component_name: str):
        """Start timing a component."""
        self.start_times[component_name] = time.time()

    def end(self, component_name: str) -> float:
        """
        End timing a component and return the elapsed time.

        Args:
            component_name: Name of the component being timed

        Returns:
            Elapsed time in seconds
        """
        if component_name not in self.start_times:
            raise ValueError(f"Timer for '{component_name}' was not started.")

        elapsed = time.time() - self.start_times[component_name]
        self.timings[component_name].append(elapsed)
        self.cumulative_times[component_name] += elapsed
        del self.start_times[component_name]
        return elapsed

    def get_stats(self, component_name: Optional[str] = None) -> Dict:
        """
        Get timing statistics for a component or all components.

        Args:
            component_name: Name of the component. If None, returns stats for all components.

        Returns:
            Dictionary containing timing statistics
        """
        if component_name:
            timings = self.timings[component_name]
            if not timings:
                return {}
            return {
                "count": len(timings),
                "total": sum(timings),
                "mean": np.mean(timings),
                "std": np.std(timings),
                "min": min(timings),
                "max": max(timings),
                "median": np.median(timings),
            }
        else:
            return {name: self.get_stats(name) for name in self.timings.keys()}

    def get_cumulative_times(self) -> Dict[str, float]:
        """Get cumulative times for all components."""
        return dict(self.cumulative_times)

    def reset(self):
        """Reset all timing data."""
        self.timings = defaultdict(list)
        self.start_times = {}
        self.cumulative_times = defaultdict(float)

    def print_report(self, component_name: Optional[str] = None):
        """Print a formatted timing report."""
        stats = self.get_stats(component_name)

        if component_name:
            print(f"\n=== Timing Report for '{component_name}' ===")
            if stats:
                print(f"  Count: {stats['count']}")
                print(f"  Total: {stats['total']:.4f}s")
                print(f"  Mean: {stats['mean']:.4f}s")
                print(f"  Std: {stats['std']:.4f}s")
                print(f"  Min: {stats['min']:.4f}s")
                print(f"  Max: {stats['max']:.4f}s")
                print(f"  Median: {stats['median']:.4f}s")
            else:
                print("  No timing data available.")
        else:
            print("\n=== Overall Timing Report ===")
            cumulative = self.get_cumulative_times()
            total_time = sum(cumulative.values())

            for name, times in stats.items():
                if times:
                    percentage = (
                        (cumulative[name] / total_time * 100) if total_time > 0 else 0
                    )
                    print(f"\n{name}:")
                    print(f"  Count: {times['count']}")
                    print(f"  Total: {times['total']:.4f}s ({percentage:.1f}%)")
                    print(f"  Mean: {times['mean']:.4f}s")
                    print(f"  Std: {times['std']:.4f}s")
                    print(f"  Min: {times['min']:.4f}s")
                    print(f"  Max: {times['max']:.4f}s")
                    print(f"  Median: {times['median']:.4f}s")

            print(f"\nTotal evaluation time: {total_time:.4f}s")
