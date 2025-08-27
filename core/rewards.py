"""
Reward functions for Project Autodidact.
Implements intrinsic and extrinsic rewards:
- Curiosity Reward
- Goal Success Reward
- Computational Cost Penalty
"""

import math
import time
from typing import Any, Dict


class RewardSystem:
    def __init__(self):
        self.last_compute_time = None

    def curiosity_reward(self, predicted_prob: float) -> float:
        """
        Curiosity reward: -log(P_predicted)
        Higher reward for surprising events (low predicted probability).
        """
        if predicted_prob <= 0:
            predicted_prob = 1e-9
        return -math.log(predicted_prob)

    def goal_success_reward(self, success: bool) -> float:
        """
        Goal success reward: +1 for achieving a goal, else 0.
        """
        return 1.0 if success else 0.0

    def compute_cost_penalty(self, compute_time: float) -> float:
        """
        Computational cost penalty: -0.01 * compute_time (seconds).
        """
        return -0.01 * compute_time

    def start_timing(self):
        """Mark the start of a computation for cost penalty measurement."""
        self.last_compute_time = time.time()

    def end_timing_and_penalty(self) -> float:
        """Mark the end of computation and return cost penalty."""
        if self.last_compute_time is None:
            return 0.0
        elapsed = time.time() - self.last_compute_time
        self.last_compute_time = None
        return self.compute_cost_penalty(elapsed)

    def total_reward(self, predicted_prob: float, success: bool, compute_time: float) -> float:
        """
        Combine all rewards into a single scalar.
        """
        return (
            self.curiosity_reward(predicted_prob)
            + self.goal_success_reward(success)
            + self.compute_cost_penalty(compute_time)
        )


if __name__ == "__main__":
    # Example usage
    rewards = RewardSystem()
    rewards.start_timing()
    # Simulate computation
    time.sleep(0.05)
    penalty = rewards.end_timing_and_penalty()
    total = rewards.total_reward(predicted_prob=0.2, success=True, compute_time=0.05)
    print(f"Penalty: {penalty}, Total Reward: {total}")
