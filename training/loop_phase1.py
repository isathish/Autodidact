"""
Phase 1, Milestone M1.4 — First Learning Loop: Change Detection Reward
Integrates perception, motor, and RL agent to learn basic browser interaction.
"""

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../perception")))
from perception.vision_cnn import VisionCNN
from perception.dom_gnn import DOMGNN
from motor.actions import MotorSystem
from rl.ppo_agent import PPOAgent
from core.rewards import RewardSystem


def frame_difference(vec1, vec2):
    return torch.norm(vec1 - vec2).item()


def run_phase1_episode():
    # Initialize components
    vision = VisionCNN()
    dom_encoder = DOMGNN()
    motor = MotorSystem(headless=True)
    agent = PPOAgent(input_dim=512, action_dim=6)
    rewards = RewardSystem()

    # Example: navigate to a nursery site
    motor.navigate("https://example.com")

    # Dummy perception vectors (replace with real pixel+DOM extraction)
    prev_state = torch.randn(1, 512)
    total_reward = 0

    for step in range(5):
        action, log_prob, value = agent.select_action(prev_state)
        # Map action index to motor command
        if action == 0:
            motor.scroll(500)
        elif action == 1:
            motor.scroll(-500)
        elif action == 2:
            motor.click(100, 200)
        elif action == 3:
            motor.type_text("hello")
        elif action == 4:
            motor.press_enter()
        elif action == 5:
            motor.go_back()

        # Simulate new perception
        new_state = torch.randn(1, 512)

        # Reward: change detection
        change = frame_difference(prev_state, new_state)
        reward = change  # Positive if screen changes
        total_reward += reward

        prev_state = new_state

    motor.close()
    return total_reward


if __name__ == "__main__":
    total = run_phase1_episode()
    print("Total reward from episode:", total)
