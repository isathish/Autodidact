"""
Phase 4, Milestone M4.3 — Self-Improvement Loop
Continuously improves the agent's world model, goals, and learning parameters.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from goals.generator import GoalGenerator
from meta.optimizer import MetaOptimizer


class SelfImprovementLoop:
    def __init__(self, agent_class, env_fn, world_model=None, knowledge_graph=None):
        self.agent_class = agent_class
        self.env_fn = env_fn
        self.world_model = world_model
        self.knowledge_graph = knowledge_graph
        self.goal_generator = GoalGenerator()
        self.meta_optimizer = MetaOptimizer()

    def run_iteration(self):
        # Generate a goal
        goal = self.goal_generator.generate_goal(self.world_model, self.knowledge_graph)
        print(f"Generated Goal: {goal}")

        # Optimize learning parameters
        best_params = self.meta_optimizer.suggest_new_params()
        print(f"Using parameters: {best_params}")

        # Train agent with new parameters
        agent = self.agent_class(**best_params)
        reward = self.env_fn(agent)
        print(f"Reward from environment: {reward}")

        # Update meta-learning history
        self.meta_optimizer.history.append((best_params["lr"], reward))

    def run_loop(self, iterations=10):
        for i in range(iterations):
            print(f"\n--- Iteration {i+1} ---")
            self.run_iteration()


if __name__ == "__main__":
    # Dummy test
    def dummy_env(agent):
        return 1.0  # Simulated reward

    class DummyAgent:
        def __init__(self, lr):
            self.lr = lr

    loop = SelfImprovementLoop(DummyAgent, dummy_env)
    loop.run_loop(3)
