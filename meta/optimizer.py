"""
Phase 4, Milestone M4.2 — Meta-Learning Optimizer
Tunes learning parameters based on intrinsic curiosity reward improvements.
"""

import random


class MetaOptimizer:
    def __init__(self, learning_rates=None):
        self.learning_rates = learning_rates or [1e-4, 3e-4, 1e-3]
        self.history = []

    def evaluate(self, agent_class, env_fn, episodes=5):
        """
        Evaluate each learning rate and return the best one.
        """
        best_lr = None
        best_score = float("-inf")
        for lr in self.learning_rates:
            agent = agent_class(lr=lr)
            total_reward = 0
            for _ in range(episodes):
                reward = env_fn(agent)
                total_reward += reward
            avg_reward = total_reward / episodes
            self.history.append((lr, avg_reward))
            if avg_reward > best_score:
                best_score = avg_reward
                best_lr = lr
        return best_lr

    def suggest_new_params(self):
        """
        Suggest new hyperparameters based on history.
        """
        if not self.history:
            return {"lr": random.choice(self.learning_rates)}
        best_lr = max(self.history, key=lambda x: x[1])[0]
        return {"lr": best_lr}


if __name__ == "__main__":
    # Dummy test
    def dummy_env(agent):
        return random.uniform(0, 1)

    class DummyAgent:
        def __init__(self, lr):
            self.lr = lr

    optimizer = MetaOptimizer()
    best = optimizer.evaluate(DummyAgent, dummy_env)
    print("Best LR:", best)
    print("Suggested params:", optimizer.suggest_new_params())
