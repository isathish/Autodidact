"""
Phase 1, Milestone M1.3 — PPO Reinforcement Learning Agent
Maps perception vectors to actions using Proximal Policy Optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, lr=3e-4):
        super(PPOAgent, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        probs = self.policy_net(x)
        dist = Categorical(probs)
        value = self.value_net(x)
        return dist, value

    def select_action(self, state):
        dist, value = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def update(self, trajectories, clip_epsilon=0.2, gamma=0.99, lam=0.95):
        states, actions, log_probs_old, returns, advantages = trajectories
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - log_probs_old)

        # Policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = (returns - values.squeeze()).pow(2).mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = PPOAgent(input_dim=512, action_dim=6)
    dummy_state = torch.randn(1, 512)
    action, log_prob, value = agent.select_action(dummy_state)
    print("Action:", action, "Value:", value.item())
