"""
Phase 2, Milestone M2.2 — Recurrent State-Space Model (RSSM)
Predicts next state and perception from current state and action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, state_dim=128, action_dim=6, hidden_dim=256, perception_dim=512):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Recurrent core
        self.rnn = nn.GRUCell(state_dim + action_dim, hidden_dim)

        # State prediction head
        self.state_head = nn.Linear(hidden_dim, state_dim)

        # Perception prediction head
        self.perception_head = nn.Linear(hidden_dim, perception_dim)

    def forward(self, state, action):
        """
        state: Tensor (batch, state_dim)
        action: Tensor (batch, action_dim) one-hot
        """
        x = torch.cat([state, action], dim=-1)
        h = self.rnn(x, state)
        next_state = self.state_head(h)
        predicted_perception = self.perception_head(h)
        return next_state, predicted_perception


if __name__ == "__main__":
    model = RSSM()
    state = torch.randn(1, 128)
    action = torch.zeros(1, 6)
    action[0, 2] = 1  # Example one-hot action
    next_state, pred_perception = model(state, action)
    print("Next state shape:", next_state.shape)
    print("Predicted perception shape:", pred_perception.shape)
