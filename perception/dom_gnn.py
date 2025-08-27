"""
Phase 1, Milestone M1.1 — DOM GNN Encoder
Processes the DOM tree into a latent representation Z_dom.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DOMGNN(nn.Module):
    def __init__(self, node_feature_dim=64, hidden_dim=128, latent_dim=256):
        super(DOMGNN, self).__init__()
        self.fc_node = nn.Linear(node_feature_dim, hidden_dim)
        self.fc_message = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, node_features, adjacency_matrix):
        """
        node_features: Tensor of shape (num_nodes, node_feature_dim)
        adjacency_matrix: Tensor of shape (num_nodes, num_nodes)
        """
        h = F.relu(self.fc_node(node_features))
        # Simple message passing: sum of neighbor features
        messages = torch.matmul(adjacency_matrix, h)
        h = F.relu(self.fc_message(messages))
        # Global pooling (mean)
        graph_embedding = torch.mean(h, dim=0)
        z = self.fc_output(graph_embedding)
        return z


if __name__ == "__main__":
    # Example usage
    num_nodes = 5
    node_feature_dim = 64
    node_features = torch.randn(num_nodes, node_feature_dim)
    adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    model = DOMGNN(node_feature_dim=node_feature_dim)
    output = model(node_features, adjacency_matrix)
    print("Latent vector shape:", output.shape)
