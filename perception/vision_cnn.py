"""
Phase 1, Milestone M1.1 — Vision CNN Encoder
Processes raw browser window pixels into a latent representation Z_pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionCNN(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super(VisionCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 3, H, W)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        z = self.fc(x)
        return z


if __name__ == "__main__":
    # Example usage
    model = VisionCNN()
    dummy_input = torch.randn(1, 3, 84, 84)  # Example pixel input
    output = model(dummy_input)
    print("Latent vector shape:", output.shape)
