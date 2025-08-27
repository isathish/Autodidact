"""
Phase 2, Milestone M2.4 — Next-Character Predictor
Trains a simple LSTM to predict the next character in a sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def predict(self, char_idx, hidden=None, top_k=None):
        x = torch.tensor([[char_idx]])
        out, hidden = self.forward(x, hidden)
        probs = F.softmax(out[:, -1], dim=-1).data
        if top_k is not None:
            probs, top_idx = probs.topk(top_k)
            top_idx = top_idx[0]
            char_idx = top_idx[torch.multinomial(probs, 1)]
        else:
            char_idx = torch.multinomial(probs, 1)
        return char_idx.item(), hidden


if __name__ == "__main__":
    vocab_size = 30  # Example small vocab
    model = CharRNN(vocab_size)
    dummy_input = torch.randint(0, vocab_size, (1, 10))
    output, _ = model(dummy_input)
    print("Output shape:", output.shape)
