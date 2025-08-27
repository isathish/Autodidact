"""
Phase 3, Milestone M3.1 — Cross-Modal Grounding
Learns to align text and perception embeddings using contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGrounding(nn.Module):
    def __init__(self, text_dim=256, image_dim=256, embed_dim=256):
        super(CrossModalGrounding, self).__init__()
        self.text_encoder = nn.Linear(text_dim, embed_dim)
        self.image_encoder = nn.Linear(image_dim, embed_dim)

    def forward(self, text_features, image_features):
        text_emb = F.normalize(self.text_encoder(text_features), dim=-1)
        image_emb = F.normalize(self.image_encoder(image_features), dim=-1)
        return text_emb, image_emb

    def contrastive_loss(self, text_emb, image_emb, temperature=0.07):
        logits = torch.matmul(text_emb, image_emb.T) / temperature
        labels = torch.arange(len(text_emb)).to(text_emb.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.T, labels)
        return (loss_t + loss_i) / 2


if __name__ == "__main__":
    model = CrossModalGrounding()
    text_feats = torch.randn(4, 256)
    img_feats = torch.randn(4, 256)
    text_emb, img_emb = model(text_feats, img_feats)
    loss = model.contrastive_loss(text_emb, img_emb)
    print("Loss:", loss.item())
