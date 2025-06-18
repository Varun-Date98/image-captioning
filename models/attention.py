import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attn_dim):
        super(AdditiveAttention, self).__init__()
        self.encoder_attn = nn.Linear(feature_dim, attn_dim)
        self.decoder_attn = nn.Linear(hidden_dim, attn_dim)
        self.full_attn = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden):
        attn1 = self.encoder_attn(features)
        attn2 = self.decoder_attn(hidden).unsqueeze(1)
        energy = self.full_attn(self.relu(attn1 + attn2)).squeeze(2)
        alpha = self.softmax(energy)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha
