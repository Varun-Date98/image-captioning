import torch
import torch.nn as nn

from attention import AdditiveAttention


class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, attention_dim, feature_dim=2048, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = AdditiveAttention(feature_dim, hidden_dim, attention_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )
        self.skip_projection = nn.Linear(512 + 1024, 512)

    def forward(self, features, captions):
        batch_size, seq_len = captions.shape
        embeddings = self.dropout(self.embedding(captions))

        h = torch.zeros(2, batch_size, self.lstm.hidden_size, device=captions.device)
        c = torch.zeros(2, batch_size, self.lstm.hidden_size, device=captions.device)

        outputs = []
        for t in range(seq_len):
            word = embeddings[:, t, :].unsqueeze(1)
            context, _ = self.attention(features, h[-1])
            context = self.projection(context)

            lstm_input = self.dropout(torch.cat((word.squeeze(1), context), dim=1).unsqueeze(1))
            _, (h, c) = self.lstm(lstm_input, (h, c))

            # Skip connection
            skip_input = torch.cat((h[-1], lstm_input.squeeze(1)), dim=1)
            combined = self.layer_norm(self.skip_projection(skip_input))
            out = self.fc(self.dropout(combined))
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)
