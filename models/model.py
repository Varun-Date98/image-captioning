import torch.nn as nn
from encoder import EncoderCNN
from decoder import DecoderLSTM


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, attention_dim=256, train_cnn=False):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(train_cnn=train_cnn)
        self.decoder = DecoderLSTM(embed_dim, hidden_dim, vocab_size, attention_dim)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
