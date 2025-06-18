import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, train_cnn=False):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.backbone.parameters():
            param.requires_grad = train_cnn

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(2).permute(0, 2, 1)
        return features
