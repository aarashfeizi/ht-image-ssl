from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, cfg):
        super(ResNet50, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.eval()

    def forward(self, x):
        out = self.backbone(x)
        return out