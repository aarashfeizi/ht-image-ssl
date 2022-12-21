from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import  ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import torch.nn as nn

RESNETS = {
    'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
    'resnet101': resnet101(weights=ResNet101_Weights.IMAGENET1K_V2),
}

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.backbone = RESNETS[cfg.emb_model.name]
        self.fc = nn.Identity()
        self.backbone.eval()

    def forward(self, x):
        out = self.backbone(x)
        return out