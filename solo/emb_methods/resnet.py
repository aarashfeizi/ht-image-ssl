from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import  ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import torch.nn as nn

RESNETS = {
    'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
    'resnet101': resnet101(weights=ResNet101_Weights.IMAGENET1K_V2),
}

RESNETS_RANDOM = {
    'resnet18': resnet18(weights=None),
    'resnet50': resnet50(weights=None),
    'resnet101': resnet101(weights=None),
}

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        if cfg.emb_model.pretrained == 'true':
            self.backbone = RESNETS[cfg.emb_model.name]
        else:
            self.backbone = RESNETS_RANDOM[cfg.emb_model.name]
        
        if cfg.emb_model.train:
            if cfg.emb_model.train_method == 'supervised':
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features=in_features, out_features=cfg.data.num_classes)
            else:
                raise Exception('Unsupported training method')
            
            self.backbone.train()
        else:
            self.backbone.fc = nn.Identity()
            self.backbone.eval()

    def forward(self, x):
        out = self.backbone(x)
        return out
    
    def eval(self):
        # super.eval()
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
