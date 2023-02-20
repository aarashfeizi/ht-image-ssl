from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import  ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import torch.nn as nn
import torch
import os

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
        if cfg.emb_model.pretrained == 'false':
            self.backbone = RESNETS_RANDOM[cfg.emb_model.name]
        elif cfg.emb_model.pretrained == 'true' or cfg.emb_model.pretrained == 'imagenet':
            self.backbone = RESNETS[cfg.emb_model.name]
        else:
            self.backbone = RESNETS_RANDOM[cfg.emb_model.name]
            model_path = os.path.join(cfg.emb_model.ckpt_path, f'{cfg.emb_model.name}_{cfg.emb_model.pretrained}.ckpt')

            print(f'Loading {model_path}')
            assert os.path.exists(model_path), f'{model_path} does not exist! :('
            checkpoint = torch.load(model_path)

            new_ckpt_dict = self.__fix_keys(checkpoint['state_dict'], 'backbone.', '')
            
            mk = self.backbone.load_state_dict(new_ckpt_dict, strict=False)
            assert set(mk.missing_keys) == {'fc.weight', 'fc.bias'}, f'Missing keys are {mk.missing_keys}'

        
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
        self.train(False)
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
    
    def __fix_keys(self, d, old, new):
        new_d = {}
        for k, v in d.items():
            new_k = k.replace(old, new)
            new_d[new_k] = v
        
        new_d.pop('fc.weight')
        new_d.pop('fc.bias')
        return new_d