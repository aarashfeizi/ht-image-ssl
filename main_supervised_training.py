import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from solo.utils import misc
from pytorch_lightning.loggers import WandbLogger
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.strategies.ddp import DDPStrategy
import os


import wandb
import argparse

RESNETS = {'resnet18': resnet18,
            'resnet50': resnet50}

DATASETS = {'cifar100': datasets.CIFAR100,
            'cifar10': datasets.CIFAR10,
            'svhn': datasets.SVHN}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ResNet(pl.LightningModule):
    def __init__(self, config, no_classes=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = RESNETS[config.backbone]()
        self.feature_size = self.backbone.fc.in_features
        self.class_num = no_classes
        fc = nn.Linear(self.feature_size, self.class_num)
        self.backbone.fc = fc
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.train_acc = Accuracy(task='multiclass', num_classes=self.no_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.no_classes)   
        self.test_acc = Accuracy(task='multiclass', num_classes=self.no_classes)

    
    def forward(self, x):
        pred = self.backbone(x)
        return pred

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        return {'loss': loss, 'preds': y_scores, 'target': y_true.int()}
    
    def training_step_end(self, outputs):
        # update and log
        self.train_acc(outputs['preds'], outputs['target'])
        self.log("train_loss", outputs['loss'], prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        loss = F.cross_entropy(y_logits, y)
        return {'loss': loss, 'preds': y_scores, 'target': y_true.int()}

    def validation_step_end(self, outputs):
        # update and log
        self.val_acc(outputs['preds'], outputs['target'])
        self.log("val_loss", outputs['loss'], prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_scores = torch.sigmoid(y_hat)
        y_true = y.view((-1, 1)).type_as(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss, 'preds': y_scores, 'target': y_true.int()}

    def test_step_end(self, outputs):
        # update and log
        self.test_acc(outputs['preds'], outputs['target'])
        self.log("test_loss", outputs['loss'], prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--num_workers', default=10,  type=int)
    parser.add_argument('--weight_decay', default=1e-5,  type=float)
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18, resnet50'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--dataset_path', default='../../scratch/')
    parser.add_argument('--save_path', default='../../scratch/ht-image-ssl/supervised_training/')


    args = parser.parse_args()

    if args.wandb:
        wandb.init(config=args, dir=args.save_path, mode='online')
        args = wandb.config

    return args
    
def main():
    args = get_args()

    misc.make_dirs(args.save_path)
    misc.make_dirs(os.path.join(args.save_path, 'checkpoints'))

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    svhn_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                # transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    inat_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop((224, 224)),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    dataset_args = {}

    no_classes = -1
    if args.dataset.startswith('cifar'):
        dataset_args['root'] = args.dataset_path
        dataset_args['train'] = True
        dataset_args['transform'] = cifar_pipeline['T_train']
        if args.dataset == 'cifar10':
            no_classes = 10
        elif args.dataset == 'cifar100':
            no_classes = 100
    elif args.dataset.startswith('svhn'):
        dataset_args['root'] = args.dataset_path
        dataset_args['split'] = 'train'
        dataset_args['transform'] = svhn_pipeline['T_train']
        no_classes = 10
    
    train_dataset = DATASETS[args.dataset](**dataset_args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    dataset_args = {}

    if args.dataset.startswith('cifar'):
        dataset_args['root'] = args.dataset_path
        dataset_args['train'] = False
        dataset_args['transform'] = cifar_pipeline['T_val']
    elif args.dataset.startswith('svhn'):
        dataset_args['root'] = args.dataset_path
        dataset_args['split'] = 'test'
        dataset_args['transform'] = svhn_pipeline['T_val']
    
    test_dataset = DATASETS[args.dataset](**dataset_args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    model = ResNet(args, no_classes=no_classes)

    if args.wandb:
        wandb_logger = WandbLogger(
                    project='ht-image-ssl',
                    entity='aarashfeizi'
                )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

    trainer_kwargs = (
        {
            "logger": wandb_logger if args.wandb else None,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=True),
        }
    )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

if __name__ == '__main__':
    main()