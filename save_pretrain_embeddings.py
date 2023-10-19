from solo.utils import misc
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision import models
import clip
import numpy as np
import os
import argparse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# model = input('Model:', )
# image_size = int(input('Image size:', ))
# train_path = input('Train Path:', )
# save_path = input('Train Path:', )

def main(args):
    model = args.model
    image_size = args.image_size
    train_path = args.train_path
    save_path = args.save_path

    if train_path == '':
        print("Printing train_path to '/network/datasets/imagenet.var/imagenet_torchvision/train/'")
        train_path = '/network/datasets/imagenet.var/imagenet_torchvision/train/'

    if save_path == '':
        save_path = '/network/scratch/f/feiziaar/ht-image-ssl/logs/cache/'

    if model == 'sup':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        t = transforms.Compose([transforms.Resize(image_size, image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),])
    elif model == 'clip':
        model, t = clip.load('RN50', device='cpu')

    ds = datasets.ImageFolder(train_path, transform=t)
    dl = DataLoader(ds, batch_size=256, pin_memory=True, num_workers=4)
    model.eval()
    model.cuda()

    if model == 'clip':
        embs = misc.get_clip_embeddings(model, dataloader=dl, device='cuda')
    else:
        embs = misc.get_pretrained_model_embeddings(model, dataloader=dl, device='cuda')


    full_path = os.path.join(save_path, f'imagenet_rn50_{model}.npy')
    print('saving!')
    np.save(full_path, embs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sup', choices=['sup', 'clip'])
    parser.add_argument('--image_size', default='224', type=int)
    parser.add_argument('--train_path', default='/network/datasets/imagenet.var/imagenet_torchvision/train/')
    parser.add_argument('--save_path', default='/network/scratch/f/feiziaar/ht-image-ssl/logs/cache/')

    args = parser.parse_args()

    main(args)