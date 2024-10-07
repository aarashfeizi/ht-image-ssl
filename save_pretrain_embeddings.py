from solo.utils import misc
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision import models
import clip
import transformers
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
    model_name = args.model
    image_size = args.image_size
    train_path = args.train_path
    save_path = args.save_path

    print('getting model...')
    if model_name == 'sup-rn50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        t = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),])
    elif model_name == 'clip-rn50':
        model, t = clip.load('RN50', device='cpu')
    
    elif model_name == 'clip-vit-b16':
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
        t = preprocess_val
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    elif model_name == 'clip-convnext_b':
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
        t = preprocess_val
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K')

    print('Loading Datasets...')
    ds = datasets.ImageFolder(train_path, transform=t)
    dl = DataLoader(ds, batch_size=256, pin_memory=True, num_workers=4)
    model.eval()
    model.cuda()

    full_path = os.path.join(save_path, f'imagenet_rn50_{model_name}.npy')
    print('full_path to save:', full_path)

    print('Getting Embeddings...')
    if model_name.startswith('clip'):
        embs = misc.get_clip_embeddings(model, dataloader=dl, device='cuda')
    else:
        embs = misc.get_pretrained_model_embeddings(model, dataloader=dl, device='cuda')

    print('saving!')
    np.save(full_path, embs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sup-rn50', choices=['sup-rn50',
                                                                'clip-rn50',
                                                                'clip-vit-b16',
                                                                'clip-convnext_b',
                                                                ])
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--train_path', default='/network/datasets/imagenet.var/imagenet_torchvision/train/')
    parser.add_argument('--save_path', default='/network/scratch/f/feiziaar/ht-image-ssl/logs/cache/')

    args = parser.parse_args()

    main(args)