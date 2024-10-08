from solo.utils import misc
import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tv_dataset
from torchvision import transforms
import datasets
from torchvision import models
import clip
import transformers
import numpy as np
import os
import argparse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def collate_fn(batch, transform=None, img_label='image', lbl_label='variant'):
    
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    

    images = []
    labels = []

    # Process each item in the batch
    for item in batch:
        # Convert PIL image to tensor
        image = transform(item[img_label])
        images.append(image)

        # Append variants
        labels.append(item[lbl_label])

    # Stack images and convert variants to a tensor
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    # Return only images and variants as a list
    return [images, labels]


# model = input('Model:', )
# image_size = int(input('Image size:', ))
# train_path = input('Train Path:', )
# save_path = input('Train Path:', )

def main(args):
    model_name = args.model
    image_size = args.image_size
    batch_size = args.batch_size
    train_path = args.train_path
    save_path = args.save_path
    dataset = args.dataset
    split = args.split

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
    if dataset == 'imagenet':
        ds = tv_dataset.ImageFolder(train_path, transform=t)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=4)
    elif dataset == 'aircrafts':
        # ds = datasets.FGVCAircraft(train_path,
        #                            split=split,
        #                            transform=t)
        ds = datasets.load_dataset('HuggingFaceM4/FGVC-Aircraft', split=split)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        pin_memory=True,
                        num_workers=4,
                        collate_fn=lambda batch: collate_fn(batch, t), shuffle=False)
    elif dataset == 'pathmnist':
        import medmnist
        ds = medmnist.PathMNIST(split=split,
                                transform=t,
                                root=train_path, 
                                download=True)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=False)
    elif dataset == 'tissuemnist':
        import medmnist
        ds = medmnist.TissueMNIST(split=split,
                                transform=t,
                                root=train_path, 
                                download=True)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=False)
    elif dataset == 'cifar10':
        ds = datasets.load_dataset('uoft-cs/cifar10',
                                   split=split)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=False,
                                   collate_fn=lambda batch: collate_fn(batch, t,
                                        img_label='img',
                                        lbl_label='label'))
    #     ds = datasets.CIFAR10(train_path,
    #                                split=split,
    #                                transform=t)
    elif dataset == 'cifar100':
        ds = datasets.load_dataset('uoft-cs/cifar100', split=split)
        # ds = datasets.CIFAR100(train_path,
        #                            split=split,
        #                            transform=t)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=False,
                                   collate_fn=lambda batch: collate_fn(batch, t,
                                        img_label='img',
                                        lbl_label='label'))
        
    model.eval()
    model.cuda()

    full_path = os.path.join(save_path, f'{dataset}_{split}_{model_name}.npy')
    label_path = os.path.join(save_path, f'{dataset}_{split}_labels.npy')
    print('full_path to save:', full_path)

    labels = None
    print('Getting Embeddings...')
    if model_name.startswith('clip'):
        embs, labels = misc.get_clip_embeddings(model, dataloader=dl, device='cuda', labels=True)
    else:
        embs = misc.get_pretrained_model_embeddings(model, dataloader=dl, device='cuda')

    print('saving!')
    np.save(full_path, embs)
    if labels:
        np.save(label_path, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sup-rn50', choices=['sup-rn50',
                                                                'clip-rn50',
                                                                'clip-vit-b16',
                                                                'clip-convnext_b',
                                                                ])
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--train_path', default='/network/datasets/imagenet.var/imagenet_torchvision/train/')
    parser.add_argument('--train_path', default='/network/scratch/f/feiziaar/.cache/huggingface/')
    parser.add_argument('--save_path', default='/network/scratch/f/feiziaar/ht-image-ssl/logs/cache/')
    parser.add_argument('--dataset', default='imagenet', choices=['aircrafts', 'imagenet', 'pathmnist', 'tissuemnist', 'cifar10', 'cifar100'])
    
    parser.add_argument('--split', default='train', choices=['train', 'val', 'trainval', 'test', 'train+validation'])

    args = parser.parse_args()

    main(args)