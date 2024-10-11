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
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import argparse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Function to check if the path exists and give a warning if not
def load_data_with_warning(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: The {label} file at '{path}' does not exist!")
    return np.load(path)

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
    num_workers = args.num_workers
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
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    elif dataset == 'aircrafts':
        # ds = datasets.FGVCAircraft(train_path,
        #                            split=split,
        #                            transform=t)
        ds = datasets.load_dataset('HuggingFaceM4/FGVC-Aircraft', split=split)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        pin_memory=True,
                        num_workers=num_workers,
                        collate_fn=lambda batch: collate_fn(batch, t), shuffle=False)
    elif dataset == 'pathmnist':
        import medmnist
        ds = medmnist.PathMNIST(split=split,
                                transform=t,
                                root=train_path, 
                                download=True)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False)
    elif dataset == 'tissuemnist':
        import medmnist
        ds = medmnist.TissueMNIST(split=split,
                                transform=t,
                                root=train_path, 
                                download=True)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False)
    elif dataset == 'octmnist':
        import medmnist
        ds = medmnist.OCTMNIST(split=split,
                                transform=t,
                                root=train_path, 
                                download=True)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False)
    elif dataset == 'cifar10':
        ds = datasets.load_dataset('uoft-cs/cifar10',
                                   split=split)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False,
                                   collate_fn=lambda batch: collate_fn(batch, t,
                                        img_label='img',
                                        lbl_label='label'))
    elif dataset == 'food101':
        assert split in ['train', 'validation']
        ds = datasets.load_dataset('ethz/food101', split=split)
        
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False,
                                   collate_fn=lambda batch: collate_fn(batch, t,
                                        img_label='image',
                                        lbl_label='label'))
    #     ds = datasets.CIFAR10(train_path,
    #                                split=split,
    #                                transform=t)
    elif dataset == 'cifar100':
        ds = datasets.load_dataset('uoft-cs/cifar100', split=split)
        # ds = datasets.CIFAR100(train_path,
        #                            split=split,
        #                            transform=t)
        dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=False,
                                   collate_fn=lambda batch: collate_fn(batch, t,
                                        img_label='img',
                                        lbl_label='label'))
    
    model.eval()
    model.cuda()

    full_path = os.path.join(save_path, f'{dataset}_{split}_{model_name.replace("-", "_")}.npy')
    label_path = os.path.join(save_path, f'{dataset}_{split}_labels.npy')

    labels = None
    if not os.path.exists(full_path):
        print('full_path to save:', full_path)
        print('Getting Embeddings...')
        if model_name.startswith('clip'):
            embs, labels = misc.get_clip_embeddings(model, dataloader=dl, device='cuda', labels=True)
        else:
            embs = misc.get_pretrained_model_embeddings(model, dataloader=dl, device='cuda')

        print('saving!')
        np.save(full_path, embs)
        if labels is not None:
            np.save(label_path, labels)
    else:
        print('full_path already exists.:', full_path)


    if args.eval:
        train_path = os.path.join(save_path, f'{dataset}_{args.eval_train}_{model_name.replace("-", "_")}.npy')
        train_labels_path = os.path.join(save_path, f'{dataset}_{args.eval_train}_labels.npy')
        test_path = os.path.join(save_path, f'{dataset}_{args.eval_test}_{model_name.replace("-", "_")}.npy')
        test_labels_path = os.path.join(save_path, f'{dataset}_{args.eval_test}_labels.npy')
        
        # Load the datasets and labels with warnings if paths are invalid
        train_data = load_data_with_warning(train_path, "train embeddings")
        train_labels = load_data_with_warning(train_labels_path, "train labels")
        test_data = load_data_with_warning(test_path, "test embeddings")
        test_labels = load_data_with_warning(test_labels_path, "test labels")

        # Check if any of the data was not loaded due to missing files
        if None in [train_data, train_labels, test_data, test_labels]:
            print("Error: One or more datasets could not be loaded. Please check the warnings and the file paths.")
        else:
            # Define and train the Logistic Regression model
            model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
            model.fit(train_data, train_labels)

            # Make predictions on the test set
            test_predictions = model.predict(test_data)

            # Calculate accuracy
            accuracy = accuracy_score(test_labels, test_predictions)
            acc_message = f'Test Accuracy {model_name} trained on {dataset}_{args.eval_train} and tested on {dataset}_{args.eval_test}: {accuracy * 100:.2f}%\n'
            print(acc_message)

            # Save accuracy to a file
            file_name = f'LP_Acc_{model_name.replace("-", "_")}_{dataset}_Train-{args.eval_train}_Test-{args.eval_test}.txt'
            with open(file_name, 'w') as f:
                f.write(acc_message)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sup-rn50', choices=['sup-rn50',
                                                                'clip-rn50',
                                                                'clip-vit-b16',
                                                                'clip-convnext_b',
                                                                ])
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # parser.add_argument('--train_path', default='/network/datasets/imagenet.var/imagenet_torchvision/train/')
    parser.add_argument('--train_path', default='/network/scratch/f/feiziaar/.cache/huggingface/')
    parser.add_argument('--save_path', default='/network/scratch/f/feiziaar/ht-image-ssl/logs/cache/')
    parser.add_argument('--dataset', default='imagenet', choices=['aircrafts', 'food101', 'imagenet', 'pathmnist', 'octmnist', 'tissuemnist', 'cifar10', 'cifar100'])
    
    parser.add_argument('--split', default='train', choices=['train', 'val', 'trainval', 'test', 'train+validation', 'validation'])
    
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--eval_train', default='train', choices=['train', 'trainval', 'train+validation'])
    parser.add_argument('--eval_test', default='test', choices=['test', 'validation'])
    

    args = parser.parse_args()

    main(args)