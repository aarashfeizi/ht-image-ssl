import datasets
from torch.utils.data import Dataset

class BaseWrapper(Dataset):
    def __init__(self,
                path, # irrelevant now, automatically uses Huggingface cache
                split="train", # 'train', 'val', 'trainval', 'test'
                transform=None) -> None:
        
        self.dataset = None
        self.image_label = ''
        self.target_label = ''
        self.transform = transform
    
    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        image = item[self.image_label]
        label = item[self.target_label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FGVCAircraft(BaseWrapper):
    
    def __init__(self,
                train_data_path, # irrelevant now, automatically uses Huggingface cache
                split="train", # 'train', 'val', 'trainval', 'test'
                transform=None) -> None:
        
        super(FGVCAircraft, self).__init__(train_data_path, split, transform)
        
        torchvision_to_hf_split_mapper = {
            "train": "train",
            "val": "validation",
            "trainval": "train+validation",
            "test": "test"
        }
        
        self.dataset = datasets.load_dataset('HuggingFaceM4/FGVC-Aircraft',
                                             split=torchvision_to_hf_split_mapper[split])
        self.image_label = 'image'
        self.target_label = 'variant'

class Cifar10(BaseWrapper):
    def __init__(self,
                train_data_path, # irrelevant now, automatically uses Huggingface cache
                split="train", # 'train', 'val', 'trainval', 'test'
                transform=None) -> None:
                
        super(Cifar10, self).__init__(train_data_path, split, transform)
        
        self.dataset = datasets.load_dataset('uoft-cs/cifar10',
                                             split=split)
        self.image_label = 'img'
        self.target_label = 'label'


class Cifar100(BaseWrapper):
    def __init__(self,
                train_data_path, # irrelevant now, automatically uses Huggingface cache
                split="train", # 'train', 'val', 'trainval', 'test'
                transform=None) -> None:
                
        super(Cifar100, self).__init__(train_data_path, split, transform)
        
        self.dataset = datasets.load_dataset('uoft-cs/cifar100',
                                             split=split)
        self.image_label = 'img'
        self.target_label = 'label'