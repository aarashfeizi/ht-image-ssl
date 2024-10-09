import datasets
from torch.utils.data import Dataset

class Food101(Dataset):
    def __init__(self, split='train', transform=None) -> None:
        """
            split: either 'train' or 'validation'
        """
        super().__init__()
        assert split in ['train', 'validation']
        self.dataset = datasets.load_dataset('ethz/food101', split=split)
        self.transform = transform

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        ret_item = self.dataset.__getitem__(int(index))
        img, target = ret_item['image'].convert('RGB'), ret_item['label']

        if self.transform:
            img = self.transform(img)
        
        return img, target

class Country211(Dataset):
    def __init__(self, split='train', transform=None) -> None:
        """
            split: either 'train' or 'test'
        """
        super().__init__()
        assert split in ['train', 'test']
        self.dataset = datasets.load_dataset('clip-benchmark/wds_country211', split=split)
        self.transform = transform

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        ret_item = self.dataset.__getitem__(int(index))
        img, target = ret_item['jpg'].convert('RGB'), ret_item['cls']

        if self.transform:
            img = self.transform(img)
        
        return img, target
