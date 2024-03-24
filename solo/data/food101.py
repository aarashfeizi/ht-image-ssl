import datasets
from torch.utils.data import Dataset

class Food101(Dataset):
    def __init__(self, split='train', transform=None) -> None:
        """
            split: either 'train' or 'validation'
        """
        super().__init__()
        assert split in ['train', 'validation']
        self.dataset = datasets.load_dataset('food101', split=split)
        self.transform = transform

    def __getitem__(self, index):
        ret_item = self.dataset.__getitem__(index)
        img, target = ret_item['image'], ret_item['label']

        if self.transform:
            img = self.transform(img)
        
        return img, target