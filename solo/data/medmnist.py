from medmnist import PathMNIST as PathMNIST_T
from medmnist import TissueMNIST as TissueMNIST_T
import pdb

class PathMNIST(PathMNIST_T):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = target[0]
        return img, target 

class TissueMNIST(TissueMNIST_T):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = target[0]
        return img, target