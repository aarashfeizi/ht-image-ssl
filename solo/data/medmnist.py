from medmnist import PathMNIST as PathMNIST_T
from medmnist import TissueMNIST as TissueMNIST_T
import pdb

class PathMNIST(PathMNIST_T):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        pdb.set_trace()
        return batch

class TissueMNIST(TissueMNIST_T):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        pdb.set_trace()
        return batch