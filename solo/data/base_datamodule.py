import pytorch_lightning as pl
import torch
from solo.utils.misc import get_embeddings, get_sim_matrix
from solo.data.nnclr2_dataset import NNCLR2_Dataset_Wrapper
from solo.data.pretrain_dataloader import prepare_dataloader


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, model=None, filter_sim_matrix=True):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.emb_train_loader = None
        self.train_loader = None
        self.val_loader = None
        self.epoch = -1 # once for sanity check
        self.model = model
        self.filter_sim_matrix = filter_sim_matrix

    def set_emb_dataloder(self, loader):
        self.emb_train_loader = loader
    
    def set_val_loader(self, loader):
        self.val_loader = loader
    
    def set_train_loader(self, loader):
        self.train_loader = loader

    # overwriting
    def val_dataloader(self):
        return self.val_loader

    # overwriting
    def train_dataloader(self):
        self.epoch += 1
        if self.epoch < 2:
            return self.train_loader
        else:
            print('Updating train_loader sim_matrix on epoch = ', self.epoch)

            assert self.emb_train_loader is not None

            embeddings = get_embeddings(self.model, self.emb_train_loader)
            _, emb_sim_matrix = get_sim_matrix(embeddings, k=self.train_loader.dataset.num_nns_choice, gpu=torch.cuda.is_available())
            train_dataset = NNCLR2_Dataset_Wrapper(dataset=self.train_loader.dataset.dataset,
                                                    sim_matrix=emb_sim_matrix,
                                                    num_nns=self.train_loader.dataset.num_nns,
                                                    num_nns_choice=self.train_loader.dataset.num_nns_choice,
                                                    filter_sim_matrix=self.filter_sim_matrix)

            self.train_loader = prepare_dataloader(
                train_dataset, batch_size=self.train_loader.batch_size, num_workers=self.train_loader.num_workers)
            
            return self.train_loader
