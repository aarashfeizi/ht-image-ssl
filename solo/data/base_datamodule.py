import pytorch_lightning as pl
import torch
from solo.utils.misc import get_embeddings, get_sim_matrix, get_clusters
from solo.data.nnclr2_dataset import NNCLR2_Dataset_Wrapper
from solo.data.pretrain_dataloader import prepare_dataloader
import numpy as np

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_transforms=None, 
                    val_transforms=None,
                    test_transforms=None,
                    dims=None,
                    model=None,
                    filter_sim_matrix=True,
                    subsample_by=1,
                    num_clusters=1,
                    nn_threshold=-1,
                    threshold_mode='fixed',
                    clustering_algo=None):
        
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.emb_train_loader = None
        self.train_loader = None
        self.val_loader = None
        self.epoch = -1 # once for sanity check
        self.model = model
        self.filter_sim_matrix = filter_sim_matrix
        self.subsample_by = subsample_by
        self.num_clusters = num_clusters
        self.nn_threshold = nn_threshold
        self.threshold_mode = threshold_mode
        self.clustering_algo = clustering_algo

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
            extra_info = {}
            embeddings = get_embeddings(self.model, self.emb_train_loader)['embs']
            emb_dist_matrix, emb_sim_matrix = get_sim_matrix(embeddings, gpu=torch.cuda.is_available())
            clust_dist, clust_lbls = None, None
            if self.num_clusters > 1:
                clust_dist, clust_lbls = get_clusters(embeddings, k=self.num_clusters, gpu=torch.cuda.is_available())
            
            if self.threshold_mode == 'adaptive':
                threshold = np.mean(emb_dist_matrix[:, 1:21]) + np.std(emb_dist_matrix[:, 1:21])
                extra_info['emb_dist_AVG'] = np.mean(emb_dist_matrix[:, 1:21])
                extra_info['emb_dist_STD'] = np.std(emb_dist_matrix[:, 1:21])
                extra_info['emb_dist_VAR'] = np.var(emb_dist_matrix[:, 1:21])
                print(f'Seeting threshold to {threshold}')
            elif self.threshold_mode == 'fixed':
                threshold = self.nn_threshold

            train_dataset = NNCLR2_Dataset_Wrapper(dataset=self.train_loader.dataset.dataset,
                                                    sim_matrix=emb_sim_matrix,
                                                    dist_matrix=emb_dist_matrix,
                                                    cluster_lbls=clust_lbls,
                                                    nn_threshold=threshold,
                                                    num_nns=self.train_loader.dataset.num_nns,
                                                    num_nns_choice=self.train_loader.dataset.num_nns_choice,
                                                    filter_sim_matrix=self.filter_sim_matrix,
                                                    subsample_by=self.subsample_by,
                                                    clustering_algo=self.clustering_algo,
                                                    extra_info=extra_info)
            
            print('Relevant class percentage: ', train_dataset.relevant_classes)
            print('Not from cluster percentage: ', train_dataset.not_from_cluster_percentage)
            print('Number of nns: ', train_dataset.no_nns)
            self.train_loader = prepare_dataloader(
                train_dataset, batch_size=self.train_loader.batch_size, num_workers=self.train_loader.num_workers)
            
            return self.train_loader
