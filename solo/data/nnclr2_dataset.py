from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets

class NNCLR2_Dataset_Wrapper(Dataset):
    def __init__(self, dataset, sim_matrix, cluster_lbls=None, num_nns=1, num_nns_choice=1, filter_sim_matrix=True, subsample_by=1) -> None:
        super().__init__()
        self.sim_matrix = sim_matrix
        self.clusters = cluster_lbls

        if filter_sim_matrix:
            self._filter_sim_matrix()
        else:
            self.sim_matrix = sim_matrix[:, :-1]

        self.not_from_cluster_percentage = {'avg': 0, 'var': 0, 'median': 0 }
        self._filter_sim_matrix_by_nnc()

        self.num_nns = num_nns
        self.num_nns_choice = num_nns_choice
        assert num_nns_choice >= num_nns
        
        self.dataset = dataset
        self.dataset_type = type(self.dataset).__bases__[0]
        self.subsample_by = subsample_by
        if subsample_by > 1:
            self.__subsample_dataset()

        self.labels = self.__get_labels()
        
        self.relevant_classes = self.get_class_percentage()
        

    def get_class_percentage(self):
        all_lbls_sim_matrix = self.labels[self.sim_matrix]
        all_lbls_true = self.labels.repeat(self.num_nns_choice).reshape(-1, self.num_nns_choice)
        correct_lbls = (all_lbls_true == all_lbls_sim_matrix)
        avg = correct_lbls.sum() / (len(self.labels) * self.num_nns_choice)
        all_percentages = correct_lbls.sum(axis=1) / self.num_nns_choice
        median = np.median(all_percentages)
        var = np.var(all_percentages)
        return {'avg': avg, 'median': median, 'var': var}

    
    def __subsample_dataset(self):
        # currently only for inat
        if self.dataset_type is not datasets.INaturalist:
            print(f'Subsampling not supported for {self.dataset_type}')
            return
        
        labels = self.__get_labels()
        imgs = np.array(list(list(zip(*self.dataset.index))[1]))
        no_classes = len(np.unique(labels))
        assert no_classes > labels.max()
        new_no_classes = no_classes // self.subsample_by
        new_imgs = imgs[labels <= new_no_classes]
        new_labels = labels[labels <= new_no_classes]
        new_index = list(zip(new_labels, new_imgs))
        self.dataset.index = new_index
        return
  

    def __get_labels(self):
        if self.dataset_type is datasets.CIFAR10 or \
            self.dataset_type is datasets.CIFAR100:
            return np.array(self.dataset.targets)
        elif self.dataset_type is datasets.SVHN:
            return np.array(self.dataset.labels)
        elif self.dataset_type is datasets.INaturalist:
            return np.array(list(list(zip(*self.dataset.index))[0]))
        else:
            return self.dataset.labels


    def __getitem__(self, index):
        sim_index_idxes = np.random.randint(0, len(self.sim_matrix[index]), self.num_nns)
        sim_index = self.sim_matrix[index][sim_index_idxes]
        all_idxs = []
        all_xs = []
        all_ys = []
        idx1, x1, y1 = self.dataset.__getitem__(index)
        if len(x1) != 1:
            print('Warning, More than 1 augmentations found, using first one')
        x1 = x1[0]
        all_idxs.append(idx1)
        all_xs.append(x1)
        all_ys.append(y1)

        for i in sim_index:
            idx2, x2, y2 = self.dataset.__getitem__(i)
            x2 = x2[0]
            all_idxs.append(idx2)
            all_xs.append(x2)
            all_ys.append(y2)

        return all_idxs, all_xs, all_ys


    def _filter_sim_matrix(self):
        """
        remove datapoint itself from its nearest neighbors (default should be False)
        """
        new_sim_idices = []
        for idx, row in enumerate(self.sim_matrix):
            if idx in row:
                new_row = np.concatenate([row[:np.argwhere(row == idx)[0][0]], row[np.argwhere(row == idx)[0][0] + 1:]])
            else:
                new_row = row[:-1]
            new_sim_idices.append(new_row)
        
        new_sim_matrix = np.stack(new_sim_idices, axis=0)
        assert new_sim_matrix.shape[0] == self.sim_matrix.shape[0]
        assert new_sim_matrix.shape[1] == (self.sim_matrix.shape[1] - 1)
        self.sim_matrix = new_sim_matrix
        return
    
    
    def _filter_sim_matrix_by_nnc(self):
        not_from_cluster = []
        if self.clusters is not None:
            new_sim_idices = []
            for idx, row in enumerate(self.sim_matrix):
                row_clusters = self.clusters[row]
                idx_from_same_cluster = idx[row_clusters == row_clusters[0]]
                new_row = idx_from_same_cluster[:self.num_nns_choice]
                if len(new_row) < self.num_nns_choice:
                    diff = self.num_nns_choice - len(new_row)
                    repeated_idxs = new_row[np.random.randint(0, len(new_row), diff)]
                    new_row = np.hstack([new_row, repeated_idxs])

                new_sim_idices.append(new_row)

                not_from_cluster.append(len(set(new_row) - set(row[:self.num_nns_choice])))
            
            new_sim_matrix = np.stack(new_sim_idices, axis=0)
            assert new_sim_matrix.shape[0] == self.sim_matrix.shape[0]
            assert new_sim_matrix.shape[1] == (self.num_nns_choice)
            self.sim_matrix = new_sim_matrix

            not_from_cluster = np.array(not_from_cluster) / self.num_nns_choice 

            self.not_from_cluster_percentage['avg'] = not_from_cluster.mean()
            self.not_from_cluster_percentage['median'] = np.median(not_from_cluster)
            self.not_from_cluster_percentage['var'] = np.var(not_from_cluster)
        else:
            self.sim_matrix = self.sim_matrix[:, :self.num_nns_choice]

        return
    

    def __len__(self):
        return len(self.dataset)