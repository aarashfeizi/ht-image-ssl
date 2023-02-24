from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets

class NNCLR2_Dataset_Wrapper(Dataset):
    def __init__(self, dataset, sim_matrix, num_nns=1, num_nns_choice=1, filter_sim_matrix=True, subsample_by=1) -> None:
        super().__init__()
        self.sim_matrix = sim_matrix
        if filter_sim_matrix:
            self._filter_sim_matrix()
        else:
            self.sim_matrix = sim_matrix[:, :-1]
        self.num_nns = num_nns
        self.num_nns_choice = num_nns_choice
        assert num_nns_choice >= num_nns
        self.sim_matrix = self.sim_matrix[:, :self.num_nns_choice]
        self.dataset = dataset
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
        parent_class_type = type(self.dataset).__bases__[0] # type(dataset) will be DatasetWithIndex
        if parent_class_type is datasets.CIFAR10 or \
            parent_class_type is datasets.CIFAR100:
            return np.array(self.dataset.targets)
        elif parent_class_type is datasets.SVHN:
            return np.array(self.dataset.labels)
        elif parent_class_type is datasets.INaturalist:
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
    
    def __len__(self):
        return len(self.dataset)