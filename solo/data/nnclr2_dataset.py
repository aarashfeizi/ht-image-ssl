from torch.utils.data import Dataset
import numpy as np

class NNCLR2_Dataset_Wrapper(Dataset):
    def __init__(self, dataset, sim_matrix) -> None:
        super().__init__()
        self.sim_matrix = sim_matrix
        self._filter_sim_matrix()

        self.dataset = dataset

    def __getitem__(self, index):
        sim_index = self.sim_matrix[index, 0]
        idx1, data1 = self.dataset.__getitem__(index)
        idx2, data2 = self.dataset.__getitem__(sim_index)

        return [idx1, idx2], [data1, data2]

    def _filter_sim_matrix(self):
        new_sim_idices = []
        for idx, row in enumerate(self.sim_matrix):
            if idx in row:
                new_row = np.concatenate([row[:np.argwhere(row == idx)[0][0]], row[np.argwhere(row == idx)[0][0] + 1:]])
            else:
                new_row = row[:-1]
            new_sim_idices.append(new_row)
        
        new_sim_matrix = np.concatenate(new_sim_idices)
        assert new_sim_matrix.shape[0] == self.sim_matrix.shape[0]
        assert new_sim_matrix.shape[1] == (self.sim_matrix.shape[1] - 1)
        self.sim_matrix = new_sim_matrix
        return
    
    def __len__(self):
        return len(self.dataset)