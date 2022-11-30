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
        idx1, x1, y1 = self.dataset.__getitem__(index)
        idx2, x2, y2 = self.dataset.__getitem__(sim_index)

        assert len(x1) == 1
        assert len(x2) == 1

        x1 = x1[0]
        x2 = x2[0]

        return [idx1, idx2], [x1, x2], [y1, y2]

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