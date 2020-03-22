import numpy as np
import os

from torch.utils import data
from torch.utils.data import Dataset

class BridgeData(data.Dataset):
    def __init__(self, root, set_name):
        self.root = os.path.expanduser(root)
        self.set_name = set_name  # training set or test set

        # now load the picked numpy arrays
        rng = np.random.RandomState(seed=0)
      
        all_data = np.loadtxt(root+'/all_year')

        if self.set_name is 'train':
            train_sample_idx = rng.choice(a=[i for i in np.arange(29866)], size=29866, replace=False)
            all_data = all_data[train_sample_idx]
        elif self.set_name is 'val':
            val_sample_idx = rng.choice(a=[i for i in np.arange(29866, 34130)], size=4264, replace=False)
            all_data = all_data[val_sample_idx]
        else:
            test_sample_idx = rng.choice(a=[i for i in np.arange(34130, 42658)], size=8528, replace=False)
            all_data = all_data[test_sample_idx]
            
        self.data = all_data[:, :372]
        self.deal_labels = all_data[:, 372:-38]
        self.auction_labels = all_data[:, -38:]

        print(set_name, self.data.shape)
        print(set_name, self.deal_labels.shape)
        print(set_name, self.auction_labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, deal_label, auction_label) where target is index of the target class.
        """
        data, deal_label, auction_label = self.data[index], self.deal_labels[index], self.auction_labels[index]

        # doing this so that it is consistent with all other datasets
        return data, deal_label, auction_label

    def __len__(self):
        return len(self.data)