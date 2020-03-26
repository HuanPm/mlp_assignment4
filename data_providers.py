import numpy as np
import os

from torch.utils import data
from torch.utils.data import Dataset

class BridgeData(data.Dataset):
    def __init__(self, root, set_name, use_features):
        """
        Args:
            root (string): stored data folder name
            set_name (string): data set name (train, val, test)
            use_features (int): 1 - only features
                                2 - deal + features
                                else - deal
        """
        
        
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
        
        if use_features == 1:
            '''
            HCP(4): 0 ~ 3
            CONTROLS(4): 4 ~ 7
            HC(4): 8 ~ 11
            HONOR(4): 12 ~ 15
            RKCB(4): 16 ~ 19
            ACE(4): 24, 29, 34, 39
            KING(4): 23, 28, 33, 38
            QUEEN(4): 22, 27, 32, 37
            JACK(4): 21, 26, 31, 36
            TEN(4): 20, 25, 30, 35
            
            '''
            self.features = np.zeros((all_data.shape[0], 40))
            for i in np.arange(4):
                self.features[:, i] = self.data[:, i * 13 + 9] + 2 * self.data[:, i * 13 + 10] + \
                                        3 * self.data[:, i * 13 + 11] + 4 * self.data[:, i * 13 + 12] # HCP
                self.features[:, i + 4] = self.data[:, i * 13 + 11] + 2 * self.data[:, i * 13 + 12] # CONTROLS                     
                self.features[:, i + 8] = self.data[:, i * 13 + 10] + self.data[:, i * 13 + 11] + \
                                            self.data[:, i * 13 + 12] # HC
                self.features[:, i + 12] = self.features[:, i + 8] + \
                                            self.data[:, i * 13 + 8] + self.data[:, i * 13 + 9] # HONOR
                self.features[:, i + 16] = self.data[:, 12] + self.data[:, 25] + \
                                            self.data[:, 38] + self.data[:, 51] + \
                                            self.data[:, i * 13 + 11] # RKCB
            self.features[:, 20:25] = self.data[:, 8:13]
            self.features[:, 25:30] = self.data[:, 21:26]
            self.features[:, 30:35] = self.data[:, 34:39]
            self.features[:, 35:40] = self.data[:, 47:52]
        
            self.data = np.concatenate((self.features, self.data[:, 52:]), axis=1)
        elif use_features == 2:
            '''
            HCP(4): 0 ~ 3
            CONTROLS(4): 4 ~ 7
            HC(4): 8 ~ 11
            HONOR(4): 12 ~ 15
            RKCB(4): 16 ~ 19
            ACE(4): 24, 29, 34, 39
            KING(4): 23, 28, 33, 38
            QUEEN(4): 22, 27, 32, 37
            JACK(4): 21, 26, 31, 36
            TEN(4): 20, 25, 30, 35
            
            '''
            self.features = np.zeros((all_data.shape[0], 40))
            for i in np.arange(4):
                self.features[:, i] = self.data[:, i * 13 + 9] + 2 * self.data[:, i * 13 + 10] + \
                                        3 * self.data[:, i * 13 + 11] + 4 * self.data[:, i * 13 + 12] # HCP
                self.features[:, i + 4] = self.data[:, i * 13 + 11] + 2 * self.data[:, i * 13 + 12] # CONTROLS                     
                self.features[:, i + 8] = self.data[:, i * 13 + 10] + self.data[:, i * 13 + 11] + \
                                            self.data[:, i * 13 + 12] # HC
                self.features[:, i + 12] = self.features[:, i + 8] + \
                                            self.data[:, i * 13 + 8] + self.data[:, i * 13 + 9] # HONOR
                self.features[:, i + 16] = self.data[:, 12] + self.data[:, 25] + \
                                            self.data[:, 38] + self.data[:, 51] + \
                                            self.data[:, i * 13 + 11] # RKCB
            self.features[:, 20:25] = self.data[:, 8:13]
            self.features[:, 25:30] = self.data[:, 21:26]
            self.features[:, 30:35] = self.data[:, 34:39]
            self.features[:, 35:40] = self.data[:, 47:52]
        
            self.data = np.concatenate((self.data[:, :52], self.features[:, :20], self.data[:, 52:]), axis=1)

        print(set_name, 'data', self.data.shape)
        print(set_name, 'deal_labels', self.deal_labels.shape)
        print(set_name, 'auction_labels', self.auction_labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data, deal_label, auction_label) where target is index of the target class.
        """
        data = self.data[index]
        deal_label = self.deal_labels[index]
        auction_label = self.auction_labels[index]

        return data, deal_label, auction_label

    def __len__(self):
        return len(self.data)