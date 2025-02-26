# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TabularDataset(Dataset):
    def __init__(self, csv_file, log=False):
        super().__init__()
        df = pd.read_csv(csv_file)
        numpy_array = df.values
        self.X_num = torch.tensor(numpy_array[:,:1], dtype=torch.float)
        if log:
            self.X_num = torch.log(self.X_num+1)
        self.X_cat = torch.tensor(numpy_array[:,1:], dtype=torch.long)

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

data_dict = {
    'taxi':[10],
    'amazon':[16],
    'stackoverflow':[22],
    'retweet':[3],
    'taobao':[20]
}