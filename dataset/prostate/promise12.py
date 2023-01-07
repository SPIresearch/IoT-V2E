from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch

class Promise12(Dataset):
    
    def __init__(self, root, modality='t2w', transforms=None, file ='dataset/prostate/PROMISE12.csv' ):
        self.root = os.path.join(root, 'PROMISE12')
        assert modality == 't2w'
        self.modality = modality
        self.transforms = transforms
        self.df = pd.read_csv(os.path.join(root, file) )
        self.xs = self.df[self.modality]
        self.ys = self.df['mask']
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        x = os.path.join(self.root, self.xs[index])
        y = os.path.join(self.root, self.ys[index])
        x = np.load(x).transpose(0,2,1)
        y = np.load(y).transpose(0,2,1)
        if self.transforms is not None:
            x, y = self.transforms(x ,y)
        return x.unsqueeze(0), y.unsqueeze(0), index
