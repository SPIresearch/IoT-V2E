import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pandas as pd

class UCLm(Dataset):
    def __init__(self, root, transforms=None, file='dataset/prostate/ucl.csv'):
        self.root = os.path.join(root, 'ucl_dataset')
        self.transforms = transforms
        self.df = pd.read_csv(os.path.join(root, file) )
        self.xs_t2w = self.df['t2w']
        self.xs_adc = self.df['adc']
        self.ys = self.df['prostate_mask']
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        t2w = os.path.join(self.root, self.xs_t2w[index])
        adc = os.path.join(self.root, self.xs_adc[index])
        y = os.path.join(self.root, self.ys[index])
        t2w = nib.load(t2w).get_data().transpose(2,0,1)
        adc = nib.load(adc).get_data().transpose(2,0,1)
        y = nib.load(y).get_data().transpose(2,0,1)
        if self.transforms is not None:
            t2w, adc, y = self.transforms(t2w, adc, y)
        return t2w.unsqueeze(0), adc.unsqueeze(0), y.unsqueeze(0), index
