import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pandas as pd

class UCL(Dataset):
    def __init__(self, root, modality, transforms=None, file='dataset/prostate/ucl.csv'):
        self.root = os.path.join(root, 'ucl_dataset')
        print(file)
        assert modality in ['t2w', 'adc']
        self.modality = modality
        self.transforms = transforms
        self.df = pd.read_csv(os.path.join(root, file) )
        self.xs = self.df[self.modality]
        self.ys = self.df['prostate_mask']
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        x = os.path.join(self.root, self.xs[index])
        y = os.path.join(self.root, self.ys[index])
        x = nib.load(x).get_data().transpose(2,0,1)
        y = nib.load(y).get_data().transpose(2,0,1)
        if self.transforms is not None:
            x, y = self.transforms(x ,y)
        # x = x.permute(1,2,0) 
        # y = y.permute(1,2,0)
        return x.unsqueeze(0), y.unsqueeze(0), index
