from dataset.prostate import UCL, Promise12
import dataset.transforms as T
from torch.utils import data
import torch
import argparse
from model.unet3d import UNet3D 
import numpy as np
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/workspace',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='prostate')
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=100,
                        help="epoch number (default: 100)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 50)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 10)")

    return parser
args = get_argparser().parse_args()
train_transform = T.Compose([
            # T.RandomResize(),
            # T.RandomCrop((16,96,96)),
            # T.RandomFlip(),
            # T.RandomRotate(),
            # T.RandomRotate90(),
            # T.Normalize(),
            # T.Standardize(),
            # T.AdditiveGaussianNoise(),
            T.ToTensor()
        ])
dataset = UCL(args.data_root, 't2w', train_transform)
for i in range(len(dataset)):
    x, y, _ = dataset.__getitem__(i)
    print(y.shape)