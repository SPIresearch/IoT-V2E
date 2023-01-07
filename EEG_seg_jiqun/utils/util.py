import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import random

import pandas as pd
import os
import numpy as np
from glob import glob
import math
import pdb


def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print ("============== ERROR =================")


    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

def load_folds_data_MyVersion(np_data_path):
    np_traindata_path = np_data_path + '/train'
    np_testdata_path = np_data_path + '/test'
    train_files = sorted(glob(os.path.join(np_traindata_path, "*.npz")))
    test_files = sorted(glob(os.path.join(np_testdata_path, "*.npz")))
    train_files_dict = dict()
    test_files_dict = dict()
    for i in train_files:
        file_name = os.path.split(i)[-1]
        file_num = file_name[0:8]
        train_files_dict[file_num] = [i]
    for j in test_files:
        file_name = os.path.split(j)[-1]
        file_num = file_name[0:8]
        test_files_dict[file_num] = [j]

    train_files_pairs = []
    test_files_pairs = []
    for key in train_files_dict:
        train_files_pairs.append(train_files_dict[key])
    for key in test_files_dict:
        test_files_pairs.append(test_files_dict[key])

    train_files_pairs = np.array(train_files_pairs)
    test_files_pairs = np.array(test_files_pairs)
    # files_pairs = files_pairs[r_permute]
    # files_pairs = random.shuffle(files_pairs) # zhe zhong da luan fang shi bu xing
    # pdb.set_trace()

    train_files = [item for sublist in train_files_pairs for item in sublist]
    test_files = [item for sublist in test_files_pairs for item in sublist]

    folds_data = [train_files, test_files]

    return folds_data


def load_folds_data_Mat(np_data_path, train_id, test_id, Train_test):

    all_files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    all_files = np.asarray(all_files)
    train_files = all_files[train_id]
    test_files = all_files[test_id]



    train_files_dict = dict()
    test_files_dict = dict()
    for i in train_files:
        file_name = os.path.split(i)[-1]
        file_num = file_name[0:6]
        train_files_dict[file_num] = [i]
    for j in test_files:
        file_name = os.path.split(j)[-1]
        file_num = file_name[0:6]
        test_files_dict[file_num] = [j]

    train_files_pairs = []
    test_files_pairs = []
    for key in train_files_dict:
        train_files_pairs.append(train_files_dict[key])
    for key in test_files_dict:
        test_files_pairs.append(test_files_dict[key])

    train_files_pairs = np.array(train_files_pairs)
    test_files_pairs = np.array(test_files_pairs)

    train_files = [item for sublist in train_files_pairs for item in sublist]
    test_files = [item for sublist in test_files_pairs for item in sublist]

    folds_data = [train_files, test_files]

    return folds_data




def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.lr_now = 0

    def __call__(self, optimizer, i, epoch):
        T = (epoch-1) * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** ((epoch-1) // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        self.lr_now = lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
        test = 1
