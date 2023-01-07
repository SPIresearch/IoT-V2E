import pdb

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.signal as signal
from glob import glob

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset, target_channel_num):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        # X_train = np.load(np_dataset[0])["x"]
        X_train = np.load(np_dataset[0])["x"][:, target_channel_num, :]
        # pdb.set_trace()
        # X_train = X_train[:, 0, :] # zanshiqu
        y_train = np.load(np_dataset[0])["y"]
        # X_train:(1009, 3840)

        for np_file in np_dataset[1:]:
            # X_train = np.vstack((X_train, np.load(np_file)["x"]))
            X_train = np.vstack((X_train, np.load(np_file)["x"][:, target_channel_num, :]))
            y_train = np.append(y_train, np.load(np_file)["y"])
        # X_train:(25756, 3840)

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()

        self.len = X_train.shape[0]
        # self.x_data = torch.from_numpy(X_train)
        # self.y_data = torch.from_numpy(y_train).long()
        self.x_data = X_train.type(torch.FloatTensor) # 1079*3840*1
        # self.y_data = y_train.type(torch.FloatTensor)
        self.y_data = y_train

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size):
    train_dataset = LoadDataset_from_mat(training_files)
    test_dataset = LoadDataset_from_mat(subject_files)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts

class LoadDataset_from_mat(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_mat, self).__init__()

        # load files
        # X_train = np.load(np_dataset[0])["x"]
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]
        # X_train:(1009, 3840)

        for np_file in np_dataset[1:]:
            # X_train = np.vstack((X_train, np.load(np_file)["x"]))
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])
        # X_train:(25756, 3840)

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()

        self.len = X_train.shape[0]
        # self.x_data = torch.from_numpy(X_train)
        # self.y_data = torch.from_numpy(y_train).long()
        self.x_data = X_train.type(torch.FloatTensor) # 1079*3840*1
        # self.y_data = y_train.type(torch.FloatTensor)
        self.y_data = y_train

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 4:

            self.x_data = self.x_data.unsqueeze(1)
        test = 1
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




class LoadDataset_from_numpy_SleepEDF(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset, target_channel_num):
        super(LoadDataset_from_numpy_SleepEDF, self).__init__()

        # load files
        # X_train = np.load(np_dataset[0])["x"]
        X_train_temp = np.load(np_dataset[0])["x"][:,:, target_channel_num]
        X_train = signal.resample_poly(X_train_temp,up=128,down=100,axis=1)
        # pdb.set_trace()
        # X_train = X_train[:, 0, :] # zanshiqu
        y_train = np.load(np_dataset[0])["y"]
        # X_train:(1009, 3840)

        for np_file in np_dataset[1:]:
            # X_train = np.vstack((X_train, np.load(np_file)["x"]))
            X_train_temp = signal.resample_poly(np.load(np_file)["x"][:, :, target_channel_num],up=128,down=100,axis=1)
            X_train = np.vstack((X_train, X_train_temp))
            y_train = np.append(y_train, np.load(np_file)["y"])
        # X_train:(25756, 3840)

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()

        self.len = X_train.shape[0]
        # self.x_data = torch.from_numpy(X_train)
        # self.y_data = torch.from_numpy(y_train).long()
        self.x_data = X_train.type(torch.FloatTensor) # 1079*3840*1
        # self.y_data = y_train.type(torch.FloatTensor)
        self.y_data = y_train

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)




    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



class LoadDataset_from_numpy_FeatureExtract_my(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset, target_channel_num):
        super(LoadDataset_from_numpy_FeatureExtract_my, self).__init__()

        # load files
        # X_train = np.load(np_dataset[0])["x"]
        X_train = np.load(np_dataset[0])["x"][:, target_channel_num, :]
        # pdb.set_trace()
        # X_train = X_train[:, 0, :] # zanshiqu
        y_train = np.load(np_dataset[0])["y"]
        filp_counter = len(y_train) # 记录该用户有多少片段
        # X_train:(1009, 3840)

        for np_file in np_dataset[1:]:
            # X_train = np.vstack((X_train, np.load(np_file)["x"]))
            X_train = np.vstack((X_train, np.load(np_file)["x"][:, target_channel_num, :]))
            y_train = np.append(y_train, np.load(np_file)["y"])
            filp_counter = np.append(filp_counter, len(y_train))
        # X_train:(25756, 3840)

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()

        self.len = X_train.shape[0]
        self.filp_counter = filp_counter
        # self.len = filp_counter

        # self.x_data = torch.from_numpy(X_train)
        # self.y_data = torch.from_numpy(y_train).long()
        self.x_data = X_train.type(torch.FloatTensor) # 1079*3840*1
        # self.y_data = y_train.type(torch.FloatTensor)
        self.y_data = y_train

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np_OnlyShhs(subject_files_shhs):

    test_dataset = LoadDataset_from_numpy_FeatureExtract_shhs(subject_files_shhs)

    # to calculate the ratio for the CAL
    # all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = test_dataset.y_data
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]


    return test_dataset, counts

# 加载路径
def load_folds_data_shhs(np_data_path):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    print('shhs subjects:' + str(len(files)))

    r_p_path = r"guorongxiao/EEG_Video_Fusion/EEG_seg_jiqun/data_loader/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)

    # import pdb
    # pdb.set_trace()
    # npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    npzfiles = np.asarray(files , dtype='<U200')

    # train_files = np.array_split(npzfiles, n_folds)
    # folds_data = {}
    # for fold_id in range(n_folds):
    #     subject_files = train_files[fold_id]
    #     training_files = list(set(npzfiles) - set(subject_files))
    #     folds_data[fold_id] = [training_files, subject_files]
    return npzfiles



class LoadDataset_from_numpy_FeatureExtract_shhs(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset_shhs):
        super(LoadDataset_from_numpy_FeatureExtract_shhs, self).__init__()

        # load files
        # 加载SHHS
        X_train_shhs = np.load(np_dataset_shhs[0])["x"][:, :, :]
        X_train_shhs = np.squeeze(X_train_shhs)
        y_train_shhs = np.load(np_dataset_shhs[0])["y"]
        # X_train:(, 3840)
        filp_counter = len(y_train_shhs) # 记录该用户有多少片段

        for np_file in np_dataset_shhs[1:]:
            # X_train = np.vstack((X_train, np.load(np_file)["x"]))
            X_train_shhs_Temp = np.load(np_file)["x"][:, : , :]
            X_train_shhs_Temp = np.squeeze(X_train_shhs_Temp)
            X_train_shhs = np.vstack((X_train_shhs, X_train_shhs_Temp))
            y_train_shhs = np.append(y_train_shhs, np.load(np_file)["y"])
            filp_counter = np.append(filp_counter,len(y_train_shhs))
        # X_train:(, 3840)

        #叠加my和shhs
        X_train = X_train_shhs
        y_train = y_train_shhs

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()

        self.len = X_train.shape[0]
        self.filp_counter = filp_counter
        # self.x_data = torch.from_numpy(X_train)
        # self.y_data = torch.from_numpy(y_train).long()
        self.x_data = X_train.type(torch.FloatTensor) # 1079*3840*1
        # self.y_data = y_train.type(torch.FloatTensor)
        self.y_data = y_train

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


class LoadDataset_index_shhs(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset_shhs):
        super(LoadDataset_index_shhs, self).__init__()

        self.eeg_paths = sorted(glob(np_dataset_shhs + '/*/*.npy'))
        print("shhs:" + str(len(self.eeg_paths)))

        self.StageLabel =  {
                            'W': 0,
                            '1': 1,
                            '2': 2,
                            '3': 3,
                            'R': 4,
                           }

    def __load_eeg(self,single_path):
        path, file = os.path.split(single_path)
        name, suff = os.path.splitext(file)
        x = np.load(single_path)
        x = np.expand_dims(x,axis=0)
        y = self.StageLabel[name[-1]]
        return x, y

    def __getitem__(self, index):

        x, y = self.__load_eeg(self.eeg_paths[index])
        x = x.astype(np.float32)

        return x, y, index

    def __len__(self):
        return len(self.eeg_paths)