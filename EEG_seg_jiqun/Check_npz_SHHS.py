import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from data_loader.data_loaders import *
# from utils.util import *


# orig_data_path = r'D:\code\EEG_Video_Fusion\DataSample\睡眠数据（EEG与Video）\SHHS_C4_Big'
# output_path = r'D:\code\EEG_Video_Fusion\DataSample\睡眠数据（EEG与Video）\SHHS_C4_Slice'
orig_data_path = r'LD-Data/SHHS/Resample329'
output_path = r'guorongxiao/EEG_Video_Fusion/Data/SHHS_Clip_C4'
folds_data = load_folds_data_shhs(orig_data_path)
# path_npz = r"/media/mtc206/83bf0d0c-3ad8-4186-8a37-4a38046cb44c/MySleepAttn_NoVali/Data_Prepare/NPZ_data_TrainTest/test/20210413.npz"

Channel_name = {
    0 : 'ECG1-ECG2',
    1 : 'C3-M2',
    2 : 'C4-M1',
    3 : 'E1-M2',
    4 : 'E2-M2',
    5 : 'Chin 1-Chin 2',
    6 : 'SpO2',
    7 : 'Pressure',
    8 : 'Therm',
    9 : 'Thor',
    10 : 'Abdo',
    11 : 'Snore',
    12 : 'PositionSen'
}

StageLabel ={
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "R",
}






for index_ii, ii in enumerate(folds_data):
    My_npz = np.load(ii)
    X_sample = My_npz["x"]
    X_sample = np.squeeze(X_sample)
    Y_sample = My_npz["y"]
    freqs = My_npz["fs"]
    count = index_ii + 1

    path_temp, file_temp = os.path.split(ii)
    date_temp, suffix_temp = file_temp.split('.')
    output_path_temp = os.path.join(output_path, date_temp)
    if not os.path.isdir(output_path_temp):
        os.makedirs(output_path_temp)

    for jj in range(0,len(Y_sample)):
        x_temp = np.squeeze(X_sample[jj,:])
        y_temp = Y_sample[jj]
        label_str = StageLabel[y_temp]
        epoch_str = str(jj)
        output_filename_temp = 'EEG_' + date_temp + '_' + epoch_str + '_' + label_str
        np.save(os.path.join(output_path_temp, output_filename_temp), x_temp)

    print('Subject {} has fininshed'.format(str(count)))

