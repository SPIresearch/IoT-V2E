import numpy as np

path = r'D:\code\EEG_Video_Fusion\DataSample\睡眠数据（EEG与Video）\SHHS_C4_Big\shhs1-200010.npz'
aa = np.load(path)
x = aa['x']
y = aa['y']