import SimpleITK as sitk
from skimage.exposure import equalize_adapthist
import os
import numpy as np
import cv2
from tqdm import tqdm
def data_to_array(base_path, store_path, img_rows, img_cols):
    fileList =  os.listdir(base_path)
    fileList = sorted((x for x in fileList if '.mhd' in x))

    for filename in fileList:
        itkimage = sitk.ReadImage(os.path.join(base_path, filename))
        imgs = sitk.GetArrayFromImage(itkimage)
        print(imgs.shape)
        # if 'segm' in filename.lower():
        #     imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
        # else:
        #     imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        # save_name = filename.split('.')[0]+'.npy'
        # np.save(os.path.join(store_path, save_name), imgs)

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
            img = smooth_images(img)

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """
    img = sitk.GetImageFromArray(imgs)
    img = sitk.CurvatureFlow(image1=img,timeStep=t_step,numberOfIterations=n_iter)
    return sitk.GetArrayFromImage(img)



data_dir = r'C:\Users\69068\Downloads\pro12'
np_data_path = r'C:\Users\69068\Downloads\npy'
data_to_array(data_dir, np_data_path, 256, 256)