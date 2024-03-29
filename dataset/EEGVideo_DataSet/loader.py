import io
import numpy as np
from os import path
import h5py
from PIL import Image


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path,'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path))


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = path.join(video_path, self.image_name_formatter(i))
            if path.exists(image_path):
                video.append(self.image_loader(image_path))

        return video


class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:

            flow_data = []
            for flow in self.flows:
                flow_data.append(f[f'video_{flow}'])

            video = []
            for i in frame_indices:
                if i < len(flow_data[0]):
                    frame = [
                        Image.open(io.BytesIO(video_data[i]))
                        for video_data in flow_data
                    ]
                    frame.append(frame[-1])  # add dummy data into third channel
                    video.append(Image.merge('RGB', frame))

        return video


class EEGFeatureLoader(object):
    # load an audio feature stored as numpy file ('.npy)
    def __init__(self):
        self.npyloader = NumpyLoader()

    def __call__(self, filename):
        if path.isfile(filename):
            features = self.npyloader(filename)
        else:
            features = None
        return features

class EEGClipLoader(object):
    # load eeg signal stored as numpy file ('.npy)
    def __init__(self):
        self.npyloader = NumpyLoader()

    def __call__(self, filename):
        if path.isfile(filename):
            features = self.npyloader(filename)
            features = np.expand_dims(features,axis=0)
            test = 1
        else:
            features = None
            print("对应的脑电数据为空")
        return features


class NumpyLoader(object):

    def __call__(self, path):
       return np.load(path)