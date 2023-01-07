import numpy as np
import random
from scipy.ndimage import rotate
import torch
import scipy.ndimage as ndimage
from skimage.transform import resize

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m1, m2, lbl):
        for t in self.transforms:
            m1, m2, lbl = t(m1, m2, lbl)
        return m1, m2, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomFlip:
    def __init__(self, p=0.5, **kwargs):
        self.axes = (0, 1, 2)
        self.p = p

    def __call__(self, m1, m2, lbl):
        for axis in self.axes:
            if random.random() < self.p:
                m1 = np.flip(m1, axis)
                m2 = np.flip(m2, axis)
                lbl = np.flip(lbl, axis)
        return m1, m2, lbl
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotate90:
    def __init__(self):
        self.axis = (1, 2)

    def __call__(self, m1, m2, lbl):
        k = random.randint(0,3)
        m1 = np.rot90(m1, k, self.axis)
        m2 = np.rot90(m2, k, self.axis)
        lbl = np.rot90(lbl, k, self.axis)
        return m1, m2, lbl

    def __repr__(self):
        return self.__class__.__name__

class RandomRotate:
    def __init__(self, angle_spectrum=15, mode='reflect', order=0, **kwargs):
        self.angle_spectrum = angle_spectrum
        self.axes = [(1, 0), (2, 1), (2, 0)]
        self.mode = mode
        self.order = order

    def __call__(self, m1, m2, lbl):
        axis = self.axes[random.randint(0,2)]
        angle = random.randint(-self.angle_spectrum, self.angle_spectrum)
        m1 = rotate(m1, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        m2 = rotate(m2, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        lbl = rotate(lbl, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        return m1, m2, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(angle={})'.format(self.angle_spectrum)


class Normalize:
    def __init__(self):
        pass

    def __call__(self, m1, m2, lbl):
        
        return (m1 - m1.min()) / (m1.max() - m1.min()), (m2 - m2.min()) / (m2.max() - m2.min()), lbl
    
    def __repr__(self):
        return self.__class__.__name__

class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, **kwargs):
        self.eps = eps

    def __call__(self, m1, m2, lbl):
        mean1 = np.mean(m1)
        std1 = np.std(m1)
        m1 = (m1 - mean1) / np.clip(std1, a_min=self.eps, a_max=None)

        mean2 = np.mean(m2)
        std2 = np.std(m2)
        m2 = (m2 - mean2) / np.clip(std2, a_min=self.eps, a_max=None)
        return m1, m2, lbl

    def __repr__(self):
        return self.__class__.__name__

class AdditiveGaussianNoise:
    def __init__(self, scale=(0.3, 0.7), p=0.1, **kwargs):
        self.p = p
        self.scale = scale

    def __call__(self, m1, m2, lbl):
        if random.random() < self.p:
            std = random.random()*( self.scale[1]-  self.scale[0]) + self.scale[0]
            gaussian_noise =  np.random.normal(0, std, size=m1.shape)
            return m1 + gaussian_noise, m2 + gaussian_noise, lbl
        return m1, m2, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(scale={}, {})'.format(self.scale[0], self.scale[1])

class RandomCrop:
    def __init__(self, size=(16, 128, 128)):
        self.size = size

    def __call__(self, m1, m2, lbl):
        depth, height, width  = m1.shape
        if depth<self.size[0]:
            m1 = np.pad(m1, (self.size[0]-depth, 0), mode='reflect')
            m2 = np.pad(m2, (self.size[0]-depth, 0), mode='reflect')
            lbl = np.pad(lbl, (self.size[0]-depth, 0), mode='reflect')
            sz = 0
        else:
            sz = random.randint(0, depth - self.size[0]-1)
        sx = random.randint(0, height - self.size[1]-1)
        sy = random.randint(0, width - self.size[2]-1)
        m1 = m1[sz:sz + self.size[0], sx:sx + self.size[1], sy:sy + self.size[2]]
        m2 = m2[sz:sz + self.size[0], sx:sx + self.size[1], sy:sy + self.size[2]]
        lbl = lbl[sz:sz + self.size[0], sx:sx + self.size[1], sy:sy + self.size[2]]
        return m1, m2, lbl
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={}, {}, {})'.format(self.size[0], self.size[1], self.size[2])

class CenterCrop:
    def __init__(self, size=(16, 128, 128)):
        self.size = size

    def __call__(self, m1, m2, lbl):
        depth, height, width  = m1.shape
        if depth<self.size[0]:
            m1 = np.pad(m1, (self.size[0]-depth, 0), mode='reflect')
            m2 = np.pad(m2, (self.size[0]-depth, 0), mode='reflect')
            lbl = np.pad(lbl, (self.size[0]-depth, 0), mode='reflect')
            sz = 0
        else:
            sz = (depth - self.size[0] - 1) // 2

        sx = (height - self.size[1] - 1) // 2
        sy = (width - self.size[2] - 1) // 2
        m1 = m1[sz:sz + self.size[0], sx:sx + self.size[1], sy:sy + self.size[2]]
        m2 = m2[sz:sz + self.size[0], sx:sx + self.size[1], sy:sy + self.size[2]]
        lbl = lbl[sz:sz + self.size[0], sx:sx + self.size[1], sy:sy + self.size[2]]
        return m1, m2, lbl
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={}, {}, {})'.format(self.size[0], self.size[1], self.size[2])

class RandomResize:
    def __init__(self, scale = 1.3, p=0.5 ):
        self.size = scale
        self.p = p

    def __call__(self, m1, m2, lbl):
        if random.random() < self.p:
            depth, height, width  = m1.shape[-3:]
            m1 = resize(m1, (depth*self.size, height*self.size, width*self.size), preserve_range=True)
            m2 = resize(m2, (depth*self.size, height*self.size, width*self.size), preserve_range=True)
            lbl = resize(lbl, (depth*self.size, height*self.size, width*self.size), preserve_range=True)
        return m1, m2, lbl
    
    def __repr__(self):
        return self.__class__.__name__ + '(scale={})'.format(self.scale)

class ToTensor:
    def __init__(self, dtype=np.float32, **kwargs):
        self.dtype = dtype

    def __call__(self, m1, m2, lbl):
        return torch.from_numpy(m1.astype(dtype=self.dtype)), torch.from_numpy(m2.astype(dtype=self.dtype)), torch.from_numpy(lbl.astype(dtype=self.dtype))

    def __repr__(self):
        return self.__class__.__name__


def random_zoom(img_numpy, min_percentage=0.8, max_percentage=1.1):
    """
    :param img_numpy: 
    :param min_percentage: 
    :param max_percentage: 
    :return: zoom in/out aigmented img
    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomZoom(object):
    def __init__(self, min_percentage=0.8, max_percentage=1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(self, m1, m2, lbl):
        m1 = random_zoom(m1, self.min_percentage, self.max_percentage)
        m2 = random_zoom(m2, self.min_percentage, self.max_percentage)
        lbl = random_zoom(lbl, self.min_percentage, self.max_percentage)
        return m1, m2, lbl
      