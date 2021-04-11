'''Class-based methods

Some of these are adapted from torch
'''
from . import functional as F
import torch
import numpy as np
from torchvision.transforms import functional as TF


#import random_dequantize
#from .functional import to_pil_image, to_tensor, to_np_rgb_array

class RandomDequantize(torch.nn.Module):
    '''Convert a Numpy uint8 image with shape (c, h, w) to a Numpy float32,
    image using randomised method, each element should be in [0, 1) or
    [0, 1], as specified in the parameter.

    See the technical report for the rationale of this way or dequantisation.

    @param closed_one: if set, the inteval of random valuable will be [0, 1]
    '''
    def __init__(self, closed_one):
        self.closed_one = closed_one

    def __call__(self, x):
        return F.random_dequantize(x, self.closed_one)

class ToPILImage(torch.nn.Module):
    '''Kept here for consistency, DO NOT USE this in new code
    '''
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        return F.to_pil_image(pic, self.mode)

class ToNumpyRGBArray(torch.nn.Module):
    '''
    Converting a np image with shape (c, h, w) of float or int type to int type

    @param closed_one: if set, the inteval of random valuable will be [0, 1]
    '''
    def __init__(self, closed_one):
        self.closed_one = closed_one

    def __call__(self, x):
        return F.to_np_rgb_array(x, self.closed_one)


class ToTensor:
    ''' Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    The fallback function for this class is
    ``torchvision.transforms.functional.to_tensor``.

    If the input is a PIL Image, the fallback method will be called.

    If the input is a Numpy array with shape (c, h, w), to_tensor will be
    called.

    If the input is a Numpy array with other shall, the fallback method will
    be called.

    The method checks whether the input has a shape (c, h, w), using
    x.shape[0] == 1 or x.shape[0] == 3.

    NOTE: this class has different behaviour than functional.to_tensor
    '''
    def __call__(self, x):
        if type(x) == np.ndarray and (
                x.shape[0] == 1 or x.shape[0] == 3):
            return F.to_tensor(x)
        return TF.to_pil_image(pic, self.mode)


