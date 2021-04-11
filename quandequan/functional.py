from PIL import Image
import numpy as np
import torch
from pathlib import Path
import tifffile
from torchvision import transforms
from torchvision.transforms import functional_pil as F_pil
import sys

def random_dequantize(img: np.ndarray, closed_one=False):
    '''Convert a Numpy uint8 image with shape (c, h, w) to a Numpy float32,
    image using randomised method, each element should be in [0, 1) or
    [0, 1], as specified in the parameter.

    See the technical report for the rationale of this way or dequantisation.

    @param closed_one: if set, the inteval of random valuable will be [0, 1]
    '''
    if closed_one:
        raise NotImplemented('Mapping onto [0, 1] it not supported yet')
    img = img.astype('float32')
    img = img + np.random.random(size=img.shape)
    img /= 256
    return img

def to_pil_image(x, mode='F'):
    print(
            'Warning:',
            'You are trying to cast a float32 RGB image to PIL Image.',
            'This is destructive!',
            file=sys.stderr)
    if mode != 'F':
        raise NotImplemented('Mode must be float32')
    return transforms.functional.to_pil_image(x, mode)

def save_img(im, fp, format='TIFF', **params):
    '''im a np array, representing the image, with the shape (c, h, w)
    '''
    suffix = None
    if format == 'SPIDER':
        suffix = '.spi'
    elif format == 'TIFF':
        suffix = '.tiff'
    else:
        raise NotImplemented('format must be spider or TIFF')

    if format != 'TIFF':
        if type(fp) == str and not fp.endswith(suffix):
            fp = fp + suffix
        return im.save(im, fp, format=format, **params)
    else:
        tifffile.imwrite(fp, im, compression='zlib') # , metadata={'axes': 'RGB'})

def read_img(fname) -> np.ndarray:
    '''Read a image to the shape (c, h, w)

    This will NOT change the format of the image

    NOTE that if a file is saved as float32, it can be loaded as float64
    '''
    pf = Path(fname)
    if pf.suffix.lower() == '.tiff':
        return tifffile.imread(fname).astype(np.float32)
    else:
        npimg = np.array(Image.open(pf))
        return npimg.transpose((2, 0, 1))

def to_np_array(im):
    '''Convert a monochorm PIL image to floating point Numpy array
    TODO: remove the method, because it should not be used
    '''
    if im.mode == 'F':
        return np.array(im, np.float32, copy=False)
    else:
        raise NotImplemented('mode must be float32')

def to_np_rgb_array(npimg: np.ndarray, closed_one=False):
    '''
    Converting a np image with shape (c, h, w) of float or int type to int type

    @param closed_one: if set, the inteval of random valuable will be [0, 1]
    '''
    if closed_one:
        raise NotImplemented('Mapping onto [0, 1] it not supported yet')

    if npimg.dtype == np.float32 and npimg.dtype == np.float64:
        npimg = np.floor(npimg * 256)
        npimg = npimg.clip(0, 255)
        npimg = np.rint(npimg).astype('uint8')
    return npimg.transpose((1, 2, 0))



def to_tensor(x):
    if F_pil._is_pil_image(x):
        return transforms.functional.to_tensor(x)
    else:
        return torch.tensor(x)

