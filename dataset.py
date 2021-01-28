import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

'''
The default file list name is 'file_list.tsv'
'''

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self,
                 img_dir,
                 transform=None,
                 get_loader=get_default_img_loader,
                 file_list=None):
        """
        Args:
            file_list (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading file list...')
        if file_list is None:
            file_list = os.path.join(img_dir, 'file_list.tsv')
        self.data = pd.read_csv(file_list, sep='\t', header=None)
        print('%d tsv data rows are successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
        I = self.loader(image_name)
        if self.transform is not None:
            I = self.transform(I)

        mos = self.data.iloc[index, 1]
        sample = {'I': I, 'y': mos, 'name': image_name}

        return sample

    def __len__(self):
        return len(self.data.index)

