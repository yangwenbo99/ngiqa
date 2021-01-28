#!/bin/env python3
'''
This file is for generating dataset for this experiement from the LIVE dataset.
'''

import argparse
from pathlib import Path
import pandas as pd
import scipy.io
from sklearn.utils import shuffle
import json
import random
import numpy as np
from random import randrange
from PIL import Image
from PIL import ImageFilter
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

SKIP_IMG_NUM = 7

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("dir", type=str,
            help='Directory to CLIVE\'s release')
    parser.add_argument('-v', "--verbose", action='store_true')

    return parser.parse_args()

def save(moss, fnames, location: Path):
    with open(location, 'w') as f:
        for mos, fname in zip(moss, fnames):
            f.write('{}\t{}\n'.format(
                '../Images/' + str(fname[0]),
                mos / 100))

def main(config):
    dbdir = Path(config.dir)
    img_dir = dbdir / 'Images'
    data_dir = dbdir / 'Data'
    save_dir = dbdir / 'full_list'
    save_dir.mkdir(parents=True, exist_ok=True)

    moss = scipy.io.loadmat(data_dir / 'AllMOS_release.mat')['AllMOS_release'].flatten()
    fnames = scipy.io.loadmat(data_dir / 'AllImages_release.mat')['AllImages_release'].flatten()

    moss = moss[SKIP_IMG_NUM:]
    fnames = fnames[SKIP_IMG_NUM:]

    save(moss, fnames, save_dir / 'file_list.tsv')

if __name__ == '__main__':
    config = parse_config()
    main(config)
