#!/bin/env python3
'''
This file is for generating dataset for this experiement from the BID dataset
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
            help='Directory to BID\'s release')
    parser.add_argument('-v', "--verbose", action='store_true')

    return parser.parse_args()

def save(moss, fnames, location: Path):
    with open(location, 'w') as f:
        for mos, fname in zip(moss, fnames):
            f.write('{}\t{}\n'.format(
                '../ImageDatabase/' + str(fname[0]),
                mos))

def main(config):
    dbdir = Path(config.dir)
    save_dir = dbdir / 'full_list'
    save_dir.mkdir(parents=True, exist_ok=True)

    imdb = scipy.io.loadmat(dbdir / 'imdb.mat')
    moss = imdb['images'][0][0][0].flatten() / 5
    fnames = imdb['images'][0][0][1].flatten()

    save(moss, fnames, save_dir / 'file_list.tsv')

if __name__ == '__main__':
    config = parse_config()
    main(config)

