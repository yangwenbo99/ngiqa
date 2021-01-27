#!/bin/env python3
'''
This file is for generating dataset for this experiement from the live dataset.

The source dataset should have some high quality images (preferrably Waterloo
Exploration Database by Ma et al.).
'''

import argparse
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import json
import random
import numpy as np
from random import randrange
from PIL import Image
from PIL import ImageFilter
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
TRAINING_REF_NUM = 23

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("lst", type=str, help='The filelist the images',)
    parser.add_argument("dir", type=str, help='The source directory containing the images',)
    parser.add_argument('-m', "--maxrad", type=float,
            default=30.0,
            help='The maximum radius of blur')
    parser.add_argument('-v', "--verbose", action='store_true')

    return parser.parse_args()

def save(df: pd.DataFrame, location: Path):
    with open(location, 'w') as f:
        for index, row in df.iterrows():
            f.write('{}\t{}\n'.format('../' + row['filenames'], row['dmos_new'] / 100))

def main(config):
    dbdir = Path(config.dir)
    training_dir = dbdir / 'ptraining'
    testing_dir = dbdir / 'ptesting'
    training_dir.mkdir(exist_ok=True)
    testing_dir.mkdir(exist_ok=True)

    live_list = pd.read_csv(config.lst)

    refs = shuffle(live_list['ref_names'].unique())
    selected_refs = refs[:TRAINING_REF_NUM]
    training_list = live_list[live_list.ref_names.isin(selected_refs) & ~live_list.is_orgs]
    testing_list = live_list[~live_list.ref_names.isin(selected_refs) & ~live_list.is_orgs]
    save(training_list, training_dir / 'file_list.tsv')
    save(testing_list, testing_dir / 'file_list.tsv')

if __name__ == '__main__':
    config = parse_config()
    main(config)
