#!/bin/env python3
'''
This file is for generating a filelist from LIVE using Python
'''

import argparse
from pathlib import Path
import scipy.io
import pandas as pd

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
DIRECTORIES = [
        ('jp2k', (0, 227)),
        ('jpeg', (227, 460)),
        ('wn', (460, 634)),
        ('gblur', (634, 808)),
        ('fastfading', (808, 982))
        ]
SIZE = 982

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("src", type=str, help='The source directory')
    parser.add_argument("dest", type=str, help='The destination csv file')
    parser.add_argument('-v', "--verbose", action='store_true')

    return parser.parse_args()


config = parse_config()

def construct_filenames():
    filenames = []
    for dir, (s, t) in DIRECTORIES:
        for i in range(t - s):
            filenames.append('{}/img{}.bmp'.format(
                dir, (i+1)))
    return filenames

def main(config):
    in_path = Path(config.src)

    realigned_scores = scipy.io.loadmat(in_path / 'dmos_realigned.mat')
    dmos_new = realigned_scores['dmos_new'].flatten()
    dmos_std = realigned_scores['dmos_std'].flatten()
    is_orgs = realigned_scores['orgs'].flatten()
    ref_names = scipy.io.loadmat(in_path / 't.mat')['refnames_all'].flatten()
    ref_names = [str(x[0]) for x in ref_names]
    filenames = construct_filenames()
    df = pd.DataFrame({
        'filenames': filenames,
        'ref_names': ref_names,
        'is_orgs': is_orgs,
        'dmos_new': dmos_new,
        'dmos_std': dmos_std})
    df.to_csv(config.dest)






if __name__ == '__main__':
    config = parse_config()
    main(config)
