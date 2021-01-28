#!/bin/env python3
'''
This file is for generating dataset for this experiement from another dataset.

The source dataset should have some high quality images (preferrably Waterloo
Exploration Database by Ma et al.).
'''

import argparse
from pathlib import Path
import json
import random
import numpy as np
from random import randrange
from PIL import Image
from PIL import ImageFilter
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("src", type=str, help='The source directory containing the images',)
    parser.add_argument("dest", type=str, help='The destination directory (for output)')
    parser.add_argument('-m', "--maxrad", type=float,
            default=30.0,
            help='The maximum radius of blur')
    parser.add_argument('-v', "--verbose", action='store_true')
    parser.add_argument('-r', "--repeat", type=int,
            default=4,
            help='How many blurred images should be generated from the source image')
    parser.add_argument('-t', "--test", type=int,
            default=128,
            help='Generate images for testing purpose (randomly crop images to desired size)')
    parser.add_argument('-s', "--ssim", action='store_true',
            help='Use SSIM, instead of the size of radius as the score')
    parser.add_argument('-w', "--win-size", target='win_size', type=int,
            help='The window size for SSIM (not implmented yet)')

    return parser.parse_args()

def random_crop(img, matrix):
    x, y = img.size
    x1 = randrange(0, x - matrix)
    y1 = randrange(0, y - matrix)
    return img.crop((x1, y1, x1 + matrix, y1 + matrix)), (x1, y1)

def crop(img, start, matrix):
    x1, y1 = start
    return img.crop((x1, y1, x1 + matrix, y1 + matrix))

def main(config):
    src_dir = Path(config.src)
    dest_dir = Path(config.dest)
    bflist = [ ]

    dest_dir.mkdir(parents=True, exist_ok=True)

    for img_file in src_dir.iterdir():
        if img_file.suffix.lower() not in IMG_EXTENSIONS: continue
        if not img_file.is_file(): continue
        fpath = img_file.relative_to(src_dir)
        fname = str(fpath)
        if config.verbose:
            print('Processing', fname)
        im = Image.open(img_file)
        for i in range(config.repeat):
            if config.test:
                im2, cord = random_crop(im, config.test)
            else:
                im2 = im
            radius = random.uniform(0, config.maxrad)
            if config.ssim:
                imc = crop(im, cord, config.test)
                ssim_score = ssim(img_as_float(imc), img_as_float(im2), multichannel=True)
            bim = im2.filter(ImageFilter.GaussianBlur(radius))
            bfpath = fpath.with_stem('{}-{:04d}'.format(fpath.stem, i)).with_suffix('.bmp')
            bfname = str(bfpath)
            res = {
                'name': bfname,
                'radius': radius}
            if config.ssim:
                res['ssim'] = ssim_score
            bflist.append(res)
            bim.save(dest_dir / bfpath)

    with open(dest_dir / 'file_list.json', 'w') as f:
        json.dump(bflist, f)

    with open(dest_dir / 'file_list.tsv', 'w') as f:
        for item in bflist:
            if config.ssim:
                f.write('{}\t{}\n'.format(item['name'], item['ssim']))
            else:
                f.write('{}\t{}\n'.format(item['name'], item['radius']))

if __name__ == '__main__':
    config = parse_config()
    main(config)
