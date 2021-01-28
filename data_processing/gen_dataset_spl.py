#!/bin/env python3
'''
This file is for splitting a dataset to a trainig set and a testing set

Note that this script is not applicable to datasets like LIVE or CSIQ, because
many images have the same `ground truth'.
'''

import argparse
from pathlib import Path
import pandas as pd

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("src", type=str, help='The source file_list.tsv')
    parser.add_argument(
            "dest", type=str,
            help='directory where result files should be stored')
    parser.add_argument('-p', "--prefix", type=str,
            help='the prefix of created directories',
            default='p')
    parser.add_argument('-r', "--sample-rate", dest='sample_rate',
            type=float, default=0.8, help='The portion of traning (UNIQUE use 0,8 for CLIVE)')
    parser.add_argument('-v', "--verbose", action='store_true')

    return parser.parse_args()

config = parse_config()

def main(config):
    in_path = Path(config.src)
    out_path = Path(config.dest)
    out_path.mkdir(parents=True, exist_ok=True)
    trainings_dir = out_path / (config.prefix + 'training')
    trainings_path = trainings_dir / 'file_list.tsv'
    testings_dir = out_path / (config.prefix + 'testing')
    testings_path = testings_dir / 'file_list.tsv'
    trainings_dir.mkdir(parents=True, exist_ok=True)
    testings_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(in_path, sep='\t', header=None)
    trainings = data.sample(frac=config.sample_rate)
    testings = pd.concat([data, trainings]).drop_duplicates(keep=False)
    trainings.to_csv(trainings_path, sep='\t', header=False, index=False)
    testings.to_csv(testings_path, sep='\t', header=False, index=False)




if __name__ == '__main__':
    main(config)
