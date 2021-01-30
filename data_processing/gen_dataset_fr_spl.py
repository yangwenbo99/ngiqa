#!/bin/env python3
'''
This file is for splitting a dataset to a trainig set and a testing set

Note that this script is not applicable to datasets like LIVE or CSIQ, because
many images have the same `ground truth'.
'''

import argparse
from pathlib import Path
from sklearn.utils import shuffle
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
    parser.add_argument('-n', "--ntraining", type=int,
            help='The number of training items',
            required=True)
    parser.add_argument('-r', "--sample-rate", dest='sample_rate',
            type=float, default=0.8, help='The portion of traning (UNIQUE use 0,8 for CLIVE)')
    parser.add_argument('-v', "--verbose", action='store_true')

    return parser.parse_args()

config = parse_config()

def save(df: pd.DataFrame, location: Path):
    with open(location, 'w') as f:
        for index, row in df.iterrows():
            f.write('{}\t{}\n'.format(row[0], row[1]))


def main(config):
    in_path = Path(config.src)
    out_path = Path(config.dest)
    out_path.mkdir(parents=True, exist_ok=True)
    trainings_dir = out_path / (config.prefix + 'training')
    testings_dir = out_path / (config.prefix + 'testing')
    trainings_dir.mkdir(parents=True, exist_ok=True)
    testings_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(in_path, sep='\t', header=None)
    refs = shuffle(data[2].unique())
    print(refs)
    selected_refs = refs[:config.ntraining]
    training_list = data[data[2].isin(selected_refs)]
    testing_list = data[~data[2].isin(selected_refs)]
    save(training_list, trainings_dir / 'file_list.tsv')
    save(testing_list, testings_dir / 'file_list.tsv')



if __name__ == '__main__':
    main(config)
