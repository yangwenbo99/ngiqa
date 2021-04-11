#!/bin/env python3

'''
This script is used to compare the simlarity of models.


Warning: the script is only guarentee to work correctly for python 3.8 and 3.9
'''

import configargparse as argparse
import scipy.stats
import matplotlib.pyplot as plt
import model_combined as model
import numpy as np
import torch
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from typing import List
import json
from torch import nn
from pathlib import Path

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add(
            '-c', '--config', required=False,
            is_config_file=True, help='config file path')

    parser.add_argument('-o', "--model", type=str, default='VE2EUIQA')
    parser.add_argument("--backbone", type=str, default='resnet18',
            help='The backbone for some models')
    parser.add_argument("--representation", type=str, default="BCNN")
    parser.add_argument("--phase1", type=int, default=2,
            help='How many epoches should be used for phase 1 training (this is only applicable for some models)')
    parser.add_argument("--phase1_lr", type=float, default=0.0)
    parser.add_argument("--lossfn", type=str, default='default')
    parser.add_argument("--eval_lossfn", type=str, default='default')
    parser.add_argument("--loss_param1", type=float, default=1.5, help='The parameter')

    parser.add_argument("--adaptive_resize", action='store_true')

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('-e', "--eval", action='store_true')
    parser.add_argument("--test_gradient_length", action='store_true')
    parser.add_argument("--test_loss_gradient_length", action='store_true')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument('-f', "--fresh", action='store_true', help='not to resume')
    parser.add_argument("--seed", type=int, default=19901116)

    parser.add_argument("--train_live", type=str)
    parser.add_argument("--train_csiq", type=str)
    parser.add_argument("--train_kadid", type=str)
    parser.add_argument("--train_clive", type=str)
    parser.add_argument("--train_koniq", type=str)
    parser.add_argument("--train_bid", type=str)

    parser.add_argument("--test_live", type=str,
            help="There is no means to adjust the size in the test set, one must ensure it by either choosing models or process sets")
    parser.add_argument("--test_csiq", type=str)
    parser.add_argument("--test_kadid", type=str)
    parser.add_argument("--test_clive", type=str)
    parser.add_argument("--test_koniq", type=str)
    parser.add_argument("--test_bid", type=str)

    parser.add_argument("--repeat_dataset", action='store_true', help='Repeat datasets')

    parser.add_argument('--ckpt_path', default='./checkpoint_combined/', type=str,
                        metavar='PATH', help='path to checkpoints')
    # parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument('-n', '--normalize', action='store_true',
            help='Whether to normalise the input')
    parser.add_argument('-v', '--verbose', action='count', default=0)

    parser.add_argument('--adversarial', default=None, type=str,
            help='If set, the adversarial attack method will be used, currently only support FGSM')
    parser.add_argument('--adversarial_radius', default=0., type=float,
            help='The radius')
    parser.add_argument("--attack_param1", type=float, default=10, help='The parameter')

    parser.add_argument('--regularizer', default='', type=str,
            help='regularizer')
    parser.add_argument('--reg_strength', default=5e-2, type=float,
            help='The strength')

    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--crop_test", action='store_true',
            help='If set, tests will also be randomly cropped')
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)  # originall 1e-4
    parser.add_argument("--decay_interval", type=int, default=4)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)

    parser.add_argument("--test_correlation", action='store_true',
            help='If set, correlation factors will also be shown')
    parser.add_argument("--train_correlation", action='store_true',
            help='If set, correlation factors will also be shown')
    parser.add_argument("--eval_adversarial", action='store_true',
            help='If set, pair-wise accuracy will be tested')

    parser.add_argument('-j', '--json', '--json-output', dest='json_output',
            default='', type=str,
            help='if set, the test result shall be output to a JSON file')

    parser.add_argument('--md', '--markdown', dest='markdown_report',
            default='', type=str,
            help='if set, a markdown report will be written into such directory')

    parser.add_argument('--cm', '--cross-model', dest='cross_model',
            action='store_true', help='cross model')

    parser.add_argument('--sm', '--single-model', dest='single_model',
            action='store_true',
            help='if set, analysis on models will be done')

    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--smstep', '--single-model-step', dest='single_model_step', default=1, type=int)
    parser.add_argument('--advtr', '--test-adversarial-training', dest='test_adversarial_training', action='store_true')

    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--log_file", type=str, default='./train_log.txt')

    config = parser.parse_args()
    if config.eval:
        config.train = False
    if config.fresh:
        config.resume = False
    return config


def model_parameters(model: nn.Module):
    return { name: param for name, param in model.named_parameters() }

def compare_networks(trainer1: model.Trainer, trainer2: model.Trainer):
    '''
    returns:
        {
            'param_name: {
                'similarity_name': value
            }
        }
    '''

    def rmae(v1, v2):
        return torch.abs(v1 - v2).sum() / torch.sqrt(torch.abs(v1).sum() * torch.abs(v2).sum())

    model1 = trainer1.model
    model2 = trainer2.model
    params1 = model_parameters(model1)
    params2 = model_parameters(model2)
    assert(params1.keys() == params2.keys())
    res = { }
    for key in params1.keys():
        res[key] = {
                'corr': scipy.stats.mstats.pearsonr(params1[key].detach().cpu(), params2[key].detach().cpu())[0],
                'rmae': rmae(params1[key], params2[key]).detach().cpu()
                }
    return res

def get_trainer(ckpt_path, config):
    this_config = argparse.Namespace(**vars(config))
    this_config.ckpt = ckpt_path.name if ckpt_path else None
    if not ckpt_path: this_config.resume = False
    if this_config.batch_size > 32: this_config.batch_size = 32
    if this_config.test_batch_size > 32: this_config.test_batch_size = 32
    trainer = model.Trainer(this_config)
    return trainer

def get_fgsm_trainer(ckpt_path, radius, config):
    this_config = argparse.Namespace(**vars(config))
    this_config.ckpt = ckpt_path.name if ckpt_path else None
    if not ckpt_path: this_config.resume = False
    # this_config.eval = True
    # this_config.train = False
    # this_config.eval_adversarial = True
    this_config.adversarial = 'FGSM'
    this_config.adversarial_radius = radius
    if this_config.batch_size > 32: this_config.batch_size = 32
    if this_config.test_batch_size > 32: this_config.test_batch_size = 32
    trainer = model.Trainer(this_config)
    return trainer

def compare(config: argparse.Namespace):
    checkpoint_directory = Path(config.ckpt_path)
    checkpoint_paths = sorted(list(checkpoint_directory.iterdir()))
    step = config.step
    if step != 1:
        checkpoint_paths = [x for i, x in enumerate(checkpoint_paths) if i % step == 0]
    checkpoint_paths.append(None)


    res = { }
    for i in range(len(checkpoint_paths)):
        trainer1 = get_trainer(checkpoint_paths[i], config)
        for j in range(i + 1, len(checkpoint_paths)):
            trainer2 = get_trainer(checkpoint_paths[j], config)
            res[(i, j)] = {
                    'checkpoint1': checkpoint_paths[i].stem if checkpoint_paths[i] else None,
                    'checkpoint2': checkpoint_paths[j].stem if checkpoint_paths[j] else None,
                    'similarities': compare_networks(trainer1, trainer2)
                    }

    return checkpoint_paths, res

def single_model_statistics(config: argparse.Namespace):
    checkpoint_directory = Path(config.ckpt_path)
    checkpoint_paths = sorted(list(checkpoint_directory.iterdir()))
    step = config.single_model_step
    if step != 1:
        checkpoint_paths = [x for i, x in enumerate(checkpoint_paths) if i % step == 0]
    checkpoint_paths = [ None ] + checkpoint_paths

    res = [ ]

    def build_adversarial_table(radius, train=False):
        if config.verbose:
            print('Building adversarial table')
        res = { }
        adv_trainer = get_fgsm_trainer(checkpoint_path, radius, config)
        if config.verbose:
            print(adv_trainer.config)
        for dataset in adv_trainer.datasets:
            if config.verbose:
                print(f'    Dealing with {dataset.name}')
            if train:
                tab = adv_trainer.eval_adversarial(loader=dataset.train)
            else:
                tab = adv_trainer.eval_adversarial(loader=dataset.test)
            tab = {key: float(value) for key, value in tab.items()}
            res[dataset.name] = tab
        del adv_trainer
        return res

    def get_score(checkpoint_path, config):
        trainer = get_trainer(checkpoint_path, config)
        res = {
            'testing scores': {
                dataset.name: trainer.eval(dataset.test)
                for dataset in trainer.datasets if len(dataset.test) > 0
                },
            'training scores': {
                dataset.name: trainer.eval(dataset.train)
                for dataset in trainer.datasets if len(dataset.train) > 0
                }
            }
        del trainer
        return res

    for checkpoint_path in checkpoint_paths:
        x = get_score(checkpoint_path, config)
        # print(x)
        res.append({
            'checkpoint': checkpoint_path.name if checkpoint_path else None,
            'testing scores': x['testing scores'],
            'training scores': x['training scores'],
            'adversarial': {
                radius: build_adversarial_table(radius) for radius in [2e-2, 5e-2, 1e-1]
                },
            'adversarial_training': {
                radius: build_adversarial_table(radius, train=True) if config.test_adversarial_training else {} for radius in [2e-2, 5e-2, 1e-1]
                }
            })
    return checkpoint_paths, res

def shape_report(config: argparse.Namespace):
    this_config = argparse.Namespace(**vars(config))
    this_config.ckpt = None
    trainer = model.Trainer(this_config)
    res = { }
    for name, param in trainer.model.named_parameters():
        res[name] = param.shape

    return res


def res_to_jsonable(checkpoint_paths, raw_res: dict):
    processed_res = { }
    for key, value in raw_res.items():
        new_key = str(key)
        processed_similarities  = {k: {n: float(v)} for k, d in value['similarities'].items() for n, v in d.items()}
        processed_res[new_key] = {
                'checkpoint1': value['checkpoint1'],
                'checkpoint2': value['checkpoint2'],
                'similarities': processed_similarities
                }
    return processed_res

def markdown_single_model_report(report_path: Path, checkpoint_paths: List[Path], res: List[dict]):

    def plot_dict(d: dict):
        for key, value in d.items():
            plt.plot(value, label=key)
        plt.legend()

    # Generate matrices first
    testing_datasets = res[0]['testing scores'].keys()
    testing_scores = { dsname: [ x['testing scores'][dsname] for x in res ]
            for dsname in testing_datasets }
    testing_labels = [ dsname + '-' + lname for dsname in testing_datasets
            for lname in ['SRCC', 'PLCC'] ]
    training_datasets = res[0]['training scores'].keys()
    training_scores = { dsname: [ x['training scores'][dsname] for x in res ]
            for dsname in training_datasets }
    training_labels = [ dsname + '-' + lname for dsname in training_datasets
            for lname in ['SRCC', 'PLCC'] ]
    # print('DSs:', testing_datasets)
    # print('DSs:', training_datasets)

    # dataset_names = set(
    adversarial_radius = res[0]['adversarial'].keys()
    adversarials = {
            radius: {
                daname: {
                    'total': [ x['adversarial'][radius][daname]['total'] for x in res ],
                    'correct': [ x['adversarial'][radius][daname]['correct'] for x in res ],
                    'inverted': [ x['adversarial'][radius][daname]['inverted'] for x in res ],
                    }
                for daname in testing_datasets
                }
            for radius in adversarial_radius
            }
    if config.test_adversarial_training:
        adversarial_trainings = {
                radius: {
                    daname: {
                        'total': [ x['adversarial_training'][radius][daname]['total'] for x in res ],
                        'correct': [ x['adversarial_training'][radius][daname]['correct'] for x in res ],
                        'inverted': [ x['adversarial_training'][radius][daname]['inverted'] for x in res ],
                        }
                    }
                for radius in adversarial_radius for daname in training_datasets
                }
    else:
        adversarial_trainings = { }


    # Output
    report_path.mkdir(parents=True, exist_ok=True)
    img_path = report_path / 'img'
    img_path.mkdir(parents=True, exist_ok=True)

    with open(report_path / 'single_model_report.md', 'w') as f:
        f.write('### Symbols\n\n')
        for i, checkpoint_name in enumerate(checkpoint_paths):
            f.write('- `{:03}`: `{}`\n'.format(i, checkpoint_name))
        f.write('\n\n')

        f.write('### Training scores\n\n')
        img_fname = 'training_score.png'
        this_img_path = img_path / img_fname
        plot_dict(training_scores)
        # plt.legend(['SRCC', 'PLCC'])
        plt.legend(training_labels)
        plt.savefig(this_img_path)
        plt.close()
        f.write('![Training scores]({})\n\n'.format(this_img_path.relative_to(report_path)))

        f.write('### Testing scores\n\n')
        img_fname = 'testing_score.png'
        this_img_path = img_path / img_fname
        plot_dict(testing_scores)
        # plt.legend(['SRCC', 'PLCC'])
        plt.legend(testing_labels)
        plt.savefig(this_img_path)
        plt.close()
        f.write('![Testing scores]({})\n\n'.format(this_img_path.relative_to(report_path)))

        f.write('### Adversarial\n\n')
        for radius, v1 in adversarials.items():
            f.write('#### Radius = {:.2f}\n\n'.format(radius))
            for dsname, to_plot in v1.items():
                f.write('##### {}\n'.format(dsname))
                img_fname = 'adversarial_{}_{}.png'.format(radius, dsname)
                this_img_path = img_path / img_fname
                plot_dict(to_plot)
                plt.savefig(this_img_path)
                plt.close()
                f.write('![Adversarial scores]({})\n\n'.format(this_img_path.relative_to(report_path)))

        f.write('### Adversarial Training\n\n')
        for radius, v1 in adversarial_trainings.items():
            f.write('#### Radius = {:.2f}\n\n'.format(radius))
            for dsname, to_plot in v1.items():
                f.write('##### {}'.format(dsname))
                img_fname = 'adversarial_{}_{}.png'.format(radius, dsname)
                this_img_path = img_path / img_fname
                plot_dict(to_plot)
                plt.savefig(this_img_path)
                plt.close()
                f.write('![Adversarial scores]({})\n\n'.format(this_img_path.relative_to(report_path)))



def markdown_report(report_path: Path, checkpoint_paths: List[Path], shapes: dict, res: dict):
    # Generate matrices first
    sim_mats = { }
    N = len(checkpoint_paths)
    similarity_measurements = res[(0, 1)]['similarities'].keys()

    for name, shape in shapes.items():
        this_parameter_mats = {
                mes: np.ones((N, N)) if mes != 'rmae' else np.zeros((N, N)) for mes in res[(0, 1)]['similarities'][name].keys() }
        for (i, j), values in res.items():
            similarities = values['similarities'][name]
            for mes, val in similarities.items():
            # print(f'{mes}: {val}')
                this_parameter_mats[mes][i, j] = val
                this_parameter_mats[mes][j, i] = val
        sim_mats[name] = this_parameter_mats


    # Output
    report_path.mkdir(parents=True, exist_ok=True)
    img_path = report_path / 'img'
    img_path.mkdir(parents=True, exist_ok=True)

    with open(report_path / 'report.md', 'w') as f:
        f.write('### Symbols\n\n')
        for i, checkpoint_name in enumerate(checkpoint_paths):
            f.write('- `{:03}`: `{}`\n'.format(i, checkpoint_name))
        f.write('\n\n')

        f.write('### Similarities\n\n')
        for name, sim_mats in sim_mats.items():
            f.write('#### `{}`\n'.format(name))
            f.write('Shape: `{}`\n\n'.format(str(shape)))
            f.write('\n\n')
            for mes, mat in sim_mats.items():
                f.write('##### {}\n'.format(mes))
                img_fname = 'sim_mat_{}_{}.png'.format(name, mes)
                cmat = ConfusionMatrixDisplay(mat)  # Using this ti plot similarity matrix
                cmat.plot()
                this_img_path = img_path / img_fname
                plt.savefig(this_img_path)
                plt.close()
                f.write('![Similarities]({})\n\n'.format(this_img_path.relative_to(report_path)))


def main(config):
    with torch.no_grad():
        if config.cross_model:
            checkpoint_paths, raw_res = compare(config)
        if config.single_model:
            single_model_checkpoint_paths, single_model = single_model_statistics(config)
        if config.json_output:
            jsonable = {
                    'shapes': shape_report(config),
                    }
            if config.cross_model:
                jsonable['items'] = res_to_jsonable(checkpoint_paths, raw_res),
            if config.single_model:
                jsonable['each'] = single_model
                print(single_model)
            Path(config.json_output).parent.mkdir(exist_ok=True, parents=True)
            with open(config.json_output, 'w') as f:
                json.dump(jsonable, indent=4, fp=f)
        if config.markdown_report:
            if config.cross_model:
                markdown_report(Path(config.markdown_report), checkpoint_paths, shape_report(config), raw_res)
            if config.single_model:
                markdown_single_model_report(Path(config.markdown_report), single_model_checkpoint_paths, single_model)

if __name__ == "__main__":
    config = parse_config()
    if config.verbose:
        print(config)

    main(config)
