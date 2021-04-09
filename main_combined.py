#!/bin/env python3

import configargparse as argparse
import model_combined as model
import os


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
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')

    parser.add_argument('-n', '--normalize', action='store_true',
            help='Whether to normalise the input')
    parser.add_argument('-v', '--verbose', action='count')

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

    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--log_file", type=str, default='./train_log.txt')

    config = parser.parse_args()
    if config.eval:
        config.train = False
    if config.fresh:
        config.resume = False
    return config


def main(cfg):
    t = model.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        if config.eval_adversarial:
            print('Evaluating adversarial')
            for ds in t.datasets:
                print('{}:'.format(ds.name))
                t.eval_adversarial(loader=ds.test)
        elif config.test_gradient_length:
            print('Calculating gradients')
            for ds in t.datasets:
                print('{}:'.format(ds.name))
                t.test_gradient_length(loader=ds.test)
        elif config.test_loss_gradient_length:
            print('Calculating gradients')
            for ds in t.datasets:
                print('{}:'.format(ds.name))
                t.test_loss_gradient_length(loader=ds.test)
        elif config.train_correlation:
            print(t.eval_train())
        else:
            print(t.eval())


if __name__ == "__main__":
    config = parse_config()
    print(config)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    main(config)
