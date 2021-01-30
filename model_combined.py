from typing import List, Set, Tuple
import os
import time
import scipy.stats
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
import scipy.stats

from py_join import Enumerable
from sklearn.utils import shuffle

import trainer_common

# Note the difference between this implementaiton and the original
from BaseCNN import BaseCNN
from dataset import ImageDataset
from ve2euiqa import VE2EUIQA
from e2euiqa import E2EUIQA
from msve2euiqa import MSModel

class Dataset():
    def __init__(self, name, train, test):
        self.name = name
        self.train = train
        self.test = test

class RepeatedDataLoader():
    def __init__(self, loader: DataLoader, repeat=2):
        self.loader = loader
        self.repeat = repeat

    def __iter__(self):
        for i in range(self.repeat):
            for item in self.loader:
                yield item

    def __len__(self):
        return self.repeat * len(self.loader)

class CombinedDataLoader:
    def __init__(self, dss: List[Dataset]):
        self.datasets = dss

    def __iter__(self):
        indecies = [idx
                for idx, ds in enumerate(self.datasets)
                for _ in range(len(ds.train))]
        indecies = shuffle(indecies)
        iterators = [iter(ds.train) for ds in self.datasets]
        for idx in indecies:
            yield self.datasets[idx].name, next(iterators[idx])

    def __len__(self):
        return sum([len(ds.train) for ds in self.datasets])


class Trainer(trainer_common.Trainer):
    '''
    Its state contains:
        - start epoch
        - model's state_dict
        - optimiser's state_dict
        - training_lost
        - test_lost
        - initial_lr
    '''
    def __init__(self, config):
        super(Trainer, self).__init__(config)


        self.dataset_names = ['live', 'csiq', 'kadid', 'clive', 'koniq', 'bid' ]
        self.dateset_repeats = [6, 8, 1, 7, 2, 7]
        self.datasets = []

        for name, repeat in zip(self.dataset_names, self.dateset_repeats):
            if config.verbose:
                print('loading from', name)
            training_data = ImageDataset(
                    img_dir=config.__getattribute__('train_'+ name),
                    transform=self.train_transform)
            test_data = ImageDataset(
                    img_dir=config.__getattribute__('test_'+ name),
                    transform=self.test_transform)

            train_loader = DataLoader(training_data,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8)
            if config.repeat_dataset:
                train_loader = RepeatedDataLoader(train_loader, repeat)
            test_loader = DataLoader(test_data,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=1)
            self.datasets.append(Dataset(name, train_loader, test_loader))
        self.combined_trainset = CombinedDataLoader(self.datasets)
        self.test_result = None


    def train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.combined_trainset)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, (name, sample_batched) in enumerate(self.combined_trainset, 0):
            if step < self.start_step:
                continue

            x, y = sample_batched['I'], sample_batched['y']
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            yp = self.model(x)
            self.loss = self.loss_fn(y, yp)
            self.loss.backward()
            self.optimizer.step()
            if type(self.model) == E2EUIQA:
                self.model.gdn_param_proc()

            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print('{:6} '.format(name), end='')
            print(format_str % (epoch, step, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)
        self.scheduler.step()

        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            test_result = self.eval()
            out_str = 'Epoch {} Testing: Finished'.format(
                              epoch)
            self.test_result = test_result
            print(out_str)
            with open(self.log_file, 'a') as f:
                f.write(out_str)
                f.write('\n')

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_result,
            }, model_name)

        return self.loss.data.item()

    def eval(self, loader=None):
        if loader is not None:
            self.eval_common(loader)
        res = []
        for dataset in self.datasets:
            print('{:6} Test: '.format(dataset.name), end='')
            res.append(self.eval_common(dataset.test))
        return res

    def eval_train(self):
        res = []
        for dataset in self.datasets:
            print('{:6} Train: '.format(dataset.name), end='')
            res.append(self.eval_common(dataset.train))
        return res


