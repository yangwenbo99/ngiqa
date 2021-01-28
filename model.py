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

import trainer_common

# Note the difference between this implementaiton and the original
from BaseCNN import BaseCNN
from dataset import ImageDataset
from ve2euiqa import VE2EUIQA
from e2euiqa import E2EUIQA
from msve2euiqa import MSModel


SEATS = list(range(1, 30))

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

        self.training_data = ImageDataset(
                img_dir=config.trainset, transform=self.train_transform)
        self.test_data = ImageDataset(
                img_dir=config.testset, transform=self.test_transform)

        self.train_loader = DataLoader(self.training_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=8)

        self.train_test_loader = DataLoader(self.training_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)
        self.test_loader = DataLoader(self.test_data,
                                       batch_size=self.test_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)
        self.device = torch.device(config.device)


    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
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
            out_str = 'Epoch {} Testing: {:.4f}'.format(
                              epoch,
                              test_result)
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


