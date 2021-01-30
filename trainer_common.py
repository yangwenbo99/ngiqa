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
from transforms import AdaptiveCrop

# Note the difference between this implementaiton and the original
from BaseCNN import BaseCNN
from dataset import ImageDataset
from ve2euiqa import VE2EUIQA
from e2euiqa import E2EUIQA
from msve2euiqa import MSModel

import losses

LOSS_N = 1

SEATS = list(range(1, 30))

class Trainer(object):
    '''
    This class does most of the house-keeping works

    Its sub-class need to specify loaders
    and the train_single_epoch method, and can rewrite the eval method


    Its state contains:
        - start epoch
        - model's state_dict
        - optimiser's state_dict
        - training_lost
        - test_lost
        - initial_lr
    '''
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config

        train_transform_list = [
            AdaptiveCrop(config.image_size, config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        test_transform_list = [
            transforms.ToTensor()
        ]
        if config.crop_test:
            test_transform_list = \
                    [ AdaptiveCrop(config.image_size, config.image_size) ] +  \
                    test_transform_list


        if config.normalize:
            train_transform_list.append(
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)))
            test_transform_list.append(
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)))

        self.train_transform = transforms.Compose(train_transform_list)
        self.test_transform = transforms.Compose(test_transform_list)

        self.train_batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size

        self.device = torch.device(config.device)

        # initialize the model
        self.model_name_input = config.model
        self.model = self._get_model(config.model)
        self.model.to(self.device)
        self.model_name = type(self.model).__name__
        if config.verbose:
            print(self.model)

        # loss function
        self.loss_fn = self._get_loss_fn(config.lossfn)
        self.loss_fn.to(self.device)
        self.eval_loss = self._get_loss_fn(config.eval_lossfn)
        self.eval_loss.to(self.device)

        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr': lr}])

        # try load the model
        if config.resume or not config.train:

            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)

            if config.verbose:
                print('[o] Checkpoint at {}', ckpt)
            if ckpt:
                self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

        self.log_file = config.log_file


    def _get_loss_fn(self, desc: str):
        if desc.upper() == 'MAE':
            return losses.MAELoss()
        elif desc.upper() == 'MSE':
            return losses.MSELoss()
        elif desc.upper().startswith('CORR'):
            return losses.CorrelationLoss()
        elif desc.upper().startswith('MCORR'):
            return losses.CorrelationWithMeanLoss()
        elif desc.upper().startswith('MAECORR'):
            return losses.CorrelationWithMAELoss()
        elif desc.upper().startswith('SCORR'):
            return losses.SSIMCorrelationLoss()
        elif desc.upper().startswith('L2RR'):
            return losses.BatchedL2RLoss()
        elif desc.upper().startswith('EL2R'):
            return losses.BatchedL2RLoss(p=True)
        else:
            return losses.model_loss()

    def _get_model(self, desc: str):
        if desc.upper().startswith('VE2EUIQA'):
            if '-' in desc:
                extra_layer_num = int(desc.split('-')[1])
                return VE2EUIQA(extra_layer_num=extra_layer_num)
            elif '+' in desc:
                splits = desc.split('+')
                layer_num = int(splits[1])
                width = int(splits[2]) if len(splits) >= 2 else 48
                return VE2EUIQA(layer=layer_num, width=width)
            else:
                extra_layer_num = 0
                return VE2EUIQA(extra_layer_num=extra_layer_num)
        if desc.upper().startswith('E2EUIQA'):
            if '+' in desc:
                splits = desc.split('+')
                layer_num = int(splits[1])
                width = int(splits[2]) if len(splits) >= 2 else 48
                return E2EUIQA(layer=layer_num, width=width)
            else:
                return E2EUIQA()
        if desc.upper().startswith('BASECNN'):
            model = BaseCNN(self.config)
            return model
            # return nn.DataParallel(model, device_ids=[0])
        if desc.upper().startswith('SIMPLE'):
            model = SimpleModel(self.config)
            return model
        if desc.upper().startswith('MS'):
            model = MSModel()
            return model
        return None

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        if len(ckpts) == 0:
            return None
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            if self.config.verbose:
                print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            if self.config.verbose:
                print("[*] loaded checkpoint '{}' (epoch {})"
                      .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            if self.model_name_input.upper().startswith('BASECNN'):
                if epoch <= self.config.phase1:
                    self.model.freeze()
                if epoch == self.config.phase1 + 1:
                    self.model.unfreeze()
                    self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr': self.initial_lr}])
            self.train_single_epoch(epoch)

    def train_single_epoch(self, epoch):
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

    def _group(self, items: List[Tuple[float, float]], seats: List[float]):
        def get_seat(x):
            for i, v in enumerate(seats):
                if x[0] <= v: return i
            return len(seats)

        def after_transform(key, values):
            losses = [x[1] for x in values]
            mean = sum(losses) / len(losses) if len(losses) > 0 else -1
            return (key, len(losses), mean)

        e = Enumerable(items).group_by(key=get_seat, after_transform=after_transform)
        el = e.to_list()
        el.sort(key=lambda x: x[0])
        return el

    def eval_common(self, loader=None):
        if loader is None:
            loader = self.test_loader
        ys = []
        yps = []
        losses = []
        pairs = []
        ypairs = []
        with torch.no_grad():
            for step, sample_batched in enumerate(loader, 0):
                x, y = sample_batched['I'], sample_batched['y']
                x = x.to(self.device)
                y = y.to(self.device)
                yp = self.model(x)
                loss = self.eval_loss(y, yp)
                ys += list(y.flatten().detach().cpu())
                yps += list(yp.flatten().detach().cpu())
                losses.append(loss.detach())
        #! ypairs.append((y.detach().cpu(), yp.detach().cpu()))

        if self.config.verbose > 2:
            for y, loss in zip(ys, losses):
                pairs.append((y.detach().cpu(), loss.detach().cpu()))
            groups = self._group(pairs, SEATS)
            for it in groups:
                print('    Group {: 4} (length {: 5}): {:6f}'.format(it[0], it[1], it[2]))

        if self.config.test_correlation or self.config.train_correlation:
            # print(ys)
            # print(yps)
            # print(pairs)
            srcc = scipy.stats.mstats.spearmanr(x=ys, y=yps)[0]
            plcc = scipy.stats.mstats.pearsonr(x=ys, y=yps)[0]
            print('SRCC {:.6f}, PLCC: {:.6f}'.format(srcc, plcc))

        return sum(losses) / len(losses)

    def eval(self, loader=None):
        return self.eval_common(loader)


class SimpleModel(nn.Module):
    def __init__(self, config, width=3, num_layers=3):
        def weight_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data)
            elif classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight.data)

        nn.Module.__init__(self)
        self.config = config

        input_size = width * config.image_size * config.image_size

        layers = [
                nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=1, dilation=1, bias=True),
                nn.ReLU(),
                nn.Flatten()
                ]
        for i in range(num_layers):
            layers += [
                    nn.Linear(input_size, input_size),
                    nn.ReLU(),
                    ]
        layers += [
                nn.Linear(input_size, 1)
                ]
        self.nn = nn.Sequential(*layers)
        self.nn.apply(weight_init)

    def forward(self, x):
        r = self.nn(x)
        return r.unsqueeze(dim=-1)

