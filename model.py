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
from torch.autograd import Variable
from torchvision import transforms
import scipy.stats

from py_join import Enumerable

# Note the difference between this implementaiton and the original
from BaseCNN import BaseCNN
from dataset import ImageDataset
from ve2euiqa import VE2EUIQA
from msve2euiqa import MSModel

LOSS_N = 1

SEATS = list(range(1, 30))

class Trainer(object):
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
        torch.manual_seed(config.seed)
        self.config = config

        train_transform_list = [
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        test_transform_list = [
            transforms.ToTensor()
        ]
        if config.crop_test:
            test_transform_list = \
                    [ transforms.RandomCrop(config.image_size) ] +  \
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
        self.test_batch_size = 1

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

        # initialize the model
        self.model_name_input = config.model
        self.model = self._get_model(config.model)
        self.model.to(self.device)
        self.model_name = type(self.model).__name__
        if config.verbose:
            print(self.model)

        # loss function
        self.loss_fn = self._get_loss_fn(config.lossfn)
        if config.alternative_train_loss:
            self.loss_fn = AlternativeLoss()
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
            if ckpt:
                self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

        self.log_file = config.log_file


    def _get_loss_fn(self, desc: str):
        if desc.upper() == 'MAE':
            return MAELoss()
        elif desc.upper() == 'MSE':
            return MSELoss()
        elif desc.upper().startswith('CORR'):
            return CorrelationLoss()
        elif desc.upper().startswith('MCORR'):
            return CorrelationWithMeanLoss()
        else:
            return model_loss()

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
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
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
            self._train_single_epoch(epoch)

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

    def eval(self, loader=None):
        if loader is None:
            loader = self.test_loader
        losses = []
        pairs = []
        with torch.no_grad():
            for step, sample_batched in enumerate(loader, 0):
                x, y = sample_batched['I'], sample_batched['y']
                x = x.to(self.device)
                y = y.to(self.device)
                yp = self.model(x)
                loss = self.eval_loss(y, yp)
                losses.append(loss.detach())
                pairs.append((y.detach().cpu(), loss.detach().cpu()))

        if self.config.verbose:
            groups = self._group(pairs, SEATS)
            for it in groups:
                print('    Group {: 4} (length {: 5}): {:6f}'.format(it[0], it[1], it[2]))

        if self.config.test_correlation or self.config.train_correlation:
            x = [p[1] for p in pairs]
            y = [p[0] for p in pairs]
            # print(pairs)
            srcc = scipy.stats.mstats.spearmanr(x=x, y=y)[0]
            plcc = scipy.stats.mstats.pearsonr(x=x, y=y)[0]
            print('SRCC {:.6f}, PLCC: {:.6f}'.format(srcc, plcc))

        return sum(losses) / len(losses)

class model_loss(torch.nn.Module):

    def __init__(self):
        super(model_loss, self).__init__()

    def forward(self, y, yp):
        return torch.mean(((yp - y) / (y + LOSS_N)) ** 2)

class MAELoss(torch.nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y, yp):
        return torch.mean(torch.abs(yp - y))

class MSELoss(torch.nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y, yp):
        return torch.mean((yp - y) ** 2)

def cov(m, y=None):
    # https://stackoverflow.com/questions/51416825/calculate-covariance-matrix-for-complex-data-in-two-channels-no-complex-data-ty
    if y is not None:
        m = torch.cat((m.unsqueeze(0), y.unsqueeze(0)), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


class CorrelationLoss(torch.nn.Module):

    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, y, yp):
        eps = 0.00001
        y = y.flatten()
        yp = yp.flatten()
        # my = torch.mean(y)
        # myp = torch.mean(yp)
        covm = cov(y, yp)
        s2y = covm[0, 0]
        s2yp = covm[1, 1]
        syyp = covm[0, 1]

        S = (syyp / (torch.sqrt(s2y) * torch.sqrt(s2yp) + eps))
        # U = (my * myp) / (my**2 + myp**2 + eps)

        return - S # * U

class CorrelationWithMeanLoss(torch.nn.Module):

    def __init__(self):
        super(CorrelationWithMeanLoss, self).__init__()

    def forward(self, y, yp):
        eps = 0.0000001

        def l(x):
            return torch.log(torch.max(torch.zeros_like(x) + eps, x + eps))

        y = y.flatten()
        yp = yp.flatten()
        my = torch.mean(y)
        myp = torch.mean(yp)
        covm = cov(y, yp)
        s2y = covm[0, 0]
        s2yp = covm[1, 1]
        syyp = covm[0, 1]
        # print(y.isnan().any(), yp.isnan().any(), covm.isnan().any())

        # S = (syyp / (torch.sqrt(s2y) * torch.sqrt(s2yp) + eps))
        S = l(syyp + eps) -  1/2 * (torch.log(s2y + eps) + torch.log(s2yp + eps))
        # print(y.isnan().any(), yp.isnan().any(), covm.isnan().any(), S)
        # U = (my * myp) / (my**2 + myp**2 + eps)

        # return - torch.log(S) + 0.03 * torch.abs(my - myp)
        return - S + 0.03 * torch.abs(my - myp)


class AlternativeLoss(torch.nn.Module):

    def __init__(self):
        super(AlternativeLoss, self).__init__()

    def forward(self, y, yp):
        # return torch.mean((yp - y) ** 2)
        return torch.mean(torch.abs(yp - y) / (y + 1))

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




