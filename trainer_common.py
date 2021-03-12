from typing import List, Set, Tuple
import os
import time
import scipy.stats
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
import scipy.stats

from py_join import Enumerable
from transforms import AdaptiveCrop, AdaptiveResize

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
        if config.adaptive_resize:
            train_transform_list = \
                    [ AdaptiveResize(512) ] +  \
                    test_transform_list
            test_transform_list = \
                    [ AdaptiveResize(768) ] +  \
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
        self.regularizer = self._get_regularizer(config.regularizer, self.loss_fn)
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

        if config.adversarial:
            self.attacker = self._get_attacker(config.adversarial)
            if config.verbose:
                print('The adversarial attacker is', self.attacker)
        else:
            self.attacker = None


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
        elif desc.upper().startswith('PL2R'):
            return losses.PairwiseL2RLoss()
        elif desc.upper().startswith('SSRCC'):
            return losses.BatchedSRCCLoss(self.config.loss_param1)
        else:
            return losses.model_loss()

    def _get_regularizer(self, desc: str, loss_fn):
        if desc.upper() == 'LOSSGRADL1':
            if self.config.verbose:
                print('The reg is', desc)
            return losses.LossGradientL1Regularizer(loss_fn, self.config.reg_strength)
        if desc.upper() == 'MODELGRADL1':
            if self.config.verbose:
                print('The reg is', desc)
            return losses.LossGradientL1Regularizer(loss_fn, self.config.reg_strength)
        else:
            return None

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

    def _get_attacker(self, desc: str):
        if type(desc) is str:
            if desc.upper().startswith('FGSM'):
                return FGSMAttacker(self.config.adversarial_radius)
            if desc.upper().startswith('RFGSM'):
                repeat = round(self.config.loss_param1)
                return RepeatedFGSMAttacker(self.config.adversarial_radius, repeat)
            if desc.upper().startswith('RANDFGSM'):
                return RandomizedFGSMAttacker(self.config.adversarial_radius)
            if desc.upper().startswith('LINF'):
                return LimitedLinfAttacker(self.config.adversarial_radius)
            if desc.upper().startswith('SLINF'):
                return LinfNormedSearch(self.config.adversarial_radius)
        return None

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
                    if self.config.verbose:
                        print('Parameter freezed)')
                    self.model.freeze()
                if epoch == self.config.phase1 + 1:
                    if self.config.verbose:
                        print('Parameter freezed')
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

            if self.attacker:
                x = self.attacker(self.model, x, y, self.loss_fn)

            self.optimizer.zero_grad()
            if not self.regularizer:
                yp = self.model(x)
                self.loss = self.loss_fn(y, yp)
            else:
                self.loss = self.regularizer(self.model, x, y)

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

    def test_loss_gradient_length(self, loader=None):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        losses = []
        lengths = []
        for step, sample_batched in enumerate(loader, 0):
            x, y = sample_batched['I'], sample_batched['y']
            x = x.to(self.device)
            y = y.to(self.device)
            x.requires_grad_(True)
            yp = self.model(x)
            loss = self.eval_loss(y, yp)
            loss.backward()
            grad = x.grad.detach()
            with torch.no_grad():
                losses.append(loss.detach().cpu().flatten())
                lengths.append(torch.sqrt((grad ** 2).sum()).detach().cpu())
        loss = torch.tensor(losses)
        l = torch.tensor(lengths)
        if self.config.verbose > 2:
            for tmp in losses:
                print(tmp.cpu().numpy())
        print('In terms of loss')
        print('        Loss:', torch.mean(loss).numpy(), torch.std(loss).numpy())
        print('        Gradients:', torch.mean(l).numpy(), torch.std(l).numpy())

    def test_gradient_length(self, loader=None):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        yps = []
        lengths = []
        for step, sample_batched in enumerate(loader, 0):
            x, y = sample_batched['I'], sample_batched['y']
            x = x.to(self.device)
            y = y.to(self.device)
            x.requires_grad_(True)
            yp = self.model(x)
            yp.backward()
            grad = x.grad.detach()
            with torch.no_grad():
                yps.append(yp.detach().cpu().flatten())
                lengths.append(torch.sqrt((grad ** 2).sum()).detach().cpu())
        ypp = torch.tensor(yps)
        l = torch.tensor(lengths)
        if self.config.verbose > 2:
            for tmp in yps:
                print(tmp[0].cpu().numpy())
        print('In terms of y')
        print('        Ys:', torch.mean(ypp).numpy(), torch.std(ypp).numpy())
        print('        Gradients:', torch.mean(l).numpy(), torch.std(l).numpy())


    def eval_common(self, loader=None):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        ys = []
        yps = []
        losses = []
        pairs = []
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

        if self.config.debug:
            for y, yp in zip(ys, yps):
                print('{:.6f}, {:.6f}'.format(y, yp))

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

        self.model.train()

        return sum(losses) / len(losses)

    def eval(self, loader=None):
        return self.eval_common(loader)

    def eval_adversarial(self, loader=None):
        if loader is None:
            loader = self.test_loader

        self.model.eval()
        ys = []
        yps = []
        yphs = []
        ypls = []
        losses = []
        pairs = []
        with torch.no_grad():
            for step, sample_batched in enumerate(loader, 0):
                x, y = sample_batched['I'], sample_batched['y']
                x = x.to(self.device)
                y = y.to(self.device)
                yp = self.model(x)
                xh, xl = self.attacker.attack_test(self.model, x, y)
                yph = self.model(xh)
                ypl = self.model(xl)
                yps += list(yp.flatten().detach().cpu())
                yphs += list(yph.flatten().detach().cpu())
                ypls  += list(ypl.flatten().detach().cpu())
                ys += list(y.flatten().detach().cpu())

        n = len(ys)
        print('Total:', n * (n - 1) / 2)
        correct = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if ys[i] < ys[j] and yps[i] < yps[j]
                ]
        inverted = [
                (i, j)
                for i, j in correct
                if yphs[i] > ypls[j]
                ]
        print('Correct:', len(correct))
        print('Inverted:', len(inverted))

        if self.config.debug:
            for y, yp, yph, ypl in zip(ys, yps, yphs, ypls):
                print('{:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                    y, yp, yph, ypl))

        self.model.train()


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

class FGSMAttacker():
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, model, input_var, target_var, criterion):
        input_var.requires_grad_(True)
        if input_var.grad is not None:
            input_var.grad.data.zero_()
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        move = torch.sign(input_var.grad.detach()) * self.radius
        input_var = input_var.detach() + move
        return input_var

    def attack_test(self, model, input_var, target_var):
        def pf(x, y):
            return x.sum()
        def nf(x, y):
            return -x.sum()
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        h = self(model, input_var, target_var, pf)
        l = self(model, input_var, target_var, nf)

        torch.set_grad_enabled(prev)
        return h, l

class RandomizedFGSMAttacker(FGSMAttacker):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, model, input_var, target_var, criterion):
        radius = self.radius
        depart = torch.rand_like(input_var) * radius
        processed_var = super().__call__(model, input_var + depart, target_var, criterion)
        res = torch.min(torch.max(processed_var, input_var - radius), input_var + radius)
        return res

    def attack_test(self, model, input_var, target_var):
        def pf(x, y):
            return x.sum()
        def nf(x, y):
            return -x.sum()
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        h = self(model, input_var, target_var, pf)
        l = self(model, input_var, target_var, nf)

        torch.set_grad_enabled(prev)
        return h, l

class RepeatedFGSMAttacker():
    def __init__(self, radius, repeat=10):
        self.radius = radius
        self.repeat = repeat

    def __call__(self, model, input_var, target_var, criterion):
        radius_per_step = self.radius / self.repeat
        for i in range(self.repeat):
            input_var.requires_grad_(True)
            if input_var.grad is not None:
                input_var.grad.data.zero_()
            output = model(input_var)
            loss = criterion(output, target_var)
            loss.backward()
            move = torch.sign(input_var.grad.detach()) * radius_per_step
            input_var = input_var.detach() + move
        return input_var

    def attack_test(self, model, input_var, target_var):
        def pf(x, y):
            return x.sum()
        def nf(x, y):
            return -x.sum()
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        h = self(model, input_var, target_var, pf)
        l = self(model, input_var, target_var, nf)

        torch.set_grad_enabled(prev)
        return h, l


class LimitedLinfAttacker():
    def __init__(self, tol, lr=1, iter=0, his=8):
        self._tol = tol
        self._lr = lr
        self._num_iter = iter if iter != 0 else 10
        self._his = his

    def __call__(self, model, img, target_var, criterion):
        tol = self._tol
        lr = self._lr

        hist_vars = [img]
        hist_vals = [model(img)[0].clone().detach()]
        ori_img = img.clone()
        lower_bound = ori_img - tol
        upper_bound = ori_img + tol

        for i in range(self._num_iter):
            img.requires_grad_(True)
            if img.grad is not None:
                img.grad.data.zero_()
            output = model(img)
            loss = criterion(output, target_var)
            loss.backward()

            yp = img.grad.clone().detach()
            # TODO: use sum, instead of mean here
            ypabs = yp.abs()
            ypabs[(ypabs > 0.003 * tol) & (ypabs < 0.1 * tol)] = 0.1 * tol
            # Empirically, / step
            yp = yp.sign() * ypabs
            # lyp = torch.norm(yp.clamp(-self._lr, self._lr))
            img.requires_grad_(False)
            simg = img + lr * yp # / lyp
            simg = torch.max(simg, lower_bound)
            simg = torch.min(simg, upper_bound)
            img = simg

            with torch.no_grad():
                score = model(img)[0]
                hist_vals.append(score)
                hist_vars.append(img)
                if len(hist_vals) > self._his:
                    hist_vals.pop(0)
                    hist_vars.pop(0)
        selected_i = 0
        for i in range(self._his):
            if self._lr * hist_vals[i] > self._lr * hist_vals[selected_i]:
                selected_i = i
        return hist_vars[selected_i]

    def attack_test(self, model, input_var, target_var):
        def pf(x, y):
            return x.sum()
        def nf(x, y):
            return -x.sum()
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        h = self(model, input_var, target_var, pf)
        l = self(model, input_var, target_var, nf)

        torch.set_grad_enabled(prev)
        return h, l

class LinfNormedSearch:
    def __init__(
            self,
            tol, iter=0,
            stop_eps=1e-5, num_sample_point=10,
            k=0.3):
        '''
        @param k: the adjusting parameter for this method, see the technical
                  report
        '''
        self._tol = tol
        self._num_iter = iter if iter != 0 else 50
        self._stop_eps = stop_eps
        self._num_sample_point = num_sample_point
        self._k = k

    def __call__(self, model, img, target_var, criterion):
        raise NotImplemented('The search method has not been implemented for batched attack')

    def attack_test(self, model, input_var, target_var):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        h = self._attack(input_var, model, 1)
        l = self._attack(input_var, model, -1)
        torch.set_grad_enabled(prev)
        return h, l

    def _attack(self, img, model, dir):
        tol = self._tol

        ori_img = img.clone()
        lower_bound = ori_img - tol
        upper_bound = ori_img + tol

        h, w = img.shape[-2], img.shape[-1]

        def project_back(x):
            x = torch.max(x, lower_bound)
            x = torch.min(x, upper_bound)
            return x

        for i in range(self._num_iter):
            print('\r', i, end='    ')

            img.requires_grad_(True)
            if img.grad is not None:
                img.grad.data.zero_()
            y = model(img)
            print(y.detach().flatten().cpu().numpy(), end='')
            y.backward()
            yp = img.grad.clone().detach()
            img.requires_grad_(False)

            with torch.no_grad():
                y_opt = - 1e128 * dir
                x_opt = None
                # We want the 'ideal step' to have generall similar size, and
                # comparable to tol
                ypp = torch.sign(yp) * (abs(yp) ** self._k)
                ideal_step = 20 * dir * (ypp * (self._tol) / (abs(ypp).max() + self._tol))
                for isample in range(self._num_sample_point):
                    # sample_step = ideal_step * ((isample + 1) / self._num_sample_point)
                    sample_step = 500 * ideal_step * math.exp(- 10 * (isample) / self._num_sample_point)
                    # Maybe we should not foloow the gradient
                    # the effect of quantised can be decomposed:
                    #     Change of step size and direction + quantise for
                    #     number
                    x_this = project_back(img + sample_step)
                    y_this = model(x_this)
                    if dir * (y_this - y_opt) > 0:
                        x_opt = x_this
                        y_opt = y_this

                # try the same on the negative side
                for isample in range(self._num_sample_point // 2):
                    sample_step = - ideal_step * math.exp(- 10 * (isample) / self._num_sample_point)
                    x_this = project_back(img + sample_step)
                    y_this = model(x_this)
                    if dir * (y_this - y_opt) > 0:
                        x_opt = x_this
                        y_opt = y_this

                if dir * (y_opt - y) <= 0:
                    # not getting better
                    break

                # Now, we can update x
                img = x_opt
                if dir * (y_opt - y) < self._stop_eps:
                    break

        print('\r', end='')
        return img

