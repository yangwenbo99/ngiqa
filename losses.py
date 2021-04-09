import torch
import math
from fast_soft_sort.pytorch_ops import soft_rank
import torch.nn.functional as F
import numpy as np

LOSS_N = 1

class model_loss(torch.nn.Module):

    def __init__(self):
        super(model_loss, self).__init__()

    def forward(self, y, yp):
        return torch.mean(((yp - y) / (torch.abs(y) + LOSS_N)) ** 2)

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

class SSIMCorrelationLoss(torch.nn.Module):

    def __init__(self):
        super(SSIMCorrelationLoss, self).__init__()

    def forward(self, y, yp):
        eps = 0.00001
        y = y.flatten()
        yp = yp.flatten()
        my = torch.mean(y)
        myp = torch.mean(yp)
        covm = cov(y, yp)
        s2y = covm[0, 0]
        s2yp = covm[1, 1]
        syyp = covm[0, 1]

        S = \
                ((2 * my * myp + eps) / (my**2 + myp**2 + eps)) * \
                ((2 * syyp + eps) / (s2y + s2yp + eps))
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


class CorrelationWithMAELoss(torch.nn.Module):

    def __init__(self):
        super(CorrelationWithMAELoss, self).__init__()

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
        return - S + 0.1 * torch.sum(torch.abs(y - yp))

class PairwiseL2RLoss(torch.nn.Module):

    def __init__(self, p=False):
        super(PairwiseL2RLoss, self).__init__()
        self.p = p

    SQRT2 = math.sqrt(2)

    def cdf(self, x):
        return (1 + torch.erf(x / self.SQRT2)) / 2

    def forward(self, y, yp):
        eps = 0.00001
        y = y.flatten()
        yp = yp.flatten()
        n = int(y.shape[0])
        k = n // 2
        y1, y2 = y[:k], y[k:2*k]
        yp1, yp2 = yp[:k], yp[k:2*k]

        labels = torch.sign(y1 - y2)

        objcdf = self.cdf(labels * (yp1 - yp2))
        objx = torch.log(objcdf + eps)
        objs = torch.mean(objx)
        if self.p:
            objs = torch.exp(objs)

        return - objs

class PairwiseL2RWithHardFidelityLoss(torch.nn.Module):
    '''
    p (real) = 0 or 1
    '''

    def __init__(self):
        super(PairwiseL2RWithHardFidelityLoss, self).__init__()

    SQRT2 = math.sqrt(2)

    def cdf(self, x):
        return (1 + torch.erf(x / self.SQRT2)) / 2

    def forward(self, y, yp):
        eps = 1e-6
        y = y.flatten()
        yp = yp.flatten()
        n = int(y.shape[0])
        k = n // 2
        y1, y2 = y[:k], y[k:2*k]
        yp1, yp2 = yp[:k], yp[k:2*k]

        labels = torch.sign(y1 - y2)
        pp = (labels + 1) / 2
        # pp = torch.clamp(pp, 0, 1)
        p = self.cdf(yp1 - yp2)
        # p = torch.clamp(p, 0, 1)
        objx = 1 - torch.sqrt(pp * p + eps) - torch.sqrt((1 - pp) * (1 - p) + eps)

        objs = torch.mean(objx)

        return objs



class BatchedL2RLoss(torch.nn.Module):

    def __init__(self, p=False):
        super(BatchedL2RLoss, self).__init__()
        self.p = p

    def squarize(self, x):
        n = x.shape[0]
        x_usq = x.unsqueeze(-1)                # n * 0
        x_expanded = x_usq.expand(n, n)
        return x_expanded

    SQRT2 = math.sqrt(2)

    def cdf(self, x):
        return (1 + torch.erf(x / self.SQRT2)) / 2


    def forward(self, y, yp):
        eps = 0.00001
        y = y.flatten()
        yp = yp.flatten()

        y_expanded = self.squarize(y)
        yp_expanded = self.squarize(yp)

        tags = torch.sign(y_expanded - y_expanded.t())

        objcdf = self.cdf((yp_expanded - yp_expanded.t()))
        objx = torch.log(objcdf + eps) * tags
        objs = torch.mean(objx.triu(1))
        if self.p:
            objs = torch.exp(objs)

        return - objs


class AlternativeLoss(torch.nn.Module):

    def __init__(self):
        super(AlternativeLoss, self).__init__()

    def forward(self, y, yp):
        # return torch.mean((yp - y) ** 2)
        return torch.mean(torch.abs(yp - y) / (y + 1))


class BatchedSRCCLoss(torch.nn.Module):

    def __init__(self, regularization_strength=1.5):
        super(BatchedSRCCLoss, self).__init__()
        self.regularization_strength = regularization_strength

    def forward(self, y, yp):
        eps = 0.00001
        y = y.flatten().unsqueeze(0)
        yp = yp.flatten().unsqueeze(0)

        # Rank them
        # print(y)
        y = soft_rank(y, regularization_strength=0.1).squeeze()
        yp = soft_rank(yp,
                regularization_strength=self.regularization_strength).squeeze()
        # print(y, yp)

        covm = cov(y, yp)
        s2y = covm[0, 0]
        s2yp = covm[1, 1]
        syyp = covm[0, 1]

        S = (syyp / (torch.sqrt(s2y) * torch.sqrt(s2yp) + eps))
        # U = (my * myp) / (my**2 + myp**2 + eps)

        return - S # * U

'''
class ModelGradientL1Regularizer(torch.nn.Module):

    def __init__(self, regularization_strength=5e-2):
        super(ModelGradientL1Regularizer, self).__init__()
        self.regularization_strength = regularization_strength

    def forward(self, model, lossfn, x, y):
        yp = model(x)
        yp.backward()
        res = self.regularization_strength * yp.backward().abs().sum()

        return res

    '''

class LossGradientL1Regularizer():

    def __init__(self, lossfn, regularization_strength=5e-2):
        super(LossGradientL1Regularizer, self).__init__()
        self.regularization_strength = regularization_strength
        self.lossfn = lossfn

    def __call__(self, model, x, y):
        x.requires_grad_(True)
        yp = model(x)
        loss = self.lossfn(y, yp)
        gloss_x = torch.autograd.grad(loss, x, create_graph=True)[0]
        reg = gloss_x.abs().sum()
        loss2 = loss + self.regularization_strength * reg
        return loss2

class ModelGradientL1Regularizer():

    def __init__(self, lossfn, regularization_strength=5e-2):
        super(ModelGradientL1Regularizer, self).__init__()
        self.regularization_strength = regularization_strength
        self.lossfn = lossfn

    def __call__(self, model, x, y):
        x.requires_grad_(True)
        yp = model(x)
        loss = self.lossfn(y, yp)
        gyp_x = torch.autograd.grad(yp, x, create_graph=True)[0]
        reg = gyp_x.abs().sum()
        loss2 = loss + self.regularization_strength * reg
        return loss2

class ModelGradientL1CosRegularizer():
    '''
    Reference: https://arxiv.org/abs/2007.02617

    Parameter in the original paper:
        step = 8 / 255
        regularization_strength2 = 0.2
    '''

    def __init__(self, lossfn, step, regularization_strength1=5e-2, regularization_strength2=2e-1):
        super(ModelGradientL1CosRegularizer, self).__init__()
        self.regularization_strength1 = regularization_strength1
        self.regularization_strength2 = regularization_strength2
        self.lossfn = lossfn
        self.step = step

    def get_rand_input(self, x):
        eps = self.step
        delta = torch.zeros_like(x).cuda()
        delta.uniform_(-eps, eps)
        return x + delta

    def __call__(self, model, x, y):
        x.requires_grad_(True)
        yp = model(x)
        gyp_x = torch.autograd.grad(yp, x, create_graph=True)[0]
        xd = self.get_rand_input(x)
        ypd = model(xd)
        gyp_xd = torch.autograd.grad(ypd, xd, create_graph=True)[0]
        grad1, grad2 = gyp_x.reshape(len(gyp_x), -1), gyp_xd.reshape(len(gyp_xd), -1)

        loss = self.lossfn(y, yp)
        reg1 = gyp_x.abs().sum()
        reg2 = 1.0 - torch.nn.functional.cosine_similarity(grad1, grad2, 1).mean()
        loss2 = \
                loss + \
                self.regularization_strength1 * reg1 + \
                self.regularization_strength2 * reg2
        return loss2

class ModelGradientCosRegularizer():
    '''
    Reference: https://arxiv.org/abs/2007.02617

    Parameter in the original paper:
        step = 8 / 255
        regularization_strength2 = 0.2
    '''

    def __init__(self, lossfn, step, regularization_strength2=2e-1):
        super(ModelGradientL1CosRegularizer, self).__init__()
        self.regularization_strength2 = regularization_strength2
        self.lossfn = lossfn
        self.step = step

    def get_rand_input(self, x):
        eps = self.step
        delta = torch.zeros_like(x).cuda()
        delta.uniform_(-eps, eps)
        return x + delta

    def __call__(self, model, x, y):
        x.requires_grad_(True)
        yp = model(x)
        gyp_x = torch.autograd.grad(yp, x, create_graph=True)[0]
        xd = self.get_rand_input(x)
        ypd = model(xd)
        gyp_xd = torch.autograd.grad(ypd, xd, create_graph=True)[0]
        grad1, grad2 = gyp_x.reshape(len(gyp_x), -1), gyp_xd.reshape(len(gyp_xd), -1)

        loss = self.lossfn(y, yp)
        reg2 = 1.0 - torch.nn.functional.cosine_similarity(grad1, grad2, 1).mean()
        loss2 = \
                loss + \
                self.regularization_strength2 * reg2
        return loss2



# The folloing loss functions come from https://github.com/lidq92/LinearityIQA/blob/master/IQAloss.py

def norm_loss_with_normalization(y_pred, y, p=2, q=2, detach=False, exponent=True, eps=1e-8):
    """norm_loss_with_normalization: norm-in-norm"""
    y_pred = y_pred.flatten()
    y = y.flatten()
    N = y_pred.size(0)
    if N > 1:
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat  # very important!!
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  # Actually, z-score normalization is related to q = 2.
        # print('bhat = {}'.format(normalization.item()))
        y_pred = y_pred / (eps + normalization)  # very important!
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) # p, q>0

        err = y_pred - y
        if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
            err += eps
        loss0 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to PLCC
        loss0 = torch.pow(loss0, p) if exponent else loss0 #

        return loss0
    else:
        # If the batch size is too small, the calculation is not reliable
        return F.l1_loss(y_pred, y_pred.detach())


class NormInNorm(torch.nn.Module):

    def __init__(self, p, q):
        '''
        p, q as defined in
        Norm-in-Norm Loss with Faster Convergence and Better
        Performance for Image Quality Assessment

        '''
        super(NormInNorm, self).__init__()
        print(f'p = {p}, q = {q}')
        self.p = p
        self.q = q

    def forward(self, y, yp):
        return norm_loss_with_normalization(yp, y, self.p, self.q)
