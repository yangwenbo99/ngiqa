import torch

LOSS_N = 1

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



class AlternativeLoss(torch.nn.Module):

    def __init__(self):
        super(AlternativeLoss, self).__init__()

    def forward(self, y, yp):
        # return torch.mean((yp - y) ** 2)
        return torch.mean(torch.abs(yp - y) / (y + 1))


