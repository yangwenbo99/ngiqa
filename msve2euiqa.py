'''Multi-scale Varied E2EUIQA
'''

import torch
import torch.nn as nn
from torch.nn import init
from ve2euiqa import weights_init

def build_model(normc=nn.ReLU, size=3, width=48, out_width=8):
    layers = [
        nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        normc(),
        nn.MaxPool2d(kernel_size=2)
    ]

    for l in range(1, size):
        layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=1,  dilation=1, bias=True),
                normc(),
                nn.MaxPool2d(kernel_size=2)
                ]

    layers += [
            nn.Flatten(),
            nn.Linear(width, out_width, bias=True),
            ]
    net = nn.Sequential(*layers)
    net.apply(weights_init)

    return net

class MSModel(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self, size=7, inner_out_width=8):
        '''
        @param size: the size of input image shall be 2 ** size
        '''
        super(MSModel, self).__init__()
        self._size = size
        self._inner_out_width = inner_out_width
        self._fc_width = (size - 1) * self._inner_out_width
        self.cnns = torch.nn.ModuleList([
                self._build_one_scale(i) for i in range(1, size)
                ])
        self.final = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self._fc_width, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )
        # print(self)

    def _build_one_scale(self, scale):
        cnn = build_model(size=scale, out_width=self._inner_out_width)
        if scale != self._size:
            return nn.Sequential(
                    nn.AvgPool2d(kernel_size=(2 ** (self._size - scale))),
                    cnn
                    )
        else:
            return cnn

    def forward(self, x):
        ress = [m(x) for m in self.cnns]
        x_intermediate = torch.cat((ress), dim=1)
        r = self.final(x_intermediate)

        return r.unsqueeze(dim=-1)

    def init_model(self, path):
        self.cnn.load_state_dict(torch.load(path))


