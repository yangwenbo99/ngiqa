import torch
import torch.nn as nn
from torch.nn import init
from Spp import SpatialPyramidPooling2d

def build_model(normc=nn.ReLU, layer=3, width=48, extra_layer_num=0):
    layers = [
        nn.Conv2d(3, width, kernel_size=5, stride=1, padding=1, dilation=1, bias=True),
        #normc(width),
        normc(),
        nn.MaxPool2d(kernel_size=2)
    ]

    # Temporarily adding 2 layers
    for i in range(extra_layer_num):
        layers += [
                nn.Conv2d(width, width, kernel_size=5, stride=1, padding=1, dilation=1, bias=True),
                normc()
                ]

    for l in range(1, layer):
        layers += [nn.Conv2d(width,  width, kernel_size=5, stride=1, padding=1,  dilation=1, bias=True),
                #normc(width),
                normc(),
                nn.MaxPool2d(kernel_size=2)
                ]

    layers += [nn.Conv2d(width,  width, kernel_size=3, stride=1, padding=1,  dilation=1,  bias=True),
            #normc(width),
            normc(),
            SpatialPyramidPooling2d(pool_type='max_pool')
            ]
    layers += [nn.Linear(width*14, 128, bias=True),
               nn.ReLU(),
               nn.Linear(128, 1, bias=True)
               ]
    net = nn.Sequential(*layers)
    net.apply(weights_init)

    return net


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
    elif classname.find('Gdn1d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)


class VE2EUIQA(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self, extra_layer_num=0, layer=3, width=48):
        super(VE2EUIQA, self).__init__()
        self.cnn = build_model(
                extra_layer_num=extra_layer_num, layer=layer, width=width)

    def forward(self, x):
        r = self.cnn(x)
        # mean = r[:, 0].unsqueeze(dim=-1)

        return r.unsqueeze(dim=-1)

    def init_model(self, path):
        self.cnn.load_state_dict(torch.load(path))


