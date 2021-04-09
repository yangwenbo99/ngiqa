import torch.nn as nn
from torchvision import models
from BCNN import BCNN

class BaseCNN(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif self.config.backbone == 'resnet50':
             self.backbone = models.resnet50(pretrained=True)
        #     self.fc = nn.Linear(2048, 1)
        elif self.config.backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
        #     self.fc = nn.Linear(2048, 1)
        outdim = 1
        if config.representation == 'BCNN':
            # assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            if self.config.backbone in ['resnet18', 'resnet34']:
                self.representation = BCNN()
                self.fc = nn.Linear(512 * 512, outdim)
            else:
                self.representation = BCNN(input_dim=2048)
                self.fc = nn.Linear(2048 * 2048, outdim)
        else:
            self.fc = nn.Linear(512, outdim)

        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def freeze(self):
        # Freeze all previous layers.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass of the network.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        #!print('The shape of x:', x.shape)
        if self.config.representation == 'BCNN':
            x = self.representation(x)
        else:
            x = self.backbone.avgpool(x)

        x = self.fc(x)

        return x.unsqueeze(-1)


class BaseCNNWithOutputNormalization(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif self.config.backbone == 'resnet50':
             self.backbone = models.resnet50(pretrained=True)
        #     self.fc = nn.Linear(2048, 1)
        # elif self.config.backbone == 'resnet101':
        #     self.backbone = models.resnet101(pretrained=True)
        #     self.fc = nn.Linear(2048, 1)
        outdim = 1
        if config.representation == 'BCNN':
            assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            self.representation = BCNN()
            self.fc = nn.Linear(512 * 512, outdim)
        else:
            self.fc = nn.Linear(512, outdim)

        self.bn = nn.BatchNorm1d(outdim)

        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def freeze(self):
        # Freeze all previous layers.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass of the network.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        #!print('The shape of x:', x.shape)
        if self.config.representation == 'BCNN':
            x = self.representation(x)
        else:
            x = self.backbone.avgpool(x)

        x = self.fc(x)
        x = self.bn(x)

        return x.unsqueeze(-1)
