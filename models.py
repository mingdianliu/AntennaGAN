from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from torchmetrics.functional import mean_absolute_percentage_error

import numpy as np


cudnn.benchmark = True


import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image

from untils import *



cuda = 1
nc = 1
ndf = 64
ns11 = 451

device = torch.device("cuda" if cuda else "cpu")

# custom weights initialization called on netS
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# build Simulator block
class Simulator(nn.Module):
    def __init__(self, num_classes=451, ngpu=1):
        super(Simulator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*1) x 32 x 32

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(ndf * 1, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Flatten(),
            nn.Linear((ndf*8) * 4 * 4, (ndf*8) * 4 * 2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear((ndf*8) * 4 * 2, (ndf*8) * 4 * 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear((ndf*8) * 4 * 1, num_classes,            bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)


        return output.view(-1, ns11).squeeze(1)



"""ResNet reference: https://jarvislabs.ai/blogs/resnet/ """

## ResNet implementation


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x



class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 1

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        # state size. 256 x 32 x 32

        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        # state size. 512 x 16 x 16

        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        # state size. 1024 x 8 x 8

        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        # state size. 2048 x 4 x 4

        self.layer5 = self._make_layer(ResBlock, layer_list[3], planes=1024, stride=2)
        # state size. 4096 x 2 x 2


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * ResBlock.expansion, num_classes)

        # nn.Linear((ndf * 8) * 4 * 4, (ndf * 8) * 4 * 2, bias=True),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Linear((ndf * 8) * 4 * 2, (ndf * 8) * 4 * 1, bias=True),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Linear((ndf * 8) * 4 * 1, num_classes, bias=True),
        # nn.LeakyReLU(0.2, inplace=True),

    def forward(self, x):
        # print('layer 0 ', x.size())
        x = self.layer1(x)
        # print('layer 1 ', x.size())
        x = self.layer2(x)
        # print('layer 2 ', x.size())
        x = self.layer3(x)
        # print('layer 3 ', x.size())
        x = self.layer4(x)
        # print('layer 4 ', x.size())
        x = self.layer5(x)
        # print('layer 5 ', x.size())

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)





def ResSimulator(num_classes=451, channels=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, num_classes=451, channels=1, ngpu = 1):
    super().__init__()
    self.ngpu = ngpu
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 1, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      # nn.Dropout(0.1),
      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      # nn.Dropout(0.1),
      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      # nn.Dropout(0.1),
      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.2, inplace=True),
      # nn.Dropout(0.1),
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.2, inplace=True),
      # nn.Dropout(0.1),
      nn.Linear(512, num_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    if x.is_cuda and self.ngpu > 1:
        output = nn.parallel.data_parallel(self.layers, x, range(self.ngpu))
    else:
        output = self.layers(x)

    return output

def MLPSimulator(num_classes=451, channels=1):
    return MLP(num_classes, channels)



class CNN(nn.Module):
    def __init__(self, num_classes=451, channels=1, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 32 x 32``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 16 x 16``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 8 x 8``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 4 x 4``
            nn.flatten(),
            nn.Linear(1, num_classes)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.layers(input)
        return output

def CNNSimulator(num_classes=451, channels=1, ngpu=1) :
    return CNN(num_classes, channels, ngpu)


class SimpleCritic(nn.Module):
    def __init__(self, ngpu):
        super(SimpleCritic, self).__init__()
        ndf = 32
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # nn.Conv2d(ndf, ndf * 1, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 1),
            # nn.LeakyLeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:

            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# build Critic block
class Critic(nn.Module):
    def __init__(self, ngpu):
        super(Critic, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*1) x 32 x 32

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(ndf * 1, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Flatten(),
            nn.Linear((ndf*8) * 4 * 4, (ndf*8) * 4 * 2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear((ndf*8) * 4 * 2, (ndf*8) * 4 * 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear((ndf*8) * 4 * 1, 1,            bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, input):

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)

def ResCritic(num_classes=1, channels=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


class InverseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_upsample=None, stride=1):
        super(InverseBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels * self.expansion)

        self.conv2 = nn.ConvTranspose2d(out_channels * self.expansion, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1 if stride==2 else 0)
        # nn.ConvTranspose2d(self.in_channels, planes, kernel_size=1, stride=stride, output_padding=1),
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)

        self.i_upsample = i_upsample

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        # print('plane 0 ', x.size())

        x = self.relu(self.batch_norm1(self.conv1(x)))
        # print('plane 1 ', x.size())
        # print('stride ', self.stride)

        x = self.relu(self.batch_norm2(self.conv2(x)))
        # print('plane 2 ', x.size())

        x = self.relu(self.batch_norm3(self.conv3(x)))
        # print('plane 3 ', x.size())

        # upsample if needed
        if self.i_upsample is not None:
            identity = self.i_upsample(identity)
            # print('identity ', identity.size())
        # add identity
        x = x.clone() + identity
        x = self.relu(x)

        return x



class InverseResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(InverseResNet, self).__init__()
        self.in_channels = 4096

        self.fc = nn.Linear(num_classes, 4096*2*2)
        self.relu = nn.ReLU()
        # state size. 4096 x 2 x 2

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=1024, stride=2)
        # state size. 2048 x 4 x 4

        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=512, stride=2)
        # state size. 1024 x 8 x 8

        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        # state size. 512 x 16 x 16

        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=128, stride=2)
        # state size. 256 x 32 x 32


        self.conv = nn.ConvTranspose2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print('x ', x.size())
        x = self.relu(self.fc(x))
        x = x.reshape(-1, 4096, 2, 2)
        # print('layer 0 ', x.size())
        x = self.layer1(x)
        # print('layer 1 ', x.size())
        x = self.layer2(x)
        # print('layer 2 ', x.size())
        x = self.layer3(x)
        # print('layer 3 ', x.size())
        x = self.layer4(x)
        # print('layer 4 ', x.size())
        # x = self.layer5(x)
        # print('layer 5 ', x.size())

        # print(x.size())

        x = self.conv(x)
        x = x.reshape(-1, 32, 32)
        # x = torch.sign(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_upsample = None
        layers = []

        if stride != 1 or self.in_channels == planes * ResBlock.expansion:
            ii_upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, planes, kernel_size=1, stride=stride, output_padding=1),
                nn.BatchNorm2d(planes)
            )

        layers.append(ResBlock(self.in_channels, planes, i_upsample=ii_upsample, stride=stride))
        self.in_channels = planes

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResGenerator(num_classes=451, channels=1):
    return InverseResNet(InverseBottleneck, [3, 6, 4, 3, 3], num_classes, channels)


nz = 10
ngf = 128

class SimpleGenerator(nn.Module):
    def __init__(self, ngpu):
        super(SimpleGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(451+nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 1,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.linear = nn.Sequential(
            nn.Linear(451 + nz, 451 + nz, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(451 + nz, 451 + nz, bias=True),
            nn.LeakyReLU(0.2, inplace=True))


        self.main = nn.Sequential(
            # nn.Linear(451 + nz, 451 + nz, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(451 + nz, 451 + nz, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(451 + nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 1,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
            output = output.reshape(-1, 451 + nz, 1, 1)
            output = nn.parallel.data_parallel(self.main, output, range(self.ngpu))
        else:
            output = self.main(input)
        return output
