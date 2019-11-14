import torch.nn as nn
from utils.common_library import *


class _Activation_combo_BN2_ReLU(nn.Module):
    def __init__(self, out_channels, dropout=0):
        super(_Activation_combo_BN2_ReLU, self).__init__()
        self.activation = nn.Sequential(*[
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        ])

    def forward(self, x):
        return self.activation(x)

class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1):
        super(_ResBlock, self).__init__()
        padsize = int(np.floor(ksize / 2))
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padsize),
        ])
        self.conv_dual = nn.Sequential(*[
            _Activation_combo_BN2_ReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=padsize),
            _Activation_combo_BN2_ReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=padsize),
        ])

    def forward(self, x):
        conv1 = self.conv1(x)
        # print(conv.shape)
        conv2 = self.conv_dual(conv1)
        return conv1+conv2