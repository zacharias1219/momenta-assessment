# rawnet2_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000):
        super(SincConv, self).__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel")

        self.device = device
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.out_channels = out_channels + 1
        self.sample_rate = sample_rate

        NFFT = 512
        f = np.linspace(0, self.sample_rate / 2, NFFT // 2 + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 2)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        self.freq = filbandwidthsf[:self.out_channels]

        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels - 1, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.freq) - 1):
            fmin, fmax = self.freq[i], self.freq[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        filters = self.band_pass.to(self.device).view(self.out_channels - 1, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=0, dilation=1, bias=None, groups=1)


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(ResidualBlock, self).__init__()
        self.first = first
        self.lrelu = nn.LeakyReLU(0.3)

        if not self.first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])

        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], kernel_size=3, stride=1, padding=1)

        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=1)

        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            x = self.bn1(x)
            x = self.lrelu(x)

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return self.mp(out)


class RawNet2(nn.Module):
    def __init__(self, config):
        super(RawNet2, self).__init__()
        self.sinc = SincConv(device='cuda' if torch.cuda.is_available() else 'cpu',
                             out_channels=config.filts[0], kernel_size=config.first_conv)
        self.first_bn = nn.BatchNorm1d(config.filts[0])
        self.selu = nn.SELU()

        self.block0 = ResidualBlock(config.filts[1], first=True)
        self.block1 = ResidualBlock(config.filts[1])
        self.block2 = ResidualBlock(config.filts[2])
        config.filts[2][0] = config.filts[2][1]  # update to match output shape
        self.block3 = ResidualBlock(config.filts[2])

        self.bn = nn.BatchNorm1d(config.filts[2][1])
        self.gru = nn.GRU(input_size=config.filts[2][1], hidden_size=config.gru_node,
                          num_layers=config.nb_gru_layer, batch_first=True)
        self.fc1 = nn.Linear(config.gru_node, config.nb_fc_node)
        self.fc2 = nn.Linear(config.nb_fc_node, config.nb_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = torch.abs(self.sinc(x))
        x = F.max_pool1d(x, 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.bn(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x)