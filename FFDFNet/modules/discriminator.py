from typing import List, Tuple

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.utils import spectral_norm, weight_norm


def _norm_conv(module: nn.Module, use_spectral_norm: bool):
    return spectral_norm(module) if use_spectral_norm else weight_norm(module)


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            _norm_conv(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)), use_spectral_norm),
            _norm_conv(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)), use_spectral_norm),
            _norm_conv(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)), use_spectral_norm),
            # _norm_conv(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)), use_spectral_norm),
            # _norm_conv(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0)), use_spectral_norm),
        ])
        # self.output_conv = _norm_conv(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        #                               use_spectral_norm)
        self.output_conv = _norm_conv(nn.Conv2d(512, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                                      use_spectral_norm)

    def forward(self, waveform: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feature_maps = []
        batch, channels, time = waveform.shape
        if time % self.period != 0:
            pad = self.period - (time % self.period)
            waveform = F.pad(waveform, (0, pad), mode="reflect")
        waveform = waveform.view(batch, channels, -1, self.period)

        outputs = waveform
        for conv in self.convs:
            outputs = F.leaky_relu(conv(outputs), negative_slope=0.1)
            feature_maps.append(outputs)
        outputs = self.output_conv(outputs)
        feature_maps.append(outputs)

        return torch.flatten(outputs, 1, -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=period, use_spectral_norm=use_spectral_norm) for period in periods]
        )

    def forward(self, waveform: Tensor):
        logits = []
        feature_maps = []
        for discriminator in self.discriminators:
            logit, fmap = discriminator(waveform)
            logits.append(logit)
            feature_maps.append(fmap)

        return logits, feature_maps


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.convs = nn.ModuleList([
            _norm_conv(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7), use_spectral_norm),
            _norm_conv(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20), use_spectral_norm),
            _norm_conv(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20), use_spectral_norm),
            _norm_conv(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20), use_spectral_norm),
            # _norm_conv(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20), use_spectral_norm),
            # _norm_conv(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20), use_spectral_norm),
            # _norm_conv(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2), use_spectral_norm),
        ])
        # self.output_conv = _norm_conv(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.output_conv = _norm_conv(nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1), use_spectral_norm)

    def forward(self, waveform: Tensor) -> Tuple[Tensor, List[Tensor]]:
        feature_maps = []
        outputs = waveform
        for conv in self.convs:
            outputs = F.leaky_relu(conv(outputs), negative_slope=0.1)
            feature_maps.append(outputs)
        outputs = self.output_conv(outputs)
        feature_maps.append(outputs)
        return torch.flatten(outputs, 1, -1), feature_maps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm if idx == 0 else False) for idx in range(scales)]
        )
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, waveform: Tensor):
        logits = []
        feature_maps = []
        outputs = waveform
        for idx, discriminator in enumerate(self.discriminators):
            if idx > 0:
                outputs = self.pooling(outputs)
            logit, fmap = discriminator(outputs)
            logits.append(logit)
            feature_maps.append(fmap)

        return logits, feature_maps


class FFDFDiscriminator(nn.Module):
    def __init__(self, periods, scales: int, use_spectral_norm: bool = False):
        super().__init__()
        self.multi_period_discriminator = MultiPeriodDiscriminator(periods=periods,
                                                                  use_spectral_norm=use_spectral_norm)
        self.multi_scale_discriminator = MultiScaleDiscriminator(scales=scales,
                                                                 use_spectral_norm=use_spectral_norm)

    def forward(self, waveform: Tensor):
        mpd_logits, mpd_feature_maps = self.multi_period_discriminator(waveform)
        msd_logits, msd_feature_maps = self.multi_scale_discriminator(waveform)
        return {
            "logits": mpd_logits + msd_logits,
            "feature_maps": mpd_feature_maps + msd_feature_maps,
        }
