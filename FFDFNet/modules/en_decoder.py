import torch
from torch import nn, Tensor


class FullBandEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.activate = nn.ELU()

    def forward(self, complex_spectrum: Tensor):
        complex_spectrum = self.conv(complex_spectrum)
        complex_spectrum = self.norm(complex_spectrum)
        complex_spectrum = self.activate(complex_spectrum)

        return complex_spectrum


class FullBandDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2,
                              kernel_size=1, stride=1, padding=0)
        self.conv_t = nn.ConvTranspose1d(in_channels // 2, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding)
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.activate = nn.ELU()

    def forward(self, encode_complex_spectrum: Tensor, decode_complex_spectrum: Tensor):
        complex_spectrum = torch.cat([encode_complex_spectrum, decode_complex_spectrum], dim=1)
        complex_spectrum = self.conv(complex_spectrum)
        complex_spectrum = self.conv_t(complex_spectrum)
        complex_spectrum = self.norm(complex_spectrum)
        complex_spectrum = self.activate(complex_spectrum)

        return complex_spectrum


class SubBandEncoderBlock(nn.Module):
    def __init__(self, start_frequency: int,
                 end_frequency: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.activate = nn.ReLU()

    def forward(self, amplitude_spectrum: Tensor):
        sub_spectrum = amplitude_spectrum[:, :, self.start_frequency:self.end_frequency]
        sub_spectrum = self.conv(sub_spectrum)
        sub_spectrum = self.activate(sub_spectrum)

        return sub_spectrum


class SubBandDecoderBlock(nn.Module):
    def __init__(self,
                 start_idx: int,
                 end_idx: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.feature_fuse = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1),
            nn.ELU(),
        )
        self.deconv = nn.ConvTranspose1d(in_channels=hidden_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding)
        self.activate = nn.ReLU()

    def forward(self, decode_feature: Tensor, encode_amplitude_spectrum: Tensor):
        decode_feature = decode_feature[:, :, self.start_idx:self.end_idx]
        spectrum = torch.cat([decode_feature, encode_amplitude_spectrum], dim=1)
        spectrum = self.feature_fuse(spectrum)
        spectrum = self.deconv(spectrum)
        spectrum = self.activate(spectrum)

        return spectrum
