from typing import Dict, List

import torch
import torch.nn.functional as F


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self,
                 fft_sizes,
                 hop_sizes,
                 win_lengths,
                 magnitude_weight: float,
                 phase_weight: float,
                 compression_exponent: float):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.compression_exponent = compression_exponent

    def _single_resolution_loss(self,
                                estimate_waveform: torch.Tensor,
                                target_waveform: torch.Tensor,
                                fft_size: int,
                                hop_size: int,
                                win_length: int):
        window = torch.hann_window(win_length, device=estimate_waveform.device)
        estimate_stft = torch.stft(estimate_waveform, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                                   window=window, return_complex=True)
        target_stft = torch.stft(target_waveform, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                                 window=window, return_complex=True)

        estimate_mag = torch.abs(estimate_stft).clamp_min(1e-8).pow(self.compression_exponent)
        target_mag = torch.abs(target_stft).clamp_min(1e-8).pow(self.compression_exponent)

        magnitude_loss = F.mse_loss(estimate_mag, target_mag)

        estimate_phase_aware = estimate_mag * torch.exp(1j * torch.angle(estimate_stft))
        target_phase_aware = target_mag * torch.exp(1j * torch.angle(target_stft))
        phase_loss = F.mse_loss(torch.view_as_real(estimate_phase_aware), torch.view_as_real(target_phase_aware))

        total_loss = self.magnitude_weight * magnitude_loss + self.phase_weight * phase_loss
        return total_loss, magnitude_loss, phase_loss

    def forward(self, estimate_waveform: torch.Tensor, target_waveform: torch.Tensor):
        total_loss = estimate_waveform.new_tensor(0.0)
        total_mag_loss = estimate_waveform.new_tensor(0.0)
        total_phase_loss = estimate_waveform.new_tensor(0.0)

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            res_total, res_mag, res_phase = self._single_resolution_loss(
                estimate_waveform, target_waveform, fft_size, hop_size, win_length
            )
            total_loss = total_loss + res_total
            total_mag_loss = total_mag_loss + res_mag
            total_phase_loss = total_phase_loss + res_phase

        num_resolutions = len(self.fft_sizes)
        return {
            "total": total_loss / num_resolutions,
            "magnitude": total_mag_loss / num_resolutions,
            "phase": total_phase_loss / num_resolutions,
        }


def discriminator_least_squares_loss(real_logits: List[torch.Tensor], fake_logits: List[torch.Tensor]):
    loss = real_logits[0].new_tensor(0.0)
    for real_logit, fake_logit in zip(real_logits, fake_logits):
        loss = loss + torch.mean((real_logit - 1.0) ** 2) + torch.mean(fake_logit ** 2)
    return loss


def generator_least_squares_loss(fake_logits: List[torch.Tensor]):
    loss = fake_logits[0].new_tensor(0.0)
    for fake_logit in fake_logits:
        loss = loss + torch.mean((fake_logit - 1.0) ** 2)
    return loss


def generator_non_saturating_loss(fake_logits: List[torch.Tensor]):
    loss = fake_logits[0].new_tensor(0.0)
    for fake_logit in fake_logits:
        loss = loss + torch.mean(F.softplus(-fake_logit))
    return loss


def feature_matching_loss(real_feature_maps, fake_feature_maps):
    loss = real_feature_maps[0][0].new_tensor(0.0)
    for real_maps, fake_maps in zip(real_feature_maps, fake_feature_maps):
        for real_map, fake_map in zip(real_maps, fake_maps):
            loss = loss + F.l1_loss(fake_map, real_map.detach())
    return loss


def build_generator_loss(stft_losses: Dict[str, torch.Tensor],
                         adversarial_loss: torch.Tensor,
                         feature_loss: torch.Tensor,
                         adversarial_weight: float,
                         feature_weight: float):
    total_loss = stft_losses["total"] + adversarial_weight * adversarial_loss + feature_weight * feature_loss
    return {
        "total": total_loss,
        "stft_total": stft_losses["total"],
        "stft_magnitude": stft_losses["magnitude"],
        "stft_phase": stft_losses["phase"],
        "adversarial": adversarial_loss,
        "feature_matching": feature_loss,
    }
