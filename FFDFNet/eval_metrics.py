import math
from typing import Dict

import numpy as np
import torch

try:
    from pesq import pesq as pesq_fn
except ImportError:  # pragma: no cover
    pesq_fn = None

try:
    from pystoi import stoi as stoi_fn
except ImportError:  # pragma: no cover
    stoi_fn = None


def _prepare_pair(clean_waveform, enhanced_waveform):
    clean = np.asarray(clean_waveform, dtype=np.float32).reshape(-1)
    enhanced = np.asarray(enhanced_waveform, dtype=np.float32).reshape(-1)
    min_length = min(clean.shape[0], enhanced.shape[0])
    if min_length == 0:
        return None, None
    return clean[:min_length], enhanced[:min_length]


def calculate_pesq(clean_waveform, enhanced_waveform, sample_rate: int):
    if pesq_fn is None:
        return math.nan

    clean, enhanced = _prepare_pair(clean_waveform, enhanced_waveform)
    if clean is None:
        return math.nan

    mode = "wb" if sample_rate >= 16000 else "nb"
    try:
        return float(pesq_fn(sample_rate, clean, enhanced, mode))
    except Exception:
        return math.nan


def calculate_stoi(clean_waveform, enhanced_waveform, sample_rate: int, extended: bool = False):
    if stoi_fn is None:
        return math.nan

    clean, enhanced = _prepare_pair(clean_waveform, enhanced_waveform)
    if clean is None:
        return math.nan

    try:
        return float(stoi_fn(clean, enhanced, sample_rate, extended=extended))
    except Exception:
        return math.nan


def calculate_batch_metrics(clean_waveforms: torch.Tensor, enhanced_waveforms: torch.Tensor, sample_rate: int):
    clean_waveforms = clean_waveforms.detach().cpu()
    enhanced_waveforms = enhanced_waveforms.detach().cpu()

    pesq_scores = []
    stoi_scores = []

    for clean_waveform, enhanced_waveform in zip(clean_waveforms, enhanced_waveforms):
        pesq_score = calculate_pesq(clean_waveform.numpy(), enhanced_waveform.numpy(), sample_rate)
        stoi_score = calculate_stoi(clean_waveform.numpy(), enhanced_waveform.numpy(), sample_rate)
        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)

    return {
        "pesq_sum": float(np.nansum(pesq_scores)),
        "pesq_count": int(np.sum(~np.isnan(pesq_scores))),
        "stoi_sum": float(np.nansum(stoi_scores)),
        "stoi_count": int(np.sum(~np.isnan(stoi_scores))),
    }
