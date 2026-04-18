from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
import soundfile as sf

from torch.utils.data import Dataset


def _find_first_existing_dir(root_dir: Path, candidates: Iterable[str]) -> Path:
    for candidate in candidates:
        path = root_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate directories exist under {root_dir}: {list(candidates)}")


class VoiceBankDemandDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 noisy_dir_candidates,
                 clean_dir_candidates,
                 sample_rate: int,
                 segment_length: int,
                 training: bool):
        self.root_dir = Path(root_dir)
        self.noisy_dir = _find_first_existing_dir(self.root_dir, noisy_dir_candidates)
        self.clean_dir = _find_first_existing_dir(self.root_dir, clean_dir_candidates)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.training = training

        self.file_pairs = self._collect_pairs()

    def _collect_pairs(self):
        noisy_files = sorted(self.noisy_dir.glob("*.wav"))
        clean_files = {file.name: file for file in self.clean_dir.glob("*.wav")}

        file_pairs = []
        for noisy_file in noisy_files:
            clean_file = clean_files.get(noisy_file.name)
            if clean_file is not None:
                file_pairs.append((noisy_file, clean_file))

        if not file_pairs:
            raise RuntimeError(
                f"No paired wav files found between {self.noisy_dir} and {self.clean_dir}."
            )

        return file_pairs

    def _load_audio(self, path: Path):
        try:
            waveform, sample_rate = torchaudio.load(path)
            waveform = waveform.mean(dim=0)
        except Exception:
            waveform, sample_rate = sf.read(str(path), always_2d=False)
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.from_numpy(waveform)

        waveform = waveform.to(dtype=torch.float32)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        return waveform

    def _match_length(self, noisy_waveform: torch.Tensor, clean_waveform: torch.Tensor):
        min_length = min(noisy_waveform.shape[-1], clean_waveform.shape[-1])
        noisy_waveform = noisy_waveform[:min_length]
        clean_waveform = clean_waveform[:min_length]

        if self.segment_length is None:
            return noisy_waveform, clean_waveform

        if min_length >= self.segment_length:
            if self.training:
                start = torch.randint(0, min_length - self.segment_length + 1, (1,)).item()
            else:
                start = 0
            end = start + self.segment_length
            return noisy_waveform[start:end], clean_waveform[start:end]

        pad_length = self.segment_length - min_length
        noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, pad_length))
        clean_waveform = torch.nn.functional.pad(clean_waveform, (0, pad_length))
        return noisy_waveform, clean_waveform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index: int):
        noisy_path, clean_path = self.file_pairs[index]
        noisy_waveform = self._load_audio(noisy_path)
        clean_waveform = self._load_audio(clean_path)
        noisy_waveform, clean_waveform = self._match_length(noisy_waveform, clean_waveform)

        return {
            "noisy": noisy_waveform,
            "clean": clean_waveform,
            "file_name": noisy_path.name,
        }
