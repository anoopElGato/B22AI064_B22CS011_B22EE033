import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from configs.train_configs import TrainConfig
from models.ffdfnet import FFDFNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FFDFNet on a single noisy/clean audio pair.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved generator checkpoint.")
    parser.add_argument("--noisy-audio", type=str, required=True, help="Path to the noisy input wav file.")
    parser.add_argument("--clean-audio", type=str, required=True, help="Path to the clean reference wav file.")
    parser.add_argument("--output-dir", type=str, default="single_eval_outputs", help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def select_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def init_hidden_state(configs: TrainConfig, batch_size: int, frames: int, device: torch.device):
    del frames
    groups = configs.inter_frame_gru["parameters"]["groups"]
    hidden_size = configs.inter_frame_gru["parameters"]["hidden_size"]
    num_modules = configs.inter_frame_gru["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)

    return [
        [torch.zeros(1, batch_size * num_bands, hidden_size // groups, device=device) for _ in range(groups)]
        for _ in range(num_modules)
    ]


def load_audio(path: Path, target_sample_rate: int):
    waveform, sample_rate = sf.read(str(path), always_2d=False)
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def waveform_to_generator_inputs(waveform: torch.Tensor, configs: TrainConfig):
    window = torch.hamming_window(configs.n_fft, device=waveform.device)
    complex_spectrum = torch.stft(
        waveform,
        n_fft=configs.n_fft,
        hop_length=configs.hop_length,
        window=window,
        return_complex=True,
    )
    amplitude_spectrum = torch.abs(complex_spectrum)

    complex_spectrum = torch.view_as_real(complex_spectrum)
    complex_spectrum = torch.permute(complex_spectrum, dims=(0, 2, 3, 1)).contiguous()
    amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 2, 1)).unsqueeze(2).contiguous()
    return complex_spectrum, amplitude_spectrum


def generator_outputs_to_waveform(enhanced_complex: torch.Tensor, configs: TrainConfig, target_length: int):
    enhanced_complex = torch.permute(enhanced_complex, dims=(0, 3, 1, 2)).contiguous()
    enhanced_complex = torch.view_as_complex(enhanced_complex)
    window = torch.hamming_window(configs.n_fft, device=enhanced_complex.device)
    return torch.istft(
        enhanced_complex,
        n_fft=configs.n_fft,
        hop_length=configs.hop_length,
        window=window,
        length=target_length,
    )


def load_generator(checkpoint_path: Path, device: torch.device):
    configs = TrainConfig()
    model = FFDFNet(configs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, configs


@torch.no_grad()
def enhance_waveform(model: FFDFNet, configs: TrainConfig, noisy_waveform: torch.Tensor, device: torch.device):
    noisy_waveform = noisy_waveform.unsqueeze(0).to(device)
    noisy_complex, noisy_amplitude = waveform_to_generator_inputs(noisy_waveform, configs)
    hidden_state = init_hidden_state(configs, batch_size=1, frames=noisy_complex.shape[1], device=device)
    enhanced_complex, _ = model(noisy_complex, noisy_amplitude, hidden_state)
    enhanced_waveform = generator_outputs_to_waveform(enhanced_complex, configs, noisy_waveform.shape[-1])
    return enhanced_waveform.squeeze(0).cpu()


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), sample_rate)


def main():
    args = parse_args()
    device = select_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    noisy_path = Path(args.noisy_audio)
    clean_path = Path(args.clean_audio)
    output_dir = Path(args.output_dir)

    model, configs = load_generator(checkpoint_path, device)

    noisy_waveform, sample_rate = load_audio(noisy_path, configs.sample_rate)
    clean_waveform, _ = load_audio(clean_path, configs.sample_rate)

    target_length = min(noisy_waveform.shape[-1], clean_waveform.shape[-1])
    noisy_waveform = noisy_waveform[:target_length]
    clean_waveform = clean_waveform[:target_length]

    enhanced_waveform = enhance_waveform(model, configs, noisy_waveform, device)
    enhanced_waveform = enhanced_waveform[:target_length]

    save_audio(output_dir / "noisy.wav", noisy_waveform, sample_rate)
    save_audio(output_dir / "clean.wav", clean_waveform, sample_rate)
    save_audio(output_dir / "generated.wav", enhanced_waveform, sample_rate)

    print(f"Saved noisy audio to: {output_dir / 'noisy.wav'}")
    print(f"Saved clean audio to: {output_dir / 'clean.wav'}")
    print(f"Saved generated audio to: {output_dir / 'generated.wav'}")


if __name__ == "__main__":
    main()
