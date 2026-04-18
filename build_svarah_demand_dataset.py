import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio

from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a noisy accented-speech dataset by mixing clean Svarah audio with DEMAND noises."
    )
    parser.add_argument("--output-root", type=str, required=True, help="Directory where the mixed dataset is written.")
    parser.add_argument("--demand-root", type=str, required=True, help="Root directory containing DEMAND wav files.")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional Hugging Face token for gated Svarah access.")
    parser.add_argument("--split", type=str, default="test", help="Svarah split to use. Default: test")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--min-snr-db", type=float, default=0.0, help="Minimum SNR in dB.")
    parser.add_argument("--max-snr-db", type=float, default=20.0, help="Maximum SNR in dB.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of Svarah examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-noise-audio", action="store_true", help="Also save the sampled noise clip for each example.")
    return parser.parse_args()


def _resample_if_needed(waveform, sample_rate: int, target_sample_rate: int):
    waveform = waveform.to(dtype=waveform.dtype)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def load_waveform(path: Path, target_sample_rate: int):
    waveform, sample_rate = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = torch_from_numpy(waveform)
    return _resample_if_needed(waveform, sample_rate, target_sample_rate)


def load_svarah_audio(audio_entry, target_sample_rate: int, hf_token: str | None):
    if isinstance(audio_entry, dict):
        audio_path = audio_entry.get("path")
        audio_bytes = audio_entry.get("bytes")
    else:
        audio_path = audio_entry
        audio_bytes = None

    if audio_bytes is not None:
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch_from_numpy(waveform)
        return _resample_if_needed(waveform, sample_rate, target_sample_rate)

    if audio_path is None:
        raise ValueError("Svarah audio entry does not contain a usable file path.")

    candidate_path = Path(audio_path)
    if candidate_path.exists():
        waveform, sample_rate = load_waveform(candidate_path, target_sample_rate)
        return waveform, sample_rate

    if not candidate_path.is_absolute():
        downloaded_path = hf_hub_download(
            repo_id="ai4bharat/Svarah",
            repo_type="dataset",
            filename=f"audio/{candidate_path.name}",
            token=hf_token,
        )
        waveform, sample_rate = load_waveform(Path(downloaded_path), target_sample_rate)
        return waveform, sample_rate

    waveform, sample_rate = load_waveform(candidate_path, target_sample_rate)
    return waveform, sample_rate


def torch_from_numpy(array: np.ndarray):
    import torch
    return torch.from_numpy(array.copy())


def ensure_min_length(noise_waveform, target_length: int):
    if noise_waveform.shape[-1] >= target_length:
        return noise_waveform

    repeats = math.ceil(target_length / max(noise_waveform.shape[-1], 1))
    return noise_waveform.repeat(repeats)[:target_length]


def sample_noise_segment(noise_waveform, target_length: int, rng: random.Random):
    noise_waveform = ensure_min_length(noise_waveform, target_length)
    if noise_waveform.shape[-1] == target_length:
        return noise_waveform
    start = rng.randint(0, noise_waveform.shape[-1] - target_length)
    return noise_waveform[start:start + target_length]


def rms(signal):
    import torch
    return torch.sqrt(torch.mean(signal ** 2) + 1e-8)


def mix_at_snr(clean_waveform, noise_waveform, snr_db: float):
    clean_level = rms(clean_waveform)
    noise_level = rms(noise_waveform)
    desired_noise_level = clean_level / (10.0 ** (snr_db / 20.0))
    noise_scale = desired_noise_level / (noise_level + 1e-8)
    scaled_noise = noise_waveform * noise_scale
    noisy_waveform = clean_waveform + scaled_noise

    peak = noisy_waveform.abs().max().item()
    if peak > 0.999:
        noisy_waveform = noisy_waveform / peak * 0.999
        scaled_noise = scaled_noise / peak * 0.999
        clean_waveform = clean_waveform / peak * 0.999

    return clean_waveform, scaled_noise, noisy_waveform


def save_audio(path: Path, waveform, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), sample_rate)


def collect_demand_noises(demand_root: Path):
    noise_files = sorted(demand_root.rglob("*.wav"))
    if not noise_files:
        raise RuntimeError(f"No wav files found under DEMAND root: {demand_root}")
    return noise_files


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_root = Path(args.output_root)
    clean_dir = output_root / "clean"
    noisy_dir = output_root / "noisy"
    noise_dir = output_root / "noise"
    manifest_path = output_root / "manifest.jsonl"

    output_root.mkdir(parents=True, exist_ok=True)

    demand_files = collect_demand_noises(Path(args.demand_root))

    dataset = load_dataset("ai4bharat/Svarah", split=args.split, token=args.hf_token)
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=args.sample_rate, decode=False))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for index, example in enumerate(tqdm(dataset, total= len(dataset))):
            clean_waveform, _ = load_svarah_audio(example["audio_filepath"], args.sample_rate, args.hf_token)
            noise_path = rng.choice(demand_files)
            noise_waveform, _ = load_waveform(noise_path, args.sample_rate)
            noise_segment = sample_noise_segment(noise_waveform, clean_waveform.shape[-1], rng)
            snr_db = rng.uniform(args.min_snr_db, args.max_snr_db)
            clean_out, scaled_noise, noisy_waveform = mix_at_snr(clean_waveform, noise_segment, snr_db)

            file_stem = f"svarah_{index:06d}"
            clean_path = clean_dir / f"{file_stem}.wav"
            noisy_path = noisy_dir / f"{file_stem}.wav"
            noise_out_path = noise_dir / f"{file_stem}.wav"

            save_audio(clean_path, clean_out, args.sample_rate)
            save_audio(noisy_path, noisy_waveform, args.sample_rate)
            if args.save_noise_audio:
                save_audio(noise_out_path, scaled_noise, args.sample_rate)

            manifest_record = {
                "id": file_stem,
                "clean_path": str(clean_path.resolve()),
                "noisy_path": str(noisy_path.resolve()),
                "noise_path": str(noise_out_path.resolve()) if args.save_noise_audio else None,
                "sample_rate": args.sample_rate,
                "duration_seconds": float(clean_waveform.shape[-1] / args.sample_rate),
                "snr_db": snr_db,
                "noise_source_path": str(noise_path.resolve()),
                "text": example.get("text"),
                "gender": example.get("gender"),
                "age_group": example.get("age-group"),
                "primary_language": example.get("primary_language"),
                "native_place_state": example.get("native_place_state"),
                "native_place_district": example.get("native_place_district"),
                "highest_qualification": example.get("highest_qualification"),
                "job_category": example.get("job_category"),
                "occupation_domain": example.get("occupation_domain"),
            }
            manifest_file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

            # if (index + 1) % 100 == 0:
            #     print(f"Processed {index + 1} examples")

    print(f"Saved mixed dataset to: {output_root}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
