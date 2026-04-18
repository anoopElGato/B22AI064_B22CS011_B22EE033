import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from configs.train_configs import TrainConfig
from eval_metrics import calculate_pesq, calculate_stoi
from models.ffdfnet import FFDFNet

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FFDFNet on a mixed Svarah+DEMAND dataset and aggregate PESQ/STOI by accent and age group."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.jsonl of the mixed dataset.")
    parser.add_argument("--output-dir", type=str, default="svarah_eval_results", help="Directory to save reports.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of manifest entries.")
    return parser.parse_args()


def select_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_waveform(path: Path, target_sample_rate: int):
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


def init_hidden_state(configs: TrainConfig, batch_size: int, device: torch.device):
    groups = configs.inter_frame_gru["parameters"]["groups"]
    hidden_size = configs.inter_frame_gru["parameters"]["hidden_size"]
    num_modules = configs.inter_frame_gru["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)

    return [
        [torch.zeros(1, batch_size * num_bands, hidden_size // groups, device=device) for _ in range(groups)]
        for _ in range(num_modules)
    ]


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
    hidden_state = init_hidden_state(configs, batch_size=1, device=device)
    enhanced_complex, _ = model(noisy_complex, noisy_amplitude, hidden_state)
    enhanced_waveform = generator_outputs_to_waveform(enhanced_complex, configs, noisy_waveform.shape[-1])
    return enhanced_waveform.squeeze(0).cpu()


def read_manifest(manifest_path: Path, limit: int | None):
    records = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if limit is not None and idx >= limit:
                break
            records.append(json.loads(line))
    return records


def update_group_stats(group_stats: dict, group_key: str, pesq_score: float, stoi_score: float):
    stats = group_stats[group_key]
    stats["num_examples"] += 1

    if not math.isnan(pesq_score):
        stats["pesq_sum"] += pesq_score
        stats["pesq_count"] += 1

    if not math.isnan(stoi_score):
        stats["stoi_sum"] += stoi_score
        stats["stoi_count"] += 1


def finalize_group_stats(group_stats: dict):
    finalized = {}
    for group_key, stats in group_stats.items():
        finalized[group_key] = {
            "num_examples": stats["num_examples"],
            "mean_pesq": stats["pesq_sum"] / stats["pesq_count"] if stats["pesq_count"] > 0 else math.nan,
            "mean_stoi": stats["stoi_sum"] / stats["stoi_count"] if stats["stoi_count"] > 0 else math.nan,
            "pesq_count": stats["pesq_count"],
            "stoi_count": stats["stoi_count"],
        }
    return finalized


def make_stats_dict():
    return {
        "num_examples": 0,
        "pesq_sum": 0.0,
        "pesq_count": 0,
        "stoi_sum": 0.0,
        "stoi_count": 0,
    }


def main():
    args = parse_args()
    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, configs = load_generator(checkpoint_path, device)
    records = read_manifest(manifest_path, args.limit)

    accent_stats = defaultdict(make_stats_dict)
    age_stats = defaultdict(make_stats_dict)
    overall_stats = make_stats_dict()
    per_file_results = []

    for idx, record in enumerate(tqdm(records, total= len(records)), start=1):
        clean_path = Path(record["clean_path"])
        noisy_path = Path(record["noisy_path"])

        clean_waveform, sample_rate = load_waveform(clean_path, configs.sample_rate)
        noisy_waveform, _ = load_waveform(noisy_path, configs.sample_rate)

        target_length = min(clean_waveform.shape[-1], noisy_waveform.shape[-1])
        clean_waveform = clean_waveform[:target_length]
        noisy_waveform = noisy_waveform[:target_length]

        enhanced_waveform = enhance_waveform(model, configs, noisy_waveform, device)
        enhanced_waveform = enhanced_waveform[:target_length]

        pesq_score = calculate_pesq(clean_waveform.numpy(), enhanced_waveform.numpy(), sample_rate)
        stoi_score = calculate_stoi(clean_waveform.numpy(), enhanced_waveform.numpy(), sample_rate)

        primary_language = record.get("primary_language") or "unknown"
        age_group = record.get("age_group") or "unknown"

        update_group_stats(accent_stats, primary_language, pesq_score, stoi_score)
        update_group_stats(age_stats, age_group, pesq_score, stoi_score)
        update_group_stats({"overall": overall_stats}, "overall", pesq_score, stoi_score)

        per_file_results.append({
            "id": record.get("id"),
            "clean_path": str(clean_path),
            "noisy_path": str(noisy_path),
            "primary_language": primary_language,
            "age_group": age_group,
            "pesq": pesq_score,
            "stoi": stoi_score,
        })

        # if idx % 50 == 0:
        #     print(f"Processed {idx}/{len(records)} examples")

    summary = {
        "overall": finalize_group_stats({"overall": overall_stats})["overall"],
        "by_primary_language": finalize_group_stats(accent_stats),
        "by_age_group": finalize_group_stats(age_stats),
        "num_records_evaluated": len(records),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    with open(output_dir / "per_file_results.jsonl", "w", encoding="utf-8") as file:
        for row in per_file_results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved grouped evaluation summary to: {output_dir / 'summary.json'}")
    print(f"Saved per-file results to: {output_dir / 'per_file_results.jsonl'}")


if __name__ == "__main__":
    main()
