import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PESQ/STOI grouped by primary language and age group from summary.json."
    )
    parser.add_argument("--summary-json", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--output-dir", type=str, default="svarah_group_plots", help="Directory to save plots")
    parser.add_argument("--top-k-accents", type=int, default=None, help="Optional top-k accents by number of examples")
    parser.add_argument("--sort-by", type=str, default="num_examples", choices=["num_examples", "mean_pesq", "mean_stoi"],
                        help="Sort criterion for accent plots")
    return parser.parse_args()


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sort_group_items(group_dict: dict, sort_by: str, top_k: int | None):
    items = list(group_dict.items())
    items.sort(key=lambda item: item[1].get(sort_by, 0), reverse=True)
    if top_k is not None:
        items = items[:top_k]
    return items


def plot_metric_bar(items, metric_key: str, title: str, ylabel: str, output_path: Path, rotate_xticks: bool):
    labels = [item[0] for item in items]
    values = [item[1].get(metric_key, float("nan")) for item in items]

    plt.figure(figsize=(max(10, len(labels) * 0.6), 6))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Group")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        if value == value:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                     ha="center", va="bottom", fontsize=8)

    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    summary = load_summary(Path(args.summary_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accent_items = sort_group_items(summary["by_primary_language"], args.sort_by, args.top_k_accents)
    age_items = sort_group_items(summary["by_age_group"], "num_examples", None)

    plot_metric_bar(
        accent_items,
        metric_key="mean_pesq",
        title="PESQ Across Primary Languages",
        ylabel="Mean PESQ",
        output_path=output_dir / "pesq_by_primary_language.png",
        rotate_xticks=True,
    )
    plot_metric_bar(
        accent_items,
        metric_key="mean_stoi",
        title="STOI Across Primary Languages",
        ylabel="Mean STOI",
        output_path=output_dir / "stoi_by_primary_language.png",
        rotate_xticks=True,
    )
    plot_metric_bar(
        age_items,
        metric_key="mean_pesq",
        title="PESQ Across Age Groups",
        ylabel="Mean PESQ",
        output_path=output_dir / "pesq_by_age_group.png",
        rotate_xticks=False,
    )
    plot_metric_bar(
        age_items,
        metric_key="mean_stoi",
        title="STOI Across Age Groups",
        ylabel="Mean STOI",
        output_path=output_dir / "stoi_by_age_group.png",
        rotate_xticks=False,
    )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()

# python FFDFNet/eval_svarah_demand_groups.py --checkpoint "checkpoints/epoch_0095.pt" --manifest "svarah_demand\manifest.jsonl" --output-dir "svarah_group_eval" --device cuda

# python plot_svarah_group_metrics.py --summary-json "svarah_group_eval\summary.json" --output-dir "svarah_group_plots"

# python FFDFNet/eval_single_audio.py --checkpoint "checkpoints/epoch_0100.pt" --noisy-audio "voiceBank_demand/noisy_testset_wav/p232_010.wav" --clean-audio "voiceBank_demand/clean_testset_wav/p232_010.wav" --output-dir "testAudio" --device cuda