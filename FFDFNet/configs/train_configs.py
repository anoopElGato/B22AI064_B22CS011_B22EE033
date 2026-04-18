from typing import Dict, Tuple

from pydantic import BaseModel


def get_sub_band_specs(band_parameters: dict):
    group_bands = []
    group_band_width = []
    decoder_parameters = {}

    for idx, (_, value) in enumerate(band_parameters.items()):
        conv = value["conv"]
        group_width = value["group_width"]
        num_band = (group_width - conv["kernel_size"] + 2 * conv["padding"]) // conv["stride"] + 1
        sub_band_width = group_width // num_band
        output_padding = group_width - ((num_band - 1) * conv["stride"] - 2 * conv["padding"] + conv["kernel_size"])

        group_bands.append(num_band)
        group_band_width.append(sub_band_width)
        decoder_parameters[f"decoder{idx}"] = {
            "in_channels": conv["out_channels"] * 2,
            "hidden_channels": conv["out_channels"],
            "out_channels": 1,
            "kernel_size": conv["kernel_size"],
            "stride": conv["stride"],
            "padding": conv["padding"],
            "output_padding": output_padding,
        }

    return tuple(group_bands), tuple(group_band_width), decoder_parameters


class TrainConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {"in_channels": 2, "out_channels": 4, "kernel_size": 6, "stride": 2, "padding": 2},
        "encoder2": {"in_channels": 4, "out_channels": 16, "kernel_size": 8, "stride": 2, "padding": 3},
        "encoder3": {"in_channels": 16, "out_channels": 32, "kernel_size": 6, "stride": 2, "padding": 2},
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {"in_channels": 64, "out_channels": 16, "kernel_size": 6, "stride": 2, "padding": 2},
        "decoder2": {"in_channels": 32, "out_channels": 4, "kernel_size": 8, "stride": 2, "padding": 3},
        "decoder3": {"in_channels": 8, "out_channels": 2, "kernel_size": 6, "stride": 2, "padding": 2},
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {"group_width": 16, "conv": {"start_frequency": 0, "end_frequency": 16, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 4, "stride": 2, "padding": 1}},
        "encoder2": {"group_width": 18, "conv": {"start_frequency": 16, "end_frequency": 34, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 7, "stride": 3, "padding": 2}},
        "encoder3": {"group_width": 36, "conv": {"start_frequency": 34, "end_frequency": 70, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 11, "stride": 5, "padding": 2}},
        "encoder4": {"group_width": 66, "conv": {"start_frequency": 70, "end_frequency": 136, "in_channels": 1,
                                                 "out_channels": 32, "kernel_size": 20, "stride": 10, "padding": 4}},
        "encoder5": {"group_width": 121, "conv": {"start_frequency": 136, "end_frequency": 257, "in_channels": 1,
                                                  "out_channels": 32, "kernel_size": 30, "stride": 20, "padding": 5}},
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 2}

    _sub_band_specs = get_sub_band_specs(sub_band_encoder)
    bands_num_in_groups: Tuple[int, ...] = _sub_band_specs[0]
    band_width_in_groups: Tuple[int, ...] = _sub_band_specs[1]
    sub_band_decoder: Dict[str, dict] = _sub_band_specs[2]

    inter_frame_gru: dict = {
        "num_modules": 3,
        "parameters": {"input_size": 16, "hidden_size": 16, "groups": 8, "rnn_type": "GRU"},
    }
    frequency_transformer: dict = {
        "num_pre_blocks": 1,
        "num_post_blocks": 1,
        "parameters": {"input_size": 16, "hidden_size": 16, "num_heads": 4, "dropout": 0.0},
    }
    stft_losses: dict = {
        "fft_sizes": [128, 256, 512, 1024],
        "hop_sizes": [64, 128, 256, 512],
        "win_lengths": [128, 256, 512, 1024],
        "magnitude_weight": 0.3,
        "phase_weight": 0.7,
        "compression_exponent": 0.3,
    }
    adversarial_losses: dict = {
        "generator_objective": "non_saturating",
        "generator_weight": 0.05,
        "feature_matching_weight": 2.0,
    }
    discriminator: dict = {
        "periods": [2, 3, 5, 7, 11],
        "use_spectral_norm": False,
        "scales": 3,
    }
    dataset: dict = {
        "sample_rate": 16000,
        "segment_length": train_points,
        "valid_segment_length": None,
        "train_noisy_dirnames": ["noisy_trainset_28spk_wav", "train/noisy", "trainset/noisy"],
        "train_clean_dirnames": ["clean_trainset_28spk_wav", "train/clean", "trainset/clean"],
        "valid_noisy_dirnames": ["noisy_testset_wav", "test/noisy", "valid/noisy"],
        "valid_clean_dirnames": ["clean_testset_wav", "test/clean", "valid/clean"],
    }
    training: dict = {
        "batch_size": 8,
        "num_workers": 2,
        "epochs": 100,
        "generator_lr": 5e-4,
        "discriminator_lr": 1e-5,
        "betas": (0.8, 0.99),
        "weight_decay": 1e-4,
        "clip_grad_norm": 5.0,
        "pretrain_generator_epochs": 15,
        "discriminator_only_epochs": 3,
        "discriminator_warmup_epochs": 1,
        "generator_update_interval_epochs": 3,
        "discriminator_update_interval": 1,
        "checkpoint_dir": "checkpoints",
        "history_path": "checkpoints/loss_history.json",
        "checkpoint_interval": 1,
        "device": "cuda",
    }


if __name__ == "__main__":
    test_configs = TrainConfig()
    print(test_configs.model_dump())
