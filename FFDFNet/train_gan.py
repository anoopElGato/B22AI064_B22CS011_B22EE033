import argparse
import json
import math
from pathlib import Path

import torch

from torch.utils.data import DataLoader

from configs.train_configs import TrainConfig
from data.voicebank_demand_dataset import VoiceBankDemandDataset
from eval_metrics import calculate_batch_metrics
from losses import (
    MultiResolutionSTFTLoss,
    build_generator_loss,
    discriminator_least_squares_loss,
    feature_matching_loss,
    generator_least_squares_loss,
    generator_non_saturating_loss,
)
from models.ffdfnet import FFDFNet
from modules.discriminator import FFDFDiscriminator

from tqdm import tqdm
from thop import profile


def parse_args():
    parser = argparse.ArgumentParser(description="Train FFDFNet on VoiceBank-DEMAND.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--generator-lr", type=float, default=None)
    parser.add_argument("--discriminator-lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--resume-epoch", type=int, default=None)
    return parser.parse_args()


def select_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def init_hidden_state(configs: TrainConfig, batch_size: int, device: torch.device):
    groups = configs.inter_frame_gru["parameters"]["groups"]
    hidden_size = configs.inter_frame_gru["parameters"]["hidden_size"]
    num_modules = configs.inter_frame_gru["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)

    return [
        [torch.zeros(1, batch_size * num_bands, hidden_size // groups, device=device) for _ in range(groups)]
        for _ in range(num_modules)
    ]


def waveform_to_generator_inputs(waveform: torch.Tensor, configs: TrainConfig):
    window = torch.hamming_window(configs.n_fft, device=waveform.device)
    complex_spectrum = torch.stft(waveform, n_fft=configs.n_fft, hop_length=configs.hop_length,
                                  window=window, return_complex=True)
    amplitude_spectrum = torch.abs(complex_spectrum)

    complex_spectrum = torch.view_as_real(complex_spectrum)
    complex_spectrum = torch.permute(complex_spectrum, dims=(0, 2, 3, 1)).contiguous()
    amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 2, 1)).unsqueeze(2).contiguous()

    return complex_spectrum, amplitude_spectrum


def generator_outputs_to_waveform(enhanced_complex: torch.Tensor, configs: TrainConfig, target_length: int):
    enhanced_complex = torch.permute(enhanced_complex, dims=(0, 3, 1, 2)).contiguous()
    enhanced_complex = torch.view_as_complex(enhanced_complex)
    window = torch.hamming_window(configs.n_fft, device=enhanced_complex.device)
    return torch.istft(enhanced_complex, n_fft=configs.n_fft, hop_length=configs.hop_length,
                       window=window, length=target_length)


def set_requires_grad(module: torch.nn.Module, requires_grad: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def build_dataloaders(configs: TrainConfig, data_root: str, batch_size: int, num_workers: int):
    dataset_configs = configs.dataset
    valid_batch_size = 1 if dataset_configs["valid_segment_length"] is None else batch_size
    train_dataset = VoiceBankDemandDataset(
        root_dir=data_root,
        noisy_dir_candidates=dataset_configs["train_noisy_dirnames"],
        clean_dir_candidates=dataset_configs["train_clean_dirnames"],
        sample_rate=dataset_configs["sample_rate"],
        segment_length=dataset_configs["segment_length"],
        training=True,
    )
    valid_dataset = VoiceBankDemandDataset(
        root_dir=data_root,
        noisy_dir_candidates=dataset_configs["valid_noisy_dirnames"],
        clean_dir_candidates=dataset_configs["valid_clean_dirnames"],
        sample_rate=dataset_configs["sample_rate"],
        segment_length=dataset_configs["valid_segment_length"],
        training=False,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, valid_loader


def save_checkpoint(checkpoint_dir: Path,
                    epoch: int,
                    generator: FFDFNet,
                    discriminator: FFDFDiscriminator,
                    generator_optimizer: torch.optim.Optimizer,
                    discriminator_optimizer: torch.optim.Optimizer,
                    history):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "history": history,
    }, checkpoint_path)


def save_history(history_path: Path, history):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def resolve_resume_checkpoint(checkpoint_dir: Path, resume_checkpoint: str | None, resume_epoch: int | None):
    if resume_checkpoint is not None:
        return Path(resume_checkpoint)
    if resume_epoch is not None:
        return checkpoint_dir / f"epoch_{resume_epoch:04d}.pt"
    return None


def load_checkpoint(checkpoint_path: Path,
                    generator: FFDFNet,
                    discriminator: FFDFDiscriminator,
                    generator_optimizer: torch.optim.Optimizer,
                    discriminator_optimizer: torch.optim.Optimizer,
                    device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    history = checkpoint.get("history", [])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch, history


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def get_discriminator_lr(epoch: int, configs: TrainConfig):
    pretrain_epochs = configs.training["pretrain_generator_epochs"]
    discriminator_only_epochs = configs.training["discriminator_only_epochs"]
    warmup_epochs = configs.training["discriminator_warmup_epochs"]
    target_lr = configs.training["discriminator_lr"]

    if epoch <= pretrain_epochs:
        return 0.0

    if warmup_epochs <= 0:
        return target_lr

    warmup_progress = min(epoch - pretrain_epochs, warmup_epochs) / warmup_epochs
    return target_lr * warmup_progress


def get_training_phase(epoch: int, configs: TrainConfig):
    pretrain_epochs = configs.training["pretrain_generator_epochs"]
    discriminator_only_epochs = configs.training["discriminator_only_epochs"]

    if epoch <= pretrain_epochs:
        return "generator_pretrain"
    if epoch <= pretrain_epochs + discriminator_only_epochs:
        return "discriminator_bootstrap"
    return "joint_gan"


def should_update_generator_this_epoch(epoch: int, configs: TrainConfig):
    training_phase = get_training_phase(epoch, configs)
    if training_phase == "generator_pretrain":
        return True
    if training_phase == "discriminator_bootstrap":
        return False

    joint_gan_start_epoch = configs.training["pretrain_generator_epochs"] + configs.training["discriminator_only_epochs"] + 1
    generator_update_interval_epochs = max(int(configs.training["generator_update_interval_epochs"]), 1)
    joint_gan_epoch_index = epoch - joint_gan_start_epoch
    return joint_gan_epoch_index % generator_update_interval_epochs == 0


def get_generator_adversarial_loss(fake_logits, configs: TrainConfig):
    objective = configs.adversarial_losses["generator_objective"]
    if objective == "non_saturating":
        return generator_non_saturating_loss(fake_logits)
    if objective == "least_squares":
        return generator_least_squares_loss(fake_logits)
    raise ValueError(f"Unsupported generator adversarial objective: {objective}")


def train_one_epoch(generator,
                    discriminator,
                    train_loader,
                    generator_optimizer,
                    discriminator_optimizer,
                    stft_loss_fn,
                    configs,
                    device,
                    epoch: int):
    generator.train()
    discriminator.train()

    generator_loss_sums = {
        "total": 0.0,
        "stft_total": 0.0,
        "stft_magnitude": 0.0,
        "stft_phase": 0.0,
        "adversarial": 0.0,
        "feature_matching": 0.0,
    }
    discriminator_loss_sum = 0.0
    d_real_sum = 0.0
    d_fake_sum = 0.0
    d_logit_batches = 0

    adversarial_weight = configs.adversarial_losses["generator_weight"]
    feature_weight = configs.adversarial_losses["feature_matching_weight"]
    clip_grad_norm = configs.training["clip_grad_norm"]
    current_discriminator_lr = get_discriminator_lr(epoch, configs)
    use_gan = current_discriminator_lr > 0.0
    training_phase = get_training_phase(epoch, configs)
    update_generator = should_update_generator_this_epoch(epoch, configs)
    set_optimizer_lr(discriminator_optimizer, current_discriminator_lr)
    discriminator_update_interval = max(int(configs.training["discriminator_update_interval"]), 1)

    for batch_idx, batch in enumerate(tqdm(train_loader, total= len(train_loader)), start=1):
        noisy_waveform = batch["noisy"].to(device)
        clean_waveform = batch["clean"].to(device)
        clean_disc = clean_waveform.unsqueeze(1)

        noisy_complex, noisy_amplitude = waveform_to_generator_inputs(noisy_waveform, configs)
        hidden_state = init_hidden_state(configs, noisy_waveform.shape[0], device)
        enhanced_complex, _ = generator(noisy_complex, noisy_amplitude, hidden_state)
        enhanced_waveform = generator_outputs_to_waveform(enhanced_complex, configs, noisy_waveform.shape[-1])
        fake_disc = enhanced_waveform.unsqueeze(1)

        generator_optimizer.zero_grad(set_to_none=True)
        stft_losses = stft_loss_fn(enhanced_waveform, clean_waveform)
        discriminator_loss = enhanced_waveform.new_tensor(0.0)

        should_update_discriminator = use_gan and (batch_idx % discriminator_update_interval == 0)

        if should_update_discriminator:
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                detached_fake_disc = fake_disc.detach()
            real_disc_outputs = discriminator(clean_disc)
            fake_disc_outputs = discriminator(detached_fake_disc)
            discriminator_loss = discriminator_least_squares_loss(real_disc_outputs["logits"], fake_disc_outputs["logits"])
            discriminator_loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad_norm)
            discriminator_optimizer.step()

            d_real_sum += sum(float(logits.detach().mean().cpu()) for logits in real_disc_outputs["logits"]) / len(real_disc_outputs["logits"])
            d_fake_sum += sum(float(logits.detach().mean().cpu()) for logits in fake_disc_outputs["logits"]) / len(fake_disc_outputs["logits"])
            d_logit_batches += 1

        if use_gan:
            set_requires_grad(discriminator, False)
            real_disc_outputs = discriminator(clean_disc)
            fake_disc_outputs = discriminator(fake_disc)
            adversarial_loss = get_generator_adversarial_loss(fake_disc_outputs["logits"], configs)
            feature_loss = feature_matching_loss(real_disc_outputs["feature_maps"], fake_disc_outputs["feature_maps"])
            set_requires_grad(discriminator, True)
        else:
            adversarial_loss = enhanced_waveform.new_tensor(0.0)
            feature_loss = enhanced_waveform.new_tensor(0.0)

        generator_losses = build_generator_loss(stft_losses, adversarial_loss, feature_loss,
                                               adversarial_weight=adversarial_weight if use_gan else 0.0,
                                               feature_weight=feature_weight if use_gan else 0.0)
        if update_generator:
            generator_losses["total"].backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad_norm)
            generator_optimizer.step()

        for key in generator_loss_sums:
            generator_loss_sums[key] += float(generator_losses[key].detach().cpu())
        discriminator_loss_sum += float(discriminator_loss.detach().cpu())

    num_batches = len(train_loader)
    epoch_stats = {f"train_generator_{key}": value / num_batches for key, value in generator_loss_sums.items()}
    epoch_stats["train_discriminator_total"] = discriminator_loss_sum / num_batches
    epoch_stats["train_mean_d_real"] = d_real_sum / d_logit_batches if d_logit_batches > 0 else math.nan
    epoch_stats["train_mean_d_fake"] = d_fake_sum / d_logit_batches if d_logit_batches > 0 else math.nan
    epoch_stats["train_discriminator_lr"] = current_discriminator_lr
    epoch_stats["train_discriminator_update_interval"] = discriminator_update_interval
    epoch_stats["train_generator_update_interval_epochs"] = configs.training["generator_update_interval_epochs"]
    epoch_stats["train_generator_updated"] = float(update_generator)
    epoch_stats["train_training_phase"] = training_phase
    return epoch_stats


@torch.no_grad()
def validate(generator, discriminator, valid_loader, stft_loss_fn, configs, device, epoch: int):
    generator.eval()
    discriminator.eval()

    generator_loss_sums = {
        "total": 0.0,
        "stft_total": 0.0,
        "stft_magnitude": 0.0,
        "stft_phase": 0.0,
        "adversarial": 0.0,
        "feature_matching": 0.0,
    }
    discriminator_loss_sum = 0.0
    pesq_sum = 0.0
    pesq_count = 0
    stoi_sum = 0.0
    stoi_count = 0
    d_real_sum = 0.0
    d_fake_sum = 0.0
    d_logit_batches = 0

    adversarial_weight = configs.adversarial_losses["generator_weight"]
    feature_weight = configs.adversarial_losses["feature_matching_weight"]
    sample_rate = configs.dataset["sample_rate"]
    current_discriminator_lr = get_discriminator_lr(epoch, configs)
    use_gan = current_discriminator_lr > 0.0
    training_phase = get_training_phase(epoch, configs)

    for batch in tqdm(valid_loader, total= len(valid_loader)):
        noisy_waveform = batch["noisy"].to(device)
        clean_waveform = batch["clean"].to(device)
        clean_disc = clean_waveform.unsqueeze(1)

        noisy_complex, noisy_amplitude = waveform_to_generator_inputs(noisy_waveform, configs)
        hidden_state = init_hidden_state(configs, noisy_waveform.shape[0], device)
        enhanced_complex, _ = generator(noisy_complex, noisy_amplitude, hidden_state)
        enhanced_waveform = generator_outputs_to_waveform(enhanced_complex, configs, noisy_waveform.shape[-1])
        fake_disc = enhanced_waveform.unsqueeze(1)

        stft_losses = stft_loss_fn(enhanced_waveform, clean_waveform)
        if use_gan:
            real_disc_outputs = discriminator(clean_disc)
            fake_disc_outputs = discriminator(fake_disc)
            discriminator_loss = discriminator_least_squares_loss(real_disc_outputs["logits"], fake_disc_outputs["logits"])
            adversarial_loss = get_generator_adversarial_loss(fake_disc_outputs["logits"], configs)
            feature_loss = feature_matching_loss(real_disc_outputs["feature_maps"], fake_disc_outputs["feature_maps"])
            generator_losses = build_generator_loss(stft_losses, adversarial_loss, feature_loss,
                                                   adversarial_weight=adversarial_weight,
                                                   feature_weight=feature_weight)
            d_real_sum += sum(float(logits.detach().mean().cpu()) for logits in real_disc_outputs["logits"]) / len(real_disc_outputs["logits"])
            d_fake_sum += sum(float(logits.detach().mean().cpu()) for logits in fake_disc_outputs["logits"]) / len(fake_disc_outputs["logits"])
            d_logit_batches += 1
        else:
            discriminator_loss = enhanced_waveform.new_tensor(0.0)
            adversarial_loss = enhanced_waveform.new_tensor(0.0)
            feature_loss = enhanced_waveform.new_tensor(0.0)
            generator_losses = build_generator_loss(stft_losses, adversarial_loss, feature_loss,
                                                   adversarial_weight=0.0,
                                                   feature_weight=0.0)

        for key in generator_loss_sums:
            generator_loss_sums[key] += float(generator_losses[key].detach().cpu())
        discriminator_loss_sum += float(discriminator_loss.detach().cpu())
        metric_stats = calculate_batch_metrics(clean_waveform, enhanced_waveform, sample_rate)
        pesq_sum += metric_stats["pesq_sum"]
        pesq_count += metric_stats["pesq_count"]
        stoi_sum += metric_stats["stoi_sum"]
        stoi_count += metric_stats["stoi_count"]

    num_batches = len(valid_loader)
    epoch_stats = {f"valid_generator_{key}": value / num_batches for key, value in generator_loss_sums.items()}
    epoch_stats["valid_discriminator_total"] = discriminator_loss_sum / num_batches
    epoch_stats["valid_pesq"] = pesq_sum / pesq_count if pesq_count > 0 else float("nan")
    epoch_stats["valid_stoi"] = stoi_sum / stoi_count if stoi_count > 0 else float("nan")
    epoch_stats["valid_mean_d_real"] = d_real_sum / d_logit_batches if d_logit_batches > 0 else math.nan
    epoch_stats["valid_mean_d_fake"] = d_fake_sum / d_logit_batches if d_logit_batches > 0 else math.nan
    epoch_stats["valid_training_phase"] = training_phase
    return epoch_stats


def main():
    args = parse_args()
    configs = TrainConfig()

    batch_size = args.batch_size or configs.training["batch_size"]
    epochs = args.epochs or configs.training["epochs"]
    generator_lr = args.generator_lr or configs.training["generator_lr"]
    discriminator_lr = args.discriminator_lr or configs.training["discriminator_lr"]
    checkpoint_dir = Path(args.checkpoint_dir or configs.training["checkpoint_dir"])
    history_path = checkpoint_dir / Path(configs.training["history_path"]).name
    num_workers = args.num_workers if args.num_workers is not None else configs.training["num_workers"]
    save_every = args.save_every or configs.training["checkpoint_interval"]
    device = select_device(args.device or configs.training["device"])

    train_loader, valid_loader = build_dataloaders(configs, args.data_root, batch_size, num_workers)

    generator = FFDFNet(configs).to(device)
    discriminator = FFDFDiscriminator(periods=configs.discriminator["periods"],
                                      scales=configs.discriminator["scales"],
                                      use_spectral_norm=configs.discriminator["use_spectral_norm"]).to(device)

    print("\nGenerator parameters:")
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"total: {total_params}\ntrainable: {trainable_params}")

    print("Discriminator parameters:")
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"total: {total_params}\ntrainable: {trainable_params}")

    for batch in train_loader:
        noisy_waveform = batch["noisy"].to(device)

        noisy_complex, noisy_amplitude = waveform_to_generator_inputs(noisy_waveform, configs)
        hidden_state = init_hidden_state(configs, noisy_waveform.shape[0], device)
        macs, params = profile(generator, inputs=(noisy_complex, noisy_amplitude, hidden_state))
        print(f"MACs: {macs/train_loader.batch_size}")
        print(f"params: {params}")
        break
  
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=generator_lr,
        betas=configs.training["betas"],
        weight_decay=configs.training["weight_decay"],
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=configs.training["betas"],
        weight_decay=configs.training["weight_decay"],
    )

    stft_loss_fn = MultiResolutionSTFTLoss(**configs.stft_losses).to(device)

    resume_path = resolve_resume_checkpoint(checkpoint_dir, args.resume_checkpoint, args.resume_epoch)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, history = load_checkpoint(resume_path, generator, discriminator,
                                               generator_optimizer, discriminator_optimizer, device)
        print(f"Resuming training from checkpoint: {resume_path}")
        print(f"Starting at epoch {start_epoch}")
    else:
        start_epoch = 1
        history = []

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch:{epoch}/{epochs}:")
        print("Training:")
        train_stats = train_one_epoch(generator, discriminator, train_loader,
                                      generator_optimizer, discriminator_optimizer,
                                      stft_loss_fn, configs, device, epoch)
        print("\nValidation:")
        valid_stats = validate(generator, discriminator, valid_loader, stft_loss_fn, configs, device, epoch)

        epoch_stats = {"epoch": epoch, **train_stats, **valid_stats}
        history.append(epoch_stats)
        save_history(history_path, history)

        if epoch % save_every == 0:
            save_checkpoint(checkpoint_dir, epoch, generator, discriminator,
                            generator_optimizer, discriminator_optimizer, history)

        print(json.dumps(epoch_stats, indent=2))

if __name__ == "__main__":
    main()