# FFDFNet

FFDFNet is a speech enhancement project that combines:
- a full-band encoder/decoder branch
- a sub-band encoder/decoder branch
- frequency transformer blocks around an inter-frame GRU bottleneck
- optional GAN training with multi-period and multi-scale discriminators

This README explains how to:
- install dependencies
- train the model
- resume training
- run distributed training
- evaluate a single sample
- build and evaluate the Svarah+DEMAND accent dataset
- plot grouped results

## 1. Environment

Create and activate an environment, then install dependencies:

```powershell
conda create -n ffdfnet python=3.11 -y
conda activate ffdfnet
pip install -r requirements.txt
```

For GPU-enabled PyTorch, install the correct `torch`/`torchaudio` wheel for your CUDA version first if needed.

## 2. Main Files

- `models/ffdfnet.py`: generator model
- `modules/discriminator.py`: waveform discriminators
- `train_gan.py`: single-device training
- `train_gan_ddp.py`: DDP/multi-GPU training
- `eval_single_audio.py`: enhance one noisy audio file
- `eval_svarah_demand_groups.py`: grouped evaluation across accent and age groups
- `plot_svarah_group_metrics.py`: plots grouped PESQ/STOI results
- `configs/train_configs.py`: all important hyperparameters

## 3. Important Configs To Modify

The main configuration file is:

- [configs/train_configs.py](./configs/train_configs.py)

Most important fields:

### Dataset

Inside `dataset`:
- `sample_rate`
- `segment_length`
- `valid_segment_length`
- `train_noisy_dirnames`
- `train_clean_dirnames`
- `valid_noisy_dirnames`
- `valid_clean_dirnames`

Use:
- `segment_length` for random training crops
- `valid_segment_length = None` to evaluate full utterances

### Training

Inside `training`:
- `batch_size`
- `num_workers`
- `epochs`
- `generator_lr`
- `discriminator_lr`
- `betas`
- `weight_decay`
- `clip_grad_norm`
- `pretrain_generator_epochs`
- `discriminator_only_epochs`
- `discriminator_warmup_epochs`
- `generator_update_interval_epochs`
- `discriminator_update_interval`
- `checkpoint_dir`
- `checkpoint_interval`
- `device`

Recommended to inspect first:
- `pretrain_generator_epochs`
- `discriminator_only_epochs`
- `generator_update_interval_epochs`
- `generator_lr`
- `discriminator_lr`

### Adversarial loss

Inside `adversarial_losses`:
- `generator_objective`
- `generator_weight`
- `feature_matching_weight`

Currently supported generator objectives:
- `"non_saturating"`
- `"least_squares"`

### Discriminator

Inside `discriminator`:
- `periods`
- `scales`
- `use_spectral_norm`

### STFT loss

Inside `stft_losses`:
- `fft_sizes`
- `hop_sizes`
- `win_lengths`
- `magnitude_weight`
- `phase_weight`
- `compression_exponent`

## 4. Train On VoiceBank-DEMAND

Run single-device training:

```powershell
python FFDFNet/train_gan.py `
  --data-root "C:\path\to\VoiceBank-DEMAND" `
  --epochs 50 `
  --batch-size 8 `
  --generator-lr 5e-4 `
  --discriminator-lr 1e-5 `
  --checkpoint-dir "checkpoints"
```

Arguments for `train_gan.py`:
- `--data-root`: required, root directory of the dataset
- `--epochs`: final epoch number
- `--batch-size`
- `--generator-lr`
- `--discriminator-lr`
- `--checkpoint-dir`
- `--num-workers`
- `--device`
- `--save-every`
- `--resume-checkpoint`
- `--resume-epoch`

## 5. Resume Training

Resume from an explicit checkpoint:

```powershell
python FFDFNet/train_gan.py `
  --data-root "C:\path\to\VoiceBank-DEMAND" `
  --epochs 80 `
  --resume-checkpoint "checkpoints/epoch_0030.pt"
```

Resume using an epoch number:

```powershell
python FFDFNet/train_gan.py `
  --data-root "C:\path\to\VoiceBank-DEMAND" `
  --epochs 80 `
  --resume-epoch 30
```

Note:
- `--epochs` means the final epoch to train until
- if resuming from epoch 30 and you want 30 more epochs, pass `--epochs 60`

## 6. Multi-GPU / DDP Training

Run distributed training:

```powershell
torchrun --nproc_per_node=2 FFDFNet/train_gan_ddp.py `
  --data-root "C:\path\to\VoiceBank-DEMAND" `
  --epochs 50 `
  --batch-size 8 `
  --checkpoint-dir "checkpoints_ddp"
```

Arguments for `train_gan_ddp.py`:
- `--data-root`
- `--epochs`
- `--batch-size`
- `--generator-lr`
- `--discriminator-lr`
- `--checkpoint-dir`
- `--num-workers`
- `--save-every`
- `--backend`
- `--resume-checkpoint`
- `--resume-epoch`

Resume DDP training:

```powershell
torchrun --nproc_per_node=2 FFDFNet/train_gan_ddp.py `
  --data-root "C:\path\to\VoiceBank-DEMAND" `
  --epochs 80 `
  --resume-epoch 30
```

## 7. Single Audio Evaluation

Enhance a single noisy/clean pair and save:
- noisy audio
- clean audio
- generated audio

```powershell
python FFDFNet/eval_single_audio.py `
  --checkpoint "checkpoints/epoch_0030.pt" `
  --noisy-audio "path/to/noisy.wav" `
  --clean-audio "path/to/clean.wav" `
  --output-dir "single_eval_outputs" `
  --device cuda
```

Arguments:
- `--checkpoint`
- `--noisy-audio`
- `--clean-audio`
- `--output-dir`
- `--device`

## 8. Build Svarah + DEMAND Dataset

This creates a new dataset using:
- clean audio from `ai4bharat/Svarah`
- noise from a local DEMAND dataset directory

Run:

```powershell
python build_svarah_demand_dataset.py `
  --output-root "C:\path\to\svarah_demand_mixed" `
  --demand-root "C:\path\to\DEMAND" `
  --hf-token "YOUR_HF_TOKEN" `
  --split test `
  --sample-rate 16000 `
  --min-snr-db 0 `
  --max-snr-db 20 `
  --save-noise-audio
```

Arguments:
- `--output-root`: output dataset directory
- `--demand-root`: root folder containing DEMAND wav files
- `--hf-token`: Hugging Face token for Svarah access
- `--split`: Svarah split
- `--sample-rate`
- `--min-snr-db`
- `--max-snr-db`
- `--limit`
- `--seed`
- `--save-noise-audio`

Outputs:
- `clean/`
- `noisy/`
- optional `noise/`
- `manifest.jsonl`

## 9. Evaluate Across Accents and Age Groups

Use the generated `manifest.jsonl` and a trained checkpoint:

```powershell
python FFDFNet/eval_svarah_demand_groups.py `
  --checkpoint "checkpoints/epoch_0030.pt" `
  --manifest "C:\path\to\svarah_demand_mixed\manifest.jsonl" `
  --output-dir "svarah_group_eval" `
  --device cuda
```

Arguments:
- `--checkpoint`
- `--manifest`
- `--output-dir`
- `--device`
- `--limit`

Outputs:
- `summary.json`
- `per_file_results.jsonl`

`summary.json` contains:
- overall metrics
- grouped metrics by `primary_language`
- grouped metrics by `age_group`

## 10. Plot Grouped Accent/Age Results

Create bar plots from `summary.json`:

```powershell
python plot_svarah_group_metrics.py `
  --summary-json "svarah_group_eval/summary.json" `
  --output-dir "svarah_group_plots"
```

Optional:

```powershell
python plot_svarah_group_metrics.py `
  --summary-json "svarah_group_eval/summary.json" `
  --output-dir "svarah_group_plots" `
  --top-k-accents 10 `
  --sort-by mean_pesq
```

Arguments:
- `--summary-json`
- `--output-dir`
- `--top-k-accents`
- `--sort-by`

Generated plots:
- `pesq_by_primary_language.png`
- `stoi_by_primary_language.png`
- `pesq_by_age_group.png`
- `stoi_by_age_group.png`

## 11. Common Notes

- If `pesq` or `pystoi` are not installed, metric evaluation will return `NaN`
- On some systems `torchaudio`/`torchcodec` may fail; the code already falls back to `soundfile` in key places
- Validation uses full utterances when `valid_segment_length = None`
- Generator/discriminator scheduling is controlled from `train_configs.py`

## 12. Suggested Workflow

1. Set dataset paths and training schedule in `train_configs.py`
2. Train with `train_gan.py` or `train_gan_ddp.py`
3. Resume from checkpoints if needed
4. Evaluate single samples with `eval_single_audio.py`
5. Build accented noisy dataset from Svarah + DEMAND
6. Run grouped evaluation with `eval_svarah_demand_groups.py`
7. Plot grouped results with `plot_svarah_group_metrics.py`
