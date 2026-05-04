#!/bin/bash
# AfriLION TPU v4-8 Setup & SFT Training Script
# Run this once after SSH into the TPU VM
# Usage: bash scripts/tpu_setup.sh

set -euo pipefail

PROJECT="afrilion-493616"
ZONE="us-central2-b"
REPO="https://github.com/LocaleNLP/afrilion.git"
WORKDIR="/home/$USER/afrilion"
HF_MODEL="google/gemma-3-1b-it"
OUTPUT_DIR="/home/$USER/afrilion-sft-output"
LOG_FILE="/home/$USER/sft_training.log"

echo "=== AfriLION TPU Setup ==="
echo "Date: $(date)"
echo "User: $USER"
echo "Zone: $ZONE"

# ── 1. System packages ─────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl screen htop

# ── 2. Python env ──────────────────────────────────────────────────────────
echo "[2/6] Setting up Python environment..."
pip install --upgrade pip --quiet

# TPU-optimized torch + JAX
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html --quiet

# Training deps
pip install \
  transformers>=4.40.0 \
  datasets>=2.18.0 \
  accelerate>=0.29.0 \
  peft>=0.10.0 \
  trl>=0.8.6 \
  sentencepiece \
  huggingface_hub \
  wandb \
  bitsandbytes \
  --quiet

# ── 3. Clone repo ──────────────────────────────────────────────────────────
echo "[3/6] Cloning afrilion repo..."
if [ -d "$WORKDIR" ]; then
  echo "  Repo exists, pulling latest..."
  cd "$WORKDIR" && git pull --quiet
else
  git clone --depth 1 "$REPO" "$WORKDIR"
  cd "$WORKDIR"
fi

# ── 4. Hugging Face auth ───────────────────────────────────────────────────
echo "[4/6] Checking HF token..."
if [ -z "${HF_TOKEN:-}" ]; then
  echo "  WARNING: HF_TOKEN not set. Gemma gated model will fail."
  echo "  Set it with: export HF_TOKEN=hf_..."
else
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
  echo "  HF login OK"
fi

# ── 5. Verify TPU ─────────────────────────────────────────────────────────
echo "[5/6] Verifying TPU devices..."
python3 -c "
import torch_xla.core.xla_model as xm
devices = xm.get_xla_supported_devices()
print(f'  TPU devices found: {len(devices)}')
for d in devices:
    print(f'    {d}')
"

# ── 6. Launch SFT training ─────────────────────────────────────────────────
echo "[6/6] Launching SFT training in screen session..."
mkdir -p "$OUTPUT_DIR"

screen -dmS afrilion_train bash -c "
  cd $WORKDIR && \
  python training/sft_train.py \
    --model_name_or_path $HF_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset_name LocaleNLP/afrilion-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --bf16 \
    --tpu_num_cores 8 \
    --report_to wandb \
    2>&1 | tee $LOG_FILE
"

echo ""
echo "=== Setup complete! ==="
echo "Training running in screen session 'afrilion_train'"
echo "  Monitor: screen -r afrilion_train"
echo "  Logs:    tail -f $LOG_FILE"
echo "  TPU:     https://console.cloud.google.com/compute/tpus?project=$PROJECT"
