#!/bin/bash
# =============================================================================
# AfriLION TPU Setup Script
# Provisions Cloud TPU VMs using approved TRC quota
# Project: afrilion-49361e (TRC approved)
# Quota: Free 30-day trial from Google TPU Research Cloud
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID="afrilion-49361e"
TPU_NAME="${TPU_NAME:-afrilion-v4-tpu}"
TPU_ZONE="${TPU_ZONE:-us-central2-b}"
TPU_TYPE="${TPU_TYPE:-v4-32}"
RUNTIME_VERSION="${RUNTIME_VERSION:-tpu-ubuntu2204-base}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v4}"

# Approved zones and chip types from TRC email (April 2025)
# v4 chips  -> us-central2-b  (32 spot, 32 on-demand)
# v5e chips -> us-central1-a  (64 spot)
# v5e chips -> europe-west4-b (64 spot)
# v6e chips -> us-east1-d     (64 spot)
# v6e chips -> europe-west4-a (64 spot)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_gcloud() {
  command -v gcloud >/dev/null 2>&1 || die "gcloud SDK not found. Install from https://cloud.google.com/sdk"
  log "gcloud version: $(gcloud version --format='value(Google Cloud SDK)')"
}

set_project() {
  log "Setting project to ${PROJECT_ID}..."
  gcloud config set project "${PROJECT_ID}"
  gcloud config set compute/zone "${TPU_ZONE}"
}

# ---------------------------------------------------------------------------
# TPU Creation
# ---------------------------------------------------------------------------
create_tpu_v4() {
  log "Creating Cloud TPU v4-32 in ${TPU_ZONE}..."
  gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --zone="${TPU_ZONE}" \
    --accelerator-type="v4-32" \
    --version="${RUNTIME_VERSION}" \
    --preemptible \
    --project="${PROJECT_ID}"
  log "TPU v4-32 created: ${TPU_NAME}"
}

create_tpu_v5e() {
  local zone="${1:-us-central1-a}"
  local name="${TPU_NAME}-v5e"
  log "Creating Cloud TPU v5e-64 in ${zone}..."
  gcloud compute tpus tpu-vm create "${name}" \
    --zone="${zone}" \
    --accelerator-type="v5litepod-64" \
    --version="${RUNTIME_VERSION}" \
    --preemptible \
    --project="${PROJECT_ID}"
  log "TPU v5e created: ${name}"
}

create_tpu_v6e() {
  local zone="${1:-us-east1-d}"
  local name="${TPU_NAME}-v6e"
  log "Creating Cloud TPU v6e-64 in ${zone}..."
  gcloud compute tpus tpu-vm create "${name}" \
    --zone="${zone}" \
    --accelerator-type="v6e-64" \
    --version="${RUNTIME_VERSION}" \
    --project="${PROJECT_ID}"
  log "TPU v6e created: ${name}"
}

# ---------------------------------------------------------------------------
# Environment Setup on TPU VM
# ---------------------------------------------------------------------------
setup_tpu_env() {
  log "Installing AfriLION dependencies on TPU VM ${TPU_NAME}..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${TPU_ZONE}" \
    --project="${PROJECT_ID}" \
    --command="
      set -e
      # System updates
      sudo apt-get update -qq && sudo apt-get install -y git wget curl htop tmux

      # Python environment
      pip install --upgrade pip setuptools wheel

      # JAX with TPU support
      pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

      # ML libraries
      pip install transformers==4.40.0 \
                  datasets==2.19.0 \
                  tokenizers==0.19.1 \
                  sentencepiece==0.2.0 \
                  accelerate==0.29.0 \
                  evaluate==0.4.1 \
                  wandb \
                  tqdm \
                  numpy \
                  scipy

      # Clone AfriLION repo
      if [ ! -d ~/afrilion ]; then
        git clone https://github.com/LocaleNLP/afrilion.git ~/afrilion
      else
        cd ~/afrilion && git pull
      fi

      echo 'AfriLION environment setup complete!'
    "
  log "TPU environment setup complete."
}

# ---------------------------------------------------------------------------
# Verify JAX TPU connectivity
# ---------------------------------------------------------------------------
verify_tpu() {
  log "Verifying JAX TPU connectivity on ${TPU_NAME}..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${TPU_ZONE}" \
    --project="${PROJECT_ID}" \
    --command="
      python3 -c \"
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Device count:', jax.device_count())
print('TPU connectivity: OK')
\"
    "
}

# ---------------------------------------------------------------------------
# GCS Bucket Setup (for checkpoints + data)
# ---------------------------------------------------------------------------
setup_gcs_bucket() {
  local bucket="gs://afrilion-tpu-${PROJECT_ID}"
  log "Creating GCS bucket ${bucket}..."
  gsutil mb -p "${PROJECT_ID}" -l US-CENTRAL2 "${bucket}" 2>/dev/null || \
    log "Bucket ${bucket} already exists."

  # Create directory structure
  gsutil -m cp /dev/null "${bucket}/checkpoints/.keep"
  gsutil -m cp /dev/null "${bucket}/data/processed/.keep"
  gsutil -m cp /dev/null "${bucket}/tokenizer/.keep"
  gsutil -m cp /dev/null "${bucket}/logs/.keep"

  log "GCS bucket structure initialized: ${bucket}"
  echo "GCS_BUCKET=${bucket}" >> ~/.afrilion_env
}

# ---------------------------------------------------------------------------
# List all TPU VMs in project
# ---------------------------------------------------------------------------
list_tpus() {
  log "Active TPU VMs in project ${PROJECT_ID}:"
  gcloud compute tpus tpu-vm list \
    --project="${PROJECT_ID}" \
    --format="table(name,zone,acceleratorType,state,health)"
}

# ---------------------------------------------------------------------------
# SSH into TPU
# ---------------------------------------------------------------------------
ssh_tpu() {
  log "SSH-ing into ${TPU_NAME} in ${TPU_ZONE}..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${TPU_ZONE}" \
    --project="${PROJECT_ID}"
}

# ---------------------------------------------------------------------------
# Delete TPU (to avoid charges after 30-day free period)
# ---------------------------------------------------------------------------
delete_tpu() {
  log "WARNING: Deleting TPU ${TPU_NAME}..."
  read -p "Confirm deletion? (yes/no): " confirm
  if [[ "${confirm}" == "yes" ]]; then
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
      --zone="${TPU_ZONE}" \
      --project="${PROJECT_ID}" \
      --quiet
    log "TPU ${TPU_NAME} deleted."
  else
    log "Deletion cancelled."
  fi
}

# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
AfriLION TPU Setup Script

Usage: $0 <command> [options]

Commands:
  create-v4          Create v4-32 TPU in us-central2-b (spot)
  create-v5e [zone]  Create v5e-64 TPU (default: us-central1-a)
  create-v6e [zone]  Create v6e-64 TPU (default: us-east1-d)
  setup-env          Install AfriLION deps on TPU VM
  verify             Verify JAX TPU connectivity
  setup-gcs          Create GCS bucket for checkpoints/data
  list               List all TPU VMs
  ssh                SSH into TPU VM
  delete             Delete TPU VM

Environment Variables:
  TPU_NAME           TPU VM name (default: afrilion-v4-tpu)
  TPU_ZONE           Zone override
  TPU_TYPE           Accelerator type override

Examples:
  # Quick start with v4 TPU (recommended for first run)
  $0 create-v4
  $0 setup-env
  $0 verify

  # Use v5e in Europe zone
  TPU_ZONE=europe-west4-b $0 create-v5e europe-west4-b
EOF
}

main() {
  check_gcloud
  set_project

  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    create-v4)     create_tpu_v4 ;;
    create-v5e)    create_tpu_v5e "${1:-us-central1-a}" ;;
    create-v6e)    create_tpu_v6e "${1:-us-east1-d}" ;;
    setup-env)     setup_tpu_env ;;
    verify)        verify_tpu ;;
    setup-gcs)     setup_gcs_bucket ;;
    list)          list_tpus ;;
    ssh)           ssh_tpu ;;
    delete)        delete_tpu ;;
    help|--help|-h) usage ;;
    *) die "Unknown command: ${cmd}. Run '$0 help' for usage." ;;
  esac
}

main "$@"
