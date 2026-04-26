# TPU Infrastructure Setup Guide

This guide details the steps to provision and configure the TPU Research Cloud (TRC) resources for the AfriLION project.

## 1. Prerequisites
- Active Google Cloud Project with billing enabled.
- Approved TPU Research Cloud application.
- `gcloud` CLI installed and authenticated.

## 2. Provisioning TPU VMs
Use the provided script in the `training` directory to create TPU instances.

```bash
bash training/tpu_setup.sh
```

## 3. Environment Setup
The standard environment includes:
- JAX/Flax for distributed training.
- PyTorch (XLA) for compatibility.
- Hugging Face Transformers.

## 4. Monitoring & Logging
- **Weights & Biases:** Integrated into the training scripts.
- **Google Cloud Monitoring:** For hardware-level metrics.
