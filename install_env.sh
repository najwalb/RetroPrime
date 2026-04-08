#!/usr/bin/env bash
# One-shot install of the `retroprime` conda env for self-contained inference.
#
# Why Python 3.8 (not the README's 3.6)? syntheseus requires Python >= 3.8.
# torch 1.5.0 has no conda-forge build for Py 3.8, but the official PyTorch
# pip wheels for cu101 *do* have cp38 builds, so we install torch via pip.
#
# Why torchtext 0.6.0 (not 0.3.1)? 0.3.1 has no Py 3.8 wheel. 0.6.0 still
# exposes the legacy Field/Vocab API the vendored OpenNMT-py 0.4.1 uses.
#
# Pass --cpu to install the CPU-only torch build (useful for dev machines
# without a GPU). Default is the CUDA 10.1 build.

set -euo pipefail
ENV_NAME="${RETROPRIME_ENV_NAME:-retroprime}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TORCH_BUILD="${TORCH_BUILD:-cu101}"
if [[ "${1:-}" == "--cpu" ]]; then
    TORCH_BUILD="cpu"
fi

echo "==> Creating conda env: $ENV_NAME (Python 3.8, torch 1.5.0+$TORCH_BUILD)"

conda create -n "$ENV_NAME" -y --override-channels \
    -c conda-forge \
    python=3.8 rdkit=2020.09 pandas tqdm six future numpy networkx pyyaml

# Activate without sourcing the user's full bashrc.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installing torch 1.5.0+$TORCH_BUILD via pip"
pip install --no-cache-dir \
    "torch==1.5.0+${TORCH_BUILD}" \
    "torchvision==0.6.0+${TORCH_BUILD}" \
    -f https://download.pytorch.org/whl/torch_stable.html

echo "==> Installing torchtext 0.6.0 (no-deps to keep torch 1.5)"
pip install --no-cache-dir torchtext==0.6.0 --no-deps

echo "==> Installing syntheseus 0.6.0 (no-deps; layered runtime deps next)"
pip install --no-cache-dir syntheseus==0.6.0 --no-deps
pip install --no-cache-dir \
    omegaconf "hydra-core==1.3.2" more-itertools \
    requests \
    wandb
# `requests` is a transitive dep of torchtext 0.6 used by its
# `torchtext.utils.download_from_url` path; not pulled in by `--no-deps`.

echo "==> Editable installs of the RetroPrime packages"
# Note: README says `dataprocess/packeage` but the real path on disk is
# `data_process/package` — keep the corrected path here.
cd "$REPO_DIR"
pip install --no-cache-dir -e .
cd "$REPO_DIR/retroprime/data_process/package/SmilesEnumerator"
pip install --no-cache-dir -e .
cd "$REPO_DIR/retroprime/transformer_model"
pip install --no-cache-dir -e .

echo
echo "Done. To activate:"
echo "    conda activate $ENV_NAME"
echo
echo "Next: download the pretrained checkpoints from"
echo "    https://drive.google.com/file/d/1-715B8jU0rRC3YaY4p6URQcgjcRG2OlV/view"
echo "and extract them under $REPO_DIR/checkpoints/{USPTO-50K_pos_pred,USPTO-50K_S2R}/"
