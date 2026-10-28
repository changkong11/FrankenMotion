#!/usr/bin/env bash
set -euo pipefail
SETUP="${FM_SETUP_DIR:-/data_disk/wulw/FM_setup}"
ENV="${FM_ENV:-/data_disk/wulw/envs/frankenmotion}"
mkdir -p "$SETUP" /data_disk/wulw/envs
export TMPDIR="$SETUP/tmp"
export PIP_CACHE_DIR="$SETUP/pip_cache"
export HF_HOME="$SETUP/huggingface"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
if [[ ! -x "$ENV/bin/python" ]]; then
  curl -fL --retry 3 https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh -o "$SETUP/miniconda.sh"
  bash "$SETUP/miniconda.sh" -b -p "$ENV"
fi
"$ENV/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
"$ENV/bin/python" -m pip install --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  'setuptools<81' numpy==1.24.1 scipy==1.11.2 pandas==1.5.3 scikit-learn==1.3.1 \
  pytorch-lightning==2.0.9 torchmetrics==1.1.2 hydra-core==1.3.2 hydra-colorlog==1.2.0 \
  diffusers==0.33.1 transformers==4.57.6 rotary-embedding-torch==0.8.6 \
  huggingface-hub==0.36.0 einops==0.8.1 orjson==3.9.7 matplotlib==3.7.3 \
  tqdm PyYAML ftfy regex clip-anytorch smplx roma==1.4.1 pillow imageio
"$ENV/bin/python" -c 'import torch,hydra,pytorch_lightning,diffusers; print("ENV_READY",torch.__version__,torch.cuda.is_available(),flush=True)'
