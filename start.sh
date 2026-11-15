#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
APP_DIR="${APP_DIR:-/app}"
FISH_DIR="${FISH_DIR:-$APP_DIR/fish-speech}"
MODELS_DIR="${MODELS_DIR:-$APP_DIR/models}"
export HF_HOME="${HF_HOME:-$APP_DIR/cache/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

FISH_CKPT_DIR="${FISH_CKPT_DIR:-$MODELS_DIR/s2-pro}"
QWEN_CKPT_DIR="${QWEN_CKPT_DIR:-$MODELS_DIR/qwen3-tts-voicedesign}"
FISH_REPO_ID="${FISH_REPO_ID:-fishaudio/s2-pro}"
QWEN_REPO_ID="${QWEN_REPO_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"

mkdir -p "$APP_DIR" "$MODELS_DIR" "$HF_HOME"

echo "[1/5] Installing system packages..."
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v sox >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends git ffmpeg libsndfile1 sox portaudio19-dev build-essential curl ca-certificates
    rm -rf /var/lib/apt/lists/*
fi

echo "[2/5] Preparing Gateway virtualenv (/opt/venv-gateway)..."
if [ ! -d /opt/venv-gateway ]; then
    python3 -m venv --system-site-packages /opt/venv-gateway
    /opt/venv-gateway/bin/pip install --no-cache-dir -U pip
    /opt/venv-gateway/bin/pip install --no-cache-dir \
        fastapi "uvicorn[standard]" httpx ormsgpack pydantic huggingface_hub hf_transfer
fi

echo "[3/5] Installing Fish Speech S2-Pro runtime (/opt/venv-fish)..."
if [ ! -d "$FISH_DIR/.git" ]; then
    git clone --depth 1 https://github.com/fishaudio/fish-speech.git "$FISH_DIR"
fi
if [ ! -d /opt/venv-fish ]; then
    python3 -m venv --system-site-packages /opt/venv-fish
    /opt/venv-fish/bin/pip install --no-cache-dir -U pip setuptools wheel
    /opt/venv-fish/bin/pip install --no-cache-dir --no-build-isolation -e "$FISH_DIR"
    /opt/venv-fish/bin/pip install --no-cache-dir ormsgpack uvicorn fastapi httpx
fi

echo "[4/5] Installing Qwen3-TTS VoiceDesign runtime (/opt/venv-qwen)..."
if [ ! -d /opt/venv-qwen ]; then
    python3 -m venv --system-site-packages /opt/venv-qwen
    /opt/venv-qwen/bin/pip install --no-cache-dir -U pip
    /opt/venv-qwen/bin/pip install --no-cache-dir \
        qwen-tts soundfile "transformers==4.57.3" accelerate
fi

echo "[5/5] Downloading models at startup (no weights baked into Docker image)..."
/opt/venv-gateway/bin/python - <<PY
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN") or None
fish_dir = "$FISH_CKPT_DIR"
qwen_dir = "$QWEN_CKPT_DIR"

if not os.path.exists(os.path.join(fish_dir, "codec.pth")):
    print("Downloading $FISH_REPO_ID ->", fish_dir, flush=True)
    snapshot_download(repo_id="$FISH_REPO_ID", local_dir=fish_dir, token=token)
else:
    print("Fish S2-Pro checkpoint already present:", fish_dir, flush=True)

if not os.path.exists(os.path.join(qwen_dir, "config.json")):
    print("Downloading $QWEN_REPO_ID ->", qwen_dir, flush=True)
    snapshot_download(repo_id="$QWEN_REPO_ID", local_dir=qwen_dir, token=token)
else:
    print("Qwen3-TTS VoiceDesign checkpoint already present:", qwen_dir, flush=True)
PY

echo "Starting unified gateway on port ${PORT:-8000}..."
export FISH_DIR FISH_CKPT_DIR QWEN_CKPT_DIR
export FISH_PYTHON=/opt/venv-fish/bin/python
export QWEN_PYTHON=/opt/venv-qwen/bin/python
export QWEN_WORKER="$APP_DIR/qwen_worker.py"
exec /opt/venv-gateway/bin/python "$APP_DIR/gateway.py"
