"""Isolated Qwen3-TTS VoiceDesign worker invoked on demand by gateway.py."""

import argparse
import gc
import importlib.util
import json
import os
import sys

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def set_seed(seed):
    if seed is None:
        return
    try:
        value = int(seed)
    except (TypeError, ValueError):
        return
    if value < 0:
        return
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def resolve_attention_impl():
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def main():
    parser = argparse.ArgumentParser(description="Generate a reference voice sample with Qwen3-TTS VoiceDesign")
    parser.add_argument("--request", required=True, help="Path to JSON file containing instruct, text, language, seed")
    parser.add_argument("--model-dir", required=True, help="Local directory or Hugging Face repo id")
    parser.add_argument("--output", required=True, help="Destination WAV file path")
    args = parser.parse_args()

    with open(args.request, "r", encoding="utf-8") as fh:
        req = json.load(fh)

    instruct = str(req.get("instruct") or "").strip()
    text = str(req.get("text") or "").strip()
    language = str(req.get("language") or "English").strip() or "English"
    if not instruct or not text:
        print("Both instruct and text are required.", file=sys.stderr)
        return 2

    set_seed(req.get("seed"))

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    attn_impl = resolve_attention_impl()

    try:
        model = Qwen3TTSModel.from_pretrained(
            args.model_dir,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
    except Exception:
        # Fall back to SDPA if flash_attn binary mismatches the container CUDA/PyTorch ABI.
        if attn_impl != "sdpa":
            model = Qwen3TTSModel.from_pretrained(
                args.model_dir,
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa",
            )
        else:
            raise

    wavs, sr = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wavs[0], sr)

    del model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
