"""Unified SaladCloud FastAPI gateway for Fish Audio S2-Pro + Qwen3-TTS VoiceDesign."""

import asyncio
import base64
import contextlib
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import wave
from typing import Any, List, Optional

import httpx
import ormsgpack
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tts-gateway")

PORT = int(os.getenv("PORT", "8000"))
FISH_HOST = "127.0.0.1"
FISH_PORT = int(os.getenv("FISH_PORT", "8081"))
FISH_BASE_URL = f"http://{FISH_HOST}:{FISH_PORT}"

FISH_DIR = os.getenv("FISH_DIR", "/app/fish-speech")
FISH_PYTHON = os.getenv("FISH_PYTHON", "/opt/venv-fish/bin/python")
QWEN_PYTHON = os.getenv("QWEN_PYTHON", "/opt/venv-qwen/bin/python")
QWEN_WORKER = os.getenv("QWEN_WORKER", "/app/qwen_worker.py")

FISH_CKPT_DIR = os.getenv("FISH_CKPT_DIR", "/app/models/s2-pro")
QWEN_CKPT_DIR = os.getenv("QWEN_CKPT_DIR", "/app/models/qwen3-tts-voicedesign")
FISH_DECODER_CONFIG = os.getenv("FISH_DECODER_CONFIG", "modded_dac_vq")
FISH_COMPILE = os.getenv("FISH_COMPILE", "0") == "1"
FISH_HALF = os.getenv("FISH_HALF", "0") == "1"

# auto: run Qwen alongside Fish if free VRAM >= QWEN_MIN_FREE_MIB, otherwise stop Fish,
# run Qwen, and automatically restart Fish. always: always stop Fish during VoiceDesign.
VRAM_SWAP_MODE = os.getenv("VRAM_SWAP_MODE", "auto").strip().lower()
QWEN_MIN_FREE_MIB = int(os.getenv("QWEN_MIN_FREE_MIB", "6500"))
TTS_API_KEY = os.getenv("TTS_API_KEY", "").strip()

FISH_MODEL_ID = "fishaudio/s2-pro"
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
}


class ReferenceItem(BaseModel):
    audio_base64: Optional[str] = None
    audio: Optional[str] = None
    text: str = ""


class SpeechRequest(BaseModel):
    model: Optional[str] = FISH_MODEL_ID
    input: Optional[str] = None
    text: Optional[str] = None
    voice: Optional[str] = None
    reference_id: Optional[str] = None
    references: List[ReferenceItem] = Field(default_factory=list)
    response_format: Optional[str] = None
    format: Optional[str] = None
    speed: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    chunk_length: int = 200
    max_new_tokens: int = 1024
    seed: Optional[int] = None
    normalize: bool = True
    streaming: bool = False
    use_memory_cache: str = "off"


class VoiceDesignRequest(BaseModel):
    instruct: str
    text: str
    language: str = "English"
    format: str = "mp3"
    seed: Optional[int] = None


def check_auth(request: Request) -> None:
    if not TTS_API_KEY:
        return
    auth = request.headers.get("authorization", "")
    prefix = "bearer "
    token = auth[len(prefix):].strip() if auth.lower().startswith(prefix) else ""
    if not token or not secrets.compare_digest(token, TTS_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer API key.")


def decode_base64_audio(raw: str) -> bytes:
    value = str(raw or "").strip()
    if not value:
        return b""
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def normalize_format(fmt: Optional[str], default: str = "mp3") -> str:
    value = str(fmt or default).strip().lower()
    return value if value in MIME_TYPES else default


def query_free_vram_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if lines:
            return int(float(lines[0]))
    except Exception:
        pass
    return None


async def transcode_audio(raw_bytes: bytes, input_fmt: str, output_fmt: str) -> bytes:
    if input_fmt == output_fmt:
        return raw_bytes
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, f"in.{input_fmt}")
        dst = os.path.join(tmp, f"out.{output_fmt}")
        with open(src, "wb") as fh:
            fh.write(raw_bytes)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src]
        if output_fmt == "mp3":
            cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        elif output_fmt == "opus":
            cmd += ["-c:a", "libopus", "-b:a", "64k"]
        cmd.append(dst)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed converting {input_fmt} -> {output_fmt}: {err.decode(errors='ignore')}")
        with open(dst, "rb") as fh:
            return fh.read()


def wav_duration_sec(wav_path: str) -> float:
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0


class ModelManager:
    """Owns the Fish Speech subprocess lifecycle and coordinates RTX 3090 VRAM with Qwen3-TTS."""

    def __init__(self) -> None:
        self.gpu_lock = asyncio.Lock()
        self.fish_proc: Optional[asyncio.subprocess.Process] = None
        self.fish_ready = False
        self.last_error: str = ""
        self.respawn_task: Optional[asyncio.Task] = None

    def _fish_command(self) -> List[str]:
        codec_path = os.path.join(FISH_CKPT_DIR, "codec.pth")
        if not os.path.exists(codec_path) and os.path.isdir(FISH_CKPT_DIR):
            for name in sorted(os.listdir(FISH_CKPT_DIR)):
                if name.endswith(".pth"):
                    codec_path = os.path.join(FISH_CKPT_DIR, name)
                    break
        cmd = [
            FISH_PYTHON,
            os.path.join(FISH_DIR, "tools", "api_server.py"),
            "--listen",
            f"{FISH_HOST}:{FISH_PORT}",
            "--llama-checkpoint-path",
            FISH_CKPT_DIR,
            "--decoder-checkpoint-path",
            codec_path,
            "--decoder-config-name",
            FISH_DECODER_CONFIG,
        ]
        if FISH_HALF:
            cmd.append("--half")
        if FISH_COMPILE:
            cmd.append("--compile")
        return cmd

    async def probe_fish_health(self) -> bool:
        if not self.fish_proc or self.fish_proc.returncode is not None:
            self.fish_ready = False
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{FISH_BASE_URL}/v1/health")
                ok = resp.status_code == 200
                self.fish_ready = ok
                return ok
        except Exception:
            self.fish_ready = False
            return False

    async def stop_fish(self) -> None:
        if not self.fish_proc or self.fish_proc.returncode is not None:
            self.fish_proc = None
            self.fish_ready = False
            return
        log.info("Stopping Fish Speech server to free GPU VRAM...")
        self.fish_ready = False
        try:
            self.fish_proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(self.fish_proc.wait(), timeout=15)
        except Exception:
            with contextlib.suppress(Exception):
                self.fish_proc.kill()
            with contextlib.suppress(Exception):
                await self.fish_proc.wait()
        self.fish_proc = None
        # Give CUDA a brief moment to reclaim driver allocations.
        await asyncio.sleep(1.0)

    async def start_fish_and_wait(self, timeout_sec: int = 180) -> None:
        if await self.probe_fish_health():
            return
        if not self.fish_proc or self.fish_proc.returncode is not None:
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{FISH_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
            cmd = self._fish_command()
            log.info("Starting Fish Speech S2-Pro server: %s", " ".join(cmd))
            self.fish_proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=FISH_DIR,
                env=env,
            )
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.fish_proc.returncode is not None:
                self.last_error = f"Fish Speech exited unexpectedly with code {self.fish_proc.returncode}."
                raise RuntimeError(self.last_error)
            if await self.probe_fish_health():
                self.last_error = ""
                log.info("Fish Speech S2-Pro is ready on %s", FISH_BASE_URL)
                return
            await asyncio.sleep(1.0)
        self.last_error = "Timed out waiting for Fish Speech S2-Pro health check."
        raise TimeoutError(self.last_error)

    def schedule_fish_respawn(self) -> None:
        if self.respawn_task and not self.respawn_task.done():
            return

        async def _respawn() -> None:
            async with self.gpu_lock:
                try:
                    await self.start_fish_and_wait()
                except Exception as exc:
                    log.error("Background Fish respawn failed: %s", exc)

        self.respawn_task = asyncio.create_task(_respawn())

    async def run_qwen_design(self, req: VoiceDesignRequest) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            req_path = os.path.join(tmp, "request.json")
            out_wav = os.path.join(tmp, "design.wav")
            with open(req_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "instruct": req.instruct,
                        "text": req.text,
                        "language": req.language or "English",
                        "seed": req.seed,
                    },
                    fh,
                )

            async def _exec_worker() -> tuple[int, str, str]:
                proc = await asyncio.create_subprocess_exec(
                    QWEN_PYTHON,
                    QWEN_WORKER,
                    "--request",
                    req_path,
                    "--model-dir",
                    QWEN_CKPT_DIR,
                    "--output",
                    out_wav,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out, err = await proc.communicate()
                return proc.returncode or 0, out.decode(errors="ignore"), err.decode(errors="ignore")

            free_mib = query_free_vram_mib()
            stopped_fish = False
            if VRAM_SWAP_MODE == "always" or free_mib is None or free_mib < QWEN_MIN_FREE_MIB:
                await self.stop_fish()
                stopped_fish = True

            code, out, err = await _exec_worker()
            if code != 0 and not stopped_fish:
                # Retry cleanly after offloading Fish if coexisting peaked VRAM.
                log.warning("Qwen co-residency exited with code %s; unloading Fish and retrying.", code)
                await self.stop_fish()
                stopped_fish = True
                code, out, err = await _exec_worker()

            if code != 0:
                if stopped_fish:
                    self.schedule_fish_respawn()
                raise RuntimeError(err.strip() or out.strip() or f"Qwen3-TTS worker exited with code {code}")

            duration = wav_duration_sec(out_wav)
            with open(out_wav, "rb") as fh:
                raw_wav = fh.read()

            target_fmt = "wav" if normalize_format(req.format, "mp3") == "wav" else "mp3"
            audio_bytes = raw_wav if target_fmt == "wav" else await transcode_audio(raw_wav, "wav", "mp3")

            if stopped_fish:
                self.schedule_fish_respawn()

            return {
                "audio_bytes": audio_bytes,
                "format": target_fmt,
                "mime_type": MIME_TYPES[target_fmt],
                "duration_sec": duration,
            }


manager = ModelManager()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    manager.schedule_fish_respawn()
    try:
        yield
    finally:
        await manager.stop_fish()


app = FastAPI(title="SaladCloud Fish S2-Pro + Qwen3-TTS Gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/v1/health")
async def health() -> Response:
    fish_ok = await manager.probe_fish_health()
    payload = {
        "status": "ok" if fish_ok else "loading",
        "fish_ready": fish_ok,
        "models": [FISH_MODEL_ID, QWEN_MODEL_ID],
        "free_vram_mib": query_free_vram_mib(),
        **({"error": manager.last_error} if manager.last_error else {}),
    }
    return Response(
        content=json.dumps(payload),
        status_code=200 if fish_ok else 503,
        media_type="application/json",
    )


@app.get("/models")
@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    check_auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": FISH_MODEL_ID,
                "object": "model",
                "owned_by": "fishaudio",
                "capabilities": ["tts", "zero-shot-cloning", "inline-tags", "streaming"],
            },
            {
                "id": QWEN_MODEL_ID,
                "object": "model",
                "owned_by": "qwen",
                "capabilities": ["voice-design"],
            },
        ],
    }


def build_fish_payload(req: SpeechRequest, fish_format: str) -> dict[str, Any]:
    text = str(req.input if req.input is not None else (req.text or "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required.")

    refs = []
    for item in req.references:
        encoded = item.audio_base64 or item.audio or ""
        audio_bytes = decode_base64_audio(encoded)
        if audio_bytes:
            refs.append({"audio": audio_bytes, "text": str(item.text or "")})

    # Treat `voice` as `reference_id` only when caller didn't send inline references and
    # didn't send a dummy placeholder like "default".
    ref_id = (req.reference_id or "").strip() or None
    if not ref_id and not refs:
        candidate = (req.voice or "").strip()
        if candidate and candidate.lower() not in {"default", "alloy", "none"}:
            ref_id = candidate

    payload: dict[str, Any] = {
        "text": text,
        "references": refs,
        "format": fish_format,
        "chunk_length": max(100, min(300, int(req.chunk_length))),
        "max_new_tokens": max(128, min(4096, int(req.max_new_tokens))),
        "top_p": max(0.05, min(1.0, float(req.top_p))),
        "repetition_penalty": max(0.9, min(2.0, float(req.repetition_penalty))),
        "temperature": max(0.05, min(1.5, float(req.temperature))),
        "streaming": bool(req.streaming),
        "use_memory_cache": "on" if str(req.use_memory_cache).lower() == "on" else "off",
        "normalize": bool(req.normalize),
    }
    if ref_id:
        payload["reference_id"] = ref_id
    if req.seed is not None and int(req.seed) >= 0:
        payload["seed"] = int(req.seed)
    return payload


@app.post("/audio/speech")
@app.post("/tts")
@app.post("/v1/audio/speech")
@app.post("/v1/tts")
async def synthesize_speech(req: SpeechRequest, request: Request) -> Response:
    check_auth(request)
    requested_fmt = normalize_format(req.format or req.response_format, "mp3")
    # Fish native server outputs wav, mp3, or pcm. Transcode opus from wav if requested.
    fish_fmt = "wav" if requested_fmt == "opus" else requested_fmt
    payload = build_fish_payload(req, fish_fmt)
    packed = ormsgpack.packb(payload)

    if payload["streaming"] and requested_fmt != "opus":
        await manager.gpu_lock.acquire()
        client = httpx.AsyncClient(timeout=120.0)
        upstream = None
        try:
            await manager.start_fish_and_wait()
            req_out = client.build_request(
                "POST",
                f"{FISH_BASE_URL}/v1/tts",
                content=packed,
                headers={"Content-Type": "application/msgpack"},
            )
            upstream = await client.send(req_out, stream=True)
            if upstream.status_code != 200:
                body = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                manager.gpu_lock.release()
                raise HTTPException(
                    status_code=upstream.status_code,
                    detail=body.decode(errors="ignore")[:400],
                )
        except HTTPException:
            raise
        except Exception as exc:
            if upstream is not None:
                await upstream.aclose()
            await client.aclose()
            manager.gpu_lock.release()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def stream_audio():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                with contextlib.suppress(Exception):
                    await upstream.aclose()
                with contextlib.suppress(Exception):
                    await client.aclose()
                manager.gpu_lock.release()

        return StreamingResponse(stream_audio(), media_type=MIME_TYPES[requested_fmt])

    async with manager.gpu_lock:
        try:
            await manager.start_fish_and_wait()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.post(
                f"{FISH_BASE_URL}/v1/tts",
                content=packed,
                headers={"Content-Type": "application/msgpack"},
            )
            if upstream.status_code != 200:
                raise HTTPException(
                    status_code=upstream.status_code,
                    detail=upstream.text[:400] or "Fish Speech synthesis failed.",
                )
            audio = upstream.content

    if requested_fmt == "opus":
        audio = await transcode_audio(audio, "wav", "opus")
    return Response(content=audio, media_type=MIME_TYPES[requested_fmt])


@app.post("/voices/design")
@app.post("/v1/voices/design")
async def design_voice(req: VoiceDesignRequest, request: Request) -> dict[str, Any]:
    check_auth(request)
    if not req.instruct.strip() or not req.text.strip():
        raise HTTPException(status_code=400, detail="Both instruct and text are required.")

    async with manager.gpu_lock:
        try:
            result = await manager.run_qwen_design(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VoiceDesign failed: {exc}") from exc

    return {
        "model": QWEN_MODEL_ID,
        "format": result["format"],
        "mimeType": result["mime_type"],
        "durationSec": result["duration_sec"],
        "transcript": req.text.strip(),
        "instruct": req.instruct.strip(),
        "audioBase64": base64.b64encode(result["audio_bytes"]).decode("ascii"),
    }


def detect_bind_host() -> str:
    env_host = os.getenv("HOST", "").strip()
    if env_host:
        return env_host
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.close()
        return "::"
    except OSError:
        return "0.0.0.0"


if __name__ == "__main__":
    uvicorn.run(app, host=detect_bind_host(), port=PORT, log_level="info")
