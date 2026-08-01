from __future__ import annotations

import io
import logging
import os
import pickle
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import stable_whisper
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("shorts_inference")

TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "French")
VOICE_PROMPT_PATH = Path(os.getenv("VOICE_PROMPT_PATH", "/workspace/voice_clone_prompt.pkl"))

tts_device: str | None = None
whisper_device: str | None = None
tts_dtype: torch.dtype | None = None
tts_model: Any = None
whisper_model: Any = None
voice_clone_prompt: Any = None
models_ready = False
startup_error: str | None = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = None


def detect_devices() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda:0", "cuda"

    raise RuntimeError("CUDA is required. Run this server on RunPod or another NVIDIA GPU host.")


def load_tts_model() -> Qwen3TTSModel:
    try:
        return Qwen3TTSModel.from_pretrained(
            TTS_MODEL_NAME,
            device_map=tts_device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        logger.exception("flash_attention_2 failed, retrying with default attention")
        return Qwen3TTSModel.from_pretrained(
            TTS_MODEL_NAME,
            device_map=tts_device,
            dtype=torch.bfloat16,
        )


def load_whisper_model() -> Any:
    return stable_whisper.load_faster_whisper(WHISPER_MODEL_SIZE, device=whisper_device)


def load_saved_voice_prompt() -> Any:
    if not VOICE_PROMPT_PATH.exists():
        return None

    with VOICE_PROMPT_PATH.open("rb") as file_handle:
        return pickle.load(file_handle)


def persist_voice_prompt(prompt: Any) -> None:
    VOICE_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VOICE_PROMPT_PATH.open("wb") as file_handle:
        pickle.dump(prompt, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def to_mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    if waveform.ndim == 1:
        mono = waveform
    elif waveform.shape[0] == 1:
        mono = waveform.squeeze(0)
    else:
        mono = waveform.mean(dim=0)

    return mono.detach().cpu().to(torch.float32).numpy()


def wav_bytes_from_audio(audio: Any, sample_rate: int) -> bytes:
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().to(torch.float32).numpy()
    else:
        audio = np.asarray(audio, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def initialize_runtime() -> None:
    global tts_device, whisper_device, tts_dtype, tts_model, whisper_model, voice_clone_prompt, models_ready, startup_error

    try:
        tts_device, whisper_device = detect_devices()
        tts_dtype = torch.bfloat16

        logger.info(
            "Selected devices: tts_device=%s, whisper_device=%s, tts_dtype=%s",
            tts_device,
            whisper_device,
            tts_dtype,
        )

        tts_model = load_tts_model()
        whisper_model = load_whisper_model()
        voice_clone_prompt = load_saved_voice_prompt()

        if voice_clone_prompt is not None:
            logger.info("Loaded cached voice clone prompt from %s", VOICE_PROMPT_PATH)

        startup_error = None
        models_ready = True
    except Exception as exc:
        startup_error = str(exc)
        models_ready = False
        logger.exception("Runtime initialization failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=initialize_runtime, daemon=True).start()
    yield


app = FastAPI(title="YouTube Shorts Inference Server", lifespan=lifespan)


def require_models_ready() -> None:
    if models_ready:
        return

    detail = "Models are still loading. Try again in a moment."
    if startup_error:
        detail = f"Models failed to load: {startup_error}"

    raise HTTPException(status_code=503, detail=detail)


@app.post("/register-voice")
async def register_voice(
    reference_audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
) -> dict[str, str]:
    global voice_clone_prompt

    require_models_ready()

    suffix = Path(reference_audio.filename or "reference_audio.wav").suffix or ".wav"
    temp_fd, temp_input_name = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)
    temp_input_path = Path(temp_input_name)

    try:
        with temp_input_path.open("wb") as file_handle:
            file_handle.write(await reference_audio.read())

        waveform, sample_rate = torchaudio.load(str(temp_input_path))
        audio = to_mono_numpy(waveform)

        prompt_kwargs: dict[str, Any] = {
            "ref_audio": (audio, sample_rate),
        }

        cleaned_text = reference_text.strip() if reference_text else ""
        if cleaned_text:
            prompt_kwargs["ref_text"] = cleaned_text
        else:
            prompt_kwargs["x_vector_only_mode"] = True

        voice_clone_prompt = tts_model.create_voice_clone_prompt(**prompt_kwargs)
        persist_voice_prompt(voice_clone_prompt)

        return {"status": "ok", "message": f"Voice registered and saved to {VOICE_PROMPT_PATH}"}
    except Exception as exc:
        logger.exception("Failed to register voice")
        raise HTTPException(status_code=400, detail=f"Voice registration failed: {exc}") from exc
    finally:
        try:
            temp_input_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Could not remove temporary voice upload file: %s", temp_input_path)


@app.post("/tts")
async def tts(payload: TTSRequest) -> Response:
    require_models_ready()

    if voice_clone_prompt is None:
        raise HTTPException(
            status_code=400,
            detail="No voice registered yet. Call /register-voice first.",
        )

    language = payload.language or TTS_LANGUAGE

    try:
        wavs, sample_rate = tts_model.generate_voice_clone(
            text=payload.text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        wav_bytes = wav_bytes_from_audio(audio, sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc


@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe(audio: UploadFile = File(...)) -> PlainTextResponse:
    require_models_ready()

    temp_audio_fd, temp_audio_name = tempfile.mkstemp(suffix=Path(audio.filename or "audio.wav").suffix or ".wav")
    temp_srt_fd, temp_srt_name = tempfile.mkstemp(suffix=".srt")
    os.close(temp_audio_fd)
    os.close(temp_srt_fd)
    temp_audio_path = Path(temp_audio_name)
    temp_srt_path = Path(temp_srt_name)

    try:
        with temp_audio_path.open("wb") as file_handle:
            file_handle.write(await audio.read())

        result = whisper_model.transcribe(str(temp_audio_path))
        result.to_srt_vtt(str(temp_srt_path))
        subtitle_text = temp_srt_path.read_text(encoding="utf-8")
        return PlainTextResponse(content=subtitle_text)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        for temp_path in (temp_audio_path, temp_srt_path):
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temporary file: %s", temp_path)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if models_ready else "loading" if startup_error is None else "error",
        "tts_loaded": tts_model is not None,
        "whisper_loaded": whisper_model is not None,
        "voice_registered": voice_clone_prompt is not None,
        "startup_error": startup_error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))