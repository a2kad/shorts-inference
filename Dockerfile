FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    WHISPER_MODEL_SIZE=medium \
    TTS_LANGUAGE=French \
    VOICE_PROMPT_PATH=/workspace/voice_clone_prompt.pkl

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY server.py /workspace/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]