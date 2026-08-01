# YouTube Shorts Inference Server

FastAPI-сервер для пайплайна `скрипт -> озвучка -> субтитры`.

Он рассчитан на RunPod / другой NVIDIA CUDA-хост и не содержит локальной ветки запуска.

## Что умеет

- `POST /register-voice` - сохраняет voice clone prompt в памяти и на диск
- `POST /tts` - генерирует WAV по тексту и зарегистрированному голосу
- `POST /transcribe` - делает субтитры в SRT/VTT-совместимом тексте
- `GET /health` - показывает состояние моделей и регистрации голоса

## Структура

- `server.py` - FastAPI-приложение
- `requirements.txt` - Python-зависимости
- `Dockerfile` - контейнер для RunPod / любого CUDA-хоста
- `docker-compose.yml` - запуск одной командой через Docker Compose
- `Makefile` - `make up`, `make down`, `make health`

## Быстрый запуск одной командой

```bash
make up
```

После старта проверка:

```bash
make health
```

## API-контракт

### `POST /register-voice`

Multipart form-data:

- `reference_audio` - `mp3` или `wav`
- `reference_text` - опционально, строка-транскрипт

Пример:

```bash
curl -X POST http://127.0.0.1:8000/register-voice \
  -F "reference_audio=@./samples/reference.wav" \
  -F "reference_text=Hello, this is my reference voice." 
```

### `POST /tts`

JSON:

```json
{
  "text": "Write a short hook for YouTube Shorts.",
  "language": "French"
}
```

Пример:

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Write a short hook for YouTube Shorts.","language":"French"}' \
  --output out.wav
```

### `POST /transcribe`

Multipart form-data:

- `audio` - `wav`

### `GET /health`

Пример:

```bash
curl http://127.0.0.1:8000/health
```

Ответ:

```json
{
  "status": "ok",
  "tts_loaded": true,
  "whisper_loaded": true,
  "voice_registered": false
}
```

## RunPod: запуск с нуля

### Шаг 1. Создайте аккаунт и Pod

1. Зарегистрируйтесь в RunPod.
2. Создайте `Pod`.
3. Выберите GPU-класс вроде `RTX A5000`, `RTX 4090` или аналогичный.
4. Возьмите шаблон на базе PyTorch, чтобы CUDA уже была доступна.
5. Подключите `Volume Disk` и смонтируйте его в `/workspace`.
6. Откройте `HTTP Port 8000`.

### Шаг 2. Подключитесь к веб-терминалу

Обычно достаточно стандартного shell в веб-интерфейсе RunPod.

### Шаг 3. Установите проект

```bash
cd /workspace
git clone <your-repo-url> my-repo
cd my-repo/shorts-inference
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
docker compose up --build
```

Если модель очень долго скачивается, запускайте сервер в `tmux` или через `nohup`.

Через `tmux`:

```bash
tmux new -s shorts
uvicorn server:app --host 0.0.0.0 --port 8000
```

Через `nohup`:

```bash
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

### Шаг 4. Проверка

```bash
curl http://127.0.0.1:8000/health
```

### Шаг 5. Один раз зарегистрируйте голос

```bash
curl -X POST http://127.0.0.1:8000/register-voice \
  -F "reference_audio=@./samples/reference.wav" \
  -F "reference_text=Hello, this is my reference voice." 
```

### Шаг 6. Остановка между сессиями для экономии

Когда вы не работаете, остановите Pod в панели RunPod. Это дешевле, чем держать GPU включенным без нужды.

Если вы работали через `tmux`, можно отсоединиться `Ctrl+B`, затем `D`, а сам Pod выключить в UI.

## Пример: одноразовая регистрация голоса

```bash
curl -X POST http://127.0.0.1:8000/register-voice \
  -F "reference_audio=@./samples/reference.wav" \
  -F "reference_text=This is the transcript of my reference voice." 
```

## Пример: проверка здоровья

```bash
curl http://127.0.0.1:8000/health
```

## Если n8n вызывает этот сервер удаленно с VPS

Если n8n работает на VPS, а inference-сервер в RunPod, нужен публичный HTTPS-адрес или приватная сеть между ними.

## Примечания по коду

- Движок TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- На CUDA пробуется `flash_attention_2`, а затем безопасный fallback
- `voice_clone_prompt` кешируется в памяти и сохраняется в `VOICE_PROMPT_PATH`
- При старте, если файл существует, prompt загружается автоматически