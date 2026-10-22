# Lightweight container for the LeakWatch FastAPI service
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies required by OpenCV / EasyOCR / FFmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY leakwatch ./leakwatch
COPY config ./config
COPY samples ./samples
COPY scripts ./scripts
COPY service ./service
COPY docs ./docs
COPY tests ./tests
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install . \
    && python -m spacy download en_core_web_sm

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
