FROM python:3.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 annotation
WORKDIR /app
COPY backend/requirements*.txt /app/backend/
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements-annotations.txt
COPY --chown=annotation:annotation backend /app/backend
COPY --chown=annotation:annotation deployment/annotation-hosted.json /app/deployment/annotation-hosted.json
USER annotation
CMD ["uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "7861"]
