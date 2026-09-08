FROM oven/bun:1.3.14 AS build
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --network-concurrency 8
COPY . .
ARG NEXT_PUBLIC_ANNOTATE_BACKEND_URL=/api/annotation
ENV NEXT_PUBLIC_ANNOTATE_BACKEND_URL=$NEXT_PUBLIC_ANNOTATE_BACKEND_URL
RUN bun run type-check && bun run build

FROM node:22-bookworm-slim AS node

FROM python:3.13-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=build /usr/local/bin/bun /usr/local/bin/bun
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json /app/next.config.ts ./
COPY backend ./backend
COPY deployment/start-annotation-space.py ./deployment/start-annotation-space.py
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
    && pip install --no-cache-dir -r backend/requirements-annotations.txt \
    && useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["python", "deployment/start-annotation-space.py"]
