FROM oven/bun:1.3.14 AS build
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile
COPY . .
ARG NEXT_PUBLIC_ANNOTATE_BACKEND_URL=/api/annotation
ENV NEXT_PUBLIC_ANNOTATE_BACKEND_URL=$NEXT_PUBLIC_ANNOTATE_BACKEND_URL
RUN bun run type-check && bun run build

FROM oven/bun:1.3.14-slim
WORKDIR /app
COPY --from=build --chown=bun:bun /app/.next ./.next
COPY --from=build --chown=bun:bun /app/node_modules ./node_modules
COPY --from=build --chown=bun:bun /app/package.json /app/next.config.ts ./
USER bun
ENV PORT=7860
EXPOSE 7860
CMD ["bun", "run", "start", "--hostname", "0.0.0.0", "--port", "7860"]
