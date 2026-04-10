FROM python:3.13-slim

WORKDIR /app

# 시스템 의존성 (OpenCV headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 의존성 먼저 (캐시 활용)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 소스 복사
COPY src/ src/

EXPOSE 8200

CMD ["uv", "run", "python", "-m", "src.server"]
