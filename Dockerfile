FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY main.py http_server.py ./
COPY words_repo/Vocabulary/ words_repo/Vocabulary/
COPY datasets/ datasets/

EXPOSE 8421

CMD ["uv", "run", "python", "http_server.py"]
