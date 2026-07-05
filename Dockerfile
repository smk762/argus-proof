# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image) and the image
# needs the git binary for setuptools-scm to read it (slim ships without it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN uv pip install --system --no-cache ".[server,cli]"

EXPOSE 8104
CMD ["argus-proof", "serve", "--port", "8104", "--cors"]

