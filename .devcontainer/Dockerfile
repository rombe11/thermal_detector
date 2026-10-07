FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends zsh libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/zsh
