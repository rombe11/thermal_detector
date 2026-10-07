FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        zsh libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libdbus-1-3 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/zsh \
    QT_X11_NO_MITSHM=1
