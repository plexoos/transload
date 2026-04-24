# syntax=docker/dockerfile:latest

ARG OS=ubuntu24.04
ARG CUDA_VERSION=13.0.2

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
 && apt install -y g++ gcc gzip tar python3 python-is-python3 python3-pip curl git \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y vim

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ENV PROJECT_HOME=/workspaces/transload
ENV VIRTUAL_ENV=${PROJECT_HOME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

WORKDIR $PROJECT_HOME

# Install Python dependencies
COPY pyproject.toml uv.lock $PROJECT_HOME/
COPY transload $PROJECT_HOME/transload
RUN uv sync
