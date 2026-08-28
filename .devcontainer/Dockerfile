# CUDA 12.4 + cuDNN devel image: the "devel" variant (not "runtime") is
# intentional — Phase 2 fine-tunes Mask2Former, and building/using custom
# CUDA ops for segmentation models typically needs nvcc, not just the
# CUDA runtime libraries. Ubuntu 22.04 ships Python 3.10, which satisfies
# this project's requires-python >=3.10 (see pyproject.toml).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive

# python3-venv/pip: application runtime.
# gdal-bin/libgdal-dev/proj-bin/libproj-dev: system geometry libraries backing
# rasterio/pyproj (and future geopandas/shapely use in data/ground_truth and
# trajectory/ per docs/INTEGRATION_PLAN.md) — the manylinux wheels for these
# packages usually vendor their own GDAL/PROJ, but the system libs are kept
# here so native builds and any tooling that shells out to `gdalinfo`/`ogr2ogr`
# work out of the box.
# build-essential: needed to build native extensions (e.g. custom CUDA/C++
# ops some segmentation model implementations ship as source).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        gdal-bin \
        libgdal-dev \
        proj-bin \
        libproj-dev \
        git \
        curl \
        ca-certificates \
        sudo \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user: required so the container can be run with
# `claude --dangerously-skip-permissions` for unattended sessions, per
# https://code.claude.com/docs/en/devcontainer.
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# uv: single static binary, used in place of pip/venv for dependency
# resolution and installs — see uv.lock for the pinned, reproducible graph.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Isolated venv (owned by the non-root user) rather than the system
# site-packages, so `postCreateCommand` can install into it without sudo.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV} && chown -R ${USERNAME}:${USERNAME} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

USER ${USERNAME}

ENV DEBIAN_FRONTEND=dialog
WORKDIR /workspaces
