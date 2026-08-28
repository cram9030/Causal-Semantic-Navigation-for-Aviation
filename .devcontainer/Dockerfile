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

# python3-venv: uv still needs a base interpreter to build a venv against
# (it doesn't vendor one for the already-installed-Python case).
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

# uv (https://astral.sh/uv) replaces pip/venv for this project: dependency
# resolution is pinned in the committed uv.lock, so `uv sync` installs the
# exact versions that were resolved and tested rather than whatever's
# newest on PyPI that day - torch in particular moves fast enough that an
# unpinned `pip install` could silently jump to a much larger/newer CUDA
# stack between one rebuild and the next. Installed system-wide (root, via
# UV_INSTALL_DIR) so both root build steps and the non-root user below can
# call it without sudo. Version pinned to match what's been verified
# working; bump deliberately, not implicitly on a rebuild.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION=0.8.17 sh

# Non-root user: required so the container can be run with
# `claude --dangerously-skip-permissions` for unattended sessions, per
# https://code.claude.com/docs/en/devcontainer.
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Isolated venv (owned by the non-root user) rather than the system
# site-packages. UV_PROJECT_ENVIRONMENT pins uv's target venv to the same
# /opt/venv path the previous pip/venv setup used, so
# devcontainer.json's python.defaultInterpreterPath doesn't need to
# change; `postCreateCommand`'s `uv sync` (run as the non-root user, once
# the repo is mounted) populates it from uv.lock.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN mkdir -p ${UV_PROJECT_ENVIRONMENT} && chown -R ${USERNAME}:${USERNAME} ${UV_PROJECT_ENVIRONMENT}
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

USER ${USERNAME}

ENV DEBIAN_FRONTEND=dialog
WORKDIR /workspaces
