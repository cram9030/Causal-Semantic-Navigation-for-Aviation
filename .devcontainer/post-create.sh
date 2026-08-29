#!/bin/bash
# Runs once per container create/rebuild, as the non-root `vscode` user,
# into the uv-managed venv at UV_PROJECT_ENVIRONMENT (/opt/venv, on PATH
# via the Dockerfile).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --all-extras installs every optional-dependencies group declared in
# pyproject.toml:
# `dev`  - test/lint tooling (pytest, responses)
# `viz`  - Phase 1 visualization (matplotlib, folium); the trajectory/tube/
#          manifest figures and maps, and the tests that render them
# `ml`   - Phase 2 segmentation training deps (torch, torchvision,
#          transformers, scikit-learn). Installed by default so the GPU
#          path is ready to use, not something a contributor has to
#          remember to opt into.
# `dvc`  - data/pipeline version control for dvc.yaml + params.yaml (see
#          README's "Data versioning (DVC)" section).
# All are declared in pyproject.toml so it stays the single source of
# truth for versions. `uv sync` installs the exact versions pinned in the
# committed uv.lock rather than re-resolving against whatever's newest on
# PyPI - see the Dockerfile's `uv` install step for why that matters for
# `ml` (torch).
uv sync --all-extras

echo "--- GPU check ---"
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('No CUDA device visible to this container (CPU-only, or GPU passthrough not configured on the host) - fine for Phase 0/1 work, needed for Phase 2 Mask2Former fine-tuning.')
" || echo "torch import failed - check the 'uv sync --all-extras' output above."
