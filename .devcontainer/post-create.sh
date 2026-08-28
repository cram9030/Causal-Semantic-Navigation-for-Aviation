#!/bin/bash
# Runs once per container create/rebuild, as the non-root `vscode` user,
# inside the venv activated via PATH in the Dockerfile.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# `dev`  - test/lint tooling (pytest, responses)
# `ml`   - Phase 2 segmentation training deps (torch, torchvision,
#          transformers, scikit-learn). Installed by default so the GPU
#          path is ready to use, not something a contributor has to
#          remember to opt into. Both are declared in pyproject.toml, and
#          pinned (with the rest of the dependency graph) in uv.lock, so
#          those two files stay the single source of truth for versions.
# `uv sync` installs into $VIRTUAL_ENV (set in the Dockerfile) and installs
# the project itself editable by default - equivalent to the old
# `pip install -e ".[dev,ml]"`, but resolved from the committed lockfile
# instead of pip re-resolving loose `>=` bounds on every rebuild.
uv sync --extra dev --extra ml

echo "--- GPU check ---"
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('No CUDA device visible to this container (CPU-only, or GPU passthrough not configured on the host) - fine for Phase 0/1 work, needed for Phase 2 Mask2Former fine-tuning.')
" || echo "torch import failed - check the 'ml' optional-dependencies install above."
