#!/bin/bash
# Runs once per container create/rebuild, as the non-root `vscode` user,
# inside the venv activated via PATH in the Dockerfile.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The Claude Code config dir (CLAUDE_CONFIG_DIR, set in devcontainer.json) is
# bind-mounted from a named volume so auth/settings/history survive rebuilds.
# A brand-new volume (e.g. after devcontainerId changes) is seeded root:root
# by Docker if nothing pre-exists at that path in the image, which blocks the
# non-root vscode user from writing to it at all (EACCES on every write,
# including the very first one, so login "succeeds" but never persists).
# Re-asserting ownership here runs after the volume is mounted and is a
# no-op on an already-correct volume, so it self-heals on every rebuild.
sudo chown -R "$(id -u):$(id -g)" "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

# The `claude-code` and `node` dev container features both run as root
# during image build, before `remoteUser: vscode` takes effect. The node
# feature's own installs (npm, corepack, yarn) end up owned by vscode, but
# `npm install -g @anthropic-ai/claude-code` (run by the claude-code
# feature, and by pnpm alongside it) leaves its package directory under the
# global node_modules root owned by root:nvm with no group-write bit. vscode
# is in the nvm group but can only read/execute, not write, so Claude Code's
# self-update (`npm install -g` into that same directory) fails with
# "Auto-update failed: no write permission to npm prefix". Re-owning the
# global node_modules tree here runs after the feature installs and is a
# no-op once it's already vscode-owned, so it self-heals on every rebuild.
npm_global_modules="$(npm root -g)"
if [ -n "${npm_global_modules}" ] && [ -d "${npm_global_modules}" ]; then
    sudo chown -R "$(id -u):$(id -g)" "${npm_global_modules}"
fi

# Default `git commit` (no -m) to opening its message in VS Code and
# blocking until that tab is closed, instead of falling back to $EDITOR/vi
# inside the integrated terminal. `code` is put on PATH by the VS Code
# Server itself, so this can't be set in the Dockerfile (the server, and
# its remote-cli, don't exist yet at image build time). Global git config
# lives in the container's home directory, which isn't volume-mounted like
# CLAUDE_CONFIG_DIR above, so it doesn't survive a rebuild on its own -
# re-set it here every time so it self-heals the same way.
git config --global core.editor "code --wait"

# `dev`  - test/lint tooling (pytest, responses)
# `viz`  - Phase 1 visualization (matplotlib, folium); the trajectory/tube/
#          manifest figures and maps, and the tests that render them
# `ml`   - Phase 2 segmentation training deps (torch, torchvision,
#          transformers, scikit-learn). Installed by default so the GPU
#          path is ready to use, not something a contributor has to
#          remember to opt into.
# `dvc`  - data/pipeline version control for dvc.yaml + params.yaml (see
#          README's "Data versioning (DVC)" section).
# All are declared in pyproject.toml, and pinned (with the rest of the
# dependency graph) in uv.lock, so those two files stay the single source
# of truth for versions. `uv sync` installs into $VIRTUAL_ENV (set in the
# Dockerfile) and installs the project itself editable by default -
# equivalent to the old `pip install -e ".[dev,viz,ml,dvc]"`, but resolved
# from the committed lockfile instead of pip re-resolving loose `>=`
# bounds on every rebuild.
uv sync --extra dev --extra viz --extra ml --extra dvc

echo "--- GPU check ---"
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('No CUDA device visible to this container (CPU-only, or GPU passthrough not configured on the host) - fine for Phase 0/1 work, needed for Phase 2 Mask2Former fine-tuning.')
" || echo "torch import failed - check the 'ml' optional-dependencies install above."
