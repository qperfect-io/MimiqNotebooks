#!/usr/bin/env bash
# Launch JupyterLab from the strasbourg_markets_demo uv environment, with
# all Jupyter state stored under <project-root>/.jupyter/ rather than $HOME.
#
# Usage:
#   ./bin/jupyter.sh                  # JupyterLab (default)
#   ./bin/jupyter.sh notebook          # classic Notebook UI
#   ./bin/jupyter.sh -- --port 9999    # extra flags forwarded to jupyter
#
# Pre-req: direnv has loaded the flake (or you ran `nix develop`),
# and `uv sync` has been run in strasbourg_markets_demo/.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="$PROJECT_ROOT/strasbourg_markets_demo"

if [ ! -d "$DEMO_DIR" ]; then
  echo "error: $DEMO_DIR does not exist. Run the project setup first." >&2
  exit 1
fi

# Per-project Jupyter state. The flake's shellHook also sets these, but we
# re-export here so the script works when invoked outside the dev shell too.
export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/.jupyter"
export JUPYTER_DATA_DIR="$PROJECT_ROOT/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.jupyter/runtime"
export IPYTHONDIR="$PROJECT_ROOT/.ipython"

mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"

# Decide which Jupyter front-end to launch.
SUBCMD="lab"
if [ "${1:-}" = "notebook" ] || [ "${1:-}" = "lab" ] || [ "${1:-}" = "console" ]; then
  SUBCMD="$1"
  shift
fi

cd "$DEMO_DIR"

# `uv run` activates the project's venv and runs the command inside it.
# --notebook-dir keeps file browser rooted at the project, not $HOME.
exec uv run --project "$DEMO_DIR" jupyter "$SUBCMD" \
  --notebook-dir="$PROJECT_ROOT" \
  "$@"
