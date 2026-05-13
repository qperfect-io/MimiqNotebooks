#!/usr/bin/env bash
# Launch IPython from the strasbourg_markets_demo uv environment, with
# IPython history / config stored under <project-root>/.ipython/.
#
# Usage:
#   ./bin/ipython.sh                  # interactive REPL
#   ./bin/ipython.sh script.py        # run a script in the env
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

export IPYTHONDIR="$PROJECT_ROOT/.ipython"
mkdir -p "$IPYTHONDIR"

cd "$DEMO_DIR"

exec uv run --project "$DEMO_DIR" ipython "$@"
