#!/usr/bin/env -S bash --login
set -euo pipefail
# This script is used to install any custom packages required by the algorithm.

command -v curl >/dev/null 2>&1 || {
    echo "Installing curl..."
    conda install curl
}

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

basedir=$(cd "$(dirname "$0")" && pwd -P)

UV_PROJECT="$basedir" uv sync --no-dev

# unset PROJ env vars
unset PROJ_LIB
unset PROJ_DATA
