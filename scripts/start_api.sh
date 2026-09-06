#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
if [[ "${CONDA_DEFAULT_ENV:-}" != "AlphaResearch" ]]; then
  echo "Activate the environment first: conda activate AlphaResearch" >&2
  exit 2
fi
echo "AlphaLDM dashboard: http://127.0.0.1:${1:-8765}"
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i python -m alpha_research.api.app --port "${1:-8765}"
fi
exec python -m alpha_research.api.app --port "${1:-8765}"
