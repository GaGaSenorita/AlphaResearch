#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
upstream="$repo_root/external/AlphaBench"
overlays="$repo_root/overlays/alphabench"

if [[ ! -d "$upstream/ffo" ]]; then
  echo "missing external/AlphaBench; initialise the pinned submodule first" >&2
  exit 2
fi

cp "$overlays/ffo/routes/factors.py" "$upstream/ffo/routes/factors.py"
cp "$overlays/ffo/utils/utils.py" "$upstream/ffo/utils/utils.py"
echo "Applied recovered FFO integrity overlays."
