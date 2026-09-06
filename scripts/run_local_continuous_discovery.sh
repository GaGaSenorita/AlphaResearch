#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="configs/ldm_continuous_discovery/ldm_continuous_discovery_neolink-deepseek-v4-flash_2016-2025.yaml"
target_rounds="${1:-100}"
seed="${2:-42}"
ffo_url="http://127.0.0.1:19777"
data_path="$repo/../quantaalpha_qlib_csi300/cn_data"

if [[ "${CONDA_DEFAULT_ENV:-}" != "AlphaResearch" ]]; then
  echo "Activate the environment first: conda activate AlphaResearch" >&2
  exit 2
fi
if [[ ! "$target_rounds" =~ ^[1-9][0-9]*$ ]]; then
  echo "target rounds must be a positive integer" >&2
  exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -d "$data_path" ]]; then
  echo "Qlib data directory is missing: $data_path" >&2
  exit 2
fi
if [[ -z "${ALPHARESEARCH_LLM_API_KEY:-}" ]]; then
  echo "The AlphaResearch API key is unavailable in the environment." >&2
  echo "Reactivate the Conda environment so its Keychain hook can load it." >&2
  exit 2
fi

cd "$repo"
export QLIB_DATA_PATH="$data_path"
export QLIB_PROVIDER_URI="$data_path"
export FFO_QLIB_DATA_PATH="$data_path"

if ! curl -fsS --max-time 3 "$ffo_url/health" >/dev/null 2>&1; then
  echo "Starting the local AlphaBench FFO backend..."
  ppo start backend
fi

echo "Starting local continual discovery: seed=$seed target_rounds=$target_rounds"
echo "The run is resumable after every committed round. Keep this terminal open."
exec caffeinate -i python -m alpha_research.cli "$config" \
  --set "args.search-rounds=$target_rounds" \
  --set "args.ldm-random-seed=$seed"

