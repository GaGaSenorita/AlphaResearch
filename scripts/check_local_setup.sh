#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_path="$repo/../quantaalpha_qlib_csi300/cn_data"

if [[ "${CONDA_DEFAULT_ENV:-}" != "AlphaResearch" ]]; then
  echo "FAIL conda environment: expected AlphaResearch" >&2
  exit 1
fi

python -c "import alpha_research, gpytorch, qlib, torch; print('OK Python imports'); print('PyTorch', torch.__version__); print('Qlib', qlib.__version__)"
test -f "$data_path/calendars/day.txt"
echo "OK Qlib data: $data_path"

if [[ -n "${ALPHARESEARCH_LLM_API_KEY:-}" ]]; then
  echo "OK API key: loaded from macOS Keychain"
else
  echo "FAIL API key: not loaded" >&2
  exit 1
fi

if curl -fsS --max-time 3 http://127.0.0.1:19777/health >/dev/null 2>&1; then
  echo "OK FFO backend: healthy"
else
  echo "INFO FFO backend: stopped (the run script starts it automatically)"
fi

