#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/data1/fin-autoresearch/AlphaResearch
config=configs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml
target_rounds=${1:-100}
seed=${2:-42}
secret_file=/home/zsgpu/.alpharesearch_llm_key
log_dir="$repo/runtime/logs/ldm_continuous_discovery"
progress="$log_dir/progress.log"
run_name="ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed${seed}"
out="$repo/runs/ldm_continuous_discovery/$run_name"

if [[ ! "$target_rounds" =~ ^[1-9][0-9]*$ ]]; then
  echo "target rounds must be a positive integer" >&2
  exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi

cd "$repo"
mkdir -p "$log_dir"

if [[ -z "${ALPHARESEARCH_LLM_API_KEY:-}" ]]; then
  if [[ ! -r "$secret_file" ]]; then
    echo "protected LLM key file is missing: $secret_file" >&2
    exit 2
  fi
  IFS= read -r ALPHARESEARCH_LLM_API_KEY < "$secret_file" \
    || [[ -n "$ALPHARESEARCH_LLM_API_KEY" ]]
  export ALPHARESEARCH_LLM_API_KEY
fi

if [[ -d "$out" ]]; then
  if [[ ! -f "$out/resume_state.json" || ! -f "$out/factor_ledger.jsonl" ]]; then
    echo "refusing to reuse a non-resumable output directory: $out" >&2
    exit 3
  fi
  completed=$(/home/zsgpu/miniconda3/envs/fin_autoresearch/bin/python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["round_id"]))' \
    "$out/resume_state.json")
  if (( target_rounds <= completed )); then
    printf '%s seed=%s status=already_complete completed=%s target=%s\n' \
      "$(date --iso-8601=seconds)" "$seed" "$completed" "$target_rounds" | tee -a "$progress"
    exit 0
  fi
else
  mkdir -p "$out"
fi

if pgrep -af "python -m alpha_research.cli .*ldm_continuous_discovery" \
    | grep -Fq "seed${seed}"; then
  echo "continuous discovery seed $seed is already running" >&2
  exit 4
fi

printf '%s seed=%s status=queued target=%s\n' \
  "$(date --iso-8601=seconds)" "$seed" "$target_rounds" | tee -a "$progress"
while pgrep -f "python -m alpha_research.cli" >/dev/null; do
  printf '%s seed=%s status=waiting_for_existing_alpha_research_job target=%s\n' \
    "$(date --iso-8601=seconds)" "$seed" "$target_rounds" >> "$progress"
  sleep 60
done

printf '%s seed=%s status=started target=%s\n' \
  "$(date --iso-8601=seconds)" "$seed" "$target_rounds" | tee -a "$progress"
if /home/zsgpu/miniconda3/envs/fin_autoresearch/bin/python \
    -m alpha_research.cli "$config" \
    --set "args.search-rounds=$target_rounds" \
    --set "args.ldm-random-seed=$seed" \
    2>&1 | tee -a "$out/full_run.log"; then
  printf '%s seed=%s status=completed target=%s\n' \
    "$(date --iso-8601=seconds)" "$seed" "$target_rounds" | tee -a "$progress"
else
  status=${PIPESTATUS[0]}
  printf '%s seed=%s status=failed exit=%s target=%s\n' \
    "$(date --iso-8601=seconds)" "$seed" "$status" "$target_rounds" | tee -a "$progress"
  exit "$status"
fi
