#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/data1/fin-autoresearch/AlphaResearch
config=configs/ldm_rankic_rankicir_ehvi/ldm_rankic_rankicir_ehvi_deepseek-v4-pro_2016-2025.yaml
log_dir="$repo/runtime/logs/ldm_rankic_rankicir_ehvi"
progress="$log_dir/progress.log"

cd "$repo"
mkdir -p "$log_dir"

if [[ -z "${ALPHARESEARCH_LLM_API_KEY:-}" ]]; then
  echo "ALPHARESEARCH_LLM_API_KEY is not set" >&2
  exit 2
fi

for seed in "$@"; do
  out="$repo/runs/ldm_rankic_rankicir_ehvi/ldm_rankic_rankicir_ehvi_deepseek-v4-pro_2016-2025_seed${seed}"
  if [[ -e "$out" ]]; then
    echo "Refusing to overwrite existing output: $out" >&2
    exit 3
  fi

  mkdir -p "$out"
  printf '%s seed=%s status=started\n' "$(date --iso-8601=seconds)" "$seed" | tee -a "$progress"

  if /home/zsgpu/miniconda3/envs/fin_autoresearch/bin/python \
      -m alpha_research.cli "$config" \
      --set "args.ldm-random-seed=$seed" \
      2>&1 | tee "$out/full_run.log"; then
    printf '%s seed=%s status=completed\n' "$(date --iso-8601=seconds)" "$seed" | tee -a "$progress"
  else
    status=${PIPESTATUS[0]}
    printf '%s seed=%s status=failed exit=%s\n' "$(date --iso-8601=seconds)" "$seed" "$status" | tee -a "$progress"
    exit "$status"
  fi
done

printf '%s queue status=completed\n' "$(date --iso-8601=seconds)" | tee -a "$progress"
