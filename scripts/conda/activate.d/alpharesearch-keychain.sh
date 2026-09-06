#!/usr/bin/env bash

# Load the private gateway credential only while this Conda environment is
# active. The secret itself remains in the macOS login Keychain.
if command -v security >/dev/null 2>&1; then
  _alpharesearch_account="$(id -un)"
  _alpharesearch_key="$(security find-generic-password \
    -a "$_alpharesearch_account" \
    -s AlphaResearch-LiteLLM \
    -w 2>/dev/null || true)"
  if [[ -n "$_alpharesearch_key" ]]; then
    export ALPHARESEARCH_LLM_API_KEY="$_alpharesearch_key"
  fi
  unset _alpharesearch_account _alpharesearch_key
fi

