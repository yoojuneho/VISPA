#!/usr/bin/env bash
set -euo pipefail

# VISPA DirectLeakage CI seed pipeline.
# Usage:
#   export OPENAI_API_KEY="..."
#   bash run_pipeline.sh
#   bash run_pipeline.sh --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OPENAI_API_KEY:-}" && "$DRY_RUN" == "0" ]]; then
  echo "OPENAI_API_KEY is required unless --dry-run is used." >&2
  exit 1
fi

COMMON_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "[1/4] Describing clean background scenes"
python "$SCRIPT_DIR/describe_scenes.py" "${COMMON_ARGS[@]}"

echo "[2/4] Building DirectLeakage CI seed variants"
python "$SCRIPT_DIR/build_ci_seed.py" "${COMMON_ARGS[@]}"

if [[ "$DRY_RUN" == "0" ]]; then
  SEED_PATH="$SCRIPT_DIR/DirectLeakage_Seed.json"

  echo "[3/4] Generating user instructions"
  python "$SCRIPT_DIR/generate_user_instructions.py" "$SEED_PATH" --skip-existing

  echo "[4/4] Auditing CI seed"
  AUDIT_ARGS=(--seed "$SEED_PATH" --mapping "$SCRIPT_DIR/mapping.json")
  python "$SCRIPT_DIR/audit_seed.py" "${AUDIT_ARGS[@]}"
fi
