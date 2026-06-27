#!/bin/bash
set -e

SCENE=""
OUT_BASE="output/ablations"

while [[ $# -gt 0 ]]; do
    case $1 in
        -s) SCENE="$2"; shift 2 ;;
        -o) OUT_BASE="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCENE" ]]; then
    echo "Usage: bash run_ablations.sh -s <scene_path> [-o <output_base>]"
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE_NAME=$(basename "$SCENE")

run() {
    local name=$1; shift
    local out="$ROOT/$OUT_BASE/${SCENE_NAME}/${name}"
    echo ""
    echo "=== Ablation: $name → $out ==="
    python "$ROOT/train.py" -s "$SCENE" --model_path "$out" --eval "$@"
}

run full
run no_normal   --no_normal_loss
run no_alpha    --no_alpha_loss
run rho_env     --rho_weighted_env

echo ""
echo "All ablations complete. Results in $OUT_BASE/$SCENE_NAME/"
