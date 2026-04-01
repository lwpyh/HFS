#!/usr/bin/env bash
# Run HFS on VideoMME via SLURM (or directly).
#
# Usage:
#   sbatch examples/run_videomme_hfs.sh            # HFS (default)
#   sbatch examples/run_videomme_hfs.sh uniform_128 # baseline
#
#SBATCH --job-name=videomme_hfs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out

set -e

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"

# Repo root = two levels above this script (HFS/examples/ -> HFS/)
HFS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LMMS_DIR="$HFS_ROOT/hfs/lmms-eval"
METHOD="${1:-hfs}"   # hfs | uniform_128

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$HFS_ROOT/logs/videomme_${METHOD}_${TS}"
mkdir -p "$OUT_DIR"

echo "HFS_ROOT : $HFS_ROOT"
echo "LMMS_DIR : $LMMS_DIR"
echo "METHOD   : $METHOD"
echo "OUT_DIR  : $OUT_DIR"

# Make hfs package importable
export PYTHONPATH="$HFS_ROOT:${PYTHONPATH:-}"

cd "$LMMS_DIR"
$PYTHON -m pip install -e . -q --no-deps 2>&1 | tail -1

$ACCELERATE launch \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision bf16 \
    -m lmms_eval \
    --model qwen2_5_vl_hfs \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,\
max_pixels=6422528,\
min_pixels=200704,\
attn_implementation=flash_attention_2,\
method=${METHOD}" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUT_DIR"

echo "Results saved to $OUT_DIR"
