#!/usr/bin/env bash
# Build predicted-mask crop boxes and train a ResNet-50 crop classifier.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the Oxford-IIIT Pet dataset directory}"
: "${SEGMENTATION_CHECKPOINT:?Set SEGMENTATION_CHECKPOINT to the trained U-Net checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to an existing, empty experiment directory}"

EPOCHS="${EPOCHS:-30}"
HEAD_EPOCHS="${HEAD_EPOCHS:-4}"
BACKBONE_LR="${BACKBONE_LR:-0.00001}"
HEAD_LR="${HEAD_LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PATIENCE="${PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-42}"
PRECISION="${PRECISION:-bf16}"
CROP_THRESHOLD="${CROP_THRESHOLD:-0.5}"
CROP_MARGIN="${CROP_MARGIN:-0.15}"
CROP_BATCH_SIZE="${CROP_BATCH_SIZE:-64}"

if [[ ! -d "$DATA_ROOT/images" || ! -d "$DATA_ROOT/annotations" ]]; then
    echo "Dataset not found under DATA_ROOT=$DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$SEGMENTATION_CHECKPOINT" ]]; then
    echo "Segmentation checkpoint not found: $SEGMENTATION_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Create RUN_DIR before starting the experiment: $RUN_DIR" >&2
    exit 1
fi
if [[ -e "$RUN_DIR/crop" || -e "$RUN_DIR/crop-cache.json" ]]; then
    echo "RUN_DIR already contains crop output; choose a new directory: $RUN_DIR" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python - "$RUN_DIR/environment.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("A CUDA/ROCm GPU is required")
properties = torch.cuda.get_device_properties(0)
environment = {
    "python": sys.version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "hip": torch.version.hip,
    "cuda": torch.version.cuda,
    "gpu": properties.name,
    "total_memory_gib": properties.total_memory / 2**30,
}
Path(sys.argv[1]).write_text(json.dumps(environment, indent=2), encoding="utf-8")
print(json.dumps(environment, indent=2), flush=True)
PY

python -u scripts/build_mask_crop_cache.py \
    --root "$DATA_ROOT" \
    --output "$RUN_DIR/crop-cache.json" \
    --source predicted \
    --segmentation-checkpoint "$SEGMENTATION_CHECKPOINT" \
    --split trainval --image-size "$IMAGE_SIZE" \
    --threshold "$CROP_THRESHOLD" --margin "$CROP_MARGIN" \
    --batch-size "$CROP_BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --device cuda --seed "$SPLIT_SEED"

python -u -m scripts.train_classification \
    --root "$DATA_ROOT" --output "$RUN_DIR/crop" \
    --architecture resnet50 --pretrained --mode crop \
    --crop-cache "$RUN_DIR/crop-cache.json" --schedule staged \
    --epochs "$EPOCHS" --head-epochs "$HEAD_EPOCHS" \
    --backbone-learning-rate "$BACKBONE_LR" --head-learning-rate "$HEAD_LR" \
    --weight-decay "$WEIGHT_DECAY" --early-stopping-patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --image-size "$IMAGE_SIZE" --augmentation color --val-fraction 0.15 \
    --precision "$PRECISION" \
    --split-seed "$SPLIT_SEED" --seed "$TRAINING_SEED" --device cuda --fast

echo "Training complete: $RUN_DIR/crop"
echo "Predicted crop cache: $RUN_DIR/crop-cache.json"
echo "best.pt was selected by validation Macro-F1. Test evaluation was not run."
