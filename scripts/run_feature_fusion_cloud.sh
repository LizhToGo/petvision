#!/usr/bin/env bash
# Train and evaluate a feature-level raw/guided fusion head.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the Oxford-IIIT Pet dataset directory}"
: "${SEGMENTATION_CHECKPOINT:?Set SEGMENTATION_CHECKPOINT to the segmentation checkpoint}"
: "${RAW_CHECKPOINT:?Set RAW_CHECKPOINT to the raw ResNet checkpoint}"
: "${GUIDED_CHECKPOINT:?Set GUIDED_CHECKPOINT to the guided ResNet checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to an existing, empty experiment directory}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
AUGMENTATION="${AUGMENTATION:-none}"

if [[ ! -d "$DATA_ROOT/images" || ! -d "$DATA_ROOT/annotations" ]]; then
    echo "Dataset not found under DATA_ROOT=$DATA_ROOT" >&2
    exit 1
fi
for checkpoint in "$SEGMENTATION_CHECKPOINT" "$RAW_CHECKPOINT" "$GUIDED_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi
done
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Create RUN_DIR before starting the experiment: $RUN_DIR" >&2
    exit 1
fi
if [[ -e "$RUN_DIR/fusion_head" || -e "$RUN_DIR/evaluation" ]]; then
    echo "RUN_DIR already contains output; choose a new directory: $RUN_DIR" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip); assert torch.cuda.is_available(), "CUDA/ROCm GPU is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

python -u scripts/train_feature_fusion.py \
    --root "$DATA_ROOT" \
    --segmentation-checkpoint "$SEGMENTATION_CHECKPOINT" \
    --raw-checkpoint "$RAW_CHECKPOINT" \
    --guided-checkpoint "$GUIDED_CHECKPOINT" \
    --output "$RUN_DIR/fusion_head" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --learning-rate "$LEARNING_RATE" --augmentation "$AUGMENTATION" \
    --fast --device cuda

python -u scripts/evaluate_feature_fusion.py \
    --root "$DATA_ROOT" \
    --segmentation-checkpoint "$SEGMENTATION_CHECKPOINT" \
    --raw-checkpoint "$RAW_CHECKPOINT" \
    --guided-checkpoint "$GUIDED_CHECKPOINT" \
    --fusion-checkpoint "$RUN_DIR/fusion_head/best.pt" \
    --output "$RUN_DIR/evaluation" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --device cuda

echo "Experiment complete: $RUN_DIR"
