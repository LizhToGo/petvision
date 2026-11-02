#!/usr/bin/env bash
# Run the fixed ResNet-18 raw/guided experiment on a CUDA or ROCm PyTorch GPU.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the Oxford-IIIT Pet dataset directory}"
: "${SEGMENTATION_CHECKPOINT:?Set SEGMENTATION_CHECKPOINT to the trained U-Net checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to an existing, empty experiment directory}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

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
if [[ -e "$RUN_DIR/raw" || -e "$RUN_DIR/guided" || -e "$RUN_DIR/fusion" ]]; then
    echo "RUN_DIR already contains experiment output; choose a new directory: $RUN_DIR" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, torchvision; print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__, "ROCm:", torch.version.hip); assert torch.cuda.is_available(), "CUDA/ROCm GPU is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

echo "Training raw ResNet-18; results: $RUN_DIR"
python -u scripts/train_classification.py \
    --root "$DATA_ROOT" --output "$RUN_DIR/raw" \
    --architecture resnet18 --pretrained --mode raw \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --image-size 256 --learning-rate 0.0001 \
    --seed 42 --device cuda --fast

echo "Training guided ResNet-18"
python -u scripts/train_classification.py \
    --root "$DATA_ROOT" --output "$RUN_DIR/guided" \
    --architecture resnet18 --pretrained --mode guided \
    --segmenter-checkpoint "$SEGMENTATION_CHECKPOINT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --image-size 256 --learning-rate 0.0001 \
    --seed 42 --device cuda --fast

echo "Selecting fusion weight on validation and evaluating on test"
python -u scripts/evaluate_fusion.py \
    --root "$DATA_ROOT" --segmentation-checkpoint "$SEGMENTATION_CHECKPOINT" \
    --raw-checkpoint "$RUN_DIR/raw/best.pt" \
    --guided-checkpoint "$RUN_DIR/guided/best.pt" \
    --output "$RUN_DIR/fusion" --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" --lambda-step 0.05 \
    --seed 42 --device cuda

echo "Experiment complete: $RUN_DIR"
