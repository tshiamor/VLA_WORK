#!/bin/bash
# Train VLA on NVIDIA Cosmos Cube Stacking Dataset
# =================================================
# Dataset: cosmos_dataset_1k.hdf5 (3-cube stacking: blue -> red -> green)
# Model: Qwen2.5-VL-7B + Action Expert with LoRA
# GPU: Requires 24GB+ VRAM
#
# Usage:
#   bash scripts/train_cube_stacking.sh
#   bash scripts/train_cube_stacking.sh --resume checkpoints/cube_stacking_cosmos/epoch_10.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

CONFIG="configs/training/cube_stacking_cosmos.yaml"
HDF5_PATH="data/cosmos_dataset_1k.hdf5"
KEY_MAPPING="configs/key_mappings/nvidia_cosmos.json"

# Check dataset exists
if [ ! -f "$HDF5_PATH" ]; then
    echo "ERROR: Dataset not found at $HDF5_PATH"
    echo "Download it with:"
    echo "  hf download nvidia/PhysicalAI-Robotics-Manipulation-Augmented \\"
    echo "    cosmos_dataset_1k.hdf5 --repo-type dataset --local-dir data/"
    exit 1
fi

# Verify dataset
echo "Verifying dataset..."
python scripts/verify_dataset.py --hdf5_path "$HDF5_PATH" --key_mapping_file "$KEY_MAPPING" --num_samples 1

echo ""
echo "Starting VLA training..."
echo "Config: $CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Launch training (pass extra CLI args through)
python scripts/train.py \
    --config "$CONFIG" \
    --key_mapping_file "$KEY_MAPPING" \
    "$@"
