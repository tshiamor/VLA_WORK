#!/bin/bash
#
# Train VLA on MCX Card Dataset
#
# Quick start script for training on the mcx_card_demos dataset.
#
# Usage:
#   ./scripts/train_mcx_card.sh                    # Default settings
#   ./scripts/train_mcx_card.sh --batch_size 32    # Custom batch size
#   ./scripts/train_mcx_card.sh --epochs 100       # More epochs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
HDF5_PATH="/home/tshiamo/IsaacLab/demos/mcx_card_demos_vla_224.hdf5"
INSTRUCTION="pick up the blue block and place it in the first card's closest slot"
ACTION_DIM=7
PROPRIO_DIM=27
CHUNK_SIZE=16
BATCH_SIZE=16
EPOCHS=50
LR="1e-4"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/mcx_card"
WANDB_PROJECT="vla-mcx-card"
WANDB_RUN="mcx-card-$(date +%Y%m%d-%H%M%S)"

# Parse command line overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --hdf5_path)
            HDF5_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_PROJECT=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --hdf5_path PATH    Path to HDF5 dataset"
            echo "  --batch_size N      Batch size (default: 16)"
            echo "  --epochs N          Number of epochs (default: 50)"
            echo "  --lr RATE           Learning rate (default: 1e-4)"
            echo "  --resume PATH       Resume from checkpoint"
            echo "  --no-wandb          Disable wandb logging"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check dataset exists
if [ ! -f "$HDF5_PATH" ]; then
    echo "Error: Dataset not found at $HDF5_PATH"
    exit 1
fi

# Build command
CMD="python $SCRIPT_DIR/train.py"
CMD+=" --hdf5_path \"$HDF5_PATH\""
CMD+=" --instruction \"$INSTRUCTION\""
CMD+=" --action_dim $ACTION_DIM"
CMD+=" --proprio_dim $PROPRIO_DIM"
CMD+=" --chunk_size $CHUNK_SIZE"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --epochs $EPOCHS"
CMD+=" --lr $LR"
CMD+=" --checkpoint_dir \"$CHECKPOINT_DIR\""

if [ -n "$WANDB_PROJECT" ]; then
    CMD+=" --wandb_project $WANDB_PROJECT"
    CMD+=" --wandb_run $WANDB_RUN"
fi

if [ -n "$RESUME" ]; then
    CMD+=" --resume \"$RESUME\""
fi

# Print configuration
echo "============================================================"
echo "           VLA Training - MCX Card Task"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Dataset:      $HDF5_PATH"
echo "  Instruction:  $INSTRUCTION"
echo "  Action dim:   $ACTION_DIM"
echo "  Batch size:   $BATCH_SIZE"
echo "  Epochs:       $EPOCHS"
echo "  Checkpoints:  $CHECKPOINT_DIR"
if [ -n "$WANDB_PROJECT" ]; then
    echo "  Wandb:        $WANDB_PROJECT / $WANDB_RUN"
fi
echo ""
echo "============================================================"
echo ""

# Run training
cd "$PROJECT_ROOT"
eval $CMD
