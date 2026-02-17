#!/bin/bash
#
# VLA Environment Setup Script
#
# This script sets up the complete environment for VLA training.
# It can either use an existing conda environment or create a new one.
#
# Usage:
#   ./scripts/setup_env.sh              # Use existing env, install missing packages
#   ./scripts/setup_env.sh --new        # Create fresh 'vla' conda environment
#   ./scripts/setup_env.sh --env myenv  # Create/use specific environment name
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CREATE_NEW_ENV=false
ENV_NAME=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --new)
            CREATE_NEW_ENV=true
            ENV_NAME="vla"
            shift
            ;;
        --env)
            CREATE_NEW_ENV=true
            ENV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --new         Create a fresh 'vla' conda environment"
            echo "  --env NAME    Create/use a specific conda environment"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Without options, installs packages in current environment."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "============================================================"
echo "           VLA Environment Setup"
echo "============================================================"
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python package
check_package() {
    python -c "import $1" 2>/dev/null
}

# Step 1: Check conda
echo -e "${YELLOW}Step 1: Checking conda installation...${NC}"
if ! command_exists conda; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Conda found${NC}"

# Step 2: Create or activate environment
echo -e "\n${YELLOW}Step 2: Setting up conda environment...${NC}"

if [ "$CREATE_NEW_ENV" = true ]; then
    echo "Creating new conda environment: $ENV_NAME"

    # Check if env already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Activating...${NC}"
    else
        echo "Creating environment with Python 3.10..."
        conda create -n "$ENV_NAME" python=3.10 -y
    fi

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    echo -e "${GREEN}✓ Activated environment: $ENV_NAME${NC}"
else
    echo "Using current environment: $CONDA_DEFAULT_ENV"
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo -e "${YELLOW}Warning: No conda environment active. Using base Python.${NC}"
    fi
fi

# Step 3: Check/Install PyTorch
echo -e "\n${YELLOW}Step 3: Checking PyTorch...${NC}"
if check_package torch; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo -e "${GREEN}✓ PyTorch $TORCH_VERSION installed (CUDA: $CUDA_AVAILABLE)${NC}"
else
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo -e "${GREEN}✓ PyTorch installed${NC}"
fi

# Step 4: Install core dependencies
echo -e "\n${YELLOW}Step 4: Installing core dependencies...${NC}"

PACKAGES=(
    "transformers>=4.40.0"
    "accelerate"
    "bitsandbytes"
    "peft"
    "qwen-vl-utils"
    "wandb"
    "h5py"
    "Pillow"
    "numpy"
    "scipy"
    "matplotlib"
    "tqdm"
    "pyyaml"
)

for pkg in "${PACKAGES[@]}"; do
    pkg_name=$(echo "$pkg" | cut -d'>' -f1 | cut -d'=' -f1)
    echo -n "  Installing $pkg_name... "
    pip install "$pkg" -q
    echo -e "${GREEN}✓${NC}"
done

# Step 5: Install VLA package
echo -e "\n${YELLOW}Step 5: Installing VLA package...${NC}"
cd "$PROJECT_ROOT"
pip install -e . -q
echo -e "${GREEN}✓ VLA package installed${NC}"

# Step 6: Verify installation
echo -e "\n${YELLOW}Step 6: Verifying installation...${NC}"

VERIFY_SCRIPT=$(cat << 'EOF'
import sys

def check(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'OK')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: FAILED ({e})")
        return False

print("\nCore packages:")
all_ok = True
all_ok &= check("PyTorch", "torch")
all_ok &= check("Transformers", "transformers")
all_ok &= check("Accelerate", "accelerate")
all_ok &= check("PEFT", "peft")
all_ok &= check("h5py")
all_ok &= check("wandb")
all_ok &= check("PIL", "PIL")

print("\nVLA package:")
try:
    from vla import VLAModel, create_vla_model
    from vla.data import HDF5VLADataset
    print("  ✓ vla.models")
    print("  ✓ vla.data.HDF5VLADataset")
except ImportError as e:
    print(f"  ✗ VLA import failed: {e}")
    all_ok = False

print("\nGPU Status:")
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ✗ CUDA not available")

sys.exit(0 if all_ok else 1)
EOF
)

python -c "$VERIFY_SCRIPT"
VERIFY_STATUS=$?

# Step 7: Summary
echo -e "\n${BLUE}============================================================${NC}"
if [ $VERIFY_STATUS -eq 0 ]; then
    echo -e "${GREEN}           Setup Complete!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Verify your dataset:"
    echo "     python scripts/verify_dataset.py \\"
    echo "       --hdf5_path /home/tshiamo/IsaacLab/demos/mcx_card_demos_vla_224.hdf5"
    echo ""
    echo "  2. Login to wandb:"
    echo "     wandb login"
    echo ""
    echo "  3. Start training:"
    echo "     python scripts/train.py \\"
    echo "       --hdf5_path /home/tshiamo/IsaacLab/demos/mcx_card_demos_vla_224.hdf5 \\"
    echo "       --instruction \"pick up the blue block and place it in the first card's closest slot\" \\"
    echo "       --action_dim 7 \\"
    echo "       --batch_size 16 \\"
    echo "       --epochs 50 \\"
    echo "       --wandb_project vla-mcx-card"
    echo ""
else
    echo -e "${RED}           Setup had errors!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo "Please check the error messages above and try again."
    exit 1
fi
