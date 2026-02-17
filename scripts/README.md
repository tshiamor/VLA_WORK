# VLA Scripts

Command-line tools for training, evaluating, and visualizing VLA models.

---

## Overview

| Script | Purpose |
|--------|---------|
| `setup_env.sh` | **Setup conda environment and install dependencies** |
| `verify_dataset.py` | Verify HDF5 dataset compatibility |
| `train.py` | Train VLA model on demonstration data |
| `train_mcx_card.sh` | Quick-start training on MCX card dataset |
| `evaluate.py` | Evaluate trained model in Isaac Lab simulation |
| `collect_demos.py` | Collect robot demonstrations for training |
| `visualize.py` | Visualize model predictions and internals |
| `overfit_test.py` | **Overfit test**: train on N demos to verify pipeline correctness |
| `diagnose_conditioning.py` | **Diagnosis**: test VLM embeddings, condition vectors, and flow head isolation |
| `inspect_nvidia_dataset.py` | Inspect NVIDIA Cosmos HDF5 dataset structure |
| `diagnose_vla.py` | General VLA model diagnostics |

---

## Quick Start

```bash
# 1. Setup environment
./scripts/setup_env.sh

# 2. Verify your dataset
python scripts/verify_dataset.py --hdf5_path /path/to/demos.hdf5

# 3. Train
./scripts/train_mcx_card.sh
```

---

## setup_env.sh

Automated environment setup script. Installs all dependencies and verifies the installation.

### Usage

```bash
# Install in current environment
./scripts/setup_env.sh

# Create fresh 'vla' conda environment
./scripts/setup_env.sh --new

# Create custom-named environment
./scripts/setup_env.sh --env my_vla_env
```

### What It Does

1. Checks conda installation
2. Creates/activates conda environment (optional)
3. Installs PyTorch with CUDA
4. Installs transformers, accelerate, peft, wandb, etc.
5. Installs VLA package in editable mode
6. Verifies all imports work correctly

---

## verify_dataset.py

Verify HDF5 dataset structure before training.

### Usage

```bash
# Basic verification
python scripts/verify_dataset.py \
    --hdf5_path /path/to/demos.hdf5

# With visualization
python scripts/verify_dataset.py \
    --hdf5_path /path/to/demos.hdf5 \
    --visualize \
    --num_samples 5
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hdf5_path` | str | **required** | Path to HDF5 demo file |
| `--visualize` | flag | False | Show sample images |
| `--num_samples` | int | 3 | Number of samples to verify |

---

## train_mcx_card.sh

Quick-start script for training on the MCX card dataset.

### Usage

```bash
# Default settings
./scripts/train_mcx_card.sh

# Custom batch size
./scripts/train_mcx_card.sh --batch_size 32

# More epochs
./scripts/train_mcx_card.sh --epochs 100

# Resume training
./scripts/train_mcx_card.sh --resume checkpoints/mcx_card/epoch_25.pt

# Disable wandb
./scripts/train_mcx_card.sh --no-wandb
```

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | `/home/tshiamo/IsaacLab/demos/mcx_card_demos_vla_224.hdf5` |
| Instruction | "pick up the blue block and place it in the first card's closest slot" |
| Action dim | 7 |
| Batch size | 16 |
| Epochs | 50 |

---

## train.py

Train a Vision-Language-Action model on robot demonstration data.

### Basic Usage

```bash
# Train with command-line arguments
python scripts/train.py \
    --data_dir data/demos \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4

# Train with config file
python scripts/train.py --config configs/training/pretrain_config.yaml
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | None | Path to training data directory (folder-based) |
| `--hdf5_path` | str | None | Path to HDF5 file with all demos |
| `--val_data_dir` | str | None | Path to validation data directory |
| `--val_hdf5_path` | str | None | Path to validation HDF5 file |
| `--instruction` | str | "Pick up the object..." | Task instruction for VLA |
| `--vlm_model` | str | `Qwen/Qwen2.5-VL-7B-Instruct` | Vision-language model name |
| `--action_dim` | int | 7 | Robot action dimension |
| `--chunk_size` | int | 16 | Action chunk size |
| `--hidden_dim` | int | 512 | Hidden dimension for action expert |
| `--epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--warmup_steps` | int | 1000 | Number of warmup steps |
| `--gradient_accumulation` | int | 1 | Gradient accumulation steps |
| `--use_lora` | flag | True | Use LoRA adapters |
| `--lora_r` | int | 16 | LoRA rank |
| `--checkpoint_dir` | str | `checkpoints` | Directory for saving checkpoints |
| `--resume` | str | None | Path to checkpoint to resume from |
| `--wandb_project` | str | None | Wandb project name |
| `--wandb_run` | str | None | Wandb run name |
| `--config` | str | None | Path to YAML config file |
| `--device` | str | `cuda` | Device to use |

### Examples

```bash
# Train from HDF5 file (recommended)
python scripts/train.py \
    --hdf5_path /path/to/demos.hdf5 \
    --instruction "pick up the blue block and place it in the target" \
    --action_dim 7 \
    --epochs 50 \
    --wandb_project vla-training

# Full training run with validation (folder-based)
python scripts/train.py \
    --data_dir data/demos/train \
    --val_data_dir data/demos/val \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/run1 \
    --wandb_project vla-training

# Resume from checkpoint
python scripts/train.py \
    --data_dir data/demos \
    --resume checkpoints/run1/epoch_50.pt \
    --epochs 100

# Fine-tuning with lower learning rate
python scripts/train.py \
    --config configs/training/finetune_config.yaml \
    --data_dir data/demos/finetune \
    --lr 5e-5
```

### Output

- Checkpoints saved to `checkpoint_dir/` (best.pt, epoch_N.pt, step_N.pt)
- Training metrics logged to wandb (if configured)
- Console output with loss and learning rate

---

## evaluate.py

Evaluate a trained VLA model in Isaac Lab simulation.

### Basic Usage

```bash
# Evaluate on reaching task
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --robot franka \
    --task reach

# Evaluate on pick-and-place
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --robot ur5e \
    --task pick_place \
    --num_episodes 100
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to model checkpoint |
| `--robot` | str | `franka` | Robot type: `franka` or `ur5e` |
| `--task` | str | `reach` | Task: `reach` or `pick_place` |
| `--num_episodes` | int | 50 | Number of evaluation episodes |
| `--max_steps` | int | 200 | Maximum steps per episode |
| `--instruction` | str | None | Override instruction (uses task default) |
| `--headless` | flag | False | Run without visualization |
| `--save_video` | flag | False | Save evaluation videos |
| `--video_dir` | str | `videos` | Directory to save videos |
| `--device` | str | `cuda` | Device to use |

### Examples

```bash
# Quick evaluation (headless)
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --robot franka \
    --task reach \
    --num_episodes 20 \
    --headless

# Full evaluation with visualization
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --robot franka \
    --task pick_place \
    --num_episodes 100 \
    --save_video \
    --video_dir videos/eval_run1

# Custom instruction
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --robot ur5e \
    --task reach \
    --instruction "Move to the blue target on the left."
```

### Output

```
============================================================
Evaluation Results
============================================================
Success Rate: 85.0%
Mean Reward: 42.35 (+/- 12.18)
Mean Steps: 87.2
============================================================
```

---

## collect_demos.py

Collect robot demonstrations in Isaac Lab for VLA training.

### Basic Usage

```bash
# Collect with scripted policy
python scripts/collect_demos.py \
    --robot franka \
    --task reach \
    --num_episodes 100 \
    --output_dir data/demos

# Collect pick-and-place demos
python scripts/collect_demos.py \
    --robot ur5e \
    --task pick_place \
    --num_episodes 200 \
    --output_dir data/demos/pick_place
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot` | str | `franka` | Robot type: `franka` or `ur5e` |
| `--task` | str | `reach` | Task: `reach` or `pick_place` |
| `--num_episodes` | int | 100 | Number of episodes to collect |
| `--max_steps` | int | 200 | Maximum steps per episode |
| `--mode` | str | `scripted` | Collection mode: `scripted` or `teleop` |
| `--output_dir` | str | `data/demos` | Directory to save demonstrations |
| `--headless` | flag | False | Run without visualization |

### Examples

```bash
# Collect reaching demos for Franka
python scripts/collect_demos.py \
    --robot franka \
    --task reach \
    --num_episodes 500 \
    --output_dir data/demos/franka_reach \
    --headless

# Collect pick-place demos for UR5e
python scripts/collect_demos.py \
    --robot ur5e \
    --task pick_place \
    --num_episodes 300 \
    --output_dir data/demos/ur5e_pick_place

# With visualization (slower but useful for debugging)
python scripts/collect_demos.py \
    --robot franka \
    --task reach \
    --num_episodes 10
```

### Output

Demonstrations are saved as HDF5 files in the output directory:
```
data/demos/
├── episode_000000.h5
├── episode_000001.h5
├── episode_000002.h5
└── ...
```

Each HDF5 file contains:
- `context_images`: Context camera RGB images [T, H, W, 3]
- `wrist_images`: Wrist camera RGB images [T, H, W, 3]
- `actions`: Robot actions [T, action_dim]
- `proprio_states`: Proprioceptive states [T, proprio_dim]
- `instruction`: Language instruction (attribute)

---

## visualize.py

Visualize VLA model predictions, flow trajectories, and attention maps.

### Basic Usage

```bash
# Visualize action distributions
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode actions

# Visualize flow matching trajectory
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode flow

# Visualize attention (placeholder)
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode attention
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to model checkpoint |
| `--mode` | str | `actions` | Visualization mode: `actions`, `flow`, `attention` |
| `--context_image` | str | None | Path to context camera image |
| `--wrist_image` | str | None | Path to wrist camera image |
| `--instruction` | str | `Move the robot arm...` | Language instruction |
| `--output_dir` | str | `visualizations` | Directory to save visualizations |
| `--num_samples` | int | 10 | Number of action samples (for `actions` mode) |
| `--num_flow_steps` | int | 20 | Number of flow steps (for `flow` mode) |
| `--device` | str | `cuda` | Device to use |

### Visualization Modes

#### `actions` - Action Distribution
Shows the distribution of sampled action trajectories:
- Multiple sampled trajectories (blue, transparent)
- Mean trajectory (red)
- Standard deviation band (shaded)

#### `flow` - Flow Matching Trajectory
Shows the flow from noise to actions:
- Trajectory through flow time t ∈ [0, 1]
- Starting point (noise) and ending point (action)
- One plot per action dimension

#### `attention` - Attention Maps
Shows attention weights overlaid on input images:
- Context and wrist camera images
- Attention heatmap overlay

### Examples

```bash
# Visualize with custom images
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode actions \
    --context_image images/context.png \
    --wrist_image images/wrist.png \
    --instruction "Pick up the red cube."

# Detailed flow visualization
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode flow \
    --num_flow_steps 50 \
    --output_dir visualizations/flow_analysis

# Multiple action samples
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode actions \
    --num_samples 50 \
    --output_dir visualizations/action_variance
```

### Output

Visualizations are saved as PNG files:
```
visualizations/
├── action_distribution.png   # Action mode
├── input_images.png          # Input images
├── flow_trajectory.png       # Flow mode
└── attention.png             # Attention mode
```

---

## overfit_test.py

Train the VLA on a small number of demos to verify the pipeline works end-to-end.
If the model can overfit 5 demos, the architecture is correct. See [Diagnosis Report](../docs/diagnosis_overfit_test.md) for full analysis.

### Usage

```bash
python scripts/overfit_test.py \
    --hdf5_path data/cosmos_dataset_1k.hdf5 \
    --key_mapping_file configs/key_mappings/nvidia_cosmos.json \
    --num_demos 5 --max_steps 50000 --target_loss 0.10 --lr 3e-4
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hdf5_path` | str | **required** | Path to HDF5 demo file |
| `--key_mapping_file` | str | `configs/key_mappings/nvidia_cosmos.json` | Key mapping for dataset format |
| `--num_demos` | int | 5 | Number of demos to overfit on |
| `--max_steps` | int | 50000 | Maximum training steps |
| `--target_loss` | float | 0.10 | Stop when loss reaches this |
| `--lr` | float | 3e-4 | Learning rate |
| `--batch_size` | int | 4 | Batch size |

### Phases

1. **Load dataset** (N demos, no augmentation)
2. **Create fresh model** (no checkpoint)
3. **Train until convergence** (or max steps)
4. **Open-loop evaluation** on training demos
5. **Verdict**: PASS (loss < 0.1, rel error < 0.3), PARTIAL (loss < 0.3), or FAIL

---

## diagnose_conditioning.py

Three diagnostic tests to isolate conditioning pipeline issues.

### Usage

```bash
python scripts/diagnose_conditioning.py
```

### Tests

1. **VLM Embedding Variance**: Are embeddings identical across samples? (cosine similarity)
2. **Condition Vector Variance**: Does proprio info reach the flow head?
3. **Flow Head Isolation**: Can the flow head overfit with unique conditions? (no VLM needed)

---

## Common Workflows

### 1. End-to-End Training Pipeline

```bash
# Step 1: Collect demonstrations
python scripts/collect_demos.py \
    --robot franka \
    --task reach \
    --num_episodes 500 \
    --output_dir data/demos/train \
    --headless

python scripts/collect_demos.py \
    --robot franka \
    --task reach \
    --num_episodes 100 \
    --output_dir data/demos/val \
    --headless

# Step 2: Train model
python scripts/train.py \
    --data_dir data/demos/train \
    --val_data_dir data/demos/val \
    --epochs 100 \
    --checkpoint_dir checkpoints/reach_v1

# Step 3: Evaluate
python scripts/evaluate.py \
    --checkpoint checkpoints/reach_v1/best.pt \
    --robot franka \
    --task reach \
    --num_episodes 100

# Step 4: Visualize
python scripts/visualize.py \
    --checkpoint checkpoints/reach_v1/best.pt \
    --mode actions
```

### 2. Fine-tuning on New Task

```bash
# Collect task-specific demos
python scripts/collect_demos.py \
    --robot franka \
    --task pick_place \
    --num_episodes 200 \
    --output_dir data/demos/pick_place

# Fine-tune from pretrained
python scripts/train.py \
    --config configs/training/finetune_config.yaml \
    --data_dir data/demos/pick_place \
    --resume checkpoints/pretrain/best.pt \
    --lr 5e-5 \
    --epochs 50
```

### 3. Multi-Robot Evaluation

```bash
# Evaluate same checkpoint on both robots
for robot in franka ur5e; do
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --robot $robot \
        --task reach \
        --num_episodes 50 \
        --headless
done
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Select GPU(s) for training/evaluation |
| `WANDB_API_KEY` | Weights & Biases API key for logging |
| `ISAACLAB_PATH` | Path to Isaac Lab installation |

```bash
# Example: Use GPU 1, enable wandb
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --data_dir data/demos --wandb_project vla
```

---

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Increase `--gradient_accumulation`
- Use `--headless` for evaluation

### Slow Training
- Increase `--num_workers` in dataloader
- Use mixed precision (enabled by default)
- Run on faster GPU

### Isaac Lab Issues
- Ensure Isaac Lab is properly installed
- Check NVIDIA driver compatibility
- Run `--headless` on remote servers
