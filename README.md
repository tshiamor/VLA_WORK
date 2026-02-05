# VLA_WORK: Vision-Language-Action for Robot Manipulation

A Vision-Language-Action (VLA) system combining **Qwen 2.5-VL-7B** with a flow matching action expert for robot manipulation, evaluated on Franka Panda and UR5e arms in Isaac Lab.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VLA Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐                               │
│  │ Context Cam  │    │  Wrist Cam   │                               │
│  └──────┬───────┘    └──────┬───────┘                               │
│         │                   │                                       │
│         ▼                   ▼                                       │
│  ┌─────────────────────────────────────┐                            │
│  │     Qwen 2.5-VL (Frozen/LoRA)       │◄── Language Instruction    │
│  │   Vision Encoder + Language Model   │                            │
│  └─────────────────┬───────────────────┘                            │
│                    │ Visual-Language Embeddings                     │
│                    ▼                                                │
│  ┌─────────────────────────────────────┐                            │
│  │       Projection Layer (MLP)        │                            │
│  └─────────────────┬───────────────────┘                            │
│                    │                                                │
│                    ▼                                                │
│  ┌─────────────────────────────────────┐                            │
│  │    Flow Matching Action Expert      │                            │
│  │  - Transformer Decoder Blocks       │                            │
│  │  - Continuous Action Output         │                            │
│  │  - Action Chunking (16 steps)       │                            │
│  └─────────────────┬───────────────────┘                            │
│                    │                                                │
│                    ▼                                                │
│            [7-DOF Joint Actions]                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **Vision-Language Model**: Qwen 2.5-VL-7B-Instruct with frozen backbone and LoRA adapters
- **Flow Matching**: Continuous action generation using conditional flow matching
- **Action Chunking**: Predicts 16-step action sequences for temporal consistency
- **Dual Camera Input**: Context camera (third-person) + wrist camera (eye-in-hand)
- **Isaac Lab Integration**: Ready-to-use environments for Franka Panda and UR5e

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ with compatible GPU (24GB+ VRAM recommended)
- Isaac Lab (already installed per project requirements)

### Setup

```bash
# Clone the repository
cd VLA_WORK

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Verify Qwen 2.5-VL Installation

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("Qwen 2.5-VL loaded successfully!")
```

## Quick Start

### 1. Collect Demonstrations

```bash
# Collect demos using scripted policy
python scripts/collect_demos.py \
    --robot franka \
    --task reach \
    --num_episodes 100 \
    --output_dir data/demos
```

### 2. Train VLA Model

```bash
# Train with default config
python scripts/train.py \
    --data_dir data/demos \
    --epochs 100 \
    --batch_size 32

# Or use config file
python scripts/train.py --config configs/training/pretrain_config.yaml
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --robot franka \
    --task reach \
    --num_episodes 50
```

### 4. Visualize

```bash
python scripts/visualize.py \
    --checkpoint checkpoints/best.pt \
    --mode actions
```

## Project Structure

```
VLA_WORK/
├── vla/                          # Core VLA library
│   ├── models/                   # Model components
│   │   ├── vision_encoder.py     # Qwen 2.5-VL wrapper
│   │   ├── flow_matching.py      # Flow matching head
│   │   ├── action_expert.py      # Transformer action decoder
│   │   ├── projector.py          # VLM→Action projection
│   │   └── vla_model.py          # Combined VLA model
│   ├── data/                     # Data utilities
│   ├── training/                 # Training components
│   └── utils/                    # Utilities
├── isaaclab_envs/                # Isaac Lab environments
│   ├── envs/                     # Robot environments
│   ├── assets/                   # Robot + camera configs
│   └── tasks/                    # Task definitions
├── configs/                      # Configuration files
├── scripts/                      # Training/evaluation scripts
└── tutorials/                    # Learning materials
```

## Configuration

### Model Configuration

```yaml
# configs/model/vla_config.yaml
vision_encoder:
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  freeze_backbone: true
  use_lora: true
  lora_r: 16

action_expert:
  hidden_dim: 512
  num_layers: 6
  num_heads: 8

flow_matching:
  action_dim: 7
  chunk_size: 16
  num_steps: 100
```

### Training Configuration

```yaml
# configs/training/pretrain_config.yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1.0e-4
  warmup_steps: 1000
```

## API Usage

### Basic Inference

```python
from vla.models import create_vla_model
import numpy as np

# Create model
model = create_vla_model(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    action_dim=7,
    chunk_size=16,
)

# Load checkpoint
model.load_checkpoint("checkpoints/best.pt")

# Predict actions
context_img = np.random.rand(224, 224, 3)  # Your camera image
wrist_img = np.random.rand(224, 224, 3)
instruction = "Pick up the red cube and place it on the green marker."

actions = model.predict_action(
    images=[context_img, wrist_img],
    instruction=instruction,
)
# actions shape: [1, 16, 7] - 16 timesteps, 7 DOF
```

### Training Loop

```python
from vla.models import create_vla_model
from vla.training import VLATrainer, TrainerConfig
from vla.data import create_dataloader

model = create_vla_model(action_dim=7, chunk_size=16)
train_loader = create_dataloader("data/demos", batch_size=32)

config = TrainerConfig(
    learning_rate=1e-4,
    num_epochs=100,
    checkpoint_dir="checkpoints",
)

trainer = VLATrainer(model, config, train_loader)
trainer.train()
```

## References

- [Qwen 2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Pi0 Architecture](https://www.pi.website/download/pi05.pdf)
- [OpenVLA](https://github.com/openvla/openvla)
- [Flow Matching](https://github.com/facebookresearch/flow_matching)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

## License

MIT License - see LICENSE file for details.
