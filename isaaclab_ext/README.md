# VLA Isaac Lab Extension

External Isaac Lab project for Vision-Language-Action tasks with camera observations.

## Overview

This extension provides:
- **VLA-Franka-Lift-v0**: Franka Panda cube lifting task with dual cameras
- Camera observations: Context (third-person) + Wrist (eye-in-hand)
- Integration with VLA_WORK training and inference pipeline

## Setup

### 1. Set Environment Variables

```bash
# Point to your Isaac Lab installation
export ISAACLAB_PATH=/home/tshiamo/IsaacLab

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab"
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab_tasks"
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab_assets"
export PYTHONPATH="${PYTHONPATH}:/home/tshiamo/VLA_WORK"
export PYTHONPATH="${PYTHONPATH}:/home/tshiamo/VLA_WORK/isaaclab_ext"
```

### 2. Install the Extension

```bash
cd /home/tshiamo/VLA_WORK/isaaclab_ext
pip install -e .
```

### 3. Verify Installation

```bash
# List registered environments
python -c "import gymnasium as gym; import ext_vla_tasks; print([e for e in gym.envs.registry.keys() if 'VLA' in e])"
```

Expected output:
```
['VLA-Franka-Lift-v0', 'VLA-Franka-Lift-Play-v0']
```

## Usage

### Run Demo (Scripted Policy)

```bash
cd /home/tshiamo/VLA_WORK/isaaclab_ext/scripts

# With visualization
python run_vla_lift.py --mode demo --num_episodes 5

# Headless
python run_vla_lift.py --mode demo --num_episodes 5 --headless
```

### Collect Demonstrations

```bash
python run_vla_lift.py --mode collect \
    --output_dir /home/tshiamo/VLA_WORK/data/demos/lift \
    --num_episodes 100 \
    --headless
```

### Run with Trained VLA Model

```bash
python run_vla_lift.py --mode vla \
    --checkpoint /home/tshiamo/VLA_WORK/checkpoints/best.pt \
    --num_episodes 50
```

## Environment Details

### VLA-Franka-Lift-v0

| Property | Value |
|----------|-------|
| Robot | Franka Panda |
| Task | Lift cube above table |
| Action Space | 8-dim (7 joints + gripper) |
| Observation | Policy (proprio) + Camera (RGB) |
| Episode Length | 10s (1000 steps @ 100Hz) |

### Observations

```python
obs = {
    "policy": {
        "joint_pos": [num_envs, 9],      # Joint positions
        "joint_vel": [num_envs, 9],      # Joint velocities
        "object_position": [num_envs, 3], # Object pos in robot frame
        "actions": [num_envs, 8],         # Last action
    },
    "camera": {
        "context_rgb": [num_envs, 224, 224, 3],  # Third-person view
        "wrist_rgb": [num_envs, 224, 224, 3],    # Eye-in-hand view
    }
}
```

### Actions

```python
action = [
    joint_1_delta,  # Shoulder pan
    joint_2_delta,  # Shoulder lift
    joint_3_delta,  # Elbow
    joint_4_delta,  # Wrist 1
    joint_5_delta,  # Wrist 2
    joint_6_delta,  # Wrist 3
    joint_7_delta,  # Wrist 4
    gripper,        # -1 (close) to 1 (open)
]
```

### Rewards

| Reward | Weight | Description |
|--------|--------|-------------|
| `reaching_object` | 1.0 | Gaussian on EE-object distance |
| `lifting_object` | 10.0 | Bonus when object > 6cm height |
| `action_rate` | -1e-3 | Penalty for jerky motions |

## Project Structure

```
isaaclab_ext/
├── setup.py                          # Package setup
├── README.md                         # This file
├── ext_vla_tasks/
│   ├── __init__.py                   # Package init, registers envs
│   ├── vla_lift_env_cfg.py           # Base VLA lift environment
│   ├── mdp/
│   │   ├── __init__.py
│   │   ├── observations.py           # Camera + proprio observations
│   │   ├── actions.py                # Action configs
│   │   ├── rewards.py                # Reward functions
│   │   ├── events.py                 # Reset events
│   │   └── terminations.py           # Termination conditions
│   └── config/
│       └── franka/
│           ├── __init__.py           # Registers Franka envs
│           └── franka_vla_lift_env_cfg.py
└── scripts/
    └── run_vla_lift.py               # Main run script
```

## Training Workflow

### 1. Collect Demonstrations

```bash
# Collect 500 demos
python isaaclab_ext/scripts/run_vla_lift.py \
    --mode collect \
    --output_dir data/demos/lift_train \
    --num_episodes 500 \
    --headless

# Collect validation set
python isaaclab_ext/scripts/run_vla_lift.py \
    --mode collect \
    --output_dir data/demos/lift_val \
    --num_episodes 100 \
    --headless
```

### 2. Train VLA Model

```bash
python scripts/train.py \
    --data_dir data/demos/lift_train \
    --val_data_dir data/demos/lift_val \
    --action_dim 8 \
    --epochs 100 \
    --checkpoint_dir checkpoints/lift
```

### 3. Evaluate

```bash
python isaaclab_ext/scripts/run_vla_lift.py \
    --mode vla \
    --checkpoint checkpoints/lift/best.pt \
    --num_episodes 100 \
    --headless
```

## Troubleshooting

### "No module named 'ext_vla_tasks'"

Ensure PYTHONPATH includes the isaaclab_ext directory:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/tshiamo/VLA_WORK/isaaclab_ext"
```

### "No module named 'isaaclab'"

Ensure Isaac Lab is properly installed and PYTHONPATH is set:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/tshiamo/IsaacLab/source/isaaclab"
```

### Camera images are black

- Check lighting configuration in scene
- Ensure `update_period` is set correctly on cameras
- Try increasing `render_interval` in sim config

### GPU Memory Issues

- Reduce `num_envs` to 1
- Use `--headless` mode
- Reduce image resolution in camera config
