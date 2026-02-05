# VLA Isaac Lab Extension - Quick Start

## Project Structure

```
VLA_WORK/isaaclab_ext/
├── setup.py                              # Package installer
├── README.md                             # Full documentation
├── QUICKSTART.md                         # This file
├── ext_vla_tasks/
│   ├── vla_lift_env_cfg.py              # Base env with cameras
│   ├── mdp/
│   │   ├── observations.py              # Camera RGB observations
│   │   ├── rewards.py                   # Reach + lift rewards
│   │   ├── events.py                    # Reset logic
│   │   └── terminations.py              # Success/failure checks
│   └── config/franka/
│       └── franka_vla_lift_env_cfg.py   # Franka-specific config
└── scripts/
    └── run_vla_lift.py                  # Main entry point
```

---

## Quick Start

### 1. Setup Environment

```bash
# Set paths
export ISAACLAB_PATH=/home/tshiamo/IsaacLab
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab"
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab_tasks"
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab_assets"
export PYTHONPATH="${PYTHONPATH}:/home/tshiamo/VLA_WORK"
export PYTHONPATH="${PYTHONPATH}:/home/tshiamo/VLA_WORK/isaaclab_ext"

# Install extension
cd /home/tshiamo/VLA_WORK/isaaclab_ext
pip install -e .
```

### 2. Run Demo (test the environment)

```bash
cd /home/tshiamo/VLA_WORK/isaaclab_ext/scripts
python run_vla_lift.py --mode demo --num_episodes 3
```

### 3. Collect Training Data

```bash
python run_vla_lift.py --mode collect \
    --output_dir /home/tshiamo/VLA_WORK/data/demos/lift \
    --num_episodes 100 \
    --headless
```

### 4. Train VLA Model

```bash
cd /home/tshiamo/VLA_WORK
python scripts/train.py \
    --data_dir data/demos/lift \
    --action_dim 8 \
    --epochs 50
```

### 5. Evaluate with VLA

```bash
python isaaclab_ext/scripts/run_vla_lift.py \
    --mode vla \
    --checkpoint checkpoints/best.pt \
    --num_episodes 20
```

---

## Task Details

```
┌─────────┬─────────────────────────────────────┐
│ Feature │                Value                │
├─────────┼─────────────────────────────────────┤
│ Task    │ Lift red cube above table           │
├─────────┼─────────────────────────────────────┤
│ Robot   │ Franka Panda                        │
├─────────┼─────────────────────────────────────┤
│ Cameras │ Context (224×224) + Wrist (224×224) │
├─────────┼─────────────────────────────────────┤
│ Actions │ 7 joint positions + 1 gripper       │
├─────────┼─────────────────────────────────────┤
│ Success │ Object height > 15cm                │
└─────────┴─────────────────────────────────────┘
```

---

## One-Liner Setup

Copy and paste this to set up everything at once:

```bash
export ISAACLAB_PATH=/home/tshiamo/IsaacLab && \
export PYTHONPATH="${PYTHONPATH}:${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_assets:/home/tshiamo/VLA_WORK:/home/tshiamo/VLA_WORK/isaaclab_ext" && \
cd /home/tshiamo/VLA_WORK/isaaclab_ext && pip install -e .
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: isaaclab` | Set PYTHONPATH as shown above |
| `ModuleNotFoundError: ext_vla_tasks` | Run `pip install -e .` in isaaclab_ext/ |
| Black/empty camera images | Check lighting config, try `--headless` |
| GPU OOM | Reduce `num_envs` to 1 |

---

For full documentation, see [README.md](./README.md)
