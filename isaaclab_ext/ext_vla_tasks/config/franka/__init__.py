"""Franka VLA task configurations."""

import gymnasium as gym

from .franka_vla_lift_env_cfg import FrankaVLALiftEnvCfg

##
# Register Gym environments
##

gym.register(
    id="VLA-Franka-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaVLALiftEnvCfg,
    },
)

gym.register(
    id="VLA-Franka-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_vla_lift_env_cfg:FrankaVLALiftEnvCfg_PLAY",
    },
)
