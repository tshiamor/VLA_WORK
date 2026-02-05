"""
Action specifications for VLA tasks.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

# Re-export standard action configs from Isaac Lab
from isaaclab.envs.mdp.actions import (
    JointPositionActionCfg,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)


@configclass
class VLAJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action config for VLA.

    Wraps standard JointPositionActionCfg with VLA-specific defaults.
    """

    scale: float = 0.5
    use_default_offset: bool = True


@configclass
class VLAGripperActionCfg(BinaryJointPositionActionCfg):
    """Gripper action config for VLA.

    Binary open/close gripper control.
    """

    pass
