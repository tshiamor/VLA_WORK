"""
Franka Panda Robot with Dual Camera Configuration

Configures a Franka Panda arm with:
- Context camera: Third-person view of the workspace
- Wrist camera: Eye-in-hand camera mounted on the end-effector
"""

from dataclasses import MISSING
from typing import Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR


class FrankaWithCameras:
    """Franka Panda robot with dual camera setup for VLA."""

    # Robot USD path
    USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"

    # Joint names
    ARM_JOINT_NAMES = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    GRIPPER_JOINT_NAMES = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    # Default joint positions (home configuration)
    DEFAULT_JOINT_POS = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    # End-effector frame
    EE_FRAME_NAME = "panda_hand"

    # Camera settings
    CONTEXT_CAM_RESOLUTION = (224, 224)
    WRIST_CAM_RESOLUTION = (224, 224)


# Articulation configuration
FRANKA_ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FrankaWithCameras.USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=FrankaWithCameras.DEFAULT_JOINT_POS,
    ),
    actuators={
        "panda_arm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            velocity_limit=100.0,
            effort_limit={
                "panda_joint[1-4]": 87.0,
                "panda_joint[5-7]": 12.0,
            },
            stiffness={
                "panda_joint[1-4]": 80.0,
                "panda_joint[5-7]": 40.0,
            },
            damping={
                "panda_joint[1-4]": 4.0,
                "panda_joint[5-7]": 2.0,
            },
        ),
        "panda_gripper": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            velocity_limit=0.2,
            effort_limit=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)


# Context camera configuration (third-person view)
CONTEXT_CAMERA_CFG = CameraCfg(
    prim_path="/World/ContextCamera",
    update_period=0.0,  # Update every frame
    height=FrankaWithCameras.CONTEXT_CAM_RESOLUTION[0],
    width=FrankaWithCameras.CONTEXT_CAM_RESOLUTION[1],
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.2),  # Position in front and above workspace
        rot=(0.9239, 0.0, 0.3827, 0.0),  # Looking down at workspace
        convention="world",
    ),
)


# Wrist camera configuration (eye-in-hand)
WRIST_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
    update_period=0.0,
    height=FrankaWithCameras.WRIST_CAM_RESOLUTION[0],
    width=FrankaWithCameras.WRIST_CAM_RESOLUTION[1],
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 5.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.05),  # Offset from hand frame
        rot=(0.7071, 0.0, 0.7071, 0.0),  # Looking forward
        convention="ros",
    ),
)


# Complete configuration with cameras
class FRANKA_WITH_CAMERAS_CFG:
    """Complete Franka configuration with cameras."""

    robot = FRANKA_ARTICULATION_CFG
    context_camera = CONTEXT_CAMERA_CFG
    wrist_camera = WRIST_CAMERA_CFG

    # Action space info
    action_dim = 7  # 7 arm joints
    proprio_dim = 14  # 7 positions + 7 velocities

    # Joint limits
    joint_limits = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
    }

    # Velocity limits (rad/s)
    velocity_limits = {
        "panda_joint1": 2.1750,
        "panda_joint2": 2.1750,
        "panda_joint3": 2.1750,
        "panda_joint4": 2.1750,
        "panda_joint5": 2.6100,
        "panda_joint6": 2.6100,
        "panda_joint7": 2.6100,
    }


def create_franka_with_cameras(
    robot_prim_path: str = "/World/Robot",
    context_cam_pos: tuple = (1.5, 0.0, 1.2),
    context_cam_rot: tuple = (0.9239, 0.0, 0.3827, 0.0),
    enable_depth: bool = True,
) -> dict:
    """
    Create Franka robot configuration with customizable camera placement.

    Args:
        robot_prim_path: USD path for robot
        context_cam_pos: Position of context camera
        context_cam_rot: Rotation of context camera (quaternion)
        enable_depth: Whether to capture depth images

    Returns:
        Dictionary with robot and camera configurations
    """
    data_types = ["rgb", "depth"] if enable_depth else ["rgb"]

    # Update context camera position
    context_cam = CameraCfg(
        prim_path="/World/ContextCamera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=context_cam_pos,
            rot=context_cam_rot,
            convention="world",
        ),
    )

    # Wrist camera attached to robot
    wrist_cam = CameraCfg(
        prim_path=f"{robot_prim_path}/panda_hand/WristCamera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.05),
            rot=(0.7071, 0.0, 0.7071, 0.0),
            convention="ros",
        ),
    )

    return {
        "robot": FRANKA_ARTICULATION_CFG,
        "context_camera": context_cam,
        "wrist_camera": wrist_cam,
    }
