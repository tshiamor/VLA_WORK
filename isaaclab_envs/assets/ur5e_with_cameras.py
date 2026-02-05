"""
UR5e Robot with Dual Camera Configuration

Configures a Universal Robots UR5e arm with:
- Context camera: Third-person view of the workspace
- Wrist camera: Eye-in-hand camera mounted on the end-effector
"""

from dataclasses import MISSING
from typing import Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR


class UR5eWithCameras:
    """UR5e robot with dual camera setup for VLA."""

    # Robot USD path
    USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR5e/ur5e_instanceable.usd"

    # Joint names (UR5e has 6 joints)
    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Default joint positions (home configuration)
    DEFAULT_JOINT_POS = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.5708,  # -90 degrees
        "elbow_joint": 1.5708,  # 90 degrees
        "wrist_1_joint": -1.5708,  # -90 degrees
        "wrist_2_joint": -1.5708,  # -90 degrees
        "wrist_3_joint": 0.0,
    }

    # End-effector frame
    EE_FRAME_NAME = "ee_link"

    # Camera settings
    CONTEXT_CAM_RESOLUTION = (224, 224)
    WRIST_CAM_RESOLUTION = (224, 224)


# Articulation configuration
UR5E_ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5eWithCameras.USD_PATH,
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
        joint_pos=UR5eWithCameras.DEFAULT_JOINT_POS,
    ),
    actuators={
        "ur5e_arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            velocity_limit=3.14,  # rad/s
            effort_limit=150.0,  # Nm
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


# Context camera configuration (third-person view)
UR5E_CONTEXT_CAMERA_CFG = CameraCfg(
    prim_path="/World/ContextCamera",
    update_period=0.0,
    height=UR5eWithCameras.CONTEXT_CAM_RESOLUTION[0],
    width=UR5eWithCameras.CONTEXT_CAM_RESOLUTION[1],
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.2),
        rot=(0.9239, 0.0, 0.3827, 0.0),
        convention="world",
    ),
)


# Wrist camera configuration (eye-in-hand)
UR5E_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/ee_link/WristCamera",
    update_period=0.0,
    height=UR5eWithCameras.WRIST_CAM_RESOLUTION[0],
    width=UR5eWithCameras.WRIST_CAM_RESOLUTION[1],
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 5.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.05),  # Offset from EE frame
        rot=(0.7071, 0.0, 0.7071, 0.0),  # Looking forward
        convention="ros",
    ),
)


# Complete configuration with cameras
class UR5E_WITH_CAMERAS_CFG:
    """Complete UR5e configuration with cameras."""

    robot = UR5E_ARTICULATION_CFG
    context_camera = UR5E_CONTEXT_CAMERA_CFG
    wrist_camera = UR5E_WRIST_CAMERA_CFG

    # Action space info (6-DOF arm + 1 gripper = 7)
    action_dim = 6  # 6 arm joints (gripper handled separately if attached)
    proprio_dim = 12  # 6 positions + 6 velocities

    # Joint limits (radians)
    joint_limits = {
        "shoulder_pan_joint": (-6.2832, 6.2832),  # ±2π
        "shoulder_lift_joint": (-6.2832, 6.2832),
        "elbow_joint": (-3.1416, 3.1416),  # ±π
        "wrist_1_joint": (-6.2832, 6.2832),
        "wrist_2_joint": (-6.2832, 6.2832),
        "wrist_3_joint": (-6.2832, 6.2832),
    }

    # Velocity limits (rad/s)
    velocity_limits = {
        "shoulder_pan_joint": 3.14,
        "shoulder_lift_joint": 3.14,
        "elbow_joint": 3.14,
        "wrist_1_joint": 3.14,
        "wrist_2_joint": 3.14,
        "wrist_3_joint": 3.14,
    }


def create_ur5e_with_cameras(
    robot_prim_path: str = "/World/Robot",
    context_cam_pos: tuple = (1.5, 0.0, 1.2),
    context_cam_rot: tuple = (0.9239, 0.0, 0.3827, 0.0),
    enable_depth: bool = True,
) -> dict:
    """
    Create UR5e robot configuration with customizable camera placement.

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
        prim_path=f"{robot_prim_path}/ee_link/WristCamera",
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
            pos=(0.0, 0.0, 0.05),
            rot=(0.7071, 0.0, 0.7071, 0.0),
            convention="ros",
        ),
    )

    return {
        "robot": UR5E_ARTICULATION_CFG,
        "context_camera": context_cam,
        "wrist_camera": wrist_cam,
    }


# UR5e with Robotiq 2F-85 gripper configuration
class UR5eRobotiq85:
    """UR5e with Robotiq 2F-85 gripper for manipulation tasks."""

    # Note: This requires the Robotiq gripper USD to be available
    USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR5e/ur5e_robotiq_2f85.usd"

    GRIPPER_JOINT_NAMES = [
        "finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    ]

    # Combined action dim (6 arm + 1 gripper)
    ACTION_DIM = 7


def create_ur5e_robotiq_cfg() -> ArticulationCfg:
    """Create UR5e with Robotiq 2F-85 gripper configuration."""
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=UR5eRobotiq85.USD_PATH,
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
            joint_pos={
                **UR5eWithCameras.DEFAULT_JOINT_POS,
                "finger_joint": 0.0,  # Gripper open
            },
        ),
        actuators={
            "ur5e_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                velocity_limit=3.14,
                effort_limit=150.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "robotiq_gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                velocity_limit=0.2,
                effort_limit=50.0,
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )
