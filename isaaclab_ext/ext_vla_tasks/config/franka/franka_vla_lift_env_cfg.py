"""
Franka VLA Lift Environment Configuration

Franka Panda robot lifting a cube with camera observations for VLA models.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ext_vla_tasks import mdp
from ext_vla_tasks.vla_lift_env_cfg import VLALiftEnvCfg

# Pre-defined configs
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class FrankaVLALiftEnvCfg(VLALiftEnvCfg):
    """Franka Panda VLA lift environment configuration."""

    def __post_init__(self):
        # Parent post init
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for Franka
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Set object (red cube)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.055],
                rot=[1, 0, 0, 0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # End-effector frame sensor
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )


@configclass
class FrankaVLALiftEnvCfg_PLAY(FrankaVLALiftEnvCfg):
    """Play/evaluation configuration with fewer environments."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play/evaluation
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Disable observation corruption
        self.observations.policy.enable_corruption = False

        # Longer episode for evaluation
        self.episode_length_s = 15.0
