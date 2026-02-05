"""
VLA Lift Environment Configuration

A cube lifting task with camera observations for Vision-Language-Action models.
Extends the standard Isaac Lab lift task with:
- Context camera (third-person view)
- Wrist camera (eye-in-hand)
- Language instruction interface
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition with cameras
##

@configclass
class VLALiftSceneCfg(InteractiveSceneCfg):
    """Configuration for VLA lift scene with robot, object, and cameras."""

    # Robot (will be populated by specific config)
    robot: ArticulationCfg = MISSING

    # End-effector frame sensor
    ee_frame: FrameTransformerCfg = MISSING

    # Object to lift
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Context camera (third-person view)
    context_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ContextCamera",
        update_period=0.1,  # 10 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, 0.5, 0.8),  # Front-right view of workspace
            rot=(0.85, -0.15, 0.35, 0.35),  # Looking at table center
            convention="world",
        ),
    )

    # Wrist camera (eye-in-hand, attached to robot hand)
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.04),  # Slightly forward and down from hand
            rot=(0.5, 0.5, -0.5, -0.5),  # Looking forward/down
            convention="ros",
        ),
    )


##
# MDP Settings
##

@configclass
class VLAActionsCfg:
    """Action specifications for VLA."""
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class VLAObservationsCfg:
    """Observation specifications for VLA with cameras."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations for policy."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CameraCfg(ObsGroup):
        """Camera observations for VLA."""
        context_rgb = ObsTerm(
            func=mdp.camera_rgb,
            params={"sensor_cfg": SceneEntityCfg("context_camera")},
        )
        wrist_rgb = ObsTerm(
            func=mdp.camera_rgb,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera: CameraCfg = CameraCfg()


@configclass
class VLAEventCfg:
    """Event configuration for VLA environment."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class VLARewardsCfg:
    """Reward terms for VLA lift task."""

    # Reaching the object
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=1.0,
    )

    # Lifting the object
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},
        weight=10.0,
    )

    # Smooth actions
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class VLATerminationsCfg:
    """Termination conditions for VLA environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Environment Configuration
##

@configclass
class VLALiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for VLA lift environment."""

    # Scene settings
    scene: VLALiftSceneCfg = VLALiftSceneCfg(num_envs=1, env_spacing=2.5)

    # MDP settings
    observations: VLAObservationsCfg = VLAObservationsCfg()
    actions: VLAActionsCfg = VLAActionsCfg()
    rewards: VLARewardsCfg = VLARewardsCfg()
    terminations: VLATerminationsCfg = VLATerminationsCfg()
    events: VLAEventCfg = VLAEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz physics
        self.sim.render_interval = self.decimation

        # PhysX settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
