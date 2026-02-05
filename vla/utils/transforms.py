"""
Action and pose transforms for VLA
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


class SE3Transform:
    """SE(3) rigid body transformations."""

    @staticmethod
    def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion [4] (w, x, y, z) or [N, 4]

        Returns:
            Rotation matrix [3, 3] or [N, 3, 3]
        """
        if quat.ndim == 1:
            quat = quat[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Rotation matrix elements
        r00 = 1 - 2 * (y**2 + z**2)
        r01 = 2 * (x*y - z*w)
        r02 = 2 * (x*z + y*w)
        r10 = 2 * (x*y + z*w)
        r11 = 1 - 2 * (x**2 + z**2)
        r12 = 2 * (y*z - x*w)
        r20 = 2 * (x*z - y*w)
        r21 = 2 * (y*z + x*w)
        r22 = 1 - 2 * (x**2 + y**2)

        rotmat = np.stack([
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ], axis=-2)

        if squeeze:
            rotmat = rotmat[0]

        return rotmat

    @staticmethod
    def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.

        Args:
            rotmat: Rotation matrix [3, 3] or [N, 3, 3]

        Returns:
            Quaternion [4] (w, x, y, z) or [N, 4]
        """
        if rotmat.ndim == 2:
            rotmat = rotmat[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        batch_size = rotmat.shape[0]
        quat = np.zeros((batch_size, 4))

        trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]

        for i in range(batch_size):
            R = rotmat[i]
            t = trace[i]

            if t > 0:
                s = 0.5 / np.sqrt(t + 1)
                quat[i, 0] = 0.25 / s
                quat[i, 1] = (R[2, 1] - R[1, 2]) * s
                quat[i, 2] = (R[0, 2] - R[2, 0]) * s
                quat[i, 3] = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
                quat[i, 0] = (R[2, 1] - R[1, 2]) / s
                quat[i, 1] = 0.25 * s
                quat[i, 2] = (R[0, 1] + R[1, 0]) / s
                quat[i, 3] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
                quat[i, 0] = (R[0, 2] - R[2, 0]) / s
                quat[i, 1] = (R[0, 1] + R[1, 0]) / s
                quat[i, 2] = 0.25 * s
                quat[i, 3] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
                quat[i, 0] = (R[1, 0] - R[0, 1]) / s
                quat[i, 1] = (R[0, 2] + R[2, 0]) / s
                quat[i, 2] = (R[1, 2] + R[2, 1]) / s
                quat[i, 3] = 0.25 * s

        if squeeze:
            quat = quat[0]

        return quat

    @staticmethod
    def euler_to_quat(euler: np.ndarray, order: str = "xyz") -> np.ndarray:
        """
        Convert Euler angles to quaternion.

        Args:
            euler: Euler angles [3] or [N, 3] in radians
            order: Rotation order (e.g., "xyz", "zyx")

        Returns:
            Quaternion [4] or [N, 4]
        """
        if euler.ndim == 1:
            euler = euler[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        angles = {}
        for i, axis in enumerate(order):
            angles[axis] = euler[:, i]

        # Compute quaternion components
        c = {k: np.cos(v / 2) for k, v in angles.items()}
        s = {k: np.sin(v / 2) for k, v in angles.items()}

        if order == "xyz":
            w = c['x']*c['y']*c['z'] + s['x']*s['y']*s['z']
            x = s['x']*c['y']*c['z'] - c['x']*s['y']*s['z']
            y = c['x']*s['y']*c['z'] + s['x']*c['y']*s['z']
            z = c['x']*c['y']*s['z'] - s['x']*s['y']*c['z']
        elif order == "zyx":
            w = c['z']*c['y']*c['x'] + s['z']*s['y']*s['x']
            x = c['z']*c['y']*s['x'] - s['z']*s['y']*c['x']
            y = c['z']*s['y']*c['x'] + s['z']*c['y']*s['x']
            z = s['z']*c['y']*c['x'] - c['z']*s['y']*s['x']
        else:
            raise NotImplementedError(f"Euler order {order} not implemented")

        quat = np.stack([w, x, y, z], axis=-1)

        if squeeze:
            quat = quat[0]

        return quat

    @staticmethod
    def pose_to_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Convert position and quaternion to 4x4 homogeneous matrix.

        Args:
            pos: Position [3] or [N, 3]
            quat: Quaternion [4] or [N, 4]

        Returns:
            Homogeneous matrix [4, 4] or [N, 4, 4]
        """
        rotmat = SE3Transform.quat_to_rotmat(quat)

        if pos.ndim == 1:
            matrix = np.eye(4)
            matrix[:3, :3] = rotmat
            matrix[:3, 3] = pos
        else:
            batch_size = pos.shape[0]
            matrix = np.zeros((batch_size, 4, 4))
            matrix[:, :3, :3] = rotmat
            matrix[:, :3, 3] = pos
            matrix[:, 3, 3] = 1

        return matrix


class ActionTransforms:
    """Transforms between different action representations."""

    @staticmethod
    def joint_pos_to_delta(
        current_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Convert absolute joint positions to delta actions.

        Args:
            current_pos: Current joint positions [N, action_dim]
            target_pos: Target joint positions [N, action_dim]

        Returns:
            Delta actions
        """
        return target_pos - current_pos

    @staticmethod
    def delta_to_joint_pos(
        current_pos: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """
        Convert delta actions to absolute joint positions.

        Args:
            current_pos: Current joint positions
            delta: Delta actions

        Returns:
            Target joint positions
        """
        return current_pos + delta

    @staticmethod
    def ee_pose_to_joint(
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        ik_solver,  # Inverse kinematics solver
    ) -> np.ndarray:
        """
        Convert end-effector pose to joint positions using IK.

        Args:
            ee_pos: End-effector position [3]
            ee_quat: End-effector orientation [4]
            ik_solver: IK solver instance

        Returns:
            Joint positions [action_dim]
        """
        # This is a placeholder - actual IK depends on robot
        return ik_solver.solve(ee_pos, ee_quat)

    @staticmethod
    def normalize_joint_actions(
        actions: np.ndarray,
        joint_limits: np.ndarray,  # [action_dim, 2] (lower, upper)
    ) -> np.ndarray:
        """
        Normalize joint actions to [-1, 1] range.

        Args:
            actions: Joint positions [N, action_dim]
            joint_limits: Joint limits

        Returns:
            Normalized actions
        """
        lower = joint_limits[:, 0]
        upper = joint_limits[:, 1]
        range_val = upper - lower

        normalized = 2 * (actions - lower) / range_val - 1
        return normalized

    @staticmethod
    def denormalize_joint_actions(
        actions: np.ndarray,
        joint_limits: np.ndarray,
    ) -> np.ndarray:
        """
        Denormalize joint actions from [-1, 1] to actual range.

        Args:
            actions: Normalized actions [N, action_dim]
            joint_limits: Joint limits

        Returns:
            Denormalized actions
        """
        lower = joint_limits[:, 0]
        upper = joint_limits[:, 1]
        range_val = upper - lower

        denormalized = (actions + 1) / 2 * range_val + lower
        return denormalized

    @staticmethod
    def velocity_to_position(
        velocities: np.ndarray,
        current_pos: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Integrate velocities to positions.

        Args:
            velocities: Joint velocities [T, action_dim]
            current_pos: Starting position [action_dim]
            dt: Time step

        Returns:
            Positions [T+1, action_dim]
        """
        positions = [current_pos]
        pos = current_pos.copy()

        for vel in velocities:
            pos = pos + vel * dt
            positions.append(pos.copy())

        return np.stack(positions)

    @staticmethod
    def smooth_actions(
        actions: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Smooth actions using moving average.

        Args:
            actions: Action sequence [T, action_dim]
            window_size: Smoothing window size

        Returns:
            Smoothed actions
        """
        if len(actions) < window_size:
            return actions

        # Pad for edge handling
        pad_size = window_size // 2
        padded = np.pad(
            actions,
            ((pad_size, pad_size), (0, 0)),
            mode='edge'
        )

        # Moving average
        smoothed = np.zeros_like(actions)
        for i in range(len(actions)):
            smoothed[i] = padded[i:i+window_size].mean(axis=0)

        return smoothed


class ActionChunker:
    """Handles action chunking for temporal action prediction."""

    def __init__(self, chunk_size: int = 16, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._buffer = None
        self._step = 0

    def reset(self):
        """Reset chunker state."""
        self._buffer = None
        self._step = 0

    def update(self, chunk: np.ndarray) -> np.ndarray:
        """
        Update with new action chunk and return current action.

        Uses temporal ensembling when chunks overlap.

        Args:
            chunk: New action chunk [chunk_size, action_dim]

        Returns:
            Current action [action_dim]
        """
        if self._buffer is None:
            self._buffer = chunk
            self._step = 0

        # Get current action
        action = self._buffer[self._step]

        # Advance step
        self._step += 1

        # Refill buffer when needed
        if self._step >= self.chunk_size - self.overlap:
            if self.overlap > 0:
                # Blend with new chunk
                old_remaining = self._buffer[self._step:]
                new_start = chunk[:len(old_remaining)]
                blended = (old_remaining + new_start) / 2
                self._buffer = np.concatenate([blended, chunk[len(old_remaining):]])
            else:
                self._buffer = chunk
            self._step = 0

        return action

    def get_current_chunk(self) -> Optional[np.ndarray]:
        """Get remaining actions in current chunk."""
        if self._buffer is None:
            return None
        return self._buffer[self._step:]
