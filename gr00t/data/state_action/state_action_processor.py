# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative action representation conversion
- Unitree root relative 6D via Root2Rot6d (WBC robot_root / SMPL frame quat)
- Unitree root Euler via Root2Euler (quat→xyz euler; optional delta vs state)
- Action processing with state dependency
"""

from copy import deepcopy
import logging

from gr00t.configs.data.embodiment_configs import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.utils import (
    apply_sin_cos_encoding,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
import numpy as np
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)


class Root2Rot6d:
    """Convert a Unitree whole-body action between 36D absolute and 38D relative form.

    Raw action layout:
        xyz(3) + quaternion_wxyz(4)

    Processed action layout:
        local_delta_xyz(3) + relative_rotation_6d(6)

    The 6D convention matches PyTorch3D: the first two rows of the rotation
    matrix are flattened. The reference root is supplied separately and is not
    stored inside the processed action.
    """

    RAW_DIM = 7
    PROCESSED_DIM = 9
    ROOT_RAW_DIM = 7
    ROOT_PROCESSED_DIM = 9
    JOINT_DIM = 0
    EPS = 1e-8

    # robot_root (WBC 7/9D) and frame (SMPL 82/84D; body ori at [72:76]/[72:78]).
    ACTION_KEYS = frozenset({"robot_root", "frame"})
    STATE_KEY_CANDIDATES = {
        "robot_root": ("robot_root", "robot_root_current"),
        "frame": ("robot_root", "robot_root_current"),
    }
    FRAME_RAW_DIM = 82
    FRAME_PROCESSED_DIM = 84

    @classmethod
    def is_action_key(cls, key: str) -> bool:
        return key in cls.ACTION_KEYS

    @classmethod
    def pack_frame_root(cls, frame: np.ndarray, *, relative: bool = False) -> np.ndarray:
        """getitem body ori from SMPL frame into a 7D/9D root (xyz unused)."""
        if relative:
            root = np.zeros((frame.shape[0], cls.PROCESSED_DIM), dtype=frame.dtype)
            root[:, 3:9] = frame[:, 72:78]
        else:
            root = np.zeros((frame.shape[0], cls.RAW_DIM), dtype=frame.dtype)
            root[:, 3:7] = frame[:, 72:76]
        return root

    @classmethod
    def splice_frame_root(
        cls, frame: np.ndarray, root: np.ndarray, *, relative: bool = False
    ) -> np.ndarray:
        """Put root ori back into SMPL frame (72 + ori + wrist)."""
        if relative:
            return np.concatenate((frame[:, :72], root[:, 3:9], frame[:, 76:82]), axis=-1)
        return np.concatenate((frame[:, :72], root[:, 3:7], frame[:, 78:84]), axis=-1)

    @classmethod
    def _normalize_quaternion(cls, quaternion: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion)
        if quaternion.shape[-1] != 4:
            raise ValueError(f"Expected quaternion (..., 4), got {quaternion.shape}")
        if not np.all(np.isfinite(quaternion)):
            raise ValueError("Quaternion contains NaN or Inf")
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        if np.any(norm < cls.EPS):
            raise ValueError("Quaternion norm is too close to zero")
        return quaternion / norm

    @classmethod
    def quaternion_to_matrix(cls, quaternion_wxyz: np.ndarray) -> np.ndarray:
        q = cls._normalize_quaternion(quaternion_wxyz)
        w, x, y, z = np.moveaxis(q, -1, 0)

        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        row0 = np.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), axis=-1)
        row1 = np.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), axis=-1)
        row2 = np.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), axis=-1)
        return np.stack((row0, row1, row2), axis=-2)

    @staticmethod
    def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Expected rotation matrix (..., 3, 3), got {matrix.shape}")
        return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)

    @classmethod
    def rotation_6d_to_matrix(cls, rotation_6d: np.ndarray) -> np.ndarray:
        rotation_6d = np.asarray(rotation_6d)
        if rotation_6d.shape[-1] != 6:
            raise ValueError(f"Expected rotation 6D (..., 6), got {rotation_6d.shape}")

        a1 = rotation_6d[..., :3]
        a2 = rotation_6d[..., 3:6]

        norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
        fallback1 = np.zeros_like(a1)
        fallback1[..., 0] = 1.0
        b1 = np.where(norm1 > cls.EPS, a1 / np.maximum(norm1, cls.EPS), fallback1)

        projection = np.sum(b1 * a2, axis=-1, keepdims=True)
        orthogonal = a2 - projection * b1
        norm2 = np.linalg.norm(orthogonal, axis=-1, keepdims=True)

        axis_index = np.argmin(np.abs(b1), axis=-1)
        fallback_axis = np.eye(3, dtype=rotation_6d.dtype)[axis_index]
        fallback_orthogonal = fallback_axis - (
            np.sum(fallback_axis * b1, axis=-1, keepdims=True) * b1
        )
        fallback_orthogonal /= np.maximum(
            np.linalg.norm(fallback_orthogonal, axis=-1, keepdims=True), cls.EPS
        )
        b2 = np.where(
            norm2 > cls.EPS,
            orthogonal / np.maximum(norm2, cls.EPS),
            fallback_orthogonal,
        )
        b3 = np.cross(b1, b2)
        return np.stack((b1, b2, b3), axis=-2)

    @classmethod
    def matrix_to_quaternion(cls, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Expected rotation matrix (..., 3, 3), got {matrix.shape}")

        flat = matrix.reshape(-1, 3, 3)
        quaternions = np.empty((flat.shape[0], 4), dtype=matrix.dtype)
        for index, rotation in enumerate(flat):
            trace = float(np.trace(rotation))
            if trace > 0.0:
                scale = np.sqrt(trace + 1.0) * 2.0
                qw = 0.25 * scale
                qx = (rotation[2, 1] - rotation[1, 2]) / scale
                qy = (rotation[0, 2] - rotation[2, 0]) / scale
                qz = (rotation[1, 0] - rotation[0, 1]) / scale
            elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
                scale = np.sqrt(
                    max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)
                ) * 2.0
                scale = max(scale, cls.EPS)
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif rotation[1, 1] > rotation[2, 2]:
                scale = np.sqrt(
                    max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)
                ) * 2.0
                scale = max(scale, cls.EPS)
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(
                    max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)
                ) * 2.0
                scale = max(scale, cls.EPS)
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
            quaternions[index] = (qw, qx, qy, qz)

        quaternions = cls._normalize_quaternion(quaternions)
        return quaternions.reshape(*matrix.shape[:-2], 4)

    @classmethod
    def to_relative(
        cls,
        action: np.ndarray,
        reference_state: np.ndarray | None = None,
        *,
        process_xyz: bool = False,
        relative: bool = True,
    ) -> np.ndarray:
        """Convert absolute root (T, 7) to processed (T, 9).

        ``relative=True`` (default): local Δxyz + rot6d of ``R_ref^T R_act``.
        ``relative=False``: xyz zeros (unless process_xyz with a reference) +
        absolute rot6d of ``R_act`` (uniJungle-style; reference optional).
        """
        action = np.asarray(action)

        if action.ndim != 2 or action.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected Unitree action (T, {cls.RAW_DIM}), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Unitree action contains NaN or Inf")

        future_rotation = cls.quaternion_to_matrix(action[:, 3:7])

        if relative:
            if reference_state is None:
                raise ValueError("reference_state is required when relative=True")
            reference_state = np.asarray(reference_state)
            if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
                raise ValueError(
                    f"Expected reference state ({cls.RAW_DIM},), "
                    f"got {reference_state.shape}"
                )
            if not np.all(np.isfinite(reference_state)):
                raise ValueError("Unitree reference state contains NaN or Inf")

            reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
            reference_rotation = cls.quaternion_to_matrix(reference_quaternion)
            if process_xyz:
                reference_position = reference_state[:3]
                future_position = action[:, :3]
                world_delta = future_position - reference_position
                local_delta = np.einsum("ij,tj->ti", reference_rotation.T, world_delta)
            else:
                local_delta = np.zeros((action.shape[0], 3), dtype=action.dtype)
            out_rotation = np.einsum(
                "ij,tjk->tik", reference_rotation.T, future_rotation
            )
        else:
            if process_xyz:
                if reference_state is None:
                    raise ValueError(
                        "reference_state is required when process_xyz=True"
                    )
                reference_state = np.asarray(reference_state)
                reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
                reference_rotation = cls.quaternion_to_matrix(reference_quaternion)
                world_delta = action[:, :3] - reference_state[:3]
                local_delta = np.einsum("ij,tj->ti", reference_rotation.T, world_delta)
            else:
                local_delta = np.zeros((action.shape[0], 3), dtype=action.dtype)
            out_rotation = future_rotation

        rotation_6d = cls.matrix_to_rotation_6d(out_rotation)
        return np.concatenate((local_delta, rotation_6d), axis=-1)

    @classmethod
    def to_absolute(
        cls,
        action: np.ndarray,
        reference_state: np.ndarray | None = None,
        *,
        process_xyz: bool = False,
        relative: bool = True,
    ) -> np.ndarray:
        """Convert processed root (T, 9) back to absolute (T, 7).

        ``relative=True``: apply ``R_ref`` to recover world orientation (needs ref).
        ``relative=False``: rot6d→quat directly; xyz from ref if given else zeros.
        """
        action = np.asarray(action)

        if action.ndim != 2 or action.shape[-1] != cls.PROCESSED_DIM:
            raise ValueError(
                f"Expected processed Unitree action "
                f"(T, {cls.PROCESSED_DIM}), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Processed Unitree action contains NaN or Inf")

        processed_rotation = cls.rotation_6d_to_matrix(action[:, 3:9])

        if relative:
            if reference_state is None:
                raise ValueError("reference_state is required when relative=True")
            reference_state = np.asarray(reference_state)
            if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
                raise ValueError(
                    f"Expected reference state ({cls.RAW_DIM},), "
                    f"got {reference_state.shape}"
                )
            if not np.all(np.isfinite(reference_state)):
                raise ValueError("Unitree reference state contains NaN or Inf")

            reference_position = reference_state[:3]
            reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
            reference_rotation = cls.quaternion_to_matrix(reference_quaternion)

            if process_xyz:
                local_delta = action[:, :3]
                world_delta = np.einsum("ij,tj->ti", reference_rotation, local_delta)
                absolute_position = reference_position + world_delta
            else:
                absolute_position = np.repeat(
                    reference_position[None, :], action.shape[0], axis=0
                )

            absolute_rotation = np.einsum(
                "ij,tjk->tik", reference_rotation, processed_rotation
            )
            absolute_quaternion = cls.matrix_to_quaternion(absolute_rotation)
            sign = np.sum(
                absolute_quaternion * reference_quaternion[None, :],
                axis=-1,
                keepdims=True,
            )
            absolute_quaternion = np.where(
                sign < 0.0, -absolute_quaternion, absolute_quaternion
            )
        else:
            absolute_quaternion = cls.matrix_to_quaternion(processed_rotation)
            if reference_state is not None:
                reference_state = np.asarray(reference_state)
                if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
                    raise ValueError(
                        f"Expected reference state ({cls.RAW_DIM},), "
                        f"got {reference_state.shape}"
                    )
                reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
                sign = np.sum(
                    absolute_quaternion * reference_quaternion[None, :],
                    axis=-1,
                    keepdims=True,
                )
                absolute_quaternion = np.where(
                    sign < 0.0, -absolute_quaternion, absolute_quaternion
                )
                if process_xyz:
                    reference_rotation = cls.quaternion_to_matrix(reference_quaternion)
                    world_delta = np.einsum(
                        "ij,tj->ti", reference_rotation, action[:, :3]
                    )
                    absolute_position = reference_state[:3] + world_delta
                else:
                    absolute_position = np.repeat(
                        reference_state[:3][None, :], action.shape[0], axis=0
                    )
            else:
                absolute_position = np.zeros((action.shape[0], 3), dtype=action.dtype)

        return np.concatenate((absolute_position, absolute_quaternion), axis=-1)

    @classmethod
    def build_normalization_params(
        cls, raw_params: dict[str, np.ndarray], *, process_xyz: bool = False
    ) -> dict[str, np.ndarray]:
        """Build 9D relative-root normalization params from absolute 7D stats.

        Used for WBC ``robot_root`` (synthetic hip bounds). SMPL ``frame`` prefers
        :meth:`build_frame_rot6d_normalization_params` instead.
        """
        raw_min = np.asarray(raw_params["min"])
        raw_max = np.asarray(raw_params["max"])
        raw_mean = np.asarray(raw_params["mean"])
        raw_std = np.asarray(raw_params["std"])
        if raw_min.shape[0] != cls.RAW_DIM:
            raise ValueError(
                f"Unitree root conversion expects {cls.RAW_DIM}D action statistics, "
                f"got {raw_min.shape[0]}"
            )

        if process_xyz:
            translation_bound = float(np.linalg.norm(raw_max[:3] - raw_min[:3]))
            translation_bound = max(translation_bound, 1e-3)
            translation_std = max(translation_bound / 3.0, 1e-3)
        else:
            translation_bound = 1e-3
            translation_std = 1e-3

        processed_min = np.concatenate(
            (
                np.full(3, -translation_bound, dtype=raw_min.dtype),
                np.full(6, -1.0, dtype=raw_min.dtype),
            )
        )
        processed_max = np.concatenate(
            (
                np.full(3, translation_bound, dtype=raw_max.dtype),
                np.full(6, 1.0, dtype=raw_max.dtype),
            )
        )
        processed_mean = np.zeros(9, dtype=raw_mean.dtype)
        processed_std = np.concatenate(
            (
                np.full(3, translation_std, dtype=raw_std.dtype),
                np.ones(6, dtype=raw_std.dtype),
            )
        )

        return {
            "min": processed_min,
            "max": processed_max,
            "dim": np.array(cls.PROCESSED_DIM),
            "mean": processed_mean,
            "std": processed_std,
        }

    @classmethod
    def build_frame_rot6d_normalization_params(
        cls,
        frame_params: dict[str, np.ndarray],
        *,
        relative: bool = True,
        num_samples: int = 4096,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """Rewrite 82D frame stats → 84D with hip rot6d stats from quat distribution.

        Joints ``[:72]`` and wrist ``[76:82]`` keep the original 82D statistics.
        Hip ``[72:78]`` is re-estimated by sampling quaternions from the 82D
        mean/std (same empirical spirit as joints), converting to absolute or
        relative rot6d, then taking mean/std/min/max — not synthetic ±1/std=1.
        """
        dtype = np.asarray(frame_params["mean"]).dtype
        mean_q_raw = np.asarray(frame_params["mean"][72:76], dtype=np.float64)
        if float(np.linalg.norm(mean_q_raw)) < cls.EPS:
            mean_q_raw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mean_q = cls._normalize_quaternion(mean_q_raw)
        std_q = np.maximum(
            np.asarray(frame_params["std"][72:76], dtype=np.float64), 1e-4
        )
        rng = np.random.default_rng(seed)
        samples_q = mean_q + std_q * rng.standard_normal((num_samples, 4))
        sample_norms = np.linalg.norm(samples_q, axis=-1, keepdims=True)
        bad = sample_norms[..., 0] < cls.EPS
        if np.any(bad):
            samples_q[bad] = mean_q
            sample_norms = np.linalg.norm(samples_q, axis=-1, keepdims=True)
        samples_q = samples_q / np.maximum(sample_norms, cls.EPS)

        if relative:
            reference_rotation = cls.quaternion_to_matrix(mean_q)
            future_rotation = cls.quaternion_to_matrix(samples_q)
            out_rotation = np.einsum(
                "ij,njk->nik", reference_rotation.T, future_rotation
            )
        else:
            out_rotation = cls.quaternion_to_matrix(samples_q)

        rot6d = cls.matrix_to_rotation_6d(out_rotation)
        hip_mean = rot6d.mean(axis=0).astype(dtype)
        hip_std = np.maximum(rot6d.std(axis=0), 1e-3).astype(dtype)
        hip_min = rot6d.min(axis=0).astype(dtype)
        hip_max = rot6d.max(axis=0).astype(dtype)

        rewritten: dict[str, np.ndarray] = {}
        for key, hip in (
            ("min", hip_min),
            ("max", hip_max),
            ("mean", hip_mean),
            ("std", hip_std),
        ):
            src = np.asarray(frame_params[key])
            rewritten[key] = np.concatenate((src[:72], hip, src[76:82])).astype(dtype)
        rewritten["dim"] = np.array(cls.FRAME_PROCESSED_DIM)
        return rewritten


class Root2Euler:
    """Convert Unitree root between absolute quat (7D) and Euler xyz (6D).

    Raw layout:     xyz(3) + quaternion_wxyz(4)
    Processed layout: xyz(3) + euler_xyz(3)   # roll, pitch, yaw (radians)

    Modes (see StateActionProcessor ``use_relative_euler`` / ``use_state_euler``):
      - absolute Euler: learn action Euler directly (no delta vs state)
      - delta Euler: wrap(action_euler - state_euler), requires state root quat→Euler

    SMPL ``frame`` hip ori lives at [72:76] (quat) / [72:75] (euler); same pack/splice
    pattern as Root2Rot6d but quat4→euler3 (82D→81D).
    """

    RAW_DIM = 7
    PROCESSED_DIM = 6
    ROOT_RAW_DIM = 7
    ROOT_PROCESSED_DIM = 6
    EULER_DIM = 3
    EPS = 1e-8

    ACTION_KEYS = frozenset({"robot_root", "frame"})
    STATE_KEY_CANDIDATES = {
        "robot_root": ("robot_root", "robot_root_current"),
        "frame": ("robot_root", "robot_root_current"),
    }
    FRAME_RAW_DIM = 82
    FRAME_PROCESSED_DIM = 81

    @classmethod
    def is_action_key(cls, key: str) -> bool:
        return key in cls.ACTION_KEYS

    @staticmethod
    def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        return (np.asarray(angles) + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _normalize_quaternion(cls, quaternion: np.ndarray) -> np.ndarray:
        return Root2Rot6d._normalize_quaternion(quaternion)

    @classmethod
    def quaternion_to_euler(cls, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """wxyz quat → xyz Euler (roll, pitch, yaw), radians."""
        q = cls._normalize_quaternion(quaternion_wxyz)
        flat = q.reshape(-1, 4)
        # scipy expects xyzw
        rot = Rotation.from_quat(np.concatenate((flat[:, 1:], flat[:, :1]), axis=-1))
        euler = rot.as_euler("xyz")
        return euler.reshape(*q.shape[:-1], 3).astype(q.dtype, copy=False)

    @classmethod
    def euler_to_quaternion(cls, euler_xyz: np.ndarray) -> np.ndarray:
        """xyz Euler (roll, pitch, yaw) → wxyz quat."""
        euler_xyz = np.asarray(euler_xyz)
        if euler_xyz.shape[-1] != 3:
            raise ValueError(f"Expected euler (..., 3), got {euler_xyz.shape}")
        flat = euler_xyz.reshape(-1, 3)
        rot = Rotation.from_euler("xyz", flat)
        q_xyzw = rot.as_quat()
        q_wxyz = np.concatenate((q_xyzw[:, 3:], q_xyzw[:, :3]), axis=-1)
        q_wxyz = cls._normalize_quaternion(q_wxyz)
        return q_wxyz.reshape(*euler_xyz.shape[:-1], 4).astype(euler_xyz.dtype, copy=False)

    @classmethod
    def pack_frame_root(cls, frame: np.ndarray, *, relative: bool = False) -> np.ndarray:
        """Pull hip ori from SMPL frame into a 7D/6D root (xyz unused)."""
        if relative:
            root = np.zeros((frame.shape[0], cls.PROCESSED_DIM), dtype=frame.dtype)
            root[:, 3:6] = frame[:, 72:75]
        else:
            root = np.zeros((frame.shape[0], cls.RAW_DIM), dtype=frame.dtype)
            root[:, 3:7] = frame[:, 72:76]
        return root

    @classmethod
    def splice_frame_root(
        cls, frame: np.ndarray, root: np.ndarray, *, relative: bool = False
    ) -> np.ndarray:
        """Put root ori back into SMPL frame."""
        if relative:
            # absolute 82D body + processed euler root → 81D
            return np.concatenate((frame[:, :72], root[:, 3:6], frame[:, 76:82]), axis=-1)
        # processed 81D body + absolute quat root → 82D
        return np.concatenate((frame[:, :72], root[:, 3:7], frame[:, 75:81]), axis=-1)

    @classmethod
    def root_quat_to_euler_root(cls, root: np.ndarray) -> np.ndarray:
        """Convert (..., 7) xyz+quat → (..., 6) xyz+euler."""
        root = np.asarray(root)
        if root.shape[-1] != cls.RAW_DIM:
            raise ValueError(f"Expected root (..., {cls.RAW_DIM}), got {root.shape}")
        euler = cls.quaternion_to_euler(root[..., 3:7])
        return np.concatenate((root[..., :3], euler), axis=-1)

    @classmethod
    def root_euler_to_quat_root(cls, root: np.ndarray) -> np.ndarray:
        """Convert (..., 6) xyz+euler → (..., 7) xyz+quat."""
        root = np.asarray(root)
        if root.shape[-1] != cls.PROCESSED_DIM:
            raise ValueError(f"Expected root (..., {cls.PROCESSED_DIM}), got {root.shape}")
        quat = cls.euler_to_quaternion(root[..., 3:6])
        return np.concatenate((root[..., :3], quat), axis=-1)

    @classmethod
    def to_relative(
        cls,
        action: np.ndarray,
        reference_state: np.ndarray,
        *,
        process_xyz: bool = False,
        use_state_delta: bool = False,
    ) -> np.ndarray:
        """Absolute root (T, 7) → processed (T, 6).

        use_state_delta=False: output absolute Euler of action.
        use_state_delta=True:  output wrap(action_euler - state_euler).
        """
        action = np.asarray(action)
        reference_state = np.asarray(reference_state)

        if action.ndim != 2 or action.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected Unitree action (T, {cls.RAW_DIM}), got {action.shape}"
            )
        if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected reference state ({cls.RAW_DIM},), got {reference_state.shape}"
            )
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(reference_state)):
            raise ValueError("Unitree action or reference state contains NaN or Inf")

        action_euler = cls.quaternion_to_euler(action[:, 3:7])
        if use_state_delta:
            reference_euler = cls.quaternion_to_euler(reference_state[3:7])
            out_euler = cls.wrap_to_pi(action_euler - reference_euler[None, :])
        else:
            out_euler = action_euler

        if process_xyz:
            reference_rotation = Root2Rot6d.quaternion_to_matrix(
                cls._normalize_quaternion(reference_state[3:7])
            )
            world_delta = action[:, :3] - reference_state[:3]
            local_delta = np.einsum("ij,tj->ti", reference_rotation.T, world_delta)
            if use_state_delta:
                pass  # local_delta already vs reference
            else:
                # absolute mode: keep absolute xyz
                local_delta = action[:, :3]
        else:
            local_delta = np.zeros((action.shape[0], 3), dtype=action.dtype)

        return np.concatenate((local_delta, out_euler), axis=-1)

    @classmethod
    def to_absolute(
        cls,
        action: np.ndarray,
        reference_state: np.ndarray,
        *,
        process_xyz: bool = False,
        use_state_delta: bool = False,
    ) -> np.ndarray:
        """Processed root (T, 6) → absolute (T, 7)."""
        action = np.asarray(action)
        reference_state = np.asarray(reference_state)

        if action.ndim != 2 or action.shape[-1] != cls.PROCESSED_DIM:
            raise ValueError(
                f"Expected processed Unitree action (T, {cls.PROCESSED_DIM}), got {action.shape}"
            )
        if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected reference state ({cls.RAW_DIM},), got {reference_state.shape}"
            )
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(reference_state)):
            raise ValueError("Processed Unitree action or reference state contains NaN or Inf")

        reference_euler = cls.quaternion_to_euler(reference_state[3:7])
        if use_state_delta:
            absolute_euler = cls.wrap_to_pi(reference_euler[None, :] + action[:, 3:6])
        else:
            absolute_euler = action[:, 3:6]
        absolute_quaternion = cls.euler_to_quaternion(absolute_euler)

        # Prefer quaternion hemisphere closest to the reference.
        reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
        sign = np.sum(absolute_quaternion * reference_quaternion[None, :], axis=-1, keepdims=True)
        absolute_quaternion = np.where(sign < 0.0, -absolute_quaternion, absolute_quaternion)

        if process_xyz:
            if use_state_delta:
                reference_rotation = Root2Rot6d.quaternion_to_matrix(reference_quaternion)
                world_delta = np.einsum("ij,tj->ti", reference_rotation, action[:, :3])
                absolute_position = reference_state[:3] + world_delta
            else:
                absolute_position = action[:, :3]
        else:
            absolute_position = np.repeat(reference_state[:3][None, :], action.shape[0], axis=0)

        return np.concatenate((absolute_position, absolute_quaternion), axis=-1)

    @classmethod
    def build_normalization_params(
        cls,
        raw_params: dict[str, np.ndarray],
        *,
        process_xyz: bool = False,
        use_state_delta: bool = False,
    ) -> dict[str, np.ndarray]:
        """Build 6D Euler-root normalization params from absolute 7D stats."""
        raw_min = np.asarray(raw_params["min"])
        raw_max = np.asarray(raw_params["max"])
        raw_mean = np.asarray(raw_params["mean"])
        raw_std = np.asarray(raw_params["std"])
        if raw_min.shape[0] != cls.RAW_DIM:
            raise ValueError(
                f"Unitree root Euler expects {cls.RAW_DIM}D action statistics, got {raw_min.shape[0]}"
            )

        if process_xyz:
            if use_state_delta:
                translation_bound = float(np.linalg.norm(raw_max[:3] - raw_min[:3]))
                translation_bound = max(translation_bound, 1e-3)
                translation_std = max(translation_bound / 3.0, 1e-3)
                t_min = np.full(3, -translation_bound, dtype=raw_min.dtype)
                t_max = np.full(3, translation_bound, dtype=raw_max.dtype)
                t_mean = np.zeros(3, dtype=raw_mean.dtype)
                t_std = np.full(3, translation_std, dtype=raw_std.dtype)
            else:
                t_min, t_max = raw_min[:3], raw_max[:3]
                t_mean, t_std = raw_mean[:3], np.maximum(raw_std[:3], 1e-3)
        else:
            t_min = np.full(3, -1e-3, dtype=raw_min.dtype)
            t_max = np.full(3, 1e-3, dtype=raw_max.dtype)
            t_mean = np.zeros(3, dtype=raw_mean.dtype)
            t_std = np.full(3, 1e-3, dtype=raw_std.dtype)

        # Euler / delta-Euler in [-pi, pi]
        e_bound = np.pi
        e_min = np.full(3, -e_bound, dtype=raw_min.dtype)
        e_max = np.full(3, e_bound, dtype=raw_max.dtype)
        e_mean = np.zeros(3, dtype=raw_mean.dtype)
        e_std = np.full(3, e_bound / 3.0, dtype=raw_std.dtype)

        return {
            "min": np.concatenate((t_min, e_min)),
            "max": np.concatenate((t_max, e_max)),
            "dim": np.array(cls.PROCESSED_DIM),
            "mean": np.concatenate((t_mean, e_mean)),
            "std": np.concatenate((t_std, e_std)),
        }

    @classmethod
    def build_state_normalization_params(
        cls, raw_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Rewrite state robot_root 7D (xyz+quat) stats → 6D (xyz+euler)."""
        raw_min = np.asarray(raw_params["min"])
        raw_max = np.asarray(raw_params["max"])
        raw_mean = np.asarray(raw_params["mean"])
        raw_std = np.asarray(raw_params["std"])
        if raw_min.shape[0] != cls.RAW_DIM:
            raise ValueError(
                f"State root Euler expects {cls.RAW_DIM}D statistics, got {raw_min.shape[0]}"
            )
        e_bound = np.pi
        return {
            "min": np.concatenate((raw_min[:3], np.full(3, -e_bound, dtype=raw_min.dtype))),
            "max": np.concatenate((raw_max[:3], np.full(3, e_bound, dtype=raw_max.dtype))),
            "dim": np.array(cls.PROCESSED_DIM),
            "mean": np.concatenate((raw_mean[:3], np.zeros(3, dtype=raw_mean.dtype))),
            "std": np.concatenate(
                (np.maximum(raw_std[:3], 1e-3), np.full(3, e_bound / 3.0, dtype=raw_std.dtype))
            ),
        }


class StateActionProcessor:
    """Unified processor for robot state and action data."""

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
        use_relative_euler: bool = False,
        use_state_euler: bool = False,
        use_rot6d: bool = False,
        use_relative_rot6d: bool = True,
    ):
        if use_state_euler and not use_relative_euler:
            raise ValueError("use_state_euler=True requires use_relative_euler=True")
        if use_relative_rot6d and not use_rot6d:
            # Legacy callers may only set use_relative_rot6d via has_root rewrite;
            # keep permissive: relative_rot6d alone does not force use_rot6d.
            pass
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.use_relative_euler = use_relative_euler
        self.use_state_euler = use_state_euler
        self.use_rot6d = use_rot6d
        self.use_relative_rot6d = use_relative_rot6d
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        if statistics is not None:
            self.set_statistics(statistics)
        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                logger.warning(
                    "Statistics for embodiment %r already present; new stats "
                    "DISCARDED (override=False). If the new data differs from "
                    "the existing distribution this will cause silent "
                    "normalization mismatch — pass override=True (or "
                    "override_pretraining_statistics=True at the dataset level) "
                    "to use the merged stats instead.",
                    key,
                )
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}
                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])
                    range_vals = np.maximum(max_vals - min_vals, 1e-8)
                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            if "action" not in self.modality_configs[embodiment_tag]:
                continue

            modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
            action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

            if action_configs is not None:
                for key, action_config in zip(modality_keys, action_configs):
                    if (
                        action_config.rep == ActionRepresentation.RELATIVE
                        and self.use_relative_action
                    ):
                        # Unitree root uses synthesized params from absolute 7D
                        # stats (xyz+quat), not joint-style relative_stats.
                        if Root2Rot6d.is_action_key(key) or Root2Euler.is_action_key(
                            key
                        ):
                            continue
                        if "relative_action" not in self.statistics[embodiment_tag]:
                            raise ValueError(
                                f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                "but 'relative_action' not found in statistics"
                            )
                        if key not in self.statistics[embodiment_tag]["relative_action"]:
                            raise ValueError(
                                f"Relative action statistics required for key '{key}' "
                                f"in embodiment '{embodiment_tag}' but not found"
                            )
                        action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                        self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                            self.statistics[embodiment_tag]["relative_action"][key]
                        )
                        self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

            state_cfg = self.modality_configs[embodiment_tag].get("state")
            state_keys = (
                {str(k).removeprefix("state.") for k in state_cfg.modality_keys}
                if state_cfg is not None
                else set()
            )
            has_root = bool(
                state_keys.intersection(Root2Rot6d.STATE_KEY_CANDIDATES["frame"])
            )
            use_state_delta = self._wants_state_euler(embodiment_tag)

            if self.use_relative_euler:
                for key in modality_keys:
                    params = self.norm_params[embodiment_tag]["action"].get(key)
                    if params is None:
                        continue
                    if (
                        key == "robot_root"
                        and int(params["dim"].item()) == Root2Euler.RAW_DIM
                    ):
                        self.norm_params[embodiment_tag]["action"][key] = (
                            Root2Euler.build_normalization_params(
                                params,
                                process_xyz=True,
                                use_state_delta=use_state_delta,
                            )
                        )
                        logger.info(
                            "Enabled Unitree relative root Euler for %s/%s: %dD -> %dD "
                            "(use_state_delta=%s)",
                            embodiment_tag,
                            key,
                            Root2Euler.RAW_DIM,
                            Root2Euler.PROCESSED_DIM,
                            use_state_delta,
                        )
                    elif (
                        key == "frame"
                        and has_root
                        and int(params["dim"].item()) == Root2Euler.FRAME_RAW_DIM
                    ):
                        root_norm = Root2Euler.build_normalization_params(
                            {
                                "min": np.zeros(
                                    Root2Euler.RAW_DIM, dtype=params["min"].dtype
                                ),
                                "max": np.ones(
                                    Root2Euler.RAW_DIM, dtype=params["max"].dtype
                                ),
                                "mean": np.zeros(
                                    Root2Euler.RAW_DIM, dtype=params["mean"].dtype
                                ),
                                "std": np.ones(
                                    Root2Euler.RAW_DIM, dtype=params["std"].dtype
                                ),
                            },
                            process_xyz=False,
                            use_state_delta=use_state_delta,
                        )
                        rewritten = {
                            k: np.concatenate(
                                (params[k][:72], root_norm[k][3:6], params[k][76:82])
                            )
                            for k in ("min", "max", "mean", "std")
                        }
                        rewritten["dim"] = np.array(Root2Euler.FRAME_PROCESSED_DIM)
                        self.norm_params[embodiment_tag]["action"][key] = rewritten
                        logger.info(
                            "Enabled Unitree relative root Euler for %s/%s: %dD -> %dD "
                            "(use_state_delta=%s)",
                            embodiment_tag,
                            key,
                            Root2Euler.FRAME_RAW_DIM,
                            Root2Euler.FRAME_PROCESSED_DIM,
                            use_state_delta,
                        )

                if use_state_delta and "state" in self.norm_params[embodiment_tag]:
                    for state_key in ("robot_root", "robot_root_current"):
                        state_params = self.norm_params[embodiment_tag]["state"].get(state_key)
                        if (
                            state_params is not None
                            and int(state_params["dim"].item()) == Root2Euler.RAW_DIM
                        ):
                            self.norm_params[embodiment_tag]["state"][state_key] = (
                                Root2Euler.build_state_normalization_params(state_params)
                            )
                            logger.info(
                                "Enabled state root Euler for %s/%s: %dD -> %dD",
                                embodiment_tag,
                                state_key,
                                Root2Euler.RAW_DIM,
                                Root2Euler.PROCESSED_DIM,
                            )
            else:
                if self.use_relative_action:
                    for key in modality_keys:
                        params = self.norm_params[embodiment_tag]["action"].get(key)
                        if (
                            Root2Rot6d.is_action_key(key)
                            and params is not None
                            and int(params["dim"].item()) == Root2Rot6d.RAW_DIM
                        ):
                            self.norm_params[embodiment_tag]["action"][key] = (
                                Root2Rot6d.build_normalization_params(
                                    params, process_xyz=True
                                )
                            )
                            logger.info(
                                "Enabled Unitree relative root 6D processing for %s/%s: %dD -> %dD",
                                embodiment_tag,
                                key,
                                Root2Rot6d.RAW_DIM,
                                Root2Rot6d.PROCESSED_DIM,
                            )

                # SMPL frame: 82->84 for rot6d (explicit flag, or legacy: state has robot_root).
                enable_frame_rot6d = self.use_rot6d or (
                    has_root and not self.use_relative_euler
                )
                if enable_frame_rot6d and not self.use_rot6d:
                    # Legacy checkpoint: rot6d inferred from state.robot_root.
                    self.use_rot6d = True
                    self.use_relative_rot6d = True
                for key in modality_keys:
                    params = self.norm_params[embodiment_tag]["action"].get(key)
                    if (
                        key != "frame"
                        or not enable_frame_rot6d
                        or params is None
                        or int(params["dim"].item()) != Root2Rot6d.FRAME_RAW_DIM
                    ):
                        continue
                    rewritten = Root2Rot6d.build_frame_rot6d_normalization_params(
                        params, relative=self.use_relative_rot6d
                    )
                    self.norm_params[embodiment_tag]["action"][key] = rewritten
                    logger.info(
                        "Enabled Unitree frame rot6d for %s/%s: %dD -> %dD "
                        "(relative=%s; hip stats from 82D quat distribution)",
                        embodiment_tag,
                        key,
                        Root2Rot6d.FRAME_RAW_DIM,
                        Root2Rot6d.FRAME_PROCESSED_DIM,
                        self.use_relative_rot6d,
                    )

    def _root_action_is_relative(self, embodiment_tag: str) -> bool:
        action_cfg = self.modality_configs[embodiment_tag].get("action")
        if action_cfg is None or action_cfg.action_configs is None:
            return False
        for key, action_config in zip(action_cfg.modality_keys, action_cfg.action_configs):
            if Root2Euler.is_action_key(key) and (
                action_config.rep == ActionRepresentation.RELATIVE
            ):
                return True
        return False

    def _wants_state_euler(self, embodiment_tag: str) -> bool:
        """Delta Euler vs state: CLI ``use_state_euler`` or root ActionConfig.RELATIVE."""
        if not self.use_relative_euler:
            return False
        return self.use_state_euler or self._root_action_is_relative(embodiment_tag)

    def _uses_unitree_root_relative_6d(self, embodiment_tag: str, key: str) -> bool:
        """WBC/SMPL rot6d gate; ``frame`` uses processed 84D (absolute or relative)."""
        if self.use_relative_euler:
            return False
        if not Root2Rot6d.is_action_key(key):
            return False
        params = self.norm_params.get(embodiment_tag, {}).get("action", {}).get(key)
        if params is None:
            return False
        dim = int(params["dim"].item())
        if key == "robot_root":
            return self.use_relative_action and dim == Root2Rot6d.PROCESSED_DIM
        if key == "frame":
            return self.use_rot6d and dim == Root2Rot6d.FRAME_PROCESSED_DIM
        return False

    def _uses_unitree_root_relative_euler(self, embodiment_tag: str, key: str) -> bool:
        if not self.use_relative_euler or not Root2Euler.is_action_key(key):
            return False
        params = self.norm_params.get(embodiment_tag, {}).get("action", {}).get(key)
        if params is None:
            return False
        dim = int(params["dim"].item())
        if key == "robot_root":
            return dim == Root2Euler.PROCESSED_DIM
        if key == "frame":
            return dim == Root2Euler.FRAME_PROCESSED_DIM
        return False

    @staticmethod
    def _strip_state_prefix(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key.removeprefix("state."): value for key, value in state.items()}

    def _get_unitree_reference_array(
        self,
        state: dict[str, np.ndarray] | None,
        action_key: str,
    ) -> np.ndarray | None:
        if state is None:
            return None
        state = self._strip_state_prefix(state)
        candidates = Root2Rot6d.STATE_KEY_CANDIDATES[action_key]
        for candidate in candidates:
            if candidate in state:
                return np.asarray(state[candidate])
        return None

    def _get_unitree_reference_for_training(
        self,
        state: dict[str, np.ndarray] | None,
        action_key: str,
        action: np.ndarray,
    ) -> np.ndarray:
        reference_array = self._get_unitree_reference_array(state, action_key)
        if reference_array is None:
            logger.warning(
                "No Unitree root state found for %s; falling back to the first desired action as reference",
                action_key,
            )
            action = np.asarray(action)
            if action_key == "frame":
                frame0 = action[0] if action.ndim == 2 else action[0, 0]
                root = np.zeros(Root2Rot6d.RAW_DIM, dtype=action.dtype)
                if frame0.shape[-1] >= Root2Euler.FRAME_PROCESSED_DIM and (
                    self.use_relative_euler
                ):
                    # Should not happen on training apply (raw 82D); keep quat path.
                    root[3:7] = frame0[72:76] if frame0.shape[-1] >= 76 else 0
                else:
                    root[3:7] = frame0[72:76]
                return root
            return action[0]
        if reference_array.ndim == 1:
            ref = reference_array
        elif reference_array.ndim == 2:
            ref = reference_array[-1]
        else:
            raise ValueError(
                f"Expected unbatched Unitree state (D,) or (T, D), got {reference_array.shape}"
            )
        # apply_action always receives raw state (quat 7D). If somehow 6D euler leaked in,
        # convert back so RootRelative* to_relative gets (7,) quat root.
        if self.use_relative_euler and ref.shape[-1] == Root2Euler.PROCESSED_DIM:
            ref = Root2Euler.root_euler_to_quat_root(ref)
        return ref

    def _convert_unitree_to_absolute(
        self,
        action: np.ndarray,
        state: dict[str, np.ndarray] | None,
        action_key: str,
    ) -> np.ndarray:
        """Decode Unitree root; Euler path when ``use_relative_euler``."""
        reference_array = self._get_unitree_reference_array(state, action_key)
        process_xyz = action_key == "robot_root"
        use_euler = self.use_relative_euler
        absolute_rot6d = (
            self.use_rot6d
            and not self.use_relative_rot6d
            and not use_euler
            and action_key == "frame"
        )

        if reference_array is None and not absolute_rot6d:
            raise ValueError(
                f"State containing one of {Root2Rot6d.STATE_KEY_CANDIDATES[action_key]} "
                f"is required to decode relative Unitree root action '{action_key}'"
            )

        embodiment_tag = None
        for tag, cfgs in self.modality_configs.items():
            action_cfg = cfgs.get("action")
            if action_cfg is not None and action_key in action_cfg.modality_keys:
                embodiment_tag = tag
                break
        if embodiment_tag is None:
            raise ValueError(f"Cannot resolve embodiment for action key '{action_key}'")
        use_state_delta = self._wants_state_euler(embodiment_tag)

        def _coerce_reference(reference: np.ndarray) -> np.ndarray:
            if reference.shape[-1] == Root2Euler.PROCESSED_DIM and use_euler:
                return Root2Euler.root_euler_to_quat_root(reference)
            return reference

        def _one(sample_action: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
            if use_euler:
                assert reference is not None
                reference = _coerce_reference(reference)
                if action_key == "frame":
                    absolute_root = Root2Euler.to_absolute(
                        Root2Euler.pack_frame_root(sample_action, relative=True),
                        reference,
                        process_xyz=process_xyz,
                        use_state_delta=use_state_delta,
                    )
                    return Root2Euler.splice_frame_root(sample_action, absolute_root)
                return Root2Euler.to_absolute(
                    sample_action,
                    reference,
                    process_xyz=process_xyz,
                    use_state_delta=use_state_delta,
                )
            if action_key == "frame":
                absolute_root = Root2Rot6d.to_absolute(
                    Root2Rot6d.pack_frame_root(sample_action, relative=True),
                    reference,
                    process_xyz=process_xyz,
                    relative=self.use_relative_rot6d,
                )
                return Root2Rot6d.splice_frame_root(sample_action, absolute_root)
            assert reference is not None
            return Root2Rot6d.to_absolute(
                sample_action,
                reference,
                process_xyz=process_xyz,
                relative=True,
            )

        action = np.asarray(action)
        if action.ndim == 2:
            if reference_array is None:
                return _one(action, None)
            if reference_array.ndim == 1:
                reference = reference_array
            elif reference_array.ndim == 2:
                reference = reference_array[-1]
            elif reference_array.ndim == 3 and reference_array.shape[0] == 1:
                reference = reference_array[0, -1]
            else:
                raise ValueError(
                    f"Cannot match action shape {action.shape} with state shape {reference_array.shape}"
                )
            return _one(action, reference)

        if action.ndim != 3:
            raise ValueError(f"Expected Unitree action (T, D) or (B, T, D), got {action.shape}")

        batch_size = action.shape[0]
        if reference_array is None:
            return np.stack([_one(action[i], None) for i in range(batch_size)], axis=0)
        if reference_array.ndim == 1:
            references = np.repeat(reference_array[None, :], batch_size, axis=0)
        elif reference_array.ndim == 2:
            if reference_array.shape[0] == batch_size:
                references = reference_array
            else:
                references = np.repeat(reference_array[-1][None, :], batch_size, axis=0)
        elif reference_array.ndim == 3:
            if reference_array.shape[0] != batch_size:
                raise ValueError(
                    f"Action batch {batch_size} does not match state batch {reference_array.shape[0]}"
                )
            references = reference_array[:, -1]
        else:
            raise ValueError(f"Unsupported Unitree state shape {reference_array.shape}")

        return np.stack(
            [_one(sample_action, sample_reference)
             for sample_action, sample_reference in zip(action, references)],
            axis=0,
        )

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        normalized_values = {}
        state = deepcopy(state)

        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Optional: robot_root quat → euler before normalize (delta-Euler modes).
            if (
                self._wants_state_euler(embodiment_tag)
                and joint_group in Root2Euler.STATE_KEY_CANDIDATES["frame"]
                and np.asarray(state[joint_group]).shape[-1] == Root2Euler.RAW_DIM
            ):
                state[joint_group] = Root2Euler.root_quat_to_euler_root(
                    state[joint_group]
                )

            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized_values[joint_group] = normalize_values_meanstd(
                    state[joint_group], params
                )
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)
                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)
                normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        unnormalized_values = {}

        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            if sin_cos_keys and joint_group in sin_cos_keys:
                raise ValueError(
                    f"Cannot unapply sin/cos encoding for joint group '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. This transformation is not reversible."
                )
            if (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_meanstd(
                    state[joint_group], params
                )
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_minmax(
                    state[joint_group], params
                )

            # Optional: euler → quat after unnormalize (delta-Euler modes).
            if (
                self._wants_state_euler(embodiment_tag)
                and joint_group in Root2Euler.STATE_KEY_CANDIDATES["frame"]
                and unnormalized_values[joint_group].shape[-1]
                == Root2Euler.PROCESSED_DIM
            ):
                unnormalized_values[joint_group] = Root2Euler.root_euler_to_quat_root(
                    unnormalized_values[joint_group]
                )

        return unnormalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        action = deepcopy(action)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs
        special_keys = set()

        for key in modality_keys:
            if self._uses_unitree_root_relative_6d(embodiment_tag, key):
                if key not in action:
                    raise KeyError(
                        f"Joint group '{key}' not found in action dict for embodiment '{embodiment_tag}'"
                    )
                if key == "frame":
                    if self.use_relative_rot6d:
                        reference = self._get_unitree_reference_for_training(
                            state, key, action[key]
                        )
                        processed_root = Root2Rot6d.to_relative(
                            Root2Rot6d.pack_frame_root(action[key]),
                            reference,
                            relative=True,
                        )
                    else:
                        processed_root = Root2Rot6d.to_relative(
                            Root2Rot6d.pack_frame_root(action[key]),
                            relative=False,
                        )
                    action[key] = Root2Rot6d.splice_frame_root(
                        action[key], processed_root, relative=True
                    )
                else:
                    reference = self._get_unitree_reference_for_training(
                        state, key, action[key]
                    )
                    action[key] = Root2Rot6d.to_relative(
                        action[key], reference, process_xyz=True, relative=True
                    )
                special_keys.add(key)
            elif self._uses_unitree_root_relative_euler(embodiment_tag, key):
                if key not in action:
                    raise KeyError(
                        f"Joint group '{key}' not found in action dict for embodiment '{embodiment_tag}'"
                    )
                reference = self._get_unitree_reference_for_training(state, key, action[key])
                use_state_delta = self._wants_state_euler(embodiment_tag)
                if key == "frame":
                    relative_root = Root2Euler.to_relative(
                        Root2Euler.pack_frame_root(action[key]),
                        reference,
                        process_xyz=False,
                        use_state_delta=use_state_delta,
                    )
                    action[key] = Root2Euler.splice_frame_root(
                        action[key], relative_root, relative=True
                    )
                else:
                    action[key] = Root2Euler.to_relative(
                        action[key],
                        reference,
                        process_xyz=True,
                        use_state_delta=use_state_delta,
                    )
                special_keys.add(key)

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if key in special_keys:
                    continue
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )
                    state_key = action_config.state_key if action_config.state_key else key
                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )
                    action[key] = self._convert_to_relative_action(
                        action=action[key],
                        reference_state=state[state_key][-1],
                        action_type=action_config.type,
                        action_format=action_config.format,
                    )

        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )
            params = self.norm_params[embodiment_tag]["action"][joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(action[joint_group], params)
            else:
                normalized = normalize_values_minmax(action[joint_group], params)
            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)
            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
        *,
        to_absolute: bool = True,
    ) -> dict[str, np.ndarray]:
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )
            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(group_values, params)
            unnormalized_values[joint_group] = unnormalized

        special_keys = set()
        if to_absolute:
            for key in modality_keys:
                if self._uses_unitree_root_relative_6d(
                    embodiment_tag, key
                ) or self._uses_unitree_root_relative_euler(embodiment_tag, key):
                    unnormalized_values[key] = self._convert_unitree_to_absolute(
                        unnormalized_values[key], state, key
                    )
                    special_keys.add(key)

        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs
        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if key in special_keys:
                    continue
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )
                    state_key = action_config.state_key if action_config.state_key else key
                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )
                    relative_action = unnormalized_values[key]
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]
                        relative_action = relative_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]

                    absolute_actions = []
                    for sample_state, sample_action in zip(reference_state, relative_action):
                        absolute_actions.append(
                            self._convert_to_absolute_action(
                                action=sample_action,
                                reference_state=sample_state[-1],
                                action_type=action_config.type,
                                action_format=action_config.format,
                            )
                        )
                    unnormalized_values[key] = (
                        np.stack(absolute_actions, axis=0) if is_batched else absolute_actions[0]
                    )

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def unapply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        try:
            unapplied_state = self.unapply_state(state, embodiment_tag)
        except ValueError as error:
            if "sin/cos encoding" in str(error) and raw_state is None:
                raise ValueError(
                    "Cannot unapply sin/cos encoded state. Please provide raw_state parameter."
                ) from error
            raise

        state_for_action = raw_state if raw_state is not None else unapplied_state
        unapplied_action = self.unapply_action(action, embodiment_tag, state=state_for_action)
        return unapplied_state, unapplied_action

    def get_state_dim(self, embodiment_tag: str, include_sincos_expansion: bool = False) -> int:
        total_dim = 0
        state_config = self.modality_configs[embodiment_tag]["state"]
        sin_cos_keys = set()
        if self.apply_sincos_state_encoding and hasattr(state_config, "sin_cos_embedding_keys"):
            sin_cos_keys = set(state_config.sin_cos_embedding_keys)

        for joint_group in state_config.modality_keys:
            base_dim = self.norm_params[embodiment_tag]["state"][joint_group]["dim"].item()
            total_dim += (
                base_dim * 2
                if include_sincos_expansion and joint_group in sin_cos_keys
                else base_dim
            )
        return total_dim

    def get_action_dim(self, embodiment_tag: str) -> int:
        return sum(
            self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
            for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys
        )

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            action_chunking = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)
        elif action_type == ActionType.NON_EEF:
            action_chunking = JointActionChunk([JointPose(movement) for movement in action])
            reference_frame = JointPose(reference_state)
        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        return action_chunking.relative_chunking(reference_frame=reference_frame).to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            relative_action = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)
        elif action_type == ActionType.NON_EEF:
            relative_action = JointActionChunk([JointPose(pose) for pose in action])
            reference_frame = JointPose(reference_state)
        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        return relative_action.to_absolute_chunking(reference_frame=reference_frame).to(action_format)

    def __str__(self) -> str:
        return (
            "StateActionProcessor("
            f"modality_configs={self.modality_configs}, statistics={self.statistics}, "
            f"use_percentiles={self.use_percentiles}, clip_outliers={self.clip_outliers}, "
            f"apply_sincos_state_encoding={self.apply_sincos_state_encoding}, "
            f"use_relative_action={self.use_relative_action}, "
            f"use_relative_euler={self.use_relative_euler}, "
            f"use_state_euler={self.use_state_euler}, "
            f"use_rot6d={self.use_rot6d}, "
            f"use_relative_rot6d={self.use_relative_rot6d})"
        )
