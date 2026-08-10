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
- Unitree G1 root conversion from xyz + quaternion to local xyz + rotation 6D
- SMPL frame hip-root quaternion via RootRelative6D (relative to state robot_root)
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


logger = logging.getLogger(__name__)


class RootRelative6D:
    """Convert a Unitree whole-body action between 36D absolute and 38D relative form.

    Raw action layout:
        xyz(3) + quaternion_wxyz(4) + joint_qpos(29) = 36

    Processed action layout:
        local_delta_xyz(3) + relative_rotation_6d(6) + joint_qpos(29) = 38

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

    ACTION_KEYS = frozenset({"robot_root"})
    STATE_KEY_CANDIDATES = {
        "robot_root": ("robot_root", "robot_root_current"),
        "frame": ("robot_root", "robot_root_current"),
    }

    # SMPL ``action.frame`` embeds hip-root quaternion at [72:76]; reference root
    # comes from state ``robot_root`` (= robot_q_current[0:7], xyz + wxyz quat).
    SMPL_FRAME_KEY = "frame"
    SMPL_FRAME_RAW_DIM = 82
    SMPL_FRAME_PROCESSED_DIM = 84
    SMPL_FRAME_QUAT_SLICE = slice(72, 76)
    SMPL_FRAME_TAIL_RAW_SLICE = slice(76, 82)
    SMPL_FRAME_ROT6D_SLICE = slice(72, 78)
    SMPL_FRAME_TAIL_PROCESSED_SLICE = slice(78, 84)

    @classmethod
    def is_smpl_frame_key(cls, key: str) -> bool:
        return key == cls.SMPL_FRAME_KEY

    @classmethod
    def _pack_smpl_frame_quat_as_root(cls, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        pseudo_root = np.zeros((frame.shape[0], cls.RAW_DIM), dtype=frame.dtype)
        pseudo_root[:, 3:7] = frame[:, cls.SMPL_FRAME_QUAT_SLICE]
        return pseudo_root

    @classmethod
    def smpl_frame_to_relative(cls, frame: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 2 or frame.shape[-1] != cls.SMPL_FRAME_RAW_DIM:
            raise ValueError(
                f"Expected SMPL frame action (T, {cls.SMPL_FRAME_RAW_DIM}), got {frame.shape}"
            )
        if not np.all(np.isfinite(frame)):
            raise ValueError("SMPL frame action contains NaN or Inf")

        relative_root = cls.to_relative(
            cls._pack_smpl_frame_quat_as_root(frame), reference_state
        )
        return np.concatenate(
            (
                frame[:, : cls.SMPL_FRAME_QUAT_SLICE.start],
                relative_root[:, 3:9],
                frame[:, cls.SMPL_FRAME_TAIL_RAW_SLICE],
            ),
            axis=-1,
        )

    @classmethod
    def smpl_frame_to_absolute(cls, frame: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 2 or frame.shape[-1] != cls.SMPL_FRAME_PROCESSED_DIM:
            raise ValueError(
                f"Expected processed SMPL frame action (T, {cls.SMPL_FRAME_PROCESSED_DIM}), "
                f"got {frame.shape}"
            )
        if not np.all(np.isfinite(frame)):
            raise ValueError("Processed SMPL frame action contains NaN or Inf")

        pseudo_relative = np.zeros((frame.shape[0], cls.PROCESSED_DIM), dtype=frame.dtype)
        pseudo_relative[:, 3:9] = frame[:, cls.SMPL_FRAME_ROT6D_SLICE]
        absolute_root = cls.to_absolute(pseudo_relative, reference_state)
        return np.concatenate(
            (
                frame[:, : cls.SMPL_FRAME_QUAT_SLICE.start],
                absolute_root[:, 3:7],
                frame[:, cls.SMPL_FRAME_TAIL_PROCESSED_SLICE],
            ),
            axis=-1,
        )

    @classmethod
    def build_smpl_frame_normalization_params(
        cls, raw_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        raw_min = np.asarray(raw_params["min"])
        raw_max = np.asarray(raw_params["max"])
        raw_mean = np.asarray(raw_params["mean"])
        raw_std = np.asarray(raw_params["std"])
        if raw_min.shape[0] != cls.SMPL_FRAME_RAW_DIM:
            raise ValueError(
                f"SMPL frame conversion expects {cls.SMPL_FRAME_RAW_DIM}D action statistics, "
                f"got {raw_min.shape[0]}"
            )

        fake_root_params = {
            "min": np.concatenate((np.zeros(3, dtype=raw_min.dtype), raw_min[72:76])),
            "max": np.concatenate((np.zeros(3, dtype=raw_max.dtype), raw_max[72:76])),
            "mean": np.concatenate((np.zeros(3, dtype=raw_mean.dtype), raw_mean[72:76])),
            "std": np.concatenate((np.ones(3, dtype=raw_std.dtype), raw_std[72:76])),
        }
        root_norm = cls.build_normalization_params(fake_root_params)
        return {
            "min": np.concatenate(
                (raw_min[:72], root_norm["min"][3:9], raw_min[cls.SMPL_FRAME_TAIL_RAW_SLICE])
            ),
            "max": np.concatenate(
                (raw_max[:72], root_norm["max"][3:9], raw_max[cls.SMPL_FRAME_TAIL_RAW_SLICE])
            ),
            "mean": np.concatenate(
                (raw_mean[:72], root_norm["mean"][3:9], raw_mean[cls.SMPL_FRAME_TAIL_RAW_SLICE])
            ),
            "std": np.concatenate(
                (raw_std[:72], root_norm["std"][3:9], raw_std[cls.SMPL_FRAME_TAIL_RAW_SLICE])
            ),
            "dim": np.array(cls.SMPL_FRAME_PROCESSED_DIM),
        }

    @classmethod
    def is_action_key(cls, key: str) -> bool:
        return key in cls.ACTION_KEYS

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
    def to_relative(cls, action: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        reference_state = np.asarray(reference_state)

        if action.ndim != 2 or action.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected Unitree action (T, {cls.RAW_DIM}), got {action.shape}"
            )

        if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected reference state ({cls.RAW_DIM},), "
                f"got {reference_state.shape}"
            )

        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(reference_state)):
            raise ValueError(
                "Unitree action or reference state contains NaN or Inf"
            )

        reference_position = reference_state[:3]
        reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
        reference_rotation = cls.quaternion_to_matrix(reference_quaternion)

        future_position = action[:, :3]
        future_rotation = cls.quaternion_to_matrix(action[:, 3:7])

        # Root translation expressed in the reference root frame.
        world_delta = future_position - reference_position
        local_delta = np.einsum("ij,tj->ti", reference_rotation.T, world_delta)

        # Root orientation relative to the reference root orientation.
        relative_rotation = np.einsum("ij,tjk->tik", reference_rotation.T, future_rotation)
        relative_rotation_6d = cls.matrix_to_rotation_6d(relative_rotation)

        return np.concatenate((local_delta, relative_rotation_6d), axis=-1)

    @classmethod
    def to_absolute(cls, action: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        reference_state = np.asarray(reference_state)

        if action.ndim != 2 or action.shape[-1] != cls.PROCESSED_DIM:
            raise ValueError(
                f"Expected processed Unitree action "
                f"(T, {cls.PROCESSED_DIM}), got {action.shape}"
            )

        if reference_state.ndim != 1 or reference_state.shape[-1] != cls.RAW_DIM:
            raise ValueError(
                f"Expected reference state ({cls.RAW_DIM},), "
                f"got {reference_state.shape}"
            )

        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(reference_state)):
            raise ValueError(
                "Processed Unitree action or reference state contains NaN or Inf"
            )

        reference_position = reference_state[:3]
        reference_quaternion = cls._normalize_quaternion(reference_state[3:7])
        reference_rotation = cls.quaternion_to_matrix(reference_quaternion)

        local_delta = action[:, :3]
        relative_rotation = cls.rotation_6d_to_matrix(action[:, 3:9])

        world_delta = np.einsum("ij,tj->ti", reference_rotation, local_delta)
        absolute_position = reference_position + world_delta

        absolute_rotation = np.einsum("ij,tjk->tik", reference_rotation, relative_rotation)
        absolute_quaternion = cls.matrix_to_quaternion(absolute_rotation)

        # Select the quaternion sign closest to the reference quaternion.
        sign = np.sum(absolute_quaternion * reference_quaternion[None, :], axis=-1, keepdims=True)
        absolute_quaternion = np.where(sign < 0.0, -absolute_quaternion, absolute_quaternion)

        return np.concatenate((absolute_position, absolute_quaternion), axis=-1)

    @classmethod
    def build_normalization_params(cls, raw_params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raw_min = np.asarray(raw_params["min"])
        raw_max = np.asarray(raw_params["max"])
        raw_mean = np.asarray(raw_params["mean"])
        raw_std = np.asarray(raw_params["std"])
        if raw_min.shape[0] != cls.RAW_DIM:
            raise ValueError(
                f"Unitree root conversion expects {cls.RAW_DIM}D action statistics, got {raw_min.shape[0]}"
            )

        translation_bound = float(np.linalg.norm(raw_max[:3] - raw_min[:3]))
        translation_bound = max(translation_bound, 1e-3)
        translation_std = max(translation_bound / 3.0, 1e-3)

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
    ):
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
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
                        # Unitree root uses synthesized 9D params from absolute 7D
                        # stats (xyz+quat), not joint-style relative_stats which are
                        # shaped (horizon, 7) and break build_normalization_params.
                        if RootRelative6D.is_action_key(key):
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

            if self.use_relative_action:
                for key in modality_keys:
                    params = self.norm_params[embodiment_tag]["action"].get(key)
                    if (
                        RootRelative6D.is_action_key(key)
                        and params is not None
                        and int(params["dim"].item()) == RootRelative6D.RAW_DIM
                    ):
                        self.norm_params[embodiment_tag]["action"][key] = (
                            RootRelative6D.build_normalization_params(params)
                        )
                        logger.info(
                            "Enabled Unitree relative root 6D processing for %s/%s: %dD -> %dD",
                            embodiment_tag,
                            key,
                            RootRelative6D.RAW_DIM,
                            RootRelative6D.PROCESSED_DIM,
                        )

            for key in modality_keys:
                params = self.norm_params[embodiment_tag]["action"].get(key)
                if (
                    RootRelative6D.is_smpl_frame_key(key)
                    and params is not None
                    and int(params["dim"].item()) == RootRelative6D.SMPL_FRAME_RAW_DIM
                ):
                    self.norm_params[embodiment_tag]["action"][key] = (
                        RootRelative6D.build_smpl_frame_normalization_params(params)
                    )
                    logger.info(
                        "Enabled SMPL frame relative root 6D processing for %s/%s: %dD -> %dD",
                        embodiment_tag,
                        key,
                        RootRelative6D.SMPL_FRAME_RAW_DIM,
                        RootRelative6D.SMPL_FRAME_PROCESSED_DIM,
                    )

    def _uses_unitree_root_relative_6d(self, embodiment_tag: str, key: str) -> bool:
        if not self.use_relative_action or not RootRelative6D.is_action_key(key):
            return False
        params = self.norm_params.get(embodiment_tag, {}).get("action", {}).get(key)
        return params is not None and int(params["dim"].item()) == RootRelative6D.PROCESSED_DIM

    def _uses_smpl_frame_relative_6d(self, embodiment_tag: str, key: str) -> bool:
        if not RootRelative6D.is_smpl_frame_key(key):
            return False
        params = self.norm_params.get(embodiment_tag, {}).get("action", {}).get(key)
        return (
            params is not None
            and int(params["dim"].item()) == RootRelative6D.SMPL_FRAME_PROCESSED_DIM
        )

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
        for candidate in RootRelative6D.STATE_KEY_CANDIDATES[action_key]:
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
            if RootRelative6D.is_smpl_frame_key(action_key):
                if action.ndim == 2:
                    return RootRelative6D._pack_smpl_frame_quat_as_root(action)[0]
                if action.ndim == 3:
                    return RootRelative6D._pack_smpl_frame_quat_as_root(action[0])[0]
                raise ValueError(
                    f"Expected SMPL frame action (T, D) or (B, T, D), got {action.shape}"
                )
            return action[0]
        if reference_array.ndim == 1:
            return reference_array
        if reference_array.ndim == 2:
            return reference_array[-1]
        raise ValueError(
            f"Expected unbatched Unitree state (D,) or (T, D), got {reference_array.shape}"
        )

    def _convert_unitree_to_absolute(
        self,
        action: np.ndarray,
        state: dict[str, np.ndarray] | None,
        action_key: str,
    ) -> np.ndarray:
        reference_array = self._get_unitree_reference_array(state, action_key)
        if reference_array is None:
            raise ValueError(
                f"State containing one of {RootRelative6D.STATE_KEY_CANDIDATES[action_key]} "
                f"is required to decode relative Unitree root action '{action_key}'"
            )

        action = np.asarray(action)
        if action.ndim == 2:
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
            return RootRelative6D.to_absolute(action, reference)

        if action.ndim != 3:
            raise ValueError(f"Expected Unitree action (T, D) or (B, T, D), got {action.shape}")

        batch_size = action.shape[0]
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
            [
                RootRelative6D.to_absolute(sample_action, sample_reference)
                for sample_action, sample_reference in zip(action, references)
            ],
            axis=0,
        )

    def _convert_smpl_frame_to_absolute(
        self,
        action: np.ndarray,
        state: dict[str, np.ndarray] | None,
        action_key: str,
    ) -> np.ndarray:
        reference_array = self._get_unitree_reference_array(state, action_key)
        if reference_array is None:
            raise ValueError(
                f"State containing one of {RootRelative6D.STATE_KEY_CANDIDATES[action_key]} "
                f"is required to decode relative SMPL frame action '{action_key}'"
            )

        action = np.asarray(action)
        if action.ndim == 2:
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
            return RootRelative6D.smpl_frame_to_absolute(action, reference)

        if action.ndim != 3:
            raise ValueError(f"Expected SMPL frame action (T, D) or (B, T, D), got {action.shape}")

        batch_size = action.shape[0]
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
            raise ValueError(f"Unsupported SMPL frame state shape {reference_array.shape}")

        return np.stack(
            [
                RootRelative6D.smpl_frame_to_absolute(sample_action, sample_reference)
                for sample_action, sample_reference in zip(action, references)
            ],
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
            if self._uses_smpl_frame_relative_6d(embodiment_tag, key):
                if key not in action:
                    raise KeyError(
                        f"Joint group '{key}' not found in action dict for embodiment '{embodiment_tag}'"
                    )
                reference = self._get_unitree_reference_for_training(state, key, action[key])
                action[key] = RootRelative6D.smpl_frame_to_relative(action[key], reference)
                special_keys.add(key)

        for key in modality_keys:
            if self._uses_unitree_root_relative_6d(embodiment_tag, key):
                if key not in action:
                    raise KeyError(
                        f"Joint group '{key}' not found in action dict for embodiment '{embodiment_tag}'"
                    )
                reference = self._get_unitree_reference_for_training(state, key, action[key])
                action[key] = RootRelative6D.to_relative(action[key], reference)
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
        for key in modality_keys:
            if self._uses_smpl_frame_relative_6d(embodiment_tag, key):
                unnormalized_values[key] = self._convert_smpl_frame_to_absolute(
                    unnormalized_values[key], state, key
                )
                special_keys.add(key)

        for key in modality_keys:
            if self._uses_unitree_root_relative_6d(embodiment_tag, key):
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
            f"use_relative_action={self.use_relative_action})"
        )
