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

"""Modality config for Unitree G1 predicting SMPL skeletal actions.

Action (96 dims after processor; 94 raw in parquet):
  - frame:      82-dim SMPL skeletal -> 84-dim after hip-root relative rot6D
                (frame[72:76] wxyz quat, relative to state robot_root via RootRelative6D)
  - left_hand:  6-dim hand command
  - right_hand: 6-dim hand command

State (48 dims):
  - left_hand / right_hand: hand_state (12)
  - robot_root: robot_q_current[0:7] (xyz + wxyz quat) — reference for frame hip rotation
  - robot_qpos: robot_q_current[7:36] (29 joint angles)

Video: head stereo + both wrists (same as WBC).

Hand actions use ABSOLUTE. SMPL frame hip quaternion is converted with the same
RootRelative6D path as WBC robot_root (relative rotation 6D w.r.t. robot_root).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

unitree_g1_smpl_config = {
    # Video: keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_stereo_left",
            "head_stereo_right",
            "wrist_left",
            "wrist_right",
        ],
    ),
    # State: left_hand (6) + right_hand (6) + robot_root (7) + robot_qpos (29) = 48 dims
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_hand",
            "right_hand",
            "robot_root",
            "robot_qpos",
        ],
    ),
    # Action: predict horizon @ 30 fps; 84 + 6 + 6 = 96 dims after relative rot6D
    "action": ModalityConfig(
        delta_indices=list(range(0, 50)),
        modality_keys=[
            "frame",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(unitree_g1_smpl_config, embodiment_tag=EmbodimentTag.UNITREE_G1_SMPL)
