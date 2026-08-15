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

Action (dims after processor depend on ``--root-process-mode``):
  - frame:      82-dim SMPL skeletal
                - original: 82-dim absolute hip quat (state drops robot_root)
                - rot6d: → 84-dim hip relative rot6D (state.robot_root ref)
                - delta_euler / euler relative: → 81-dim Δeuler vs state
                - euler absolute: → 81-dim absolute Euler
  - left_hand:  6-dim hand command
  - right_hand: 6-dim hand command

State (48 dims when robot_root present; 41 dims in original mode):
  - left_hand / right_hand: hand_state (12)
  - robot_root: robot_q_current[0:7] (xyz + wxyz quat) — reference for frame root
  - robot_qpos: robot_q_current[7:36] (29 joint angles)

Training flags (see gr00t/configs/smpl_root_mode.py):
  - ``--root-process-mode original|rot6d|delta_euler|euler``
  - ``--action-mode absolute|relative`` (rot6d / euler; delta_euler fixed relative)
  - Checkpoint stores use_relative_euler / use_state_euler / use_rot6d /
    use_relative_rot6d + full modality_configs.

Hand actions use ABSOLUTE. frame ActionConfig stays ABSOLUTE; do not use rep=RELATIVE
for root (use root-process-mode / action-mode instead).
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
    # Action: predict horizon @ 30 fps
    # frame ActionConfig stays ABSOLUTE; root ref via ACTION_MODE + USE_RELATIVE_EULER
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
