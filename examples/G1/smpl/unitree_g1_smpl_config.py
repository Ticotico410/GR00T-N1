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

Action (94 dims):
  - frame:      82-dim SMPL skeletal representation
  - left_hand:  6-dim hand command
  - right_hand: 6-dim hand command

State (41 dims):
  - left_hand / right_hand: hand_state (12)
  - robot_qpos: robot_q_current[7:36] (29) — drops root xyz+quat (first 7)

Video: head stereo + both wrists (same as WBC).

All actions use ABSOLUTE: SMPL frame has no matching state space for relative
conversion; hand_cmd follows the same absolute convention.
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
    # State: left_hand (6) + right_hand (6) + robot_qpos (29) = 41 dims
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_hand",
            "right_hand",
            "robot_qpos",
        ],
    ),
    # Action: predict horizon @ 30 fps; 82 + 6 + 6 = 94 dims
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
