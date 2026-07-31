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

"""Modality config for Unitree G1 + GEAR-SONIC latent actions (68-dim).

Action layout (matches meta/modality.json / info.json action shape [68]):
  - motion_token:      64-dim SONIC latent
  - left_hand_joints:  2-dim
  - right_hand_joints: 2-dim

State layout (states shape [33]):
  left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7)
  + left_hand(2) + right_hand(2)

Video: stereo egocentric cameras (egocentric_left / egocentric_right).

Note: unitree_g1_sonic is already registered in embodiment_configs.py (single
ego_view + projected_gravity). This file overwrites that entry for this
stereo / consolidated-column LeRobot export.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

unitree_g1_sonic_config = {
    # Video: keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "ego_view_left",
            "ego_view_right",
        ],
    ),
    # State: 6+6+3+7+7+2+2 = 33 dims
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    # Action: 40-step horizon (SONIC VLA @ ~2.5 Hz); 64+2+2 = 68 dims
    "action": ModalityConfig(
        delta_indices=list(range(0, 40)),
        modality_keys=[
            "motion_token",
            "left_hand_joints",
            "right_hand_joints",
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

# Built-in sonic config is already registered; overwrite for this dataset layout.
MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC.value] = unitree_g1_sonic_config
