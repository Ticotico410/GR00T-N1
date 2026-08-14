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

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.state_action_processor import Root2Rot6d, Root2Euler
from gr00t.eval.action_ema import (
    SMPL_HIP_EULER_SLICE,
    SMPL_HIP_QUAT_SLICE,
    SMPL_HIP_ROT6D_SLICE,
    apply_root_xyz_ema,
    apply_smpl_frame_chunk_ema,
    ema_applies_to,
)
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
import matplotlib

# Headless save: avoid X11 BadAlloc on large multi-row plots (e.g. SMPL 94D).
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

Original SMPL open loop (decoded 82D SMPL + hip quat; no EMA/euler/rot6d):
    python gr00t/eval/open_loop_eval.py --model-path ... --root-process-mode original

WBC relative 9D root open loop (local xyz + rot6d, process_xyz=True):
    python gr00t/eval/open_loop_eval.py --model-path ... --root-process-mode trans9d

SMPL relative rot6D open loop (frame[72:76] → 6D, process_xyz=False):
    python gr00t/eval/open_loop_eval.py --model-path ... --root-process-mode rot6d

SMPL delta-Euler open loop (pred 81D Δeuler vs GT Δeuler, training target space):
    python gr00t/eval/open_loop_eval.py --model-path ... --root-process-mode delta_euler
    python gr00t/eval/open_loop_eval.py ... --root-process-mode delta_euler --ema-alpha 0.25

Plots are written under ``{model-path}/open_loop_eval/{[ema_]mode}/``:
    original: smpl_{traj_id}.jpeg (82D) and quat_{traj_id}.jpeg ([72:76])
    delta_euler: euler_{traj_id}.jpeg (predicted Δeuler, to_absolute=False)
                 quat_{traj_id}.jpeg (Root2Euler.to_absolute → hip quat)
    rot6d: rot6d_{traj_id}.jpeg (predicted hip rot6d, to_absolute=False)
           quat_{traj_id}.jpeg (Root2Rot6d.to_absolute → hip quat)
    trans9d: trans9d_{traj_id}.jpeg (predicted processed 9D vs GT)

``--ema-alpha`` applies per ``--root-process-mode`` (via ``gr00t.eval.action_ema``):
    trans9d: EMA on robot_root local xyz (may span inference chunks)
    delta_euler / rot6d: EMA on hip Δeuler or rot6d within each chunk only

Relative euler/rot6d checkpoints have no original eval option. Pred stays in processed
space; quat plots are a forward decode of that same pred, not decode-then-recompute.

"""

ROOT_ACTION_KEY = "robot_root"
SMPL_FRAME_KEY = "frame"
TRANS9D_LABELS = [
    "dx",
    "dy",
    "dz",
    "r6d0",
    "r6d1",
    "r6d2",
    "r6d3",
    "r6d4",
    "r6d5",
]
ROT6D_LABELS = ["r6d0", "r6d1", "r6d2", "r6d3", "r6d4", "r6d5"]
DELTA_EULER_LABELS = ["droll", "dpitch", "dyaw"]
EULER_ABSOLUTE_LABELS = ["roll", "pitch", "yaw"]
HIP_QUAT_LABELS = ["qw", "qx", "qy", "qz"]
VALID_ROOT_PROCESS_MODES = ("original", "trans9d", "rot6d", "delta_euler", "euler")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
PROCESSED_ROOT_MODES = ("trans9d", "rot6d", "delta_euler", "euler")
# Modes whose training target is state-relative processed root (for tests / docs).
VALID_RELATIVE_ROOT_MODES = ("trans9d", "rot6d", "delta_euler")


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
    dim_labels: list[str] | None = None,
    title_note: str = "",
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
        dim_labels: Optional per-dimension subplot titles (defaults to Action {{idx}})
        title_note: Optional suffix for the figure title
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Cap row height so high-dim actions (e.g. SMPL 94D) don't allocate huge pixmaps.
    row_h = 1.5 if num_plots > 48 else (2.5 if num_plots > 24 else 4.0)
    fig, axes = plt.subplots(
        nrows=num_plots, ncols=1, figsize=(10, row_h * num_plots), dpi=80
    )

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    title = (
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}"
    )
    if title_note:
        title = f"{title} | {title_note}"
    fig.suptitle(
        title,
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        if dim_labels is not None and action_idx < len(dim_labels):
            ax.set_title(dim_labels[action_idx])
        else:
            ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path, dpi=80, bbox_inches="tight")

    plt.close(fig)  # Close the figure to free memory


def _stack_traj_column(traj: pd.DataFrame, column: str) -> np.ndarray:
    return np.vstack([np.asarray(arr) for arr in traj[column]])


def _rotation_geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between two batches of rot6d orientations."""
    rot_a = Root2Rot6d.rotation_6d_to_matrix(rot6d_a)
    rot_b = Root2Rot6d.rotation_6d_to_matrix(rot6d_b)
    rot_err = np.einsum("tij,tkj->tik", rot_a, rot_b)
    trace = rot_err[:, 0, 0] + rot_err[:, 1, 1] + rot_err[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _smpl_frame_to_relative_rot6d(frame: np.ndarray, reference_root: np.ndarray) -> np.ndarray:
    """GT only: absolute 82D SMPL frame hip quat → relative rot6D (T, 6)."""
    frame = np.asarray(frame, dtype=np.float32)
    reference_root = np.asarray(reference_root, dtype=np.float32).reshape(-1)
    if reference_root.shape[0] != Root2Rot6d.RAW_DIM:
        raise ValueError(
            f"Expected reference robot_root ({Root2Rot6d.RAW_DIM},), got {reference_root.shape}"
        )

    squeeze = False
    if frame.ndim == 1:
        frame = frame[None, ...]
        squeeze = True

    if int(frame.shape[-1]) != Root2Rot6d.FRAME_RAW_DIM:
        raise ValueError(
            f"GT rot6d expects absolute SMPL frame ({Root2Rot6d.FRAME_RAW_DIM}D), "
            f"got {frame.shape[-1]}D"
        )

    relative_root = Root2Rot6d.to_relative(
        Root2Rot6d.pack_frame_root(frame),
        reference_root,
        process_xyz=False,
    )
    out = relative_root[:, 3:9]
    return out[0] if squeeze else out


def _pred_rot6d_from_processed_frame(frame: np.ndarray) -> np.ndarray:
    """Model output in training target space: unnormalized 84D frame hip rot6d [72:78]."""
    frame = np.asarray(frame, dtype=np.float32)
    squeeze = False
    if frame.ndim == 1:
        frame = frame[None, ...]
        squeeze = True
    dim = int(frame.shape[-1])
    if dim != Root2Rot6d.FRAME_PROCESSED_DIM:
        raise ValueError(
            f"Predicted rot6d expects processed SMPL frame "
            f"({Root2Rot6d.FRAME_PROCESSED_DIM}D), got {dim}D. "
            "Use Gr00tPolicy with to_absolute=False (training target space)."
        )
    out = frame[:, SMPL_HIP_ROT6D_SLICE]
    return out[0] if squeeze else out


def _gt_euler_delta_from_absolute_frame(
    frame: np.ndarray, reference_root: np.ndarray
) -> np.ndarray:
    """GT training target: wrap(action_euler - state_euler) from absolute 82D frame."""
    frame = np.asarray(frame, dtype=np.float32)
    reference_root = np.asarray(reference_root, dtype=np.float32).reshape(-1)
    if reference_root.shape[0] != Root2Euler.RAW_DIM:
        raise ValueError(
            f"Expected reference robot_root ({Root2Euler.RAW_DIM},), "
            f"got {reference_root.shape}"
        )

    squeeze = False
    if frame.ndim == 1:
        frame = frame[None, ...]
        squeeze = True

    if int(frame.shape[-1]) != Root2Euler.FRAME_RAW_DIM:
        raise ValueError(
            f"GT euler delta expects absolute SMPL frame "
            f"({Root2Euler.FRAME_RAW_DIM}D), got {frame.shape[-1]}D"
        )

    relative_root = Root2Euler.to_relative(
        Root2Euler.pack_frame_root(frame),
        reference_root,
        process_xyz=False,
        use_state_delta=True,
    )
    out = relative_root[:, 3:6]
    return out[0] if squeeze else out


def _pred_hip_euler_from_processed_frame(frame: np.ndarray) -> np.ndarray:
    """Model output in training target space: unnormalized 81D frame hip euler [72:75]."""
    frame = np.asarray(frame, dtype=np.float32)
    squeeze = False
    if frame.ndim == 1:
        frame = frame[None, ...]
        squeeze = True

    dim = int(frame.shape[-1])
    if dim != Root2Euler.FRAME_PROCESSED_DIM:
        raise ValueError(
            f"Predicted hip euler expects processed SMPL frame "
            f"({Root2Euler.FRAME_PROCESSED_DIM}D), got {dim}D. "
            "Use Gr00tPolicy with to_absolute=False (training target space)."
        )
    out = frame[:, 72:75]
    return out[0] if squeeze else out


def _pred_euler_delta_from_processed_frame(frame: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for delta_euler eval."""
    return _pred_hip_euler_from_processed_frame(frame)


def _gt_euler_absolute_from_absolute_frame(frame: np.ndarray) -> np.ndarray:
    """GT absolute hip euler xyz (radians) from 82D SMPL frame."""
    frame = np.asarray(frame, dtype=np.float32)
    squeeze = False
    if frame.ndim == 1:
        frame = frame[None, ...]
        squeeze = True
    if int(frame.shape[-1]) != Root2Euler.FRAME_RAW_DIM:
        raise ValueError(
            f"GT absolute euler expects SMPL frame ({Root2Euler.FRAME_RAW_DIM}D), "
            f"got {frame.shape[-1]}D"
        )
    out = Root2Euler.quaternion_to_euler(frame[:, 72:76])
    return out[0] if squeeze else out


def _euler_absolute_to_abs_quat(euler_abs: np.ndarray) -> np.ndarray:
    """Absolute euler (T, 3) → hip quat wxyz (no state reference)."""
    euler_abs = np.asarray(euler_abs, dtype=np.float32)
    squeeze = False
    if euler_abs.ndim == 1:
        euler_abs = euler_abs[None, ...]
        squeeze = True
    out = Root2Euler.euler_to_quaternion(euler_abs)
    return out[0] if squeeze else out


def _smpl_frame_to_delta_euler(frame: np.ndarray, reference_root: np.ndarray) -> np.ndarray:
    """Backward-compatible alias: absolute 82D → wrap(action_euler - state_euler)."""
    return _gt_euler_delta_from_absolute_frame(frame, reference_root)


def _hip_quat_from_absolute_frame(frame: np.ndarray) -> np.ndarray:
    """SMPL 82D frame hip quat wxyz at [72:76]."""
    frame = np.asarray(frame, dtype=np.float32)
    squeeze = False
    if frame.ndim == 1:
        frame = frame[None, ...]
        squeeze = True
    dim = int(frame.shape[-1])
    if dim != Root2Euler.FRAME_RAW_DIM:
        raise ValueError(
            f"Absolute hip quat expects {Root2Euler.FRAME_RAW_DIM}D SMPL frame, got {dim}D"
        )
    out = frame[:, SMPL_HIP_QUAT_SLICE]
    return out[0] if squeeze else out


def _euler_delta_to_abs_quat(
    euler_delta: np.ndarray, reference_root: np.ndarray
) -> np.ndarray:
    """Δeuler (T, 3) → hip quat wxyz via Root2Euler.to_absolute(use_state_delta=True)."""
    euler_delta = np.asarray(euler_delta, dtype=np.float32)
    squeeze = False
    if euler_delta.ndim == 1:
        euler_delta = euler_delta[None, ...]
        squeeze = True
    processed = np.concatenate(
        [
            np.zeros((euler_delta.shape[0], 3), dtype=np.float32),
            euler_delta,
        ],
        axis=-1,
    )
    abs_root = Root2Euler.to_absolute(
        processed,
        np.asarray(reference_root, dtype=np.float32).reshape(-1),
        process_xyz=False,
        use_state_delta=True,
    )
    out = abs_root[:, 3:7]
    return out[0] if squeeze else out


def _rot6d_to_abs_quat(rot6d: np.ndarray, reference_root: np.ndarray) -> np.ndarray:
    """Hip rot6d (T, 6) → hip quat wxyz via Root2Rot6d.to_absolute."""
    rot6d = np.asarray(rot6d, dtype=np.float32)
    squeeze = False
    if rot6d.ndim == 1:
        rot6d = rot6d[None, ...]
        squeeze = True
    processed = np.concatenate(
        [np.zeros((rot6d.shape[0], 3), dtype=np.float32), rot6d],
        axis=-1,
    )
    abs_root = Root2Rot6d.to_absolute(
        processed,
        np.asarray(reference_root, dtype=np.float32).reshape(-1),
        process_xyz=False,
    )
    out = abs_root[:, 3:7]
    return out[0] if squeeze else out


def _reconstruct_abs_quat_from_chunks(
    processed_rot: np.ndarray,
    chunk_refs: list[tuple[int, np.ndarray]],
    convert_fn,
) -> np.ndarray:
    """Apply per-chunk to_absolute using the same reference as inference."""
    segs: list[np.ndarray] = []
    offset = 0
    for horizon, reference in chunk_refs:
        chunk = processed_rot[offset : offset + horizon]
        if chunk.shape[0] != horizon:
            raise ValueError(
                f"Chunk length mismatch: expected {horizon} steps at offset {offset}, "
                f"got {chunk.shape[0]} (total {processed_rot.shape[0]})"
            )
        segs.append(convert_fn(chunk, reference))
        offset += horizon
    if offset != processed_rot.shape[0]:
        raise ValueError(
            f"Chunk refs cover {offset} steps, but processed rot has {processed_rot.shape[0]}"
        )
    return np.concatenate(segs, axis=0)


def _align_quat_hemisphere(pred_quat: np.ndarray, gt_quat: np.ndarray) -> np.ndarray:
    """Flip pred q → -q when it is in the opposite hemisphere from GT."""
    sign = np.sum(pred_quat * gt_quat, axis=-1, keepdims=True)
    return np.where(sign < 0.0, -pred_quat, pred_quat)


def _quat_geodesic_deg(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between wxyz quaternions; invariant to q vs -q."""
    q_a = q_a / np.clip(np.linalg.norm(q_a, axis=-1, keepdims=True), 1e-8, None)
    q_b = q_b / np.clip(np.linalg.norm(q_b, axis=-1, keepdims=True), 1e-8, None)
    dot = np.clip(np.abs(np.sum(q_a * q_b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.degrees(np.arccos(dot))


def _resolve_open_loop_plot_dir(
    model_path: str | None,
    root_process_mode: str,
    save_plot_path: str | None,
    ema_alpha: float | None = None,
) -> Path:
    """``{checkpoint}/open_loop_eval/{mode}``, or ``.../ema_{mode}`` when EMA is set.
    ``--save-plot-path`` overrides the directory.
    """
    if save_plot_path:
        path = Path(save_plot_path)
        return path.parent if path.suffix.lower() in IMAGE_SUFFIXES else path

    subdir = root_process_mode
    if ema_alpha is not None and root_process_mode != "original":
        subdir = f"ema_{root_process_mode}"

    if model_path:
        checkpoint_dir = Path(model_path)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = checkpoint_dir.parent
        return checkpoint_dir / "open_loop_eval" / subdir
    return Path("/tmp/open_loop_eval") / subdir


def _checkpoint_root_flags(policy: BasePolicy) -> tuple[bool, bool, bool]:
    """Return (use_relative_euler, use_state_euler, use_relative_action) from the processor."""
    processor = getattr(policy, "processor", None)
    return (
        bool(getattr(processor, "use_relative_euler", False)),
        bool(getattr(processor, "use_state_euler", False)),
        bool(getattr(processor, "use_relative_action", False)),
    )


def _checkpoint_effective_root_mode(policy: BasePolicy) -> str:
    """Infer how this ckpt actually trains root — not just the raw processor flags.

    ``use_relative_action=True`` is often left on from pretrain defaults. For SMPL
    ``frame``, Root2Rot6d only activates when state includes ``robot_root`` (82→84).
    Absolute-quat SFT drops ``robot_root`` from state, so frame stays 82D quat even
    with ``use_relative_action=True``.
    """
    use_rel_euler, use_state_euler, use_rel_action = _checkpoint_root_flags(policy)
    if use_rel_euler:
        return "delta_euler" if use_state_euler else "euler"

    modality = policy.get_modality_config()
    state_cfg = modality.get("state")
    action_cfg = modality.get("action")
    state_keys = set(state_cfg.modality_keys) if state_cfg is not None else set()
    action_keys = set(action_cfg.modality_keys) if action_cfg is not None else set()
    has_root_ref = bool(state_keys.intersection({"robot_root", "robot_root_current"}))

    processor = getattr(policy, "processor", None)
    embodiment = getattr(policy, "embodiment_tag", None)
    if processor is not None and embodiment is not None:
        tag = embodiment.value if hasattr(embodiment, "value") else str(embodiment)
        uses_frame_rot6d = getattr(processor, "_uses_unitree_root_relative_6d", None)
        if callable(uses_frame_rot6d) and SMPL_FRAME_KEY in action_keys:
            if uses_frame_rot6d(tag, SMPL_FRAME_KEY):
                return "rot6d"
        uses_root_rot6d = uses_frame_rot6d
        if callable(uses_root_rot6d) and ROOT_ACTION_KEY in action_keys:
            if uses_root_rot6d(tag, ROOT_ACTION_KEY):
                return "trans9d"

    # Fallback when norm_params not rewritten yet / PolicyClient: modality gates.
    if SMPL_FRAME_KEY in action_keys and has_root_ref:
        return "rot6d"
    if ROOT_ACTION_KEY in action_keys and use_rel_action:
        return "trans9d"
    return "original"


def _assert_mode_matches_checkpoint(policy: BasePolicy, root_process_mode: str) -> None:
    """Match eval mode to how the ckpt actually represents root (not raw flags alone)."""
    use_rel_euler, use_state_euler, use_rel_action = _checkpoint_root_flags(policy)
    effective = _checkpoint_effective_root_mode(policy)
    logging.info(
        "Checkpoint processor flags: use_relative_euler=%s use_state_euler=%s "
        "use_relative_action=%s | effective_root_mode=%s",
        use_rel_euler,
        use_state_euler,
        use_rel_action,
        effective,
    )

    if effective == "delta_euler":
        if root_process_mode != "delta_euler":
            raise ValueError(
                "This checkpoint was trained with Root2Euler Δeuler "
                f"(use_relative_euler=True, use_state_euler={use_state_euler}). "
                "Open-loop must compare predicted Δeuler "
                "(get_action to_absolute=False), not decoded 82D quat. "
                "Use --root-process-mode delta_euler."
            )
        return

    if effective == "euler":
        if root_process_mode != "euler":
            raise ValueError(
                "This checkpoint was trained with Root2Euler absolute euler "
                "(use_relative_euler=True, use_state_euler=False). "
                "Use --root-process-mode euler --action-mode absolute."
            )
        return

    if effective == "rot6d":
        if root_process_mode != "rot6d":
            raise ValueError(
                "This checkpoint trains SMPL frame hip as relative rot6d "
                "(state has robot_root → 82D→84D Root2Rot6d). "
                "Use --root-process-mode rot6d (get_action to_absolute=False)."
            )
        return

    if effective == "trans9d":
        if root_process_mode != "trans9d":
            raise ValueError(
                "This checkpoint trains WBC robot_root as relative 9D "
                "(Root2Rot6d, process_xyz=True). "
                "Use --root-process-mode trans9d (get_action to_absolute=False)."
            )
        return

    # original: direct 82D quat; use_relative_action may still be True from pretrain.
    if root_process_mode in PROCESSED_ROOT_MODES:
        raise ValueError(
            f"--root-process-mode {root_process_mode} requires a relative checkpoint "
            "(euler / rot6d with state.robot_root / WBC robot_root). "
            f"This checkpoint is effective_root_mode=original "
            f"(use_relative_action={use_rel_action} alone does not enable rot6d on "
            "SMPL frame without state.robot_root). "
            "Use --root-process-mode original."
        )


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    plot_dir: str | Path | None = None,
    root_process_mode: str = "original",
    ema_alpha: float | None = None,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    if root_process_mode == "trans9d":
        if ROOT_ACTION_KEY not in loader.modality_configs["action"].modality_keys:
            raise ValueError(
                f"--root-process-mode trans9d requires action key '{ROOT_ACTION_KEY}' "
                f"in modality config, got {loader.modality_configs['action'].modality_keys}"
            )
        if ROOT_ACTION_KEY not in loader.modality_configs["state"].modality_keys:
            raise ValueError(
                f"--root-process-mode trans9d requires state key '{ROOT_ACTION_KEY}' "
                f"in modality config, got {loader.modality_configs['state'].modality_keys}"
            )
        if modality_keys is not None and modality_keys != [ROOT_ACTION_KEY]:
            logging.warning(
                "trans9d mode only evaluates '%s'; ignoring --modality-keys %s",
                ROOT_ACTION_KEY,
                modality_keys,
            )
        action_keys = [ROOT_ACTION_KEY]
        logging.info(
            "Root eval space=trans9d: compare GT/pred in Unitree local-xyz+rot6d (9D)"
        )
    elif root_process_mode == "rot6d":
        if SMPL_FRAME_KEY not in loader.modality_configs["action"].modality_keys:
            raise ValueError(
                f"--root-process-mode rot6d requires action key '{SMPL_FRAME_KEY}' "
                f"in modality config, got {loader.modality_configs['action'].modality_keys}"
            )
        if ROOT_ACTION_KEY not in loader.modality_configs["state"].modality_keys:
            raise ValueError(
                f"--root-process-mode rot6d requires state key '{ROOT_ACTION_KEY}' "
                f"(reference for hip quat). Got state keys "
                f"{loader.modality_configs['state'].modality_keys}. "
                "Use a SMPL-rel checkpoint / modality that includes robot_root."
            )
        if modality_keys is not None and modality_keys != [SMPL_FRAME_KEY]:
            logging.warning(
                "rot6d mode only evaluates hip rot6D from '%s'; ignoring --modality-keys %s",
                SMPL_FRAME_KEY,
                modality_keys,
            )
        action_keys = [SMPL_FRAME_KEY]
        logging.info(
            "Root eval space=rot6d: pred unnormalized hip rot6d (84D [72:78]) vs "
            "GT Root2Rot6d.to_relative (process_xyz=False)"
        )
        if not isinstance(policy, Gr00tPolicy):
            raise TypeError(
                "--root-process-mode rot6d requires local Gr00tPolicy "
                "(--model-path); PolicyClient cannot return processed 84D actions."
            )
    elif root_process_mode == "delta_euler":
        if SMPL_FRAME_KEY not in loader.modality_configs["action"].modality_keys:
            raise ValueError(
                f"--root-process-mode delta_euler requires action key '{SMPL_FRAME_KEY}' "
                f"in modality config, got {loader.modality_configs['action'].modality_keys}"
            )
        if ROOT_ACTION_KEY not in loader.modality_configs["state"].modality_keys:
            raise ValueError(
                f"--root-process-mode delta_euler requires state key '{ROOT_ACTION_KEY}' "
                f"(reference for hip euler delta). Got state keys "
                f"{loader.modality_configs['state'].modality_keys}."
            )
        if modality_keys is not None and modality_keys != [SMPL_FRAME_KEY]:
            logging.warning(
                "delta_euler mode only evaluates hip wrap(action_euler-state_euler) "
                "from '%s'; ignoring --modality-keys %s",
                SMPL_FRAME_KEY,
                modality_keys,
            )
        action_keys = [SMPL_FRAME_KEY]
        logging.info(
            "Root eval space=delta_euler: compare pred processed Δeuler (81D [72:75]) "
            "vs GT wrap(action_euler - state_euler) (USE_RELATIVE_EULER=1 USE_STATE_EULER=1). "
            "Also decode pred Δeuler → hip quat via Root2Euler.to_absolute for quat_{id}.jpeg"
        )
        if not isinstance(policy, Gr00tPolicy):
            raise TypeError(
                "--root-process-mode delta_euler requires local Gr00tPolicy "
                "(--model-path); PolicyClient cannot return processed 81D actions."
            )
    elif root_process_mode == "original":
        if SMPL_FRAME_KEY not in loader.modality_configs["action"].modality_keys:
            raise ValueError(
                f"--root-process-mode original requires action key '{SMPL_FRAME_KEY}' "
                f"in modality config, got {loader.modality_configs['action'].modality_keys}"
            )
        logging.info(
            "Root eval space=original: decoded SMPL frame 82D + hip quat [72:76]; "
            "no EMA / rot6d / trans9d / euler conversion"
        )
    elif root_process_mode == "euler":
        if SMPL_FRAME_KEY not in loader.modality_configs["action"].modality_keys:
            raise ValueError(
                f"--root-process-mode euler requires action key '{SMPL_FRAME_KEY}' "
                f"in modality config, got {loader.modality_configs['action'].modality_keys}"
            )
        if modality_keys is not None and modality_keys != [SMPL_FRAME_KEY]:
            logging.warning(
                "euler mode only evaluates hip absolute euler from '%s'; ignoring --modality-keys %s",
                SMPL_FRAME_KEY,
                modality_keys,
            )
        action_keys = [SMPL_FRAME_KEY]
        logging.info(
            "Root eval space=euler: compare pred processed absolute euler (81D [72:75]) "
            "vs GT hip euler from dataset quat."
        )
        if not isinstance(policy, Gr00tPolicy):
            raise TypeError(
                "--root-process-mode euler requires local Gr00tPolicy "
                "(--model-path); PolicyClient cannot return processed 81D actions."
            )

    pred_action_across_time = []
    pred_trans9d_segments: list[np.ndarray] = []
    gt_trans9d_segments: list[np.ndarray] = []
    pred_abs_frame_segments: list[np.ndarray] = []
    root_chunk_refs: list[tuple[int, np.ndarray]] = []

    gt_root_abs = None
    state_root_abs = None
    gt_frame_abs = None
    if root_process_mode == "trans9d":
        gt_root_abs = _stack_traj_column(traj, f"action.{ROOT_ACTION_KEY}")
        state_root_abs = _stack_traj_column(traj, f"state.{ROOT_ACTION_KEY}")
    elif root_process_mode in ("rot6d", "delta_euler", "euler"):
        gt_frame_abs = _stack_traj_column(traj, f"action.{SMPL_FRAME_KEY}")
        state_root_abs = _stack_traj_column(traj, f"state.{ROOT_ACTION_KEY}")

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    
    # EMA state (xyz may span chunks; euler Δ resets every inference chunk)
    ema_xyz = None

    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        if root_process_mode in PROCESSED_ROOT_MODES:
            if not isinstance(policy, Gr00tPolicy):
                raise TypeError(
                    f"--root-process-mode {root_process_mode} requires local Gr00tPolicy "
                    "(--model-path) so get_action(to_absolute=False) can return "
                    "predicted euler/rot6d in training target space."
                )
            _action_chunk, _ = policy.get_action(
                parsed_obs, options={"to_absolute": False}
            )
        else:
            _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)

        # Last chunk may be shorter than action_horizon (e.g. step 768 with H=48, steps=800).
        horizon = min(action_horizon, actual_steps - step_count)

        if ema_alpha is not None:
            if root_process_mode == "trans9d" and ROOT_ACTION_KEY in action_keys:
                root_key = f"action.{ROOT_ACTION_KEY}"
                root_full = np.asarray(action_chunk[root_key])
                if root_full.ndim == 1:
                    root_full = root_full[None, :]
                root_smoothed, ema_xyz = apply_root_xyz_ema(
                    root_full[:horizon],
                    ema_alpha,
                    ema_xyz,
                )
                root_out = root_full.copy()
                root_out[:horizon] = root_smoothed
                action_chunk[root_key] = root_out
            elif (
                root_process_mode in ("delta_euler", "rot6d")
                and SMPL_FRAME_KEY in action_keys
            ):
                frame_key = f"action.{SMPL_FRAME_KEY}"
                action_chunk[frame_key] = apply_smpl_frame_chunk_ema(
                    action_chunk[frame_key],
                    ema_alpha,
                    root_process_mode,
                    horizon=horizon,
                )

        if root_process_mode == "trans9d":
            assert gt_root_abs is not None and state_root_abs is not None
            pred_root = np.asarray(action_chunk[f"action.{ROOT_ACTION_KEY}"])[:horizon]
            if pred_root.ndim == 1:
                pred_root = pred_root[None, :]
            if pred_root.shape[-1] != Root2Rot6d.PROCESSED_DIM:
                raise ValueError(
                    f"trans9d pred expects processed robot_root "
                    f"({Root2Rot6d.PROCESSED_DIM}D), got {pred_root.shape[-1]}D. "
                    "Need get_action(to_absolute=False)."
                )
            gt_abs_chunk = gt_root_abs[step_count : step_count + horizon]
            reference = state_root_abs[step_count]
            if reference.ndim == 2:
                reference = reference[-1]
            gt_trans9d_segments.append(
                Root2Rot6d.to_relative(gt_abs_chunk, reference, process_xyz=True)
            )
            pred_trans9d_segments.append(pred_root)
        elif root_process_mode == "rot6d":
            assert gt_frame_abs is not None and state_root_abs is not None
            pred_frame = np.asarray(action_chunk[f"action.{SMPL_FRAME_KEY}"])[:horizon]
            if pred_frame.ndim == 1:
                pred_frame = pred_frame[None, :]
            gt_frame = gt_frame_abs[step_count : step_count + horizon]
            reference = state_root_abs[step_count]
            if reference.ndim == 2:
                reference = reference[-1]
            gt_trans9d_segments.append(
                _smpl_frame_to_relative_rot6d(gt_frame, reference)
            )
            pred_trans9d_segments.append(_pred_rot6d_from_processed_frame(pred_frame))
            root_chunk_refs.append((horizon, np.asarray(reference, dtype=np.float32).copy()))
        elif root_process_mode == "delta_euler":
            assert gt_frame_abs is not None and state_root_abs is not None
            pred_frame = np.asarray(action_chunk[f"action.{SMPL_FRAME_KEY}"])[:horizon]
            if pred_frame.ndim == 1:
                pred_frame = pred_frame[None, :]
            gt_frame = gt_frame_abs[step_count : step_count + horizon]
            reference = state_root_abs[step_count]
            if reference.ndim == 2:
                reference = reference[-1]

            gt_euler_delta = _gt_euler_delta_from_absolute_frame(gt_frame, reference)
            pred_euler_delta = _pred_hip_euler_from_processed_frame(pred_frame)
            gt_trans9d_segments.append(gt_euler_delta)
            pred_trans9d_segments.append(pred_euler_delta)
            root_chunk_refs.append((horizon, np.asarray(reference, dtype=np.float32).copy()))
        elif root_process_mode == "euler":
            assert gt_frame_abs is not None
            pred_frame = np.asarray(action_chunk[f"action.{SMPL_FRAME_KEY}"])[:horizon]
            if pred_frame.ndim == 1:
                pred_frame = pred_frame[None, :]
            gt_frame = gt_frame_abs[step_count : step_count + horizon]
            gt_trans9d_segments.append(_gt_euler_absolute_from_absolute_frame(gt_frame))
            pred_trans9d_segments.append(_pred_hip_euler_from_processed_frame(pred_frame))
        else:
            for j in range(horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [
                        np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                        for key in action_keys
                    ],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)
            if SMPL_FRAME_KEY in action_keys:
                pred_abs_frame = np.asarray(action_chunk[f"action.{SMPL_FRAME_KEY}"])[:horizon]
                if pred_abs_frame.ndim == 1:
                    pred_abs_frame = pred_abs_frame[None, :]
                pred_abs_frame_segments.append(pred_abs_frame)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    if root_process_mode == "trans9d":
        # State is absolute 7D; relative action is 9D — skip state overlay in plots.
        state_joints_across_time = np.zeros((actual_steps, 0))
        gt_action_across_time = np.concatenate(gt_trans9d_segments, axis=0)[:actual_steps]
        pred_action_across_time = np.concatenate(pred_trans9d_segments, axis=0)[:actual_steps]
        dim_labels = TRANS9D_LABELS
        title_note = "trans9d (predicted processed 9D vs GT to_relative)"
    elif root_process_mode == "rot6d":
        state_joints_across_time = np.zeros((actual_steps, 0))
        gt_action_across_time = np.concatenate(gt_trans9d_segments, axis=0)[:actual_steps]
        pred_action_across_time = np.concatenate(pred_trans9d_segments, axis=0)[:actual_steps]
        dim_labels = ROT6D_LABELS
        title_note = "rot6d (predicted 84D [72:78] vs GT to_relative)"
    elif root_process_mode == "delta_euler":
        state_joints_across_time = np.zeros((actual_steps, 0))
        gt_action_across_time = np.concatenate(gt_trans9d_segments, axis=0)[:actual_steps]
        pred_action_across_time = np.concatenate(pred_trans9d_segments, axis=0)[:actual_steps]
        dim_labels = DELTA_EULER_LABELS
        title_note = "delta_euler (predicted 81D [72:75] vs GT Δeuler)"
    elif root_process_mode == "euler":
        state_joints_across_time = np.zeros((actual_steps, 0))
        gt_action_across_time = np.concatenate(gt_trans9d_segments, axis=0)[:actual_steps]
        pred_action_across_time = np.concatenate(pred_trans9d_segments, axis=0)[:actual_steps]
        dim_labels = EULER_ABSOLUTE_LABELS
        title_note = "euler (predicted 81D [72:75] vs GT absolute hip euler)"
    else:
        # plot the joints (original absolute-space logic)
        state_joints_across_time = extract_state_joints(
            traj, [f"state.{key}" for key in state_keys]
        )
        gt_action_across_time = extract_state_joints(
            traj, [f"action.{key}" for key in action_keys]
        )[:actual_steps]
        pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
        dim_labels = None
        title_note = ""

    if ema_alpha is not None and root_process_mode != "original":
        ema_target = ema_applies_to(root_process_mode)
        title_note = (
            f"{title_note} | EMA {ema_target} alpha={ema_alpha}"
        ).strip(" |")
        if root_process_mode == "trans9d":
            logging.info(
                "EMA enabled on processed robot_root local xyz (may span inference chunks)"
            )
        elif root_process_mode == "delta_euler":
            logging.info(
                "EMA enabled on processed SMPL frame hip Δeuler dims %d:%d "
                "(reset each inference chunk)",
                SMPL_HIP_EULER_SLICE.start,
                SMPL_HIP_EULER_SLICE.stop,
            )
        elif root_process_mode == "rot6d":
            logging.info(
                "EMA enabled on processed SMPL frame hip rot6d dims %d:%d "
                "(reset each inference chunk)",
                SMPL_HIP_ROT6D_SLICE.start,
                SMPL_HIP_ROT6D_SLICE.stop,
            )

    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    if root_process_mode == "delta_euler":
        wrapped_err = Root2Euler.wrap_to_pi(
            gt_action_across_time - pred_action_across_time
        )
        mse = np.mean(wrapped_err ** 2)
        mae = np.mean(np.abs(wrapped_err))
    else:
        wrapped_err = None
        mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
        mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    if root_process_mode == "trans9d":
        pos_mse = np.mean((gt_action_across_time[:, :3] - pred_action_across_time[:, :3]) ** 2)
        pos_mae = np.mean(np.abs(gt_action_across_time[:, :3] - pred_action_across_time[:, :3]))
        rot_mse = np.mean((gt_action_across_time[:, 3:] - pred_action_across_time[:, 3:]) ** 2)
        rot_mae = np.mean(np.abs(gt_action_across_time[:, 3:] - pred_action_across_time[:, 3:]))
        rot_geodesic_deg = _rotation_geodesic_deg(
            gt_action_across_time[:, 3:], pred_action_across_time[:, 3:]
        )
        logging.info(f"trans9d position MSE (dx,dy,dz): {pos_mse}")
        logging.info(f"trans9d position MAE (dx,dy,dz): {pos_mae}")
        logging.info(f"trans9d rot6d MSE: {rot_mse}")
        logging.info(f"trans9d rot6d MAE: {rot_mae}")
        logging.info(
            "trans9d rotation geodesic deg: mean=%.4f median=%.4f max=%.4f",
            float(np.mean(rot_geodesic_deg)),
            float(np.median(rot_geodesic_deg)),
            float(np.max(rot_geodesic_deg)),
        )
    elif root_process_mode == "rot6d":
        rot_geodesic_deg = _rotation_geodesic_deg(
            gt_action_across_time, pred_action_across_time
        )
        logging.info(
            "rot6d geodesic deg: mean=%.4f median=%.4f max=%.4f",
            float(np.mean(rot_geodesic_deg)),
            float(np.median(rot_geodesic_deg)),
            float(np.max(rot_geodesic_deg)),
        )
    elif root_process_mode == "delta_euler":
        assert wrapped_err is not None
        err_deg = np.degrees(np.abs(wrapped_err))
        logging.info(
            "delta_euler wrap-MAE deg: mean=%.4f median=%.4f max=%.4f "
            "(droll=%.4f dpitch=%.4f dyaw=%.4f)",
            float(np.mean(err_deg)),
            float(np.median(err_deg)),
            float(np.max(err_deg)),
            float(np.mean(err_deg[:, 0])),
            float(np.mean(err_deg[:, 1])),
            float(np.mean(err_deg[:, 2])),
        )

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    resolved_plot_dir = Path(plot_dir) if plot_dir is not None else Path("/tmp/open_loop_eval")
    empty_state = np.zeros((actual_steps, 0))

    if root_process_mode == "trans9d":
        trans9d_path = resolved_plot_dir / f"trans9d_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=action_keys,
            action_horizon=action_horizon,
            save_plot_path=str(trans9d_path),
            dim_labels=TRANS9D_LABELS,
            title_note=title_note,
        )
        logging.info("Saved predicted-trans9d plot: %s", trans9d_path)
    elif root_process_mode == "delta_euler":
        euler_path = resolved_plot_dir / f"euler_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=action_keys,
            action_horizon=action_horizon,
            save_plot_path=str(euler_path),
            dim_labels=DELTA_EULER_LABELS,
            title_note=title_note,
        )
        logging.info("Saved predicted-euler plot: %s", euler_path)

        assert gt_frame_abs is not None
        gt_quat = _hip_quat_from_absolute_frame(gt_frame_abs[:actual_steps])
        pred_quat = _reconstruct_abs_quat_from_chunks(
            pred_action_across_time, root_chunk_refs, _euler_delta_to_abs_quat
        )
        gt_recon_quat = _reconstruct_abs_quat_from_chunks(
            gt_action_across_time, root_chunk_refs, _euler_delta_to_abs_quat
        )
        pred_quat_plot = _align_quat_hemisphere(pred_quat, gt_quat)
        recon_err_deg = _quat_geodesic_deg(gt_recon_quat, gt_quat)
        pred_err_deg = _quat_geodesic_deg(pred_quat, gt_quat)
        logging.info(
            "euler→quat GT reconstruction geodesic deg: mean=%.4f median=%.4f max=%.4f",
            float(np.mean(recon_err_deg)),
            float(np.median(recon_err_deg)),
            float(np.max(recon_err_deg)),
        )
        logging.info(
            "euler→quat pred vs dataset hip quat geodesic deg: "
            "mean=%.4f median=%.4f max=%.4f (aligned quat MAE=%.6f)",
            float(np.mean(pred_err_deg)),
            float(np.median(pred_err_deg)),
            float(np.max(pred_err_deg)),
            float(np.mean(np.abs(gt_quat - pred_quat_plot))),
        )
        quat_path = resolved_plot_dir / f"quat_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_quat,
            pred_action_across_time=pred_quat_plot,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=[SMPL_FRAME_KEY],
            action_horizon=action_horizon,
            save_plot_path=str(quat_path),
            dim_labels=HIP_QUAT_LABELS,
            title_note=(
                "hip quat wxyz: GT=dataset [72:76], "
                "pred=Root2Euler.to_absolute(predicted Δeuler)"
            ),
        )
        logging.info("Saved euler→quat plot: %s", quat_path)
    elif root_process_mode == "rot6d":
        rot6d_path = resolved_plot_dir / f"rot6d_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=action_keys,
            action_horizon=action_horizon,
            save_plot_path=str(rot6d_path),
            dim_labels=ROT6D_LABELS,
            title_note=title_note,
        )
        logging.info("Saved predicted-rot6d plot: %s", rot6d_path)

        assert gt_frame_abs is not None
        gt_quat = _hip_quat_from_absolute_frame(gt_frame_abs[:actual_steps])
        pred_quat = _reconstruct_abs_quat_from_chunks(
            pred_action_across_time, root_chunk_refs, _rot6d_to_abs_quat
        )
        pred_quat_plot = _align_quat_hemisphere(pred_quat, gt_quat)
        pred_err_deg = _quat_geodesic_deg(pred_quat, gt_quat)
        logging.info(
            "rot6d→quat pred vs dataset hip quat geodesic deg: "
            "mean=%.4f median=%.4f max=%.4f (aligned quat MAE=%.6f)",
            float(np.mean(pred_err_deg)),
            float(np.median(pred_err_deg)),
            float(np.max(pred_err_deg)),
            float(np.mean(np.abs(gt_quat - pred_quat_plot))),
        )
        quat_path = resolved_plot_dir / f"quat_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_quat,
            pred_action_across_time=pred_quat_plot,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=[SMPL_FRAME_KEY],
            action_horizon=action_horizon,
            save_plot_path=str(quat_path),
            dim_labels=HIP_QUAT_LABELS,
            title_note=(
                "hip quat wxyz: GT=dataset [72:76], "
                "pred=Root2Rot6d.to_absolute(predicted rot6d)"
            ),
        )
        logging.info("Saved rot6d→quat plot: %s", quat_path)
    elif root_process_mode == "original" and SMPL_FRAME_KEY in action_keys:
        gt_frame_abs_plot = extract_state_joints(traj, [f"action.{SMPL_FRAME_KEY}"])[:actual_steps]
        pred_frame_abs = np.concatenate(pred_abs_frame_segments, axis=0)[:actual_steps]
        if pred_frame_abs.shape[-1] != Root2Euler.FRAME_RAW_DIM:
            raise ValueError(
                f"original mode expects decoded 82D SMPL frame, got {pred_frame_abs.shape[-1]}D. "
                "Use an original-quat checkpoint (not euler/rot6d)."
            )
        gt_quat = _hip_quat_from_absolute_frame(gt_frame_abs_plot)
        pred_quat = _hip_quat_from_absolute_frame(pred_frame_abs)
        logging.info(
            "original SMPL frame MAE: %.6f (hip quat MAE: %.6f)",
            float(np.mean(np.abs(gt_frame_abs_plot - pred_frame_abs))),
            float(np.mean(np.abs(gt_quat - pred_quat))),
        )
        smpl_path = resolved_plot_dir / f"smpl_{traj_id}.jpeg"
        quat_path = resolved_plot_dir / f"quat_{traj_id}.jpeg"
        smpl_labels = [f"frame_{i}" for i in range(Root2Euler.FRAME_RAW_DIM)]
        smpl_labels[72:76] = HIP_QUAT_LABELS
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_frame_abs_plot,
            pred_action_across_time=pred_frame_abs,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=[SMPL_FRAME_KEY],
            action_horizon=action_horizon,
            save_plot_path=str(smpl_path),
            dim_labels=smpl_labels,
            title_note="original decoded SMPL frame 82D (direct quat, no conversion)",
        )
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_quat,
            pred_action_across_time=pred_quat,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=[SMPL_FRAME_KEY],
            action_horizon=action_horizon,
            save_plot_path=str(quat_path),
            dim_labels=HIP_QUAT_LABELS,
            title_note="original hip quat wxyz (82D [72:76], direct prediction)",
        )
        logging.info("Saved SMPL plot: %s", smpl_path)
        logging.info("Saved quat plot: %s", quat_path)
    elif root_process_mode == "euler":
        euler_path = resolved_plot_dir / f"euler_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_action_across_time,
            pred_action_across_time=pred_action_across_time,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=action_keys,
            action_horizon=action_horizon,
            save_plot_path=str(euler_path),
            dim_labels=EULER_ABSOLUTE_LABELS,
            title_note=title_note,
        )
        logging.info("Saved predicted-euler plot: %s", euler_path)
        assert gt_frame_abs is not None
        gt_quat = _hip_quat_from_absolute_frame(gt_frame_abs[:actual_steps])
        pred_quat = _align_quat_hemisphere(
            _euler_absolute_to_abs_quat(pred_action_across_time),
            gt_quat,
        )
        quat_path = resolved_plot_dir / f"quat_{traj_id}.jpeg"
        plot_trajectory_results(
            state_joints_across_time=empty_state,
            gt_action_across_time=gt_quat,
            pred_action_across_time=pred_quat,
            traj_id=traj_id,
            state_keys=state_keys,
            action_keys=[SMPL_FRAME_KEY],
            action_horizon=action_horizon,
            save_plot_path=str(quat_path),
            dim_labels=HIP_QUAT_LABELS,
            title_note="hip quat wxyz: GT=dataset [72:76], pred=euler_to_quat(predicted euler)",
        )
        logging.info("Saved euler→quat plot: %s", quat_path)
    else:
        raise ValueError(
            f"No plot branch for root_process_mode={root_process_mode!r}. "
            "Expected original, trans9d, rot6d, delta_euler, or euler."
        )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Optional plot directory override. If a .jpeg/.png file is given, its parent is used.
    Default: ``{model-path}/open_loop_eval/{[ema_]root-process-mode}/``.
    original: ``smpl_{traj_id}.jpeg`` and ``quat_{traj_id}.jpeg``.
    delta_euler: ``euler_{traj_id}.jpeg`` + ``quat_{traj_id}.jpeg``.
    rot6d: ``rot6d_{traj_id}.jpeg`` + ``quat_{traj_id}.jpeg``.
    trans9d: ``trans9d_{traj_id}.jpeg``.
    """

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    video_backend: str = "pyav"
    """Video decode backend. Use pyav for AV1 datasets (torchcodec often fails)."""

    root_process_mode: str = "original"
    """Must match the checkpoint. Relative euler/rot6d ckpts have no original option.
    - original: direct-quat ckpt; 82D frame + hip quat [72:76]
    - trans9d: WBC Root2Rot6d; predicted processed 9D (to_absolute=False)
    - rot6d: SMPL Root2Rot6d; predicted hip rot6d 84D [72:78]; also saves decoded quat
    - delta_euler: SMPL Root2Euler Δeuler; predicted 81D [72:75]; also saves decoded quat
    - euler: SMPL Root2Euler absolute; predicted 81D [72:75]
    """

    ema_alpha: float | None = None
    """EMA alpha in (0, 1]. Target depends on --root-process-mode:
    trans9d → robot_root local xyz (may span chunks);
    delta_euler → hip Δeuler [72:75] (per inference chunk);
    rot6d → hip rot6d [72:78] (per inference chunk).
    Ignored for original / euler."""


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if args.root_process_mode not in VALID_ROOT_PROCESS_MODES:
        raise ValueError(
            f"root_process_mode must be one of {VALID_ROOT_PROCESS_MODES}, "
            f"got {args.root_process_mode!r}"
        )

    if args.ema_alpha is not None and not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError(f"--ema-alpha must be in (0, 1], got {args.ema_alpha}")
    if args.root_process_mode == "original" and args.ema_alpha is not None:
        logging.warning("--ema-alpha is ignored in --root-process-mode original.")
        args.ema_alpha = None
    elif args.root_process_mode == "euler" and args.ema_alpha is not None:
        logging.warning("--ema-alpha is ignored in --root-process-mode euler (absolute).")
        args.ema_alpha = None
    elif args.ema_alpha is not None:
        ema_target = ema_applies_to(args.root_process_mode)
        logging.info(
            "EMA alpha=%s will smooth: %s",
            args.ema_alpha,
            ema_target or "nothing (unexpected mode)",
        )

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")
    _assert_mode_matches_checkpoint(policy, args.root_process_mode)

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")
    logging.info(f"root_process_mode={args.root_process_mode}")

    plot_dir = _resolve_open_loop_plot_dir(
        model_path=local_model_path,
        root_process_mode=args.root_process_mode,
        save_plot_path=args.save_plot_path,
        ema_alpha=args.ema_alpha,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Open-loop plots directory: %s", plot_dir)

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
            plot_dir=plot_dir,
            root_process_mode=args.root_process_mode,
            ema_alpha=args.ema_alpha,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
