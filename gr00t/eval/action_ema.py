# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared EMA helpers for root-processed action smoothing (open-loop and real-robot).
"""

from __future__ import annotations

import numpy as np

SMPL_HIP_EULER_SLICE = slice(72, 75)
SMPL_HIP_QUAT_SLICE = slice(72, 76)
SMPL_HIP_ROT6D_SLICE = slice(72, 78)

SMPL_FRAME_EMA_MODES = ("rot6d", "delta_euler")


def validate_ema_alpha(ema_alpha: float) -> None:
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError(f"--ema-alpha must be in (0, 1], got {ema_alpha}")


def smpl_hip_slice_for_mode(root_process_mode: str) -> slice:
    if root_process_mode == "rot6d":
        return SMPL_HIP_ROT6D_SLICE
    if root_process_mode == "delta_euler":
        return SMPL_HIP_EULER_SLICE
    raise ValueError(
        f"SMPL frame hip EMA supports {SMPL_FRAME_EMA_MODES}, got {root_process_mode!r}"
    )


def apply_root_xyz_ema(
    root_actions: np.ndarray,
    ema_alpha: float,
    ema_xyz: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA-smooth robot_root translation (xyz). Preserves rotation dims unchanged."""
    validate_ema_alpha(ema_alpha)
    root_actions = np.asarray(root_actions).copy()
    if root_actions.ndim == 1:
        root_actions = root_actions[None, :]
    for i in range(root_actions.shape[0]):
        if ema_xyz is None:
            ema_xyz = root_actions[i, 0:3].copy()
        else:
            ema_xyz = ema_alpha * root_actions[i, 0:3] + (1.0 - ema_alpha) * ema_xyz
        root_actions[i, 0:3] = ema_xyz
    return root_actions, ema_xyz


def apply_frame_slice_ema(
    frames: np.ndarray,
    ema_alpha: float,
    value_slice: slice,
) -> np.ndarray:
    """EMA-smooth a frame slice within one inference chunk.

    Callers must reset each chunk: processed hip targets are relative to that
    chunk's state reference (delta_euler / rot6d).
    """
    validate_ema_alpha(ema_alpha)
    frames = np.asarray(frames).copy()
    if frames.ndim == 1:
        frames = frames[None, :]
    ema_state: np.ndarray | None = None
    for i in range(frames.shape[0]):
        value = frames[i, value_slice]
        if ema_state is None:
            ema_state = value.copy()
        else:
            ema_state = ema_alpha * value + (1.0 - ema_alpha) * ema_state
        frames[i, value_slice] = ema_state
    return frames


def apply_smpl_processed_frame_ema(
    frames: np.ndarray,
    ema_alpha: float,
    root_process_mode: str,
) -> np.ndarray:
    """EMA hip rot6d or Δeuler in processed SMPL frame space (per inference chunk)."""
    return apply_frame_slice_ema(
        frames,
        ema_alpha,
        smpl_hip_slice_for_mode(root_process_mode),
    )


def apply_smpl_frame_chunk_ema(
    frame_actions: np.ndarray,
    ema_alpha: float,
    root_process_mode: str,
    *,
    horizon: int | None = None,
) -> np.ndarray:
    """Apply ``apply_smpl_processed_frame_ema`` on the first ``horizon`` timesteps."""
    frame_full = np.asarray(frame_actions)
    if frame_full.ndim == 1:
        frame_full = frame_full[None, :]
    end = frame_full.shape[0] if horizon is None else min(horizon, frame_full.shape[0])
    frame_out = frame_full.copy()
    frame_out[:end] = apply_smpl_processed_frame_ema(
        frame_full[:end],
        ema_alpha,
        root_process_mode,
    )
    return frame_out


def ema_applies_to(root_process_mode: str) -> str | None:
    """Human-readable EMA target for logging/titles."""
    if root_process_mode == "trans9d":
        return "robot_root local xyz"
    if root_process_mode == "delta_euler":
        return "hip Δeuler (per inference chunk)"
    if root_process_mode == "rot6d":
        return "hip rot6d (per inference chunk)"
    return None
