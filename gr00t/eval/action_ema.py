# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared EMA helpers for root-processed action smoothing (open-loop and real-robot).

Includes a rot6d **rotation-aware** smoother (geodesic / SLERP in SO(3)), matching
uniJungleVLA ``vla_server/smoother.py``: do not EMA rot6d columns as scalars.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t.data.state_action.state_action_processor import Root2Rot6d

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
    chunk's state reference (delta_euler / relative rot6d).
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


def _slerp_rot6d(prev_r6: np.ndarray, cur_r6: np.ndarray, alpha: float) -> np.ndarray | None:
    """SO(3) blend matching uniJungle ``smoother._smooth_rotation`` (rot6d).

    ``alpha`` is the weight on the *previous* frame (larger → smoother / more lag):
    ``R_out = Exp(alpha * Log(R_cur^{-1} R_prev)) R_cur``
    i.e. slerp from ``cur`` toward ``prev`` by ``alpha``. Same as::

        rotvec = (R_prev * R_cur.inv()).as_rotvec()
        R_out = Rotation.from_rotvec(rotvec * alpha) * R_cur

    Returns None if either rot6d is degenerate.
    """
    try:
        R_prev = Root2Rot6d.rotation_6d_to_matrix(
            np.asarray(prev_r6, dtype=np.float64).reshape(1, 6)
        )[0]
        R_cur = Root2Rot6d.rotation_6d_to_matrix(
            np.asarray(cur_r6, dtype=np.float64).reshape(1, 6)
        )[0]
    except (ValueError, FloatingPointError):
        return None
    if not (np.all(np.isfinite(R_prev)) and np.all(np.isfinite(R_cur))):
        return None
    rot_prev = Rotation.from_matrix(R_prev)
    rot_cur = Rotation.from_matrix(R_cur)
    # uniJungle: from cur toward prev by alpha
    rotvec = (rot_prev * rot_cur.inv()).as_rotvec()
    rot_out = Rotation.from_rotvec(float(alpha) * rotvec) * rot_cur
    return Root2Rot6d.matrix_to_rotation_6d(rot_out.as_matrix()[None, ...])[0].astype(
        np.float32
    )


def reexpress_relative_rot6d(
    r6_rel: np.ndarray,
    old_reference: np.ndarray,
    new_reference: np.ndarray,
) -> np.ndarray:
    """Rewrite relative hip rot6d from ``old_reference`` into ``new_reference``.

    Needed when SLERP-EMA spans inference chunks in relative mode: each chunk's
    rot6d is ``R_ref^T R_abs``, so carrying the filter state across a new
    ``R_ref`` without this rewrite mixes incompatible bases and recreates the
    inference-boundary jump that ``--ema-alpha`` on trans9d xyz avoids by
    spanning in a comparable local frame.

    ``R_rel_new = R_new^T R_old R_rel_old``.
    """
    old_reference = np.asarray(old_reference, dtype=np.float64).reshape(-1)
    new_reference = np.asarray(new_reference, dtype=np.float64).reshape(-1)
    if old_reference.shape[0] != Root2Rot6d.RAW_DIM:
        raise ValueError(
            f"old_reference must be {Root2Rot6d.RAW_DIM}D, got {old_reference.shape}"
        )
    if new_reference.shape[0] != Root2Rot6d.RAW_DIM:
        raise ValueError(
            f"new_reference must be {Root2Rot6d.RAW_DIM}D, got {new_reference.shape}"
        )
    R_old = Root2Rot6d.quaternion_to_matrix(
        Root2Rot6d._normalize_quaternion(old_reference[3:7])
    )
    R_new = Root2Rot6d.quaternion_to_matrix(
        Root2Rot6d._normalize_quaternion(new_reference[3:7])
    )
    R_rel = Root2Rot6d.rotation_6d_to_matrix(
        np.asarray(r6_rel, dtype=np.float64).reshape(1, 6)
    )[0]
    R_abs = R_old @ R_rel
    R_rel_new = R_new.T @ R_abs
    return Root2Rot6d.matrix_to_rotation_6d(R_rel_new[None, ...])[0].astype(np.float32)


def apply_frame_rot6d_slerp_ema(
    frames: np.ndarray,
    slerp_alpha: float,
    *,
    rot6d_slice: slice = SMPL_HIP_ROT6D_SLICE,
    ema_rot6d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-aware EMA on hip rot6d (may span chunks; pass ``ema_rot6d`` across).

    Matches uniJungle ``SMOOTH_ALPHA_DEFAULTS['rot']`` polarity:
    ``out ≈ alpha * prev + (1-alpha) * cur`` in SO(3); ``slerp_alpha`` larger →
    smoother. Default there is ``rot=0.8``.

    For relative rot6d, callers must re-express ``ema_rot6d`` into the current
    chunk reference via :func:`reexpress_relative_rot6d` before each new chunk.
    """
    validate_ema_alpha(slerp_alpha)
    frames = np.asarray(frames).copy()
    if frames.ndim == 1:
        frames = frames[None, :]
    for i in range(frames.shape[0]):
        cur = frames[i, rot6d_slice]
        if ema_rot6d is None:
            ema_rot6d = np.asarray(cur, dtype=np.float32).copy()
        else:
            blended = _slerp_rot6d(ema_rot6d, cur, slerp_alpha)
            if blended is not None:
                ema_rot6d = blended
            else:
                # Degenerate: same scalar polarity as uniJungle (alpha on prev).
                ema_rot6d = (
                    float(slerp_alpha) * ema_rot6d + (1.0 - float(slerp_alpha)) * cur
                ).astype(np.float32)
        frames[i, rot6d_slice] = ema_rot6d
    return frames, ema_rot6d


def apply_smpl_frame_chunk_rot6d_slerp(
    frame_actions: np.ndarray,
    slerp_alpha: float,
    *,
    horizon: int | None = None,
    ema_rot6d: np.ndarray | None = None,
    chunk_reference: np.ndarray | None = None,
    ema_rot6d_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply rot6d SLERP-EMA on the first ``horizon`` timesteps of a chunk.

    Returns ``(frames, ema_rot6d, ema_rot6d_reference)``. When
    ``chunk_reference`` is set (relative mode), the carried ``ema_rot6d`` is
    rewritten into that reference before blending so inference-boundary jumps
    are damped the same way ``--ema-alpha`` spans trans9d xyz across chunks.
    Absolute mode leaves ``ema_rot6d_reference`` as None.
    """
    frame_full = np.asarray(frame_actions)
    if frame_full.ndim == 1:
        frame_full = frame_full[None, :]
    end = frame_full.shape[0] if horizon is None else min(horizon, frame_full.shape[0])
    frame_out = frame_full.copy()

    if (
        ema_rot6d is not None
        and chunk_reference is not None
        and ema_rot6d_reference is not None
    ):
        ema_rot6d = reexpress_relative_rot6d(
            ema_rot6d, ema_rot6d_reference, chunk_reference
        )

    smoothed, ema_rot6d = apply_frame_rot6d_slerp_ema(
        frame_full[:end], slerp_alpha, ema_rot6d=ema_rot6d
    )
    frame_out[:end] = smoothed
    next_ref = (
        np.asarray(chunk_reference, dtype=np.float32).copy()
        if chunk_reference is not None
        else None
    )
    return frame_out, ema_rot6d, next_ref


def ema_applies_to(root_process_mode: str) -> str | None:
    """Human-readable EMA target for logging/titles."""
    if root_process_mode == "trans9d":
        return "robot_root local xyz"
    if root_process_mode == "delta_euler":
        return "hip Δeuler (per inference chunk)"
    if root_process_mode == "rot6d":
        return "hip rot6d (per inference chunk, column-wise)"
    return None
