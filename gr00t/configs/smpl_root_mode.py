# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL root training: ``--root-process-mode`` + ``--action-mode`` → processor + modality.

Checkpoint ``processor_config.json`` still stores ``use_relative_euler`` /
``use_state_euler`` / ``use_rot6d`` / ``use_relative_rot6d`` and full
``modality_configs`` (resume uses both).

Modes:
  original     82D hip quat absolute; state drops ``robot_root``.
  rot6d        84D hip rot6d; ``--action-mode`` picks absolute vs ``R_state^T R_action``.
  delta_euler  81D wrap(action_euler - state_euler); fixed relative.
  euler        81D hip euler; action-mode picks absolute vs Δeuler.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

RootProcessMode = Literal["original", "rot6d", "delta_euler", "euler"]
ActionMode = Literal["absolute", "relative"]

VALID_ROOT_PROCESS_MODES: tuple[RootProcessMode, ...] = (
    "original",
    "rot6d",
    "delta_euler",
    "euler",
)

PROCESSOR_FLAG_USE_RELATIVE_EULER = "use_relative_euler"
PROCESSOR_FLAG_USE_STATE_EULER = "use_state_euler"
PROCESSOR_FLAG_USE_ROT6D = "use_rot6d"
PROCESSOR_FLAG_USE_RELATIVE_ROT6D = "use_relative_rot6d"
STATE_ROOT_KEYS = frozenset({"robot_root", "robot_root_current"})


@dataclass(frozen=True)
class SmplRootTrainingSetup:
    root_process_mode: RootProcessMode
    action_mode: ActionMode | None
    use_relative_euler: bool
    use_state_euler: bool
    include_state_robot_root: bool
    use_rot6d: bool = False
    use_relative_rot6d: bool = True


def resolve_smpl_root_training(
    *,
    root_process_mode: RootProcessMode = "original",
    action_mode: ActionMode | None = None,
    legacy_use_relative_euler: bool = False,
    legacy_use_state_euler: bool = False,
) -> SmplRootTrainingSetup:
    """Map CLI / legacy env to processor flags and whether state carries robot_root."""
    if root_process_mode not in VALID_ROOT_PROCESS_MODES:
        raise ValueError(
            f"root_process_mode must be one of {VALID_ROOT_PROCESS_MODES}, "
            f"got {root_process_mode!r}"
        )

    # Legacy env (USE_RELATIVE_EULER / USE_STATE_EULER) when mode left at default original.
    if (
        root_process_mode == "original"
        and legacy_use_relative_euler
        and not legacy_use_state_euler
        and action_mode is None
    ):
        root_process_mode = "euler"
        action_mode = "absolute"
    elif (
        root_process_mode == "original"
        and legacy_use_relative_euler
        and legacy_use_state_euler
    ):
        root_process_mode = "delta_euler"

    if root_process_mode == "original":
        if action_mode == "relative":
            raise ValueError(
                "action-mode=relative is invalid with root-process-mode=original "
                "(82D absolute hip quat; no state.robot_root reference)."
            )
        return SmplRootTrainingSetup(
            root_process_mode="original",
            action_mode=action_mode,
            use_relative_euler=False,
            use_state_euler=False,
            include_state_robot_root=False,
            use_rot6d=False,
            use_relative_rot6d=False,
        )

    if root_process_mode == "rot6d":
        mode = action_mode or "relative"
        if mode not in ("absolute", "relative"):
            raise ValueError(f"action_mode must be 'absolute' or 'relative', got {mode!r}")
        return SmplRootTrainingSetup(
            root_process_mode="rot6d",
            action_mode=mode,
            use_relative_euler=False,
            use_state_euler=False,
            # Absolute rot6d (uniJungle-style) needs no state.robot_root reference.
            include_state_robot_root=(mode == "relative"),
            use_rot6d=True,
            use_relative_rot6d=(mode == "relative"),
        )

    if root_process_mode == "delta_euler":
        if action_mode == "absolute":
            raise ValueError(
                "action-mode=absolute is invalid for root-process-mode=delta_euler. "
                "This mode always learns wrap(action_euler - state_euler)."
            )
        return SmplRootTrainingSetup(
            root_process_mode="delta_euler",
            action_mode="relative",
            use_relative_euler=True,
            use_state_euler=True,
            include_state_robot_root=True,
            use_rot6d=False,
            use_relative_rot6d=False,
        )

    # euler: action-mode selects absolute euler vs Δeuler (default absolute).
    mode = action_mode or "absolute"
    if mode not in ("absolute", "relative"):
        raise ValueError(f"action_mode must be 'absolute' or 'relative', got {mode!r}")
    return SmplRootTrainingSetup(
        root_process_mode="euler",
        action_mode=mode,
        use_relative_euler=True,
        use_state_euler=(mode == "relative"),
        include_state_robot_root=True,
        use_rot6d=False,
        use_relative_rot6d=False,
    )


def infer_root_process_mode_from_processor(
    processor_kwargs: dict,
    *,
    embodiment_tag: str = "unitree_g1_smpl",
) -> RootProcessMode:
    """Reconstruct training mode from a saved checkpoint processor config."""
    rel = bool(processor_kwargs.get(PROCESSOR_FLAG_USE_RELATIVE_EULER, False))
    state_euler = bool(processor_kwargs.get(PROCESSOR_FLAG_USE_STATE_EULER, False))
    use_rot6d = bool(processor_kwargs.get(PROCESSOR_FLAG_USE_ROT6D, False))
    modality = processor_kwargs.get("modality_configs", {}).get(embodiment_tag, {})
    state_keys = set(modality.get("state", {}).get("modality_keys") or [])
    has_root = bool(state_keys.intersection(STATE_ROOT_KEYS))

    if rel and state_euler:
        return "delta_euler"
    if rel and not state_euler:
        return "euler"
    if use_rot6d or (not rel and has_root):
        return "rot6d"
    return "original"


def action_mode_from_processor(
    processor_kwargs: dict,
    *,
    embodiment_tag: str = "unitree_g1_smpl",
) -> ActionMode | None:
    root_mode = infer_root_process_mode_from_processor(
        processor_kwargs, embodiment_tag=embodiment_tag
    )
    if root_mode == "original":
        return None
    if root_mode == "rot6d":
        if PROCESSOR_FLAG_USE_RELATIVE_ROT6D in processor_kwargs:
            return (
                "relative"
                if bool(processor_kwargs[PROCESSOR_FLAG_USE_RELATIVE_ROT6D])
                else "absolute"
            )
        # Legacy rot6d checkpoints were always relative.
        return "relative"
    if root_mode == "delta_euler":
        return "relative"
    state_euler = bool(processor_kwargs.get(PROCESSOR_FLAG_USE_STATE_EULER, False))
    return "relative" if state_euler else "absolute"


def patch_modality_configs_for_root_mode(
    modality_configs: dict,
    embodiment_tag: str,
    setup: SmplRootTrainingSetup,
) -> dict:
    """Ensure state.robot_root presence matches root-process-mode."""
    patched = deepcopy(modality_configs)
    if embodiment_tag not in patched:
        raise KeyError(
            f"Embodiment {embodiment_tag!r} not in modality_configs: "
            f"{sorted(patched)}"
        )
    state_cfg = patched[embodiment_tag]["state"]
    keys = list(state_cfg.modality_keys)

    if setup.include_state_robot_root:
        if not any(k in STATE_ROOT_KEYS for k in keys):
            # Insert robot_root before robot_qpos when missing.
            if "robot_qpos" in keys:
                idx = keys.index("robot_qpos")
                keys.insert(idx, "robot_root")
            else:
                keys.append("robot_root")
    else:
        keys = [k for k in keys if k not in STATE_ROOT_KEYS]

    state_cfg.modality_keys = keys
    return patched


def load_processor_kwargs_from_checkpoint_dir(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "processor_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing processor_config.json under {checkpoint_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["processor_kwargs"]


def find_latest_checkpoint_dir(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = [
        p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not candidates:
        return None

    def _step(path: Path) -> int:
        try:
            return int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=_step)


def resolve_training_output_dir(output_dir: str | Path, experiment_name: str | None) -> Path:
    base = Path(output_dir)
    if experiment_name:
        nested = base / experiment_name
        if find_latest_checkpoint_dir(nested) is not None:
            return nested
    return base


def assert_processor_kwargs_match_setup(
    processor_kwargs: dict,
    setup: SmplRootTrainingSetup,
    *,
    embodiment_tag: str = "unitree_g1_smpl",
    checkpoint_label: str = "checkpoint",
) -> None:
    """Fail fast when resume CLI disagrees with saved processor kwargs."""
    proc = processor_kwargs
    saved_mode = infer_root_process_mode_from_processor(proc, embodiment_tag=embodiment_tag)
    saved_action = action_mode_from_processor(proc, embodiment_tag=embodiment_tag)

    if saved_mode != setup.root_process_mode:
        raise ValueError(
            f"Resume root-process-mode mismatch vs {checkpoint_label}: "
            f"checkpoint={saved_mode!r}, this run={setup.root_process_mode!r}."
        )
    if setup.action_mode is not None and saved_action != setup.action_mode:
        raise ValueError(
            f"Resume action-mode mismatch vs {checkpoint_label}: "
            f"checkpoint={saved_action!r}, this run={setup.action_mode!r}."
        )

    saved_rel = bool(proc.get(PROCESSOR_FLAG_USE_RELATIVE_EULER, False))
    saved_state = bool(proc.get(PROCESSOR_FLAG_USE_STATE_EULER, False))
    if saved_rel != setup.use_relative_euler or saved_state != setup.use_state_euler:
        raise ValueError(
            f"Resume processor flag mismatch vs {checkpoint_label}: "
            f"checkpoint use_relative_euler={saved_rel}, use_state_euler={saved_state}; "
            f"this run use_relative_euler={setup.use_relative_euler}, "
            f"use_state_euler={setup.use_state_euler}."
        )

    if setup.use_rot6d:
        saved_rot6d = bool(proc.get(PROCESSOR_FLAG_USE_ROT6D, False))
        # Legacy: rot6d inferred from has_root without explicit flag.
        if PROCESSOR_FLAG_USE_ROT6D in proc and saved_rot6d != setup.use_rot6d:
            raise ValueError(
                f"Resume use_rot6d mismatch vs {checkpoint_label}: "
                f"checkpoint={saved_rot6d}, this run={setup.use_rot6d}."
            )
        if PROCESSOR_FLAG_USE_RELATIVE_ROT6D in proc:
            saved_rel_r6 = bool(proc[PROCESSOR_FLAG_USE_RELATIVE_ROT6D])
            if saved_rel_r6 != setup.use_relative_rot6d:
                raise ValueError(
                    f"Resume use_relative_rot6d mismatch vs {checkpoint_label}: "
                    f"checkpoint={saved_rel_r6}, this run={setup.use_relative_rot6d}."
                )


def assert_resume_root_setup_matches(
    *,
    output_dir: str | Path,
    experiment_name: str | None,
    setup: SmplRootTrainingSetup,
    embodiment_tag: str = "unitree_g1_smpl",
) -> None:
    """Fail fast when resume CLI disagrees with saved processor config."""
    train_dir = resolve_training_output_dir(output_dir, experiment_name)
    ckpt_dir = find_latest_checkpoint_dir(train_dir)
    if ckpt_dir is None:
        ckpt_dir = find_latest_checkpoint_dir(Path(output_dir))
    if ckpt_dir is None:
        return

    proc = load_processor_kwargs_from_checkpoint_dir(ckpt_dir)
    assert_processor_kwargs_match_setup(
        proc,
        setup,
        embodiment_tag=embodiment_tag,
        checkpoint_label=ckpt_dir.name,
    )
