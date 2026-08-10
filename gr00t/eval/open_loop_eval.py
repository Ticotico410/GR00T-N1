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
from gr00t.data.state_action.state_action_processor import RootRelative6D
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

Absolute-space open loop (default, original behavior):
    python gr00t/eval/open_loop_eval.py --model-path ... --traj-ids 0

Relative 9D root open loop (local xyz + rot6d, same transform as training):
    python gr00t/eval/open_loop_eval.py --model-path ... --root-eval-space relative9d

"""

ROOT_ACTION_KEY = "robot_root"
ROOT_RELATIVE9D_LABELS = [
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

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

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
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def _stack_traj_column(traj: pd.DataFrame, column: str) -> np.ndarray:
    return np.vstack([np.asarray(arr) for arr in traj[column]])


def _rotation_geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between two batches of rot6d orientations."""
    rot_a = RootRelative6D.rotation_6d_to_matrix(rot6d_a)
    rot_b = RootRelative6D.rotation_6d_to_matrix(rot6d_b)
    rot_err = np.einsum("tij,tkj->tik", rot_a, rot_b)
    trace = rot_err[:, 0, 0] + rot_err[:, 1, 1] + rot_err[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


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


def _apply_root_xyz_ema(
    root_actions: np.ndarray,
    ema_alpha: float,
    ema_xyz: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA-smooth robot_root translation (xyz). Preserves rotation dims unchanged."""
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


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    root_eval_space: str = "absolute",
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

    if root_eval_space == "relative9d":
        if ROOT_ACTION_KEY not in loader.modality_configs["action"].modality_keys:
            raise ValueError(
                f"--root-eval-space relative9d requires action key '{ROOT_ACTION_KEY}' "
                f"in modality config, got {loader.modality_configs['action'].modality_keys}"
            )
        if ROOT_ACTION_KEY not in loader.modality_configs["state"].modality_keys:
            raise ValueError(
                f"--root-eval-space relative9d requires state key '{ROOT_ACTION_KEY}' "
                f"in modality config, got {loader.modality_configs['state'].modality_keys}"
            )
        if modality_keys is not None and modality_keys != [ROOT_ACTION_KEY]:
            logging.warning(
                "relative9d mode only evaluates '%s'; ignoring --modality-keys %s",
                ROOT_ACTION_KEY,
                modality_keys,
            )
        action_keys = [ROOT_ACTION_KEY]
        logging.info(
            "Root eval space=relative9d: compare GT/pred in Unitree local-xyz+rot6d (9D)"
        )

    pred_action_across_time = []
    # relative9d: accumulate (H, 9) segments converted with the same reference as training
    pred_relative9d_segments: list[np.ndarray] = []
    gt_relative9d_segments: list[np.ndarray] = []

    gt_root_abs = None
    state_root_abs = None
    if root_eval_space == "relative9d":
        gt_root_abs = _stack_traj_column(traj, f"action.{ROOT_ACTION_KEY}")
        state_root_abs = _stack_traj_column(traj, f"state.{ROOT_ACTION_KEY}")

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    
    # EMA state
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
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)

        # Last chunk may be shorter than action_horizon (e.g. step 768 with H=48, steps=800).
        horizon = min(action_horizon, actual_steps - step_count)

        # Policy always decodes robot_root back to absolute 7D (xyz+quat). EMA must
        # target robot_root[:, 0:3], not concat index 0:3 (left_hand in WBC layout).
        if ema_alpha is not None and ROOT_ACTION_KEY in action_keys:
            root_key = f"action.{ROOT_ACTION_KEY}"
            root_full = np.asarray(action_chunk[root_key])
            if root_full.ndim == 1:
                root_full = root_full[None, :]
            root_smoothed, ema_xyz = _apply_root_xyz_ema(
                root_full[:horizon],
                ema_alpha,
                ema_xyz,
            )
            root_out = root_full.copy()
            root_out[:horizon] = root_smoothed
            action_chunk[root_key] = root_out

        if root_eval_space == "relative9d":
            assert gt_root_abs is not None and state_root_abs is not None
            pred_abs_chunk = np.asarray(action_chunk[f"action.{ROOT_ACTION_KEY}"])[:horizon]
            if pred_abs_chunk.ndim == 1:
                pred_abs_chunk = pred_abs_chunk[None, :]
            gt_abs_chunk = gt_root_abs[step_count : step_count + horizon]
            reference = state_root_abs[step_count]
            if reference.ndim == 2:
                reference = reference[-1]

            pred_relative9d = RootRelative6D.to_relative(
                pred_abs_chunk, reference, process_xyz=True
            )

            gt_relative9d_segments.append(
                RootRelative6D.to_relative(gt_abs_chunk, reference, process_xyz=True)
            )
            pred_relative9d_segments.append(pred_relative9d)
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

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    if root_eval_space == "relative9d":
        # State is absolute 7D; relative action is 9D — skip state overlay in plots.
        state_joints_across_time = np.zeros((actual_steps, 0))
        gt_action_across_time = np.concatenate(gt_relative9d_segments, axis=0)[:actual_steps]
        pred_action_across_time = np.concatenate(pred_relative9d_segments, axis=0)[:actual_steps]
        dim_labels = ROOT_RELATIVE9D_LABELS
        title_note = "relative9d (local xyz + rot6d)"
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

    if ema_alpha is not None:
        title_note = f"{title_note} | EMA xyz alpha={ema_alpha}".strip(" |")
        root_xyz_offset = 0
        for key in loader.modality_configs["action"].modality_keys:
            if key == ROOT_ACTION_KEY:
                break
            root_xyz_offset += len(traj[f"action.{key}"].iloc[0])
        logging.info(
            "EMA enabled on decoded robot_root xyz; in absolute concat plot dims %d:%d",
            root_xyz_offset,
            root_xyz_offset + 3,
        )

    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    if root_eval_space == "relative9d":
        pos_mse = np.mean((gt_action_across_time[:, :3] - pred_action_across_time[:, :3]) ** 2)
        pos_mae = np.mean(np.abs(gt_action_across_time[:, :3] - pred_action_across_time[:, :3]))
        rot_mse = np.mean((gt_action_across_time[:, 3:] - pred_action_across_time[:, 3:]) ** 2)
        rot_mae = np.mean(np.abs(gt_action_across_time[:, 3:] - pred_action_across_time[:, 3:]))
        rot_geodesic_deg = _rotation_geodesic_deg(
            gt_action_across_time[:, 3:], pred_action_across_time[:, 3:]
        )
        logging.info(f"relative9d position MSE (dx,dy,dz): {pos_mse}")
        logging.info(f"relative9d position MAE (dx,dy,dz): {pos_mae}")
        logging.info(f"relative9d rot6d MSE: {rot_mse}")
        logging.info(f"relative9d rot6d MAE: {rot_mae}")
        logging.info(
            "relative9d rotation geodesic deg: mean=%.4f median=%.4f max=%.4f",
            float(np.mean(rot_geodesic_deg)),
            float(np.median(rot_geodesic_deg)),
            float(np.max(rot_geodesic_deg)),
        )

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
        dim_labels=dim_labels,
        title_note=title_note,
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
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    video_backend: str = "pyav"
    """Video decode backend. Use pyav for AV1 datasets (torchcodec often fails)."""

    root_eval_space: str = "absolute"
    """Action comparison space for open-loop plots/metrics.
    - absolute: original behavior (decoded absolute actions, all modality keys)
    - relative9d: Unitree robot_root only, local xyz + rot6d (same as training target)
    """

    ema_alpha: float | None = None
    """EMA smoothing factor for translation xyz (e.g. 0.1). If provided, applies EMA to smoothing."""


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if args.root_eval_space not in ("absolute", "relative9d"):
        raise ValueError(
            f"root_eval_space must be 'absolute' or 'relative9d', got {args.root_eval_space!r}"
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

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")
    logging.info(f"root_eval_space={args.root_eval_space}")

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
            root_eval_space=args.root_eval_space,
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
