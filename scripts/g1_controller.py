from __future__ import annotations

import os
import sys


def _preload_conda_libstdcxx() -> None:
    """wbc_pico pinocchio needs conda libstdc++ (GLIBCXX_3.4.31), not system default."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    libstdcxx = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.isfile(libstdcxx):
        import ctypes

        ctypes.CDLL(libstdcxx, mode=ctypes.RTLD_GLOBAL)


_preload_conda_libstdcxx()

import time
import signal
import logging
import argparse
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

WBC_ROOT = Path(__file__).resolve().parents[2] / "wbc_pico_record"
if WBC_ROOT.is_dir() and str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

from policy_client import PolicyClient
from get_images import EpisodeDataSource, GetImages, SimGetImages
from get_states import GetStates, SimGetStates, init_robot_dds
from send_action import SendAction, SimSendAction

logger = logging.getLogger(__name__)


@dataclass
class RobotConfig:
    """Configuration for Unitree G1 robot deploy and I/O."""

    mode: str = "real"

    # DDS / robot identity
    network_interface: str = "enp5s0"
    robot_id: str = "7297"

    # Camera ZMQ on robot
    robot_ip: str = "192.168.123.102"
    robot_port: int = 5555

    # GR00T inference server
    policy_host: str = "localhost"
    policy_port: int = 5555

    # End-effector
    eef_type: str = "inspire"

    # Language instruction
    language_instruction: str = (
        "Pick up the orange bottle and put it in the pink plate."
    )

    # Inference / control loop parameters
    control_hz: float = 30.0
    body_rate_hz: int = 60
    action_horizon: int = 32
    warmup_sec: float = 2.0
    max_chunks: Optional[int] = None

    # Temporal aggregation (ACT-style ensembling over overlapping chunks)
    temporal_agg: bool = False
    temporal_agg_k: float = 0.05

    # Simulation specific parameters
    episode_path: str = ""
    max_frames: Optional[int] = None
    compare_gt: bool = False

    @property
    def is_sim(self) -> bool:
        return self.mode == "sim"

    @property
    def is_real(self) -> bool:
        return self.mode == "real"

    @classmethod
    def from_cli(cls, argv: Optional[list[str]] = None) -> "RobotConfig":
        base = cls()
        parser = argparse.ArgumentParser(
            description="Unitree G1 GR00T deploy: dataset/real obs -> policy -> action.",
        )
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--real",
            action="store_const",
            const="real",
            dest="mode",
            help="Real robot: ZMQ cameras + DDS states + FSM 504 (default).",
        )
        mode_group.add_argument(
            "--sim",
            action="store_const",
            const="sim",
            dest="mode",
            help="Simulation: MCAP obs + Meshcat predicted action playback.",
        )
        parser.set_defaults(mode=base.mode)
        parser.add_argument("--net", type=str, default=base.network_interface)
        parser.add_argument("--robot-id", type=str, default=base.robot_id)
        parser.add_argument("--robot-ip", type=str, default=base.robot_ip)
        parser.add_argument("--robot-port", type=int, default=base.robot_port)
        parser.add_argument("--policy-host", type=str, default=base.policy_host)
        parser.add_argument("--policy-port", type=int, default=base.policy_port)
        parser.add_argument("--eef", type=str, default=base.eef_type, choices=["inspire", "dex1", "brainco"])
        parser.add_argument("--language", type=str, default=base.language_instruction)
        parser.add_argument("--control-hz", type=float, default=base.control_hz)
        parser.add_argument("--action-horizon", type=int, default=base.action_horizon)
        parser.add_argument("--max-chunks", type=int, default=base.max_chunks)
        parser.add_argument("--body-rate-hz", type=int, default=base.body_rate_hz)
        parser.add_argument("--warmup-sec", type=float, default=base.warmup_sec)
        parser.add_argument("--episode", type=str, default=base.episode_path)
        parser.add_argument("--max-frames", type=int, default=base.max_frames)
        parser.add_argument("--compare-gt", action="store_true")
        parser.add_argument("--temporal-agg", action="store_true")
        parser.add_argument("--temporal-agg-k", type=float, default=base.temporal_agg_k)
        args = parser.parse_args(argv)

        return cls(
            mode=args.mode,
            network_interface=args.net,
            robot_id=args.robot_id,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            eef_type=args.eef,
            language_instruction=args.language,
            control_hz=args.control_hz,
            action_horizon=args.action_horizon,
            max_chunks=args.max_chunks,
            body_rate_hz=args.body_rate_hz,
            warmup_sec=args.warmup_sec,
            episode_path=args.episode,
            max_frames=args.max_frames,
            compare_gt=args.compare_gt,
            temporal_agg=args.temporal_agg,
            temporal_agg_k=args.temporal_agg_k,
        )


def build_obs(
    get_images: Union[GetImages, SimGetImages],
    get_states: Union[GetStates, SimGetStates],
    language_instruction: str,
) -> Dict[str, Any]:
    obs: Dict[str, Any] = {}
    obs.update(get_images.process_images_obs())
    obs.update(get_states.process_states_obs())
    obs["annotation.human.task_description"] = [language_instruction]
    return obs


ACTION_CHUNK_KEYS = ("action.robot_q", "action.left_hand", "action.right_hand")


def _action_chunk_to_steps(value: Any) -> list[np.ndarray]:
    arr = np.asarray(value)
    if arr.ndim == 1:
        return [arr.reshape(-1)]
    if arr.ndim == 2:
        return [arr[i].reshape(-1) for i in range(arr.shape[0])]
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {arr.shape}.")
        return [arr[0, i].reshape(-1) for i in range(arr.shape[1])]
    raise ValueError(f"Unsupported action shape {arr.shape}.")


def _as_single_step_action(step_action: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Wrap 1D vectors as (1, dim) for send_action.send_action(..., step=0)."""
    return {key: np.asarray(vec, dtype=np.float32)[np.newaxis, :] for key, vec in step_action.items()}


class TemporalActionAggregator:
    """ACT-style buffer: all_time_actions[infer_t, t] = prediction for global step t."""

    def __init__(self, horizon: int, exp_k: float = 0.01) -> None:
        self.horizon = horizon
        self.exp_k = exp_k
        self._cells: Dict[tuple[int, int], Dict[str, np.ndarray]] = {}

    def record_chunk(self, inference_t: int, action: Dict[str, Any], chunk_len: int) -> None:
        steps_per_key: Dict[str, list[np.ndarray]] = {}
        for key in ACTION_CHUNK_KEYS:
            if key not in action:
                continue
            steps_per_key[key] = _action_chunk_to_steps(action[key])

        step_lengths = [len(v) for v in steps_per_key.values()]
        n_steps = min(chunk_len, *step_lengths) if step_lengths else chunk_len
        for offset in range(n_steps):
            global_t = inference_t + offset
            cell: Dict[str, np.ndarray] = {}
            for key, steps in steps_per_key.items():
                cell[key] = np.asarray(steps[offset], dtype=np.float64).reshape(-1)
            self._cells[(inference_t, global_t)] = cell

    def aggregate(self, global_t: int) -> Optional[Dict[str, np.ndarray]]:
        rows: list[tuple[int, Dict[str, np.ndarray]]] = []
        for infer_t in range(max(0, global_t - self.horizon + 1), global_t + 1):
            cell = self._cells.get((infer_t, global_t))
            if cell is not None:
                rows.append((infer_t, cell))

        if not rows:
            return None

        weights = np.exp(-self.exp_k * np.arange(len(rows), dtype=np.float64))
        weights /= weights.sum()

        out: Dict[str, np.ndarray] = {}
        for key in ACTION_CHUNK_KEYS:
            stacked = np.stack([cell[key] for _, cell in rows if key in cell], axis=0)
            if stacked.size == 0:
                continue
            w = weights[: stacked.shape[0]][:, np.newaxis]
            out[key] = (stacked * w).sum(axis=0)
        return out if out else None


def infer_action_horizon(action: Dict[str, Any], default: int = 16) -> int:
    robot_q = action.get("action.robot_q")
    if robot_q is None:
        return default

    arr = np.asarray(robot_q)
    if arr.ndim == 1:
        return 1
    if arr.ndim == 2:
        return int(arr.shape[0])
    if arr.ndim == 3:
        return int(arr.shape[1])

    return default


def log_sim_gt_error(frame_idx: int, action: Dict[str, Any], frame: dict) -> None:
    gt_error = np.asarray(frame.get("actions", {}).get("robot_q_desired", []), dtype=np.float32).reshape(-1)
    if gt_error.size != 36:
        return
    pred_error = np.asarray(action["action.robot_q"], dtype=np.float32)[0].reshape(-1)
    logger.info(
        "Sim frame %d | GT error: total=%.4f | pos=%.4f | joint=%.4f",
        frame_idx,
        float(np.linalg.norm(pred_error - gt_error)),
        float(np.linalg.norm(pred_error[:3] - gt_error[:3])),
        float(np.linalg.norm(pred_error[7:] - gt_error[7:])),
    )


def run(config: RobotConfig, policy) -> None:
    hand_ctrl = None
    get_states = None
    get_images = None
    send_action = None
    episode: Optional[EpisodeDataSource] = None
    language = config.language_instruction

    running = True

    def _request_stop(signum, frame) -> None:
        nonlocal running
        logger.info("Stop signal received (%s), exiting...", signum)
        running = False

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        if config.is_sim:
            if not config.episode_path:
                raise ValueError("Sim mode requires episode_path (set in RobotConfig or --episode).")
            episode = EpisodeDataSource(config.episode_path, config.language_instruction)
            get_images = SimGetImages(episode)
            get_states = SimGetStates(episode)
            send_action = SimSendAction(rate_hz=config.body_rate_hz)
            language = episode.language_instruction
            logger.info(
                "Sim | episode=%s frames=%d | policy=%s:%d",
                config.episode_path,
                episode.num_frames,
                config.policy_host,
                config.policy_port,
            )
        else:
            from eef.inspire.ftp_hand import InspireFTPHandController

            init_robot_dds(config.network_interface)
            hand_ctrl = InspireFTPHandController()
            get_states = GetStates(config, hand_ctrl)
            get_images = GetImages(config)
            enc_model_path = str(WBC_ROOT / "models" / "model.enc")
            send_action = SendAction(
                config,
                hand_ctrl,
                get_states,
                rate_hz=config.body_rate_hz,
                enc_model_path=enc_model_path,
            )
            logger.info(
                "Real | camera=%s:%d | policy=%s:%d",
                config.robot_ip,
                config.robot_port,
                config.policy_host,
                config.policy_port,
            )
            logger.info("Waiting for camera and lowstate (%.1fs)...", config.warmup_sec)
            get_images.wait_for_images(timeout=max(config.warmup_sec, 5.0))
            get_states.wait_for_states(timeout=max(config.warmup_sec, 5.0))
            get_states.reset_episode_transform()

        send_action.start(send_initial_pose=True)
        send_action.speak(
            "GR00T sim deploy started" if config.is_sim else "GR00T deploy started"
        )

        dt = 1.0 / float(config.control_hz)
        chunk_idx = 0
        real_step_t = 0
        temporal: Optional[TemporalActionAggregator] = None
        if config.temporal_agg:
            temporal = TemporalActionAggregator(
                horizon=config.action_horizon,
                exp_k=config.temporal_agg_k,
            )
        logger.info(
            "Loop: control_hz=%.1f action_horizon=%d temporal_agg=%s mode=%s",
            config.control_hz,
            config.action_horizon,
            config.temporal_agg,
            config.mode,
        )

        while running:
            if config.is_sim and episode is not None and episode.exhausted():
                break
            if config.max_chunks is not None and chunk_idx >= config.max_chunks:
                logger.info("Reached max_chunks=%d, stopping.", config.max_chunks)
                break

            obs = build_obs(get_images, get_states, language)

            infer_start = time.monotonic()
            action = policy.get_action(obs)
            infer_ms = (time.monotonic() - infer_start) * 1000.0

            chunk_len = min(
                infer_action_horizon(action, default=config.action_horizon),
                config.action_horizon,
            )
            if chunk_len <= 0:
                logger.warning("Empty action chunk, skipping.")
                continue

            inference_t = episode.frame_idx if episode is not None else real_step_t
            if temporal is not None:
                temporal.record_chunk(inference_t, action, chunk_len)

            for step in range(chunk_len):
                if not running:
                    break
                global_t = inference_t + step
                step_start = time.monotonic()
                if temporal is not None:
                    merged = temporal.aggregate(global_t)
                    if merged is None:
                        send_action.send_action(action, step=step)
                    else:
                        send_action.send_action(_as_single_step_action(merged), step=0)
                else:
                    send_action.send_action(action, step=step)
                sleep_time = dt - (time.monotonic() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if episode is None:
                    real_step_t = global_t + 1

            chunk_idx += 1
            logger.info(
                "Chunk %d | infer=%.0fms steps=%d",
                chunk_idx,
                infer_ms,
                chunk_len,
            )

            if config.is_sim and episode is not None:
                if config.compare_gt:
                    if temporal is not None:
                        merged = temporal.aggregate(inference_t)
                        if merged is not None:
                            log_sim_gt_error(
                                inference_t,
                                _as_single_step_action(merged),
                                episode.current_frame(),
                            )
                        else:
                            log_sim_gt_error(
                                inference_t, action, episode.current_frame()
                            )
                    else:
                        log_sim_gt_error(
                            inference_t, action, episode.current_frame()
                        )
                episode.advance(chunk_len)
                if config.max_frames is not None and episode.frame_idx >= config.max_frames:
                    logger.info("Reached max_frames=%d", config.max_frames)
                    break
                if episode.exhausted():
                    logger.info(
                        "Episode done at frame %d/%d",
                        episode.frame_idx,
                        episode.num_frames,
                    )
                    break

    except Exception:
        logger.exception("Deploy failed")
        raise
    finally:
        logger.info("Shutting down...")
        if send_action is not None:
            try:
                send_action.stop()
            except Exception as e:
                logger.warning("send_action.stop failed: %s", e)
        if get_images is not None:
            try:
                get_images.close()
            except Exception as e:
                logger.warning("get_images.close failed: %s", e)
        if get_states is not None:
            try:
                get_states.close()
            except Exception as e:
                logger.warning("get_states.close failed: %s", e)
        if hand_ctrl is not None and hasattr(hand_ctrl, "stop"):
            try:
                hand_ctrl.stop()
            except Exception as e:
                logger.warning("hand_ctrl.stop failed: %s", e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = RobotConfig.from_cli()
    policy = PolicyClient(
        host=config.policy_host,
        port=config.policy_port,
    )
    run(config, policy)


if __name__ == "__main__":
    main()
