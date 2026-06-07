from __future__ import annotations

import os
import sys
import json
import time
import logging
import threading
import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional
from scipy.spatial.transform import Rotation as R

from get_states import GetStates

WBC_ROOT = Path(__file__).resolve().parents[2] / "wbc_pico_record"
if WBC_ROOT.is_dir() and str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

logger = logging.getLogger(__name__)


def load_g1_urdf(urdf_path: Optional[str] = None) -> str:
    if urdf_path is not None and os.path.isfile(urdf_path):
        return urdf_path
    gr00t_urdf = Path(__file__).resolve().parent.parent / "assets/g1/g1_body29_hand14.urdf"
    wbc_urdf = WBC_ROOT / "assets/g1/g1_body29_hand14.urdf"
    if gr00t_urdf.is_file():
        return str(gr00t_urdf)
    if wbc_urdf.is_file():
        return str(wbc_urdf)
    raise FileNotFoundError(
        f"URDF not found. Tried:\n  {gr00t_urdf}\n  {wbc_urdf}"
    )


class SimSendAction:
    """Meshcat playback via SecureMotionInferencer."""

    def __init__(self, rate_hz: int = 60):
        from utils.inference import SecureMotionInferencer

        self.rate_hz = rate_hz
        self.started = False
        self.running = False
        self.cmd_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.a_key_zero_rotation = None
        self.viz = None

        self.initial_qpos = np.array(
            [
                -0.1465, -0.0014, 0.0332, 0.2938, -0.1785, 0.0235, -0.1917, 0.0155, -0.0031,
                0.3137, -0.1658, -0.0386, 0.0096, -0.0004, 0.0113, 0.2502, 0.2712, -0.0894,
                0.8251, -0.0052, 0.0093, -0.0061, 0.2587, -0.2819, 0.0896, 0.7900, 0.0069,
                0.0108, 0.0038,
            ],
            dtype=np.float32,
        )
        self.initial_root_pose = np.array(
            [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.initial_robot_q = np.concatenate(
            [self.initial_root_pose, self.initial_qpos], dtype=np.float32
        )
        self.cmd = self.initial_robot_q.copy()

        urdf_path = load_g1_urdf()
        enc_model_path = str(WBC_ROOT / "models" / "model.enc")
        if not os.path.exists(enc_model_path):
            raise FileNotFoundError(f"model.enc not found: {enc_model_path}")

        self.inferencer = SecureMotionInferencer(urdf_path, enc_model_path)

        try:
            import pinocchio as pin
            from pinocchio.visualize import MeshcatVisualizer

            import meshcat  # noqa: F401

            model, _, visual_model = pin.buildModelsFromUrdf(
                urdf_path, os.path.dirname(urdf_path), pin.JointModelFreeFlyer()
            )
            self.viz = MeshcatVisualizer(model, None, visual_model)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            logger.info("SimSendAction: Meshcat viewer started.")
        except ImportError as exc:
            logger.warning(
                "Meshcat unavailable (%s). Running sim headless. "
                "Install: pip install meshcat",
                exc,
            )

    def start(self, send_initial_pose: bool = True) -> None:
        if self.started:
            return
        self.a_key_zero_rotation = None
        if send_initial_pose:
            self.send_initial_pose()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.started = False

    def speak(self, text: str) -> None:
        logger.info("[Sim] %s", text)

    def send_initial_pose(self) -> None:
        with self.cmd_lock:
            self.cmd = self.initial_robot_q.copy()

    def send_robot_q(self, robot_q: np.ndarray) -> None:
        if not self.started:
            raise RuntimeError("SimSendAction not started.")
        robot_q = np.asarray(robot_q, dtype=np.float32).reshape(-1)
        if robot_q.shape != (36,):
            raise ValueError(f"robot_q must be (36,), got {robot_q.shape}")
        with self.cmd_lock:
            self.cmd = robot_q.copy()

    def send_hand_cmd(self, hand_cmd: np.ndarray) -> None:
        return

    def send_action(self, action: Dict[str, Any], step: int = 0) -> None:
        robot_q = np.asarray(action["action.robot_q"], dtype=np.float32)[step]
        self.send_robot_q(robot_q)

    def _loop(self) -> None:
        dt_target = 1.0 / float(self.rate_hz)
        last_time = time.time()
        while self.running:
            t0 = time.time()
            with self.cmd_lock:
                cmd = self.cmd.copy()
            now = time.time()
            dt = max(now - last_time, 1e-4)
            last_time = now
            self._process_frame(cmd, dt)
            sleep_t = dt_target - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _process_frame(self, robot_q: np.ndarray, dt: float) -> None:
        motion_vq, root_pose, cmd_wrist, q_vis = self.inferencer.process(robot_q, dt)
        if self.viz is not None and q_vis is not None:
            self.viz.display(q_vis)
        if motion_vq is None:
            return
        raw_quat_wxyz = root_pose[3:7]
        raw_quat_xyzw = np.array(
            [raw_quat_wxyz[1], raw_quat_wxyz[2], raw_quat_wxyz[3], raw_quat_wxyz[0]],
            dtype=np.float64,
        )
        raw_rot = R.from_quat(raw_quat_xyzw)
        if self.a_key_zero_rotation is None:
            self.a_key_zero_rotation = raw_rot
        delta_rot = self.a_key_zero_rotation.inv() * raw_rot
        quat_xyzw = delta_rot.as_quat()
        root_pose[3:7] = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=root_pose.dtype,
        )


class SendAction:
    """Send GR00T inferred actions to the Unitree G1 robot."""

    def __init__(
        self,
        config: Any,
        hand_ctrl,
        get_states: GetStates,
        rate_hz: int = 60,
        urdf_path: Optional[str] = None,
        enc_model_path: str = "models/model.enc",
    ):
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from utils.inference import SecureMotionInferencer

        self.config = config
        self.rate_hz = rate_hz
        self.hand_ctrl = hand_ctrl
        self.get_states = get_states
        self.eef_type = config.eef_type
        self.current_fsm_mode = 504
        logger.info("System initialized. Default mode: 504.")
        self.started = False
        self.running = False
        self.cmd_lock = threading.Lock()
        self.thread = None

        self.initial_qpos = np.array(
            [
                -0.1465, -0.0014, 0.0332, 0.2938, -0.1785, 0.0235, -0.1917, 0.0155, -0.0031,
                0.3137, -0.1658, -0.0386, 0.0096, -0.0004, 0.0113, 0.2502, 0.2712, -0.0894,
                0.8251, -0.0052, 0.0093, -0.0061, 0.2587, -0.2819, 0.0896, 0.7900, 0.0069,
                0.0108, 0.0038,
            ],
            dtype=np.float32,
        )
        self.initial_root_pose = np.array(
            [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.initial_robot_q = np.concatenate(
            [self.initial_root_pose, self.initial_qpos], dtype=np.float32
        )
        self.cmd = self.initial_robot_q.copy()
        self.a_key_zero_rotation = None
        self.motion_vq_full = None

        self.sport_client = LocoClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("LocoClient Ready!")

        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()
        self.audio_client.SetVolume(100)

        self.m_msg_publisher_ = ChannelPublisher("rt/fsm/teleop/cmd", String_)
        self.m_msg_publisher_.Init()
        self.m_msg = String_(data=str(None))

        urdf_path = load_g1_urdf(urdf_path)
        self.inferencer = SecureMotionInferencer(urdf_path, enc_model_path)

    def speak(self, text: str) -> None:
        try:
            self.audio_client.TtsMaker(text, 1)
        except Exception as e:
            print(f"[Warning] Audio announcement failed: {e}")

    def switch_fsm(self, fsm_id: int) -> bool:
        print(f"[SendAction] Switching FSM to {fsm_id}...")
        try:
            self.sport_client.SetTimeout(3.0)
            ret = self.sport_client.SetFsmId(fsm_id)
            print(f"[SendAction] Switch result: {ret}")
            self.sport_client.SetTimeout(0.01)
        except Exception as e:
            print(f"[SendAction] FSM Switch Error: {e}")
            return False
        return True

    def enter_teleop_mode(self) -> None:
        self.current_fsm_mode = 504

        if self.eef_type == "inspire":
            self.hand_ctrl.change_open_pose(self.current_fsm_mode)
            self.hand_ctrl.set_gripper_ratios(0.0, 0.0)
        elif self.eef_type == "dex1":
            self.hand_ctrl.change_open_pose(self.current_fsm_mode)
            self.hand_ctrl.set_gripper_ratios(5.5, 5.5)
        elif self.eef_type == "brainco":
            self.hand_ctrl.change_open_pose(self.current_fsm_mode)
            self.hand_ctrl.set_gripper_ratios(0.0, 0.0)

        if not self.switch_fsm(504):
            raise RuntimeError("Failed to enter FSM 504 teleop mode.")
        logger.info("Enter teleop mode (504)")

    def enter_walking_mode(self) -> None:
        self.current_fsm_mode = 801

        if self.eef_type == "inspire":
            self.hand_ctrl.change_open_pose(self.current_fsm_mode)
            self.hand_ctrl.set_gripper_ratios(0.0, 0.0)
        elif self.eef_type == "dex1":
            self.hand_ctrl.change_open_pose(self.current_fsm_mode)
            self.hand_ctrl.set_gripper_ratios(0.0, 0.0)
        elif self.eef_type == "brainco":
            self.hand_ctrl.change_open_pose(self.current_fsm_mode)
            self.hand_ctrl.set_gripper_ratios(0.0, 0.0)

        if not self.switch_fsm(801):
            raise RuntimeError("Failed to enter FSM 801 walking mode.")
        logger.info("Enter walking mode (801)")

    def dump_json_traj(self, traj_name: str, traj_frames: list):
        p = {"frame": traj_frames, "name": traj_name}
        return json.dumps(p, separators=(",", ":"))

    def process_frame(self, robot_q: np.ndarray, dt: float):
        if self.current_fsm_mode == 801:
            return

        if self.current_fsm_mode == 504:
            motion_vq, root_pose, cmd_wrist, _ = self.inferencer.process(robot_q, dt)

            if motion_vq is not None:
                raw_quat_wxyz = root_pose[3:7]
                raw_quat_xyzw = np.array(
                    [
                        raw_quat_wxyz[1],
                        raw_quat_wxyz[2],
                        raw_quat_wxyz[3],
                        raw_quat_wxyz[0],
                    ],
                    dtype=np.float64,
                )
                raw_rot = R.from_quat(raw_quat_xyzw)

                if self.a_key_zero_rotation is None:
                    self.a_key_zero_rotation = raw_rot
                    logger.info("Captured initial zero rotation.")

                if self.a_key_zero_rotation is not None:
                    delta_rot = self.a_key_zero_rotation.inv() * raw_rot
                    corrected_quat_xyzw = delta_rot.as_quat()
                    root_pose[3:7] = np.array(
                        [
                            corrected_quat_xyzw[3],
                            corrected_quat_xyzw[0],
                            corrected_quat_xyzw[1],
                            corrected_quat_xyzw[2],
                        ],
                        dtype=root_pose.dtype,
                    )

                self.motion_vq_full = np.concatenate(
                    [motion_vq, root_pose, cmd_wrist], axis=-1
                )

                if self.motion_vq_full is not None:
                    self.m_msg.data = self.dump_json_traj(
                        "default", self.motion_vq_full.tolist()
                    )
                    self.m_msg_publisher_.Write(self.m_msg)

    def _loop(self):
        dt_target = 1.0 / float(self.rate_hz)
        last_time = time.time()

        while self.running:
            loop_start = time.time()

            with self.cmd_lock:
                cmd = self.cmd.copy()

            now = time.time()
            dt = now - last_time
            last_time = now

            self.process_frame(cmd, dt)

            sleep_time = dt_target - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self, send_initial_pose: bool = True):
        if self.started:
            return

        self.a_key_zero_rotation = None
        self.motion_vq_full = None

        if send_initial_pose:
            self.send_initial_pose()

        self.enter_teleop_mode()

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return

        self.running = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.enter_walking_mode()
        time.sleep(0.5)

        self.thread = None
        self.started = False

    def send_robot_q(self, robot_q: np.ndarray):
        if not self.started:
            raise RuntimeError("SendAction has not been started.")
        if self.current_fsm_mode != 504:
            raise RuntimeError("Body action can only be submitted while FSM is 504.")

        robot_q = np.asarray(robot_q, dtype=np.float32).reshape(-1)

        if robot_q.shape != (36,):
            raise ValueError(f"send_robot_q expects shape (36,), got {robot_q.shape}.")
        if not np.all(np.isfinite(robot_q)):
            raise ValueError("robot_q contains NaN or infinite values.")
        if np.linalg.norm(robot_q[3:7]) < 1e-8:
            raise ValueError("robot_q contains an invalid zero-norm quaternion.")

        with self.cmd_lock:
            self.cmd = robot_q.copy()

        self.get_states.update_previous_action_root_xyz(robot_q)

    def send_hand_cmd(self, hand_cmd: np.ndarray):
        if not self.started:
            raise RuntimeError("SendAction has not been started.")
        if self.current_fsm_mode != 504:
            raise RuntimeError("Hand action can only be submitted while FSM is 504.")

        hand_cmd = np.asarray(hand_cmd, dtype=np.float64).reshape(-1)

        if hand_cmd.shape != (12,):
            raise ValueError(f"send_hand_cmd expects shape (12,), got {hand_cmd.shape}.")
        if not np.all(np.isfinite(hand_cmd)):
            raise ValueError("hand_cmd contains NaN or infinite values.")

        self.hand_ctrl.set_hand_targets(hand_cmd * 1000.0)

    def send_initial_pose(self) -> None:
        with self.cmd_lock:
            self.cmd = self.initial_robot_q.copy()

    def send_action(self, action: Dict[str, Any], step: int = 0) -> None:
        robot_q = np.asarray(action["action.robot_q"], dtype=np.float64)[step]
        left_hand = np.asarray(action["action.left_hand"], dtype=np.float64)[step]
        right_hand = np.asarray(action["action.right_hand"], dtype=np.float64)[step]

        hand_cmd = np.concatenate(
            [
                np.asarray(left_hand, dtype=np.float64).reshape(-1),
                np.asarray(right_hand, dtype=np.float64).reshape(-1),
            ],
            axis=0,
        )

        self.send_robot_q(robot_q)
        self.send_hand_cmd(hand_cmd)
