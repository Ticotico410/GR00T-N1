from __future__ import annotations

import sys
import time
import threading

import numpy as np

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Tuple

from get_images import EpisodeDataSource

WBC_ROOT = Path(__file__).resolve().parents[2] / "wbc_pico_record"
if WBC_ROOT.is_dir() and str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))


def init_robot_dds(network_interface: str) -> None:
    """Initialize CycloneDDS once before any ChannelPublisher/Subscriber."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    net = network_interface.strip()
    try:
        ChannelFactoryInitialize(0, None if net.lower() == "lo" else net)
    except Exception:
        print("[Warning] ChannelFactoryInitialize failed or already initialized.")


class CoordinateTransform:
    """Initialize the first frame and apply the relative transform to the rest frames."""

    def __init__(self) -> None:
        self.first_pos = None
        self.first_rot = None
        self.first_imu_quat = None

    def reset(self) -> None:
        self.first_pos = None
        self.first_rot = None
        self.first_imu_quat = None

    def _apply_qpos_transform(self, qpos):
        if len(qpos) < 7:
            return qpos

        qpos = np.array(qpos).copy()
        curr_pos = qpos[0:3].copy()
        curr_quat_wxyz = qpos[3:7].copy()

        curr_rot = R.from_quat([
            curr_quat_wxyz[1],
            curr_quat_wxyz[2],
            curr_quat_wxyz[3],
            curr_quat_wxyz[0],
        ])

        if self.first_pos is None:
            self.first_pos = curr_pos.copy()
            self.first_rot = curr_rot
            qpos[0:3] = np.array([0.0, 0.0, curr_pos[2]])
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            return qpos.tolist()

        relative_pos_world = curr_pos - self.first_pos
        first_yaw = self.first_rot.as_euler("xyz")[2]
        yaw_rot = R.from_euler("z", first_yaw)
        relative_pos_body = yaw_rot.inv().apply(relative_pos_world)
        new_pos = np.array([
            relative_pos_body[0],
            relative_pos_body[1],
            curr_pos[2],
        ])

        relative_rot = self.first_rot.inv() * curr_rot
        final_rot = R.from_quat([0, 0, 0, 1]) * relative_rot

        final_xyzw = final_rot.as_quat()
        new_quat_wxyz = np.array([
            final_xyzw[3],
            final_xyzw[0],
            final_xyzw[1],
            final_xyzw[2],
        ])
        qpos[0:3] = new_pos
        qpos[3:7] = new_quat_wxyz

        return qpos.tolist()

    def _apply_imu_quat_transform(self, quat_wxyz):
        quat_wxyz = np.array(quat_wxyz).copy()

        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ])
        curr_rot = R.from_quat(quat_xyzw)

        if self.first_imu_quat is None:
            self.first_imu_quat = curr_rot
            return np.array([1.0, 0.0, 0.0, 0.0])

        relative_rot = self.first_imu_quat.inv() * curr_rot
        relative_xyzw = relative_rot.as_quat()
        new_quat_wxyz = np.array([
            relative_xyzw[3],
            relative_xyzw[0],
            relative_xyzw[1],
            relative_xyzw[2],
        ])
        return new_quat_wxyz


class GetStates:
    """Get real-time robot state via DDS and construct GR00T state observations."""

    GR00T_STATE_KEYS = ("state.left_hand", "state.right_hand", "state.robot_q")

    def __init__(self, config: Any, hand_ctrl):
        self.config = config
        self.closed = False
        self.state_ready = False
        self.state_lock = threading.Lock()
        self.coord_transform = CoordinateTransform()
        self.previous_action_root_xyz = np.array([0.0, 0.0, 0.74], dtype=np.float32)
        self.first_root_observation = True
        self.current_joint_pos = np.zeros(29, dtype=np.float32)
        self.current_imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.eef_type = config.eef_type
        self.hand_ctrl = hand_ctrl
        self.low_state_sub = None
        self.init_dds()
        print("DDS publishers initialized.")

    def init_dds(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

        self._LowStateHG = LowStateHG

        init_robot_dds(self.config.network_interface)

        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.low_state_sub.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg) -> None:
        if self.closed:
            return

        self.low_state = msg

        if len(self.low_state.motor_state) < 29:
            return

        actual_joint_pos = np.array(
            [self.low_state.motor_state[i].q for i in range(29)],
            dtype=np.float32,
        )

        imu_msg = self.low_state.imu_state
        actual_imu_quat = np.array(
            [
                imu_msg.quaternion[0],
                imu_msg.quaternion[1],
                imu_msg.quaternion[2],
                imu_msg.quaternion[3],
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(actual_joint_pos)):
            return

        if not np.all(np.isfinite(actual_imu_quat)):
            return

        if np.linalg.norm(actual_imu_quat) < 1e-8:
            return

        with self.state_lock:
            self.current_joint_pos = actual_joint_pos
            self.current_imu_quat = actual_imu_quat
            self.state_ready = True

    def wait_for_states(
        self, timeout: float = 5.0, poll_interval: float = 0.01
    ) -> None:
        if self.state_ready:
            return

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.closed:
                raise RuntimeError("GetStates has already been closed.")

            with self.state_lock:
                if self.state_ready:
                    return

            time.sleep(poll_interval)

        raise TimeoutError("No valid rt/lowstate received before timeout.")

    def get_hand_state(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.hand_ctrl is None:
            raise RuntimeError("Hand controller has not been initialized.")
        l_hand_state, r_hand_state = self.hand_ctrl.get_hand_states()
        l_hand_state = np.asarray(l_hand_state, dtype=np.float32).reshape(-1)
        r_hand_state = np.asarray(r_hand_state, dtype=np.float32).reshape(-1)
        return l_hand_state, r_hand_state

    def reset_episode_transform(self) -> None:
        self.coord_transform.reset()

        with self.state_lock:
            self.previous_action_root_xyz = np.array(
                [0.0, 0.0, 0.74],
                dtype=np.float32,
            )
            self.first_root_observation = True

    def update_previous_action_root_xyz(self, action_robot_q: np.ndarray) -> None:
        action_robot_q = np.asarray(action_robot_q, dtype=np.float32)

        if action_robot_q.shape != (36,):
            raise ValueError(
                "update_previous_action_root_xyz expects one executed "
                f"action.robot_q with shape (36,), got {action_robot_q.shape}."
            )

        with self.state_lock:
            self.previous_action_root_xyz = action_robot_q[:3].copy()

    def build_robot_q_current(self) -> np.ndarray:
        with self.state_lock:
            previous_action_root_xyz = self.previous_action_root_xyz.copy()
            first_root_observation = self.first_root_observation
            actual_full = self.current_joint_pos.copy()
            actual_imu_quat = self.current_imu_quat.copy()

        if first_root_observation:
            desired_full = np.zeros(7 + 29, dtype=np.float32)
            desired_full[0:3] = np.array([0.0, 0.0, 0.74], dtype=np.float32)
            desired_full[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            desired_full_transformed = self.coord_transform._apply_qpos_transform(
                desired_full
            )
            root_xyz = np.asarray(
                desired_full_transformed[:3],
                dtype=np.float32,
            )

            with self.state_lock:
                self.first_root_observation = False
        else:
            root_xyz = previous_action_root_xyz

        if actual_imu_quat is not None:
            transformed_imu_quat = self.coord_transform._apply_imu_quat_transform(
                actual_imu_quat
            )
        else:
            transformed_imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        actual_full_combined = np.concatenate([
            root_xyz,
            transformed_imu_quat,
            actual_full,
        ]).astype(np.float32, copy=False)

        return actual_full_combined

    def process_states_obs(self) -> Dict[str, np.ndarray]:
        if self.closed:
            raise RuntimeError("GetStates has already been closed.")

        self.wait_for_states()

        left_hand, right_hand = self.get_hand_state()
        robot_q_current = self.build_robot_q_current()

        return {
            "state.left_hand": left_hand[np.newaxis, :],
            "state.right_hand": right_hand[np.newaxis, :],
            "state.robot_q": robot_q_current[np.newaxis, :],
        }

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True

        if self.low_state_sub is not None:
            if hasattr(self.low_state_sub, "Close"):
                self.low_state_sub.Close()
            elif hasattr(self.low_state_sub, "Stop"):
                self.low_state_sub.Stop()

        self.hand_ctrl = None
        self.low_state_sub = None


class SimGetStates:
    def __init__(self, episode: EpisodeDataSource):
        self.episode = episode
        self.closed = False

    def wait_for_states(
        self, timeout: float = 5.0, poll_interval: float = 0.01
    ) -> None:
        return

    def process_states_obs(self) -> Dict[str, np.ndarray]:
        if self.closed:
            raise RuntimeError("SimGetStates is closed.")
        states = self.episode.current_frame().get("states", {})
        robot_q = np.asarray(states.get("robot_q_current", []), dtype=np.float32).reshape(-1)
        hand = np.asarray(states.get("hand_state", []), dtype=np.float32).reshape(-1)
        if robot_q.size != 36:
            raise ValueError(f"robot_q_current must be 36D, got {robot_q.size}")
        if hand.size != 12:
            raise ValueError(f"hand_state must be 12D, got {hand.size}")
        return {
            "state.robot_q": robot_q[np.newaxis, :],
            "state.left_hand": hand[:6][np.newaxis, :],
            "state.right_hand": hand[6:12][np.newaxis, :],
        }

    def reset_episode_transform(self) -> None:
        return

    def close(self) -> None:
        self.closed = True
