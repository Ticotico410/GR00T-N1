from __future__ import annotations

import os
import sys
import json
import time
import signal
import pickle
import shutil
import logging
import argparse
import threading
import onnxruntime

import numpy as np
import multiprocessing as mp

from pathlib import Path
from threading import Lock
from collections import deque
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R

# Hand controllers
from eef.dex1.dex1 import Dex1
from eef.brainco.brainco import Brainco
from eef.inspire.ftp_hand import InspireFTPHandController

# Image client
from teleop.image_server.image_client import ImageClient

# WBC teleoperation API
from utils.inference import SecureMotionInferencer

# Unitree SDK 2.0 tools
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_ as IMUStateHG
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__IMUState_


# wbc_pico_record (ImageClient, Inspire hand, DDS helpers)
WBC_ROOT = Path(__file__).resolve().parents[2] / "wbc_pico_record"
if WBC_ROOT.is_dir() and str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

GR00T_ROOT = Path(__file__).resolve().parents[1]
if str(GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(GR00T_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class RobotConfig:
    """Configuration for Unitree G1 robot."""
    # Robot communication parameters
    network_interface: str = "enp5s0"
    robot_id: str = "7297"
    robot_port: int = 5555
    robot_ip: str = "192.168.123.102"

    # End-effector type
    eef_type: str = "inspire"
    
    # Language observation
    language_instruction: str = "Pick up the orange bottle and put it in the pink plate."
    


class GetImages:
    """
    Get images from the robot's camera via ZMQ protocol.
    """
    # unitree_g1_wbc video keys
    GR00T_VIDEO_KEYS = {
        "color_0": "video.head_stereo_left",
        "color_1": "video.head_stereo_right",
        "color_2": "video.wrist_left",
        "color_3": "video.wrist_right",
    }
    COLOR_VIEW_KEYS = ("color_0", "color_1", "color_2", "color_3")


    def __init__(self, config: RobotConfig):
        # End-effector type
        self.eef_type = config.eef_type

        # runtime status
        self.images_ready = False
        self.closed = False
        self.receive_thread = None
        self.img_client = None

        # Head camera
        self.tv_img_shape = (480, 1280, 3)
        self.tv_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(self.tv_img_shape) * np.uint8().itemsize
        )
        self.tv_img = np.ndarray(
            self.tv_img_shape, dtype=np.uint8, buffer=self.tv_shm.buf
        )
        self.tv_img.fill(0)

        # Wrist camera
        self.wrist_img_shape = (480, 1280, 3)
        self.wrist_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(self.wrist_img_shape) * np.uint8().itemsize
        )
        self.wrist_img = np.ndarray(
            self.wrist_img_shape, dtype=np.uint8, buffer=self.wrist_shm.buf
        )
        self.wrist_img.fill(0)

        if self.eef_type == "inspire":
            self.img_client = ImageClient(
                tv_img_shape=self.tv_img_shape,
                tv_img_shm_name=self.tv_shm.name,
                wrist_img_shape=self.wrist_img_shape,
                wrist_img_shm_name=self.wrist_shm.name
            )

        self.receive_thread = threading.Thread(
            target=self.img_client.receive_process,
            daemon=True
        )
        self.receive_thread.start()


    def wait_for_images(self, timeout: float = 5.0, poll_interval: float = 0.01) -> None:
        """
        Wait until the image client has written at least one frame into shared memory.
        """
        if self.images_ready:
            return

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.closed:
                raise RuntimeError("GetImages has already been closed.")

            head = self.tv_img.copy()
            wrist = self.wrist_img.copy()

            if np.any(head) and np.any(wrist):
                self.images_ready = True
                return

            time.sleep(poll_interval)

        raise TimeoutError("No valid head/wrist camera frames received before timeout.")


    def process_images_obs(self) -> Dict[str, np.ndarray]:
        """
        Process images from the robot's camera to GR00T video.* observation keys.

        Returns:
            {
                "video.head_stereo_left": np.ndarray, shape (1, H, W, 3), dtype uint8,
                "video.head_stereo_right": np.ndarray, shape (1, H, W, 3), dtype uint8,
                "video.wrist_left": np.ndarray, shape (1, H, W, 3), dtype uint8,
                "video.wrist_right": np.ndarray, shape (1, H, W, 3), dtype uint8,
            }
        """
        if self.closed:
            raise RuntimeError("GetImages has already been closed.")

        # Only blocks during startup. After the first valid frame this returns directly.
        self.wait_for_images()

        head = self.tv_img.copy()
        wrist = self.wrist_img.copy()

        mid_h = head.shape[1] // 2
        mid_w = wrist.shape[1] // 2

        views = {
            "color_0": head[:, :mid_h],
            "color_1": head[:, mid_h:],
            "color_2": wrist[:, :mid_w],
            "color_3": wrist[:, mid_w:],
        }

        images_obs = {}

        for k in self.COLOR_VIEW_KEYS:
            images_obs[self.GR00T_VIDEO_KEYS[k]] = np.ascontiguousarray(
                views[k][np.newaxis, ...],
                dtype=np.uint8,
            )

        return images_obs


    def shutdown_image_client(self, join_timeout: float = 2.0) -> None:
        """
        Stop ZMQ receive loop and make sure the receive thread has exited
        before shared memory can be released.
        """
        client = self.img_client
        thread = self.receive_thread

        if client is None:
            return

        # Request the receive loop to stop.
        if hasattr(client, "running"):
            client.running = False

        # Close the ZMQ socket/context first so that a blocking receive call
        # can return and the receive thread can terminate.
        if hasattr(client, "_close"):
            try:
                client._close()
            except Exception as e:
                print(f"[Warning] Failed to close image client transport: {e}")
        elif hasattr(client, "close"):
            try:
                client.close()
            except Exception as e:
                print(f"[Warning] Failed to close image client transport: {e}")

        # Shared memory must not be released while this thread can still write to it.
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)

        if thread is not None and thread.is_alive():
            raise RuntimeError(
                "Image receive thread is still running. "
                "Shared memory was not released to avoid concurrent access."
            )

        self.img_client = None
        self.receive_thread = None


    @staticmethod
    def release_shared_memory(shm: Optional[shared_memory.SharedMemory]) -> None:
        if shm is None:
            return

        try:
            shm.close()
        except Exception:
            pass

        try:
            shm.unlink()
        except FileNotFoundError:
            pass


    def close(self) -> None:
        """
        Stop image client, confirm the receive thread has exited, then release shared memory.
        """
        if self.closed:
            return

        try:
            self.shutdown_image_client()
        except Exception:
            raise

        self.closed = True

        self.release_shared_memory(self.tv_shm)
        self.release_shared_memory(self.wrist_shm)

        self.tv_shm = None
        self.wrist_shm = None
        self.tv_img = None
        self.wrist_img = None



class CoordinateTransform:
    """
    Initialize the first frame and apply the relative transform to the rest frames.
    """
    def __init__(self) -> None:
        self.first_pos = None  # The position of the first frame
        self.first_rot = None  # The rotation of the first frame
        self.first_imu_quat = None  # The IMU quaternion (wxyz) of the first frame


    def reset(self) -> None:
        self.first_pos = None
        self.first_rot = None
        self.first_imu_quat = None


    def _apply_qpos_transform(self, qpos):
        """
        Transform the root pose to the desired pose.
        """
        if len(qpos) < 7:
            return qpos

        qpos = np.array(qpos).copy()
        curr_pos = qpos[0:3].copy()
        curr_quat_wxyz = qpos[3:7].copy()

        # wxyz -> xyzw
        curr_rot = R.from_quat([
            curr_quat_wxyz[1],
            curr_quat_wxyz[2],
            curr_quat_wxyz[3],
            curr_quat_wxyz[0]
        ])

        # Initialize the first frame
        if self.first_pos is None:
            self.first_pos = curr_pos.copy()
            self.first_rot = curr_rot

            # First frame: x,y = 0, keep z
            qpos[0:3] = np.array([0.0, 0.0, curr_pos[2]])
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            return qpos.tolist()
        
        # ---------- Relative translation under the world frame ----------
        relative_pos_world = curr_pos - self.first_pos

        # Take yaw angle from the first frame only
        first_yaw = self.first_rot.as_euler("xyz")[2]
        yaw_rot = R.from_euler("z", first_yaw)

        # Transform from world frame to body frame
        relative_pos_body = yaw_rot.inv().apply(relative_pos_world)
        new_pos = np.array([
            relative_pos_body[0], 
            relative_pos_body[1], 
            curr_pos[2]
        ])
        
        # ---------- Relative orientation under the first frame ----------
        relative_rot = self.first_rot.inv() * curr_rot
        final_rot = R.from_quat([0, 0, 0, 1]) * relative_rot

        final_xyzw = final_rot.as_quat()
        new_quat_wxyz = np.array([
            final_xyzw[3], 
            final_xyzw[0], 
            final_xyzw[1], 
            final_xyzw[2]
        ])
        # Update the qpos
        qpos[0:3] = new_pos
        qpos[3:7] = new_quat_wxyz

        return qpos.tolist()


    def _apply_imu_quat_transform(self, quat_wxyz):
        """
        Transform the IMU quaternion to the desired quaternion.
        """
        quat_wxyz = np.array(quat_wxyz).copy()

        # wxyz -> xyzw
        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0]
        ])
        curr_rot = R.from_quat(quat_xyzw)

        # Initialize the first frame
        if self.first_imu_quat is None:
            self.first_imu_quat = curr_rot
            # Return the first frame with identity quaternion (wxyz)
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        # Compute the relative rotation of the rest frames
        # relative_rot = first_rot^{-1} * curr_rot
        relative_rot = self.first_imu_quat.inv() * curr_rot
        
        # Convert back to wxyz format
        relative_xyzw = relative_rot.as_quat()
        new_quat_wxyz = np.array([
            relative_xyzw[3], 
            relative_xyzw[0], 
            relative_xyzw[1], 
            relative_xyzw[2]
        ])
        return new_quat_wxyz



class GetStates:
    """
    Get real-time robot state via DDS protocol and construct GR00T state observations.
    """
    GR00T_STATE_KEYS = ("state.left_hand", "state.right_hand", "state.robot_q")


    def __init__(self, config: RobotConfig, hand_ctrl):
        self.config = config
        self.closed = False

        # Runtime status
        self.state_ready = False
        self.state_lock = threading.Lock()

        # Episode-relative coordinate transform
        self.coord_transform = CoordinateTransform()

        # Cache root xyz in GR00T action / policy coordinate frame.
        # The first observation uses initialized root xyz; later observations
        # use the previously executed action.robot_q[:3].
        self.previous_action_root_xyz = np.array([0.0, 0.0, 0.74],dtype=np.float32)
        self.first_root_observation = True

        # Cached actual robot feedback
        self.current_joint_pos = np.zeros(29, dtype=np.float32)
        self.current_imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Initialize dextrous hand controller
        self.eef_type = config.eef_type
        self.hand_ctrl = hand_ctrl

        # Subscribe to robot low-state feedback
        self.low_state_sub = None

        # Initialize DDS and subscribe to robot low-state feedback
        self.init_dds()
        print("DDS publishers initialized.")


    def init_dds(self) -> None:
        """
        Initialize DDS and subscribe to robot low-state feedback.
        """        
        # Initialize DDS
        net = self.config.network_interface.strip()
        # "lo" = offline / no robot Ethernet NIC; CycloneDDS uses default interface.
        try:
            ChannelFactoryInitialize(0, None if net.lower() == "lo" else net)
        except Exception:
            print("[Warning] ChannelFactoryInitialize failed or already initialized.")

        # Subscribe to robot low-state feedback
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.low_state_sub.Init(self.low_state_handler, 10)


    def low_state_handler(self, msg: LowStateHG) -> None:
        """
        DDS callback for rt/lowstate.
            - 29 actual motor joint positions
            - actual IMU quaternion in wxyz order
        """
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
    

    def wait_for_states(self, timeout: float = 5.0, poll_interval: float = 0.01) -> None:
        """
        Wait until valid rt/lowstate feedback has been received.
        """
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
        """
        Get 12-dof left/right dextrous hand states.
        """
        if self.hand_ctrl is None:
            raise RuntimeError("Hand controller has not been initialized.")
        l_hand_state, r_hand_state = self.hand_ctrl.get_hand_states()

        # Fit with Gr00t required format
        l_hand_state = np.asarray(l_hand_state, dtype=np.float32).reshape(-1)
        r_hand_state = np.asarray(r_hand_state, dtype=np.float32).reshape(-1)

        return l_hand_state, r_hand_state


    # Reset the root-state cache together with CoordinateTransform 
    # at the beginning of each new inference episode.
    def reset_episode_transform(self) -> None:
        """
        Reset the episode-relative coordinate transform and root-state cache.
        """
        self.coord_transform.reset()

        with self.state_lock:
            self.previous_action_root_xyz = np.array(
                [0.0, 0.0, 0.74],
                dtype=np.float32,
            )
            self.first_root_observation = True


    # Cache the xyz from the single GR00T action step that has
    # actually been executed. This is used by the next observation frame.
    def update_previous_action_root_xyz(self, action_robot_q: np.ndarray) -> None:
        """
        Update cached root xyz using the executed action.robot_q step.
        """
        action_robot_q = np.asarray(action_robot_q, dtype=np.float32)

        if action_robot_q.shape != (36,):
            raise ValueError(
                "update_previous_action_root_xyz expects one executed "
                f"action.robot_q with shape (36,), got {action_robot_q.shape}."
            )

        with self.state_lock:
            self.previous_action_root_xyz = action_robot_q[:3].copy()


    # Removed raw_qpos input.
    # Root xyz is initialized once through _apply_qpos_transform(),
    # then directly taken from the previously executed action.robot_q[:3].
    def build_robot_q_current(self) -> np.ndarray:
        """
        Build the 36-dof robot_q_current observation.
        """
        with self.state_lock:
            previous_action_root_xyz = self.previous_action_root_xyz.copy()
            first_root_observation = self.first_root_observation
            actual_full = self.current_joint_pos.copy()
            actual_imu_quat = self.current_imu_quat.copy()
        
        # For the first observation only, initialize root xyz and establish
        # the qpos-transform baseline. Later action xyz is already in the
        # GR00T policy coordinate frame, so no repeated qpos transform is used.
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

        # Apply coordinate transform to actual_imu_quat
        if actual_imu_quat is not None:
            transformed_imu_quat = self.coord_transform._apply_imu_quat_transform(actual_imu_quat)
        else:
            transformed_imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        actual_full_combined = np.concatenate([
            root_xyz,  # initialized root xyz or previous executed action.robot_q[:3]
            transformed_imu_quat,
            actual_full,
        ]).astype(np.float32, copy=False)

        return actual_full_combined


    # Removed raw_qpos input because root xyz is now managed internally.
    def process_states_obs(self) -> Dict[str, np.ndarray]:
        """
        Build GR00T state.* observations.

        Returns:
            {
                "state.left_hand":  np.ndarray, shape (1, 6),  dtype float32,
                "state.right_hand": np.ndarray, shape (1, 6),  dtype float32,
                "state.robot_q":    np.ndarray, shape (1, 36), dtype float32,
            }
        """
        if self.closed:
            raise RuntimeError("GetStates has already been closed.")

        self.wait_for_states()

        left_hand, right_hand = self.get_hand_state()
        robot_q_current = self.build_robot_q_current()

        states_obs = {
            "state.left_hand": left_hand[np.newaxis, :],     # (1, 6)
            "state.right_hand": right_hand[np.newaxis, :],   # (1, 6)
            "state.robot_q": robot_q_current[np.newaxis, :], # (1, 36)
        }

        return states_obs


    def close(self) -> None:
        """
        Stop hand-state access and release class references.
        """
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



def _select_action_step(value: Any, step: int = 0) -> np.ndarray:
    """
    Select one action step from common GR00T output layouts:
        (D,), (1, D), (T, D), (1, T, D)
    """
    if value is None:
        raise KeyError("Missing action field in policy output.")

    arr = np.asarray(value)

    if arr.ndim == 1:
        if step != 0:
            raise IndexError(
                f"Cannot select step={step} from a single-step action with shape {arr.shape}."
            )
        return arr

    if arr.ndim == 2:
        if arr.shape[0] == 1:
            if step != 0:
                raise IndexError(
                    f"Cannot select step={step} from action with shape {arr.shape}."
                )
            return arr[0]

        if not 0 <= step < arr.shape[0]:
            raise IndexError(
                f"Action step={step} is out of range for shape {arr.shape}."
            )
        return arr[step]

    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for action tensor, got shape {arr.shape}."
            )

        if not 0 <= step < arr.shape[1]:
            raise IndexError(
                f"Action step={step} is out of range for shape {arr.shape}."
            )
        return arr[0, step]

    raise ValueError(f"Unsupported action tensor shape {arr.shape}.")


class SendAction:
    """
    Send GR00T inferred actions to the Unitree G1 robot.
    """

    INITIAL_ROBOT_Q = np.array(
        [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0] + [0.0] * 29,
        dtype=np.float32,
    )

    def __init__(
        self,
        config: RobotConfig,
        hand_ctrl,
        get_states: GetStates,
        rate_hz: int = 60,
        urdf_path: Optional[str] = None,
        enc_model_path: str = "models/model.enc",
    ):
        self.config = config
        self.rate_hz = rate_hz
        self.hand_ctrl = hand_ctrl
        self.get_states = get_states
        self.eef_type = config.eef_type

        if self.hand_ctrl is None:
            raise ValueError("SendAction requires the shared Inspire hand controller.")

        if self.get_states is None:
            raise ValueError("SendAction requires the GetStates instance.")

        # FSM / runtime status
        self.current_fsm_mode = 504
        logger.info("System initialized. Default mode: 504.")
        self.started = False
        self.running = False

        # Body command thread control
        self.cmd_lock = threading.Lock()
        self.thread = None

        # Same default body command as robot_upper_controller.
        self.cmd = self.INITIAL_ROBOT_Q.copy()

        # Root quaternion zero-reference logic used by the original controller.
        self.a_key_zero_rotation = None
        self.motion_vq_full = None

        # Setup LocoClient for FSM switching
        self.sport_client = LocoClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("LocoClient Ready!")

        # Setup voice announcements module
        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()
        self.audio_client.SetVolume(100)

        # Setup body action DDS publisher.
        self.m_msg_publisher_ = ChannelPublisher("rt/fsm/teleop/cmd", String_)
        self.m_msg_publisher_.Init()
        self.m_msg = String_(data=str(None))

        # Setup secure motion inferencer.
        if urdf_path is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_path)
            urdf_path = os.path.join(project_root, "assets/g1/g1_body29_hand14.urdf")

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        # Load the secure motion inferencer.
        self.inferencer = SecureMotionInferencer(urdf_path, enc_model_path)


    def speak(self, text: str) -> None:
        """
        Play a spoken status message through the robot audio client.
        """
        try:
            self.audio_client.TtsMaker(text, 1)
        except Exception as e:
            print(f"[Warning] Audio announcement failed: {e}")


    def switch_fsm(self, fsm_id: int) -> bool:
        """
        Switch robot FSM ID (504: teleop, 801: locomotion/walking).
        """
        print(f"[SendAction] Switching FSM to {fsm_id}...")

        try:
            # Temporarily increase timeout to ensure the switch command is delivered
            self.sport_client.SetTimeout(3.0)
            ret = self.sport_client.SetFsmId(fsm_id)
            print(f"[SendAction] Switch result: {ret}")
            # Restore short timeout for real-time control
            self.sport_client.SetTimeout(0.01)
        except Exception as e:
            print(f"[SendAction] FSM Switch Error: {e}")
            return False

        return True

    def enter_teleop_mode(self) -> None:
        """
        Enter FSM 504 teleop mode.
        """
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
        """
        Enter FSM 801 walking / safe-exit mode.
        """
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
        p = {}
        p["frame"] = traj_frames
        p["name"] = traj_name
        parameter = json.dumps(p, separators=(',', ':'))
        return parameter


    def process_frame(self, robot_q: np.ndarray, dt: float):
        """
        Process one current body target and publish the resulting command.
        """
        if self.current_fsm_mode == 801:
            return
       
        elif self.current_fsm_mode == 504:
            motion_vq, root_pose, cmd_wrist, _ = self.inferencer.process(robot_q, dt)

            if motion_vq is not None:
                raw_quat_wxyz = root_pose[3:7]
                raw_quat_xyzw = np.array(
                    [
                        raw_quat_wxyz[1], 
                        raw_quat_wxyz[2], 
                        raw_quat_wxyz[3], 
                        raw_quat_wxyz[0]
                    ], 
                    dtype=np.float64
                )
                raw_rot = R.from_quat(raw_quat_xyzw)

                # Zero-point calibration logic
                if self.a_key_zero_rotation is None:
                    self.a_key_zero_rotation = raw_rot
                    logger.info("Captured initial zero rotation.")

                if self.a_key_zero_rotation is not None:
                    delta_rot = self.a_key_zero_rotation.inv() * raw_rot
                    corrected_quat_xyzw = delta_rot.as_quat()
                    
                    # Convert back to wxyz
                    root_pose[3:7] = np.array(
                        [
                            corrected_quat_xyzw[3],
                            corrected_quat_xyzw[0],
                            corrected_quat_xyzw[1],
                            corrected_quat_xyzw[2],
                        ],
                        dtype=root_pose.dtype,
                    )
                
                # Build outgoing command
                self.motion_vq_full = np.concatenate([motion_vq, root_pose, cmd_wrist], axis=-1)
                
                if self.motion_vq_full is not None:
                    self.m_msg.data = self.dump_json_traj("default", self.motion_vq_full.tolist())
                    self.m_msg_publisher_.Write(self.m_msg)


    def _loop(self):
        """
        Continuously execute the most recent body target at rate_hz.
        """
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
        """
        Prepare an initial standing command, enter FSM 504, and start 
        the 60 Hz body execution thread.
        """
        if self.started:
            return

        self.a_key_zero_rotation = None
        self.motion_vq_full = None

        if send_initial_pose:
            self.send_initial_pose()

        self.enter_teleop_mode()

        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self.thread.start()

        self.started = True


    def stop(self) -> None:
        """
        Stop the body execution thread and return the robot to FSM 801.
        """
        if not self.started:
            return

        self.running = False

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Match the shutdown hand-pose handling in the original system.
        self.enter_walking_mode()

        time.sleep(0.5)

        self.thread = None
        self.started = False


    def send_robot_q(self, robot_q: np.ndarray):
        """
        Submit one GR00T action.robot_q step as the current body target.
        """
        if not self.started:
            raise RuntimeError("SendAction has not been started.")

        if self.current_fsm_mode != 504:
            raise RuntimeError(
                "Body action can only be submitted while FSM is 504."
            )

        robot_q = np.asarray(robot_q, dtype=np.float32).reshape(-1)

        if robot_q.shape != (36,):
            raise ValueError(
                f"send_robot_q expects shape (36,), got {robot_q.shape}."
            )

        if not np.all(np.isfinite(robot_q)):
            raise ValueError("robot_q contains NaN or infinite values.")

        if np.linalg.norm(robot_q[3:7]) < 1e-8:
            raise ValueError("robot_q contains an invalid zero-norm quaternion.")

        with self.cmd_lock:
            self.cmd = robot_q.copy()

        # GR00T action.robot_q[:3] is already in the policy coordinate frame.
        # The next state observation directly uses this previously executed xyz.
        self.get_states.update_previous_action_root_xyz(robot_q)


    def send_hand_cmd(self, hand_cmd: np.ndarray):
        """
        Send one 12-dof hand action.
        """
        if not self.started:
            raise RuntimeError("SendAction has not been started.")

        if self.current_fsm_mode != 504:
            raise RuntimeError(
                "Hand action can only be submitted while FSM is 504."
            )

        hand_cmd = np.asarray(hand_cmd, dtype=np.float64).reshape(-1)

        if hand_cmd.shape != (12,):
            raise ValueError(
                f"send_hand_cmd expects shape (12,), got {hand_cmd.shape}."
            )

        if not np.all(np.isfinite(hand_cmd)):
            raise ValueError("hand_cmd contains NaN or infinite values.")

        self.hand_ctrl.set_hand_targets(hand_cmd * 1000.0)


    def send_initial_pose(self) -> None:
        """
        Set the initial standing body command.
        """
        with self.cmd_lock:
            self.cmd = self.INITIAL_ROBOT_Q.copy()


    def send_action(self, action: Dict[str, Any], step: int = 0) -> None:
        """
        Execute one step from Gr00tPolicy output.

        Expected policy keys:
            action.robot_q    -> (T, 36) or compatible single/batched layout
            action.left_hand  -> (T, 6) or compatible single/batched layout
            action.right_hand -> (T, 6) or compatible single/batched layout
        """
        robot_q = _select_action_step(action.get("action.robot_q"), step=step)
        left_hand = _select_action_step(action.get("action.left_hand"), step=step)
        right_hand = _select_action_step(action.get("action.right_hand"), step=step)

        hand_cmd = np.concatenate(
            [
                np.asarray(left_hand, dtype=np.float64).reshape(-1),
                np.asarray(right_hand, dtype=np.float64).reshape(-1),
            ],
            axis=0,
        )

        self.send_robot_q(robot_q)
        self.send_hand_cmd(hand_cmd)


def build_obs(get_images: GetImages, get_states: GetStates, language_instruction: str) -> Dict[str, Any]:
    """
    Assemble GR00T unitree_g1_wbc observation dict for policy.get_action().
    """
    obs: Dict[str, Any] = {}
    obs.update(get_images.process_images_obs())
    obs.update(get_states.process_states_obs())
    obs["annotation.human.task_description"] = [language_instruction]
    return obs


def infer_action_horizon(action: Dict[str, Any], default: int = 16) -> int:
    """
    Infer how many steps are in the action chunk returned by the policy.
    """
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


def run(
    config: RobotConfig,
    policy,
    *,
    control_hz: float = 30.0,
    action_horizon: int = 16,
    max_chunks: Optional[int] = None,
    body_rate_hz: int = 60,
    warmup_sec: float = 2.0,
) -> None:
    """
    One policy inference per chunk, execute steps 0..horizon-1 at control_hz.
    """
    if config.eef_type != "inspire":
        raise ValueError(
            f"Strategy B deploy currently supports eef_type='inspire' only, "
            f"got {config.eef_type!r}."
        )

    hand_ctrl = None
    get_states = None
    get_images = None
    send_action = None

    running = True

    def _request_stop(signum, frame) -> None:
        nonlocal running
        logger.info("Stop signal received (%s), exiting deploy loop...", signum)
        running = False

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    dt = 1.0 / float(control_hz)
    chunk_idx = 0

    try:
        hand_ctrl = InspireFTPHandController()
        get_states = GetStates(config, hand_ctrl)
        get_images = GetImages(config)

        enc_model_path = str(WBC_ROOT / "models" / "model.enc")
        send_action = SendAction(
            config,
            hand_ctrl,
            get_states,
            rate_hz=body_rate_hz,
            enc_model_path=enc_model_path,
        )

        logger.info("Waiting for camera and lowstate (%.1fs)...", warmup_sec)
        get_images.wait_for_images(timeout=max(warmup_sec, 5.0))
        get_states.wait_for_states(timeout=max(warmup_sec, 5.0))

        get_states.reset_episode_transform()
        send_action.start(send_initial_pose=True)
        send_action.speak("GR00T deploy started")

        logger.info(
            "Strategy B loop: control_hz=%.1f action_horizon=%d",
            control_hz,
            action_horizon,
        )

        while running:
            if max_chunks is not None and chunk_idx >= max_chunks:
                logger.info("Reached max_chunks=%d, stopping.", max_chunks)
                break

            obs = build_obs(
                get_images,
                get_states,
                config.language_instruction,
            )

            infer_start = time.monotonic()
            action = policy.get_action(obs)
            infer_ms = (time.monotonic() - infer_start) * 1000.0

            chunk_len = min(
                infer_action_horizon(action, default=action_horizon),
                action_horizon,
            )

            if chunk_len <= 0:
                logger.warning("Empty action chunk, skipping.")
                continue

            for step in range(chunk_len):
                if not running:
                    break

                step_start = time.monotonic()
                send_action.send_action(action, step=step)

                sleep_time = dt - (time.monotonic() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            chunk_idx += 1
            logger.info(
                "Chunk %d done: infer=%.0fms executed_steps=%d",
                chunk_idx,
                infer_ms,
                chunk_len,
            )

    finally:
        logger.info("Shutting down deploy...")

        if send_action is not None:
            try:
                send_action.stop()
            except Exception as e:
                logger.warning("SendAction.stop failed: %s", e)

        if get_images is not None:
            try:
                get_images.close()
            except Exception as e:
                logger.warning("GetImages.close failed: %s", e)

        if get_states is not None:
            try:
                get_states.close()
            except Exception as e:
                logger.warning("GetStates.close failed: %s", e)

        if hand_ctrl is not None and hasattr(hand_ctrl, "stop"):
            try:
                hand_ctrl.stop()
            except Exception as e:
                logger.warning("Hand controller stop failed: %s", e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Unitree G1 GR00T deploy (open-loop action chunks).")
    parser.add_argument("--net", type=str, default="enp5s0", help="DDS network interface (use 'lo' for offline).")
    parser.add_argument("--eef", type=str, default="inspire", choices=["inspire", "dex1", "brainco"], help="End-effector type.")
    parser.add_argument("--policy-host", type=str, default="localhost", help="GR00T inference server host.")
    parser.add_argument("--policy-port", type=int, default=5555, help="GR00T inference server port.")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Execution rate for each step inside an action chunk.")
    parser.add_argument("--action-horizon", type=int, default=16, help="Max steps to execute per inferred chunk.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Stop after this many replans (default: run until Ctrl+C).")
    parser.add_argument("--body-rate-hz", type=int, default=60, help="SendAction body inferencer thread rate.")
    parser.add_argument("--warmup-sec", type=float, default=2.0, help="Seconds to wait for first camera/lowstate frames.")
    parser.add_argument("--language", type=str, default=None, help="Language instruction (overrides RobotConfig default).")
    args = parser.parse_args()

    config = RobotConfig(
        network_interface=args.net,
        eef_type=args.eef,
        robot_port=args.policy_port,
    )
    if args.language is not None:
        config.language_instruction = args.language

    from gr00t.eval.robot import RobotInferenceClient

    logger.info(
        "Connecting to policy server at %s:%d",
        args.policy_host,
        args.policy_port,
    )
    policy = RobotInferenceClient(host=args.policy_host, port=args.policy_port)

    run(
        config,
        policy,
        control_hz=args.control_hz,
        action_horizon=args.action_horizon,
        max_chunks=args.max_chunks,
        body_rate_hz=args.body_rate_hz,
        warmup_sec=args.warmup_sec,
    )


if __name__ == "__main__":
    main()