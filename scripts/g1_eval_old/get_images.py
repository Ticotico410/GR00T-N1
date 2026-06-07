from __future__ import annotations

import os
import cv2
import sys
import json
import time
import base64
import logging
import threading
import numpy as np

from pathlib import Path
from mcap.reader import make_reader
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

WBC_ROOT = Path(__file__).resolve().parents[2] / "wbc_pico_record"
if WBC_ROOT.is_dir() and str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

logger = logging.getLogger(__name__)


class GetImages:
    """Get images from the robot's camera via ZMQ protocol."""

    GR00T_VIDEO_KEYS = {
        "color_0": "video.head_stereo_left",
        "color_1": "video.head_stereo_right",
        "color_2": "video.wrist_left",
        "color_3": "video.wrist_right",
    }
    COLOR_VIEW_KEYS = ("color_0", "color_1", "color_2", "color_3")

    def __init__(self, config: Any):
        self.eef_type = config.eef_type
        self.images_ready = False
        self.closed = False
        self.receive_thread = None
        self.img_client = None

        self.tv_img_shape = (480, 1280, 3)
        self.tv_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(self.tv_img_shape) * np.uint8().itemsize,
        )
        self.tv_img = np.ndarray(
            self.tv_img_shape, dtype=np.uint8, buffer=self.tv_shm.buf
        )
        self.tv_img.fill(0)

        self.wrist_img_shape = (480, 1280, 3)
        self.wrist_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(self.wrist_img_shape) * np.uint8().itemsize,
        )
        self.wrist_img = np.ndarray(
            self.wrist_img_shape, dtype=np.uint8, buffer=self.wrist_shm.buf
        )
        self.wrist_img.fill(0)

        if self.eef_type == "inspire":
            from teleop.image_server.image_client import ImageClient

            self.img_client = ImageClient(
                tv_img_shape=self.tv_img_shape,
                tv_img_shm_name=self.tv_shm.name,
                wrist_img_shape=self.wrist_img_shape,
                wrist_img_shm_name=self.wrist_shm.name,
                server_address=config.robot_ip,
                port=config.robot_port,
            )

    def _ensure_receive_thread(self) -> None:
        if self.receive_thread is not None or self.img_client is None:
            return
        self.receive_thread = threading.Thread(
            target=self.img_client.receive_process,
            daemon=True,
        )
        self.receive_thread.start()

    def wait_for_images(self, timeout: float = 5.0, poll_interval: float = 0.01) -> None:
        if self.images_ready:
            return

        self._ensure_receive_thread()

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

        raise TimeoutError(f"No valid head/wrist camera frames within {timeout}s")

    def process_images_obs(self) -> Dict[str, np.ndarray]:
        if self.closed:
            raise RuntimeError("GetImages has already been closed.")

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
        client = self.img_client
        thread = self.receive_thread

        if client is None:
            return

        if hasattr(client, "running"):
            client.running = False

        # Let receive_process exit its loop and _close() in its own thread (RCVTIMEO).
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)

        if thread is not None and thread.is_alive():
            logger.warning(
                "Image receive thread did not exit in %.1fs; skipping external ZMQ close.",
                join_timeout,
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
        if self.closed:
            return

        try:
            self.shutdown_image_client()
        except Exception as e:
            logger.warning("shutdown_image_client: %s", e)

        self.closed = True
        self.release_shared_memory(self.tv_shm)
        self.release_shared_memory(self.wrist_shm)

        self.tv_shm = None
        self.wrist_shm = None
        self.tv_img = None
        self.wrist_img = None


def decode_mcap_image(img_record: Optional[dict]) -> Optional[np.ndarray]:
    if img_record is None or not isinstance(img_record, dict):
        return None
    data_b64 = img_record.get("data")
    if not data_b64:
        return None
    raw = base64.b64decode(data_b64)
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_mcap_episode(mcap_path: str) -> Tuple[List[dict], dict, dict]:
    mcap_path = os.path.abspath(mcap_path)
    if not os.path.exists(mcap_path):
        raise FileNotFoundError(f"MCAP episode not found: {mcap_path}")

    frames: List[dict] = []
    info: dict = {}
    text: dict = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            topic = channel.topic
            payload = json.loads(message.data.decode("utf-8"))
            if topic == "/episode/meta":
                info = payload.get("info", {})
                text = payload.get("text", {})
            elif topic == "/whole_body/frame":
                frames.append(payload)
    if not frames:
        raise ValueError(f"No frames in MCAP: {mcap_path}")
    return frames, info, text


class EpisodeDataSource:
    def __init__(self, episode_path: str, language_instruction: str):
        self.frames, self.info, self.text = load_mcap_episode(episode_path)
        self.frame_idx = 0
        goal = self.text.get("goal") or self.text.get("task_description")
        self.language_instruction = str(goal) if goal else language_instruction

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def current_frame(self) -> dict:
        if self.frame_idx >= len(self.frames):
            raise IndexError(f"frame_idx={self.frame_idx} >= {len(self.frames)}")
        return self.frames[self.frame_idx]

    def advance(self, steps: int = 1) -> None:
        self.frame_idx = min(self.frame_idx + steps, len(self.frames))

    def exhausted(self) -> bool:
        return self.frame_idx >= len(self.frames)


class SimGetImages:
    # Make sure the input images format fit with the GR00T requirement
    GR00T_VIDEO_KEYS = GetImages.GR00T_VIDEO_KEYS
    COLOR_VIEW_KEYS = GetImages.COLOR_VIEW_KEYS

    def __init__(self, episode: EpisodeDataSource, show_video: bool = True):
        self.episode = episode
        self.show_video = show_video
        self._gui_ok: Optional[bool] = None
        self.closed = False

    def wait_for_images(
        self, timeout: float = 5.0, poll_interval: float = 0.01) -> None:
        return

    def process_images_obs(self) -> Dict[str, np.ndarray]:
        if self.closed:
            raise RuntimeError("SimGetImages is closed.")
        colors = self.episode.current_frame().get("colors", {})
        images_obs: Dict[str, np.ndarray] = {}
        preview_imgs = []
        for key in self.COLOR_VIEW_KEYS:
            bgr = decode_mcap_image(colors.get(key))
            if bgr is None:
                raise ValueError(
                    f"Frame {self.episode.frame_idx}: missing color '{key}'."
                )
            rgb = bgr[..., ::-1].copy()
            images_obs[self.GR00T_VIDEO_KEYS[key]] = np.ascontiguousarray(
                rgb[np.newaxis, ...], dtype=np.uint8
            )
            preview_imgs.append(bgr)
        if self.show_video and preview_imgs and self._gui_ok is not False:
            try:
                cv2.imshow("sim_obs_cameras", np.hstack(preview_imgs))
                cv2.waitKey(1)
                self._gui_ok = True
            except cv2.error as exc:
                if self._gui_ok is not False:
                    logger.warning(
                        "OpenCV GUI unavailable (%s). Disabling sim camera preview.",
                        exc,
                    )
                self._gui_ok = False
                self.show_video = False
        return images_obs

    def close(self) -> None:
        self.closed = True
        if self.show_video:
            try:
                cv2.destroyWindow("sim_obs_cameras")
            except Exception:
                pass
