"""ZMQ client for inference_service."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import torch
import zmq


class PolicyClient:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        buf = BytesIO()
        torch.save({"endpoint": "get_action", "data": observations}, buf)
        self._socket.send(buf.getvalue())

        reply = self._socket.recv()
        if reply == b"ERROR":
            raise RuntimeError("Inference server returned ERROR")
        return torch.load(BytesIO(reply), weights_only=False)
