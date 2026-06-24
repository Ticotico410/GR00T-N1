"""
RTC-capable inference service for the original GR00T-N1 codebase

The server bootstrap deliberately follows NVIDIA's original N1
``scripts/inference_service.py``:

    DATA_CONFIG_MAP
        -> Gr00tPolicy
        -> RobotInferenceServer

RTC is added without replacing that protocol:

* ``get_action`` remains the original N1 endpoint.
* ``get_modality_config`` remains the original N1 endpoint.
* ``get_action_rtc`` adds RTC-guided flow-matching inference.
* ``rtc_capabilities`` lets the real-robot client verify RTC support.
* ``reset_rtc`` clears the server-side previous-chunk cache.

This implementation mirrors that original N1 path and inserts RTC guidance
inside each flow-matching Euler step.
"""

from __future__ import annotations

import math
import time
import argparse
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import torch
from torch import Tensor
from transformers.feature_extraction_utils import BatchFeature

from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass(frozen=True)
class RTCConfig:
    """Server-side RTC configuration."""

    enabled: bool = True
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10
    debug: bool = False

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(
                "max_guidance_weight must be positive, got "
                f"{self.max_guidance_weight}"
            )
        if self.execution_horizon <= 0:
            raise ValueError(
                "execution_horizon must be positive, got "
                f"{self.execution_horizon}"
            )


class GR00TN1RTCProcessor:
    """RTC guidance for original GR00T-N1's 0 -> 1 flow convention.

    Original N1 starts from Gaussian noise and integrates

        x <- x + dt * v(x, t)

    from t=0 toward t=1. At a current time t, the estimated final action is

        x_1 = x_t + (1 - t) * v_t.

    RTC encourages the overlapping prefix of this estimate to follow the
    unconsumed part of the previous normalized action chunk.
    """

    def __init__(self, config: RTCConfig) -> None:
        self.config = config
        self.last_debug: dict[str, Any] = {}

    def denoise_step(
        self,
        *,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        t_cont: float,
        denoise_fn,
        execution_horizon: int | None = None,
    ) -> Tensor:
        if prev_chunk_left_over is None:
            return denoise_fn(x_t).detach()

        if x_t.ndim != 3:
            raise ValueError(
                f"x_t must have shape (B,T,A), got {tuple(x_t.shape)}"
            )

        prev = prev_chunk_left_over
        if prev.ndim == 2:
            prev = prev.unsqueeze(0)
        if prev.ndim != 3:
            raise ValueError(
                "prev_chunk_left_over must have shape (T,A) or (B,T,A), got "
                f"{tuple(prev.shape)}"
            )

        batch_size, chunk_size, action_dim = x_t.shape
        prev = prev.to(device=x_t.device, dtype=x_t.dtype)
        if prev.shape[0] == 1 and batch_size > 1:
            prev = prev.expand(batch_size, -1, -1)
        if prev.shape[0] != batch_size:
            raise ValueError(
                "RTC batch mismatch: "
                f"current batch={batch_size}, previous batch={prev.shape[0]}"
            )

        # Match LeRobot RTC behavior: pad a short previous tail to the current
        # chunk shape.  Zero-padded positions receive zero prefix weight.
        padded_prev = torch.zeros_like(x_t)
        copied_steps = min(chunk_size, prev.shape[1])
        copied_dims = min(action_dim, prev.shape[2])
        padded_prev[:, :copied_steps, :copied_dims] = prev[
            :, :copied_steps, :copied_dims
        ]

        overlap_horizon = (
            self.config.execution_horizon
            if execution_horizon is None
            else int(execution_horizon)
        )
        overlap_horizon = max(
            0,
            min(overlap_horizon, copied_steps, chunk_size),
        )
        if overlap_horizon == 0:
            return denoise_fn(x_t).detach()

        inference_delay = max(0, min(int(inference_delay), overlap_horizon))
        weights = self.get_prefix_weights(
            inference_delay,
            overlap_horizon,
            chunk_size,
        ).to(device=x_t.device, dtype=x_t.dtype)
        weights = weights.view(1, chunk_size, 1)

        # Model parameters are frozen by RTCGr00tPolicy, but gradients with
        # respect to the action latent are needed for the RTC correction.
        with torch.enable_grad():
            x_var = x_t.detach().clone().requires_grad_(True)
            base_velocity = denoise_fn(x_var)
            remaining_time = max(0.0, 1.0 - float(t_cont))
            x1_t = x_var + remaining_time * base_velocity
            endpoint_error = (padded_prev - x1_t) * weights
            correction = torch.autograd.grad(
                outputs=x1_t,
                inputs=x_var,
                grad_outputs=endpoint_error.detach(),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        guidance_weight = self._guidance_weight(t_cont)
        guided_velocity = base_velocity + guidance_weight * correction

        if self.config.debug:
            self.last_debug = {
                "time": float(t_cont),
                "guidance_weight": float(guidance_weight),
                "inference_delay": int(inference_delay),
                "execution_horizon": int(overlap_horizon),
                "weighted_endpoint_error_mean": float(
                    endpoint_error.detach().abs().mean().float().cpu()
                ),
                "correction_mean": float(
                    correction.detach().abs().mean().float().cpu()
                ),
            }

        return guided_velocity.detach()

    def _guidance_weight(self, t_cont: float) -> float:
        """Compute the bounded RTC guidance gain.

        LeRobot's published RTC processor uses the opposite denoising-time
        direction.  For original N1, ``tau`` is the direct 0 -> 1 progress.
        """
        tau = float(np.clip(t_cont, 0.0, 1.0))
        max_weight = float(self.config.max_guidance_weight)

        # Match torch.nan_to_num behavior at the end point.  Original N1 never
        # evaluates exactly t=1 in its loop, but this keeps the helper robust.
        if tau >= 1.0:
            return 0.0
        if tau <= 0.0:
            return max_weight

        one_minus_tau = 1.0 - tau
        inv_r2 = (
            one_minus_tau**2 + tau**2
        ) / max(one_minus_tau**2, 1e-12)
        c = one_minus_tau / max(tau, 1e-12)
        weight = c * inv_r2
        if not math.isfinite(weight):
            return max_weight
        return float(min(weight, max_weight))

    def get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(int(start), int(end))
        end = int(np.clip(end, 0, total))
        start = int(np.clip(start, 0, end))

        schedule = self.config.prefix_attention_schedule
        if schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
            return weights

        if schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
            return weights

        transition = self._linweights(start, end, total)
        if schedule == RTCAttentionSchedule.EXP:
            transition = transition * torch.expm1(transition).div(math.e - 1)

        weights = self._add_trailing_zeros(transition, total, end)
        return self._add_leading_ones(weights, start, total)

    @staticmethod
    def _linweights(start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start
        if end <= start or linspace_steps <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    @staticmethod
    def _add_trailing_zeros(weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zeros_len)])

    @staticmethod
    def _add_leading_ones(weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return torch.cat([torch.ones(ones_len), weights])


def get_action_rtc(
    *,
    action_head,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    prev_chunk_left_over: Tensor | None,
    inference_delay: int,
    rtc_processor: GR00TN1RTCProcessor,
    execution_horizon: int,
) -> BatchFeature:
    """Original N1 action-head inference with RTC guidance.

    This function intentionally mirrors the original N1
    ``FlowmatchingActionHead.get_action`` implementation:

    * no ``process_backbone_output``
    * no ``future_tokens``
    * concatenate only ``state_features`` and ``action_features``
    """
    vl_embeds = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    state_features = action_head.state_encoder(
        action_input.state,
        embodiment_id,
    )

    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    actions = torch.randn(
        size=(
            batch_size,
            action_head.action_horizon,
            action_head.action_dim,
        ),
        dtype=vl_embeds.dtype,
        device=device,
    )

    num_steps = int(action_head.num_inference_timesteps)
    if num_steps <= 0:
        raise ValueError(
            "num_inference_timesteps must be positive, got "
            f"{num_steps}"
        )
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,),
            fill_value=t_discretized,
            device=device,
            dtype=torch.long,
        )

        def denoise_fn(current_actions: Tensor) -> Tensor:
            action_features = action_head.action_encoder(
                current_actions,
                timesteps_tensor,
                embodiment_id,
            )

            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1],
                    dtype=torch.long,
                    device=device,
                )
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Exact original N1 sequence construction.
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
            pred = action_head.action_decoder(model_output, embodiment_id)
            return pred[:, -action_head.action_horizon :]

        velocity = rtc_processor.denoise_step(
            x_t=actions,
            prev_chunk_left_over=prev_chunk_left_over,
            inference_delay=inference_delay,
            t_cont=t_cont,
            denoise_fn=denoise_fn,
            execution_horizon=execution_horizon,
        )
        actions = (actions + dt * velocity).detach()

    return BatchFeature(data={"action_pred": actions})


def _unsqueeze_n1_observation(data: Mapping[str, Any]) -> dict[str, Any]:
    """Match original N1's ``unsqueeze_dict_values`` behavior."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = np.expand_dims(value, axis=0)
        elif torch.is_tensor(value):
            result[key] = value.unsqueeze(0)
        else:
            result[key] = value
    return result


def _squeeze_n1_action(data: Mapping[str, Any]) -> dict[str, Any]:
    """Match original N1's ``squeeze_dict_values`` behavior."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = np.squeeze(value)
        elif torch.is_tensor(value):
            result[key] = value.squeeze()
        else:
            result[key] = value
    return result


class RTCGr00tPolicy(Gr00tPolicy):
    """Original N1 ``Gr00tPolicy`` extended with stateful RTC inference."""

    def __init__(self, *args, rtc_config: RTCConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rtc_config = rtc_config
        self._rtc_prev_normalized_action: Tensor | None = None
        self._rtc_prev_sequence_id: int | None = None

        # RTC needs gradients with respect to the action latent only.  Freezing
        # weights avoids parameter-gradient allocation while preserving that
        # latent Jacobian path.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def reset_rtc(self) -> dict[str, Any]:
        self._rtc_prev_normalized_action = None
        self._rtc_prev_sequence_id = None
        return {"status": "ok"}

    def rtc_capabilities(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_family": "GR00T-N1",
            "rtc": True,
            "endpoint": "get_action_rtc",
            "flow_matching_guidance": True,
            "stateful_normalized_chunk_cache": True,
            "schedules": [item.value for item in RTCAttentionSchedule],
            "default_execution_horizon": self.rtc_config.execution_horizon,
            "default_max_guidance_weight": self.rtc_config.max_guidance_weight,
            "default_prefix_attention_schedule": (
                self.rtc_config.prefix_attention_schedule.value
            ),
        }

    def get_action_rtc(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run original N1 inference with RTC guidance.

        Expected request:

        ``payload["observation"]``
            The same flat observation dictionary accepted by N1 ``get_action``.

        ``payload["rtc"]``
            Sequence/cache identifiers plus runtime delay and guidance values.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("RTC request payload must be a mapping")

        observation_value = payload.get("observation")
        rtc_value = payload.get("rtc", {})
        if not isinstance(observation_value, Mapping):
            raise TypeError(
                "RTC payload must contain an 'observation' mapping"
            )
        if not isinstance(rtc_value, Mapping):
            raise TypeError("RTC payload 'rtc' must be a mapping")

        observations = dict(observation_value)
        rtc = dict(rtc_value)

        sequence_id = int(rtc.get("sequence_id", 0))
        previous_sequence_id = rtc.get("previous_sequence_id")
        if previous_sequence_id is not None:
            previous_sequence_id = int(previous_sequence_id)

        previous_action_index = max(
            0,
            int(rtc.get("previous_action_index", 0)),
        )
        inference_delay = max(0, int(rtc.get("inference_delay", 0)))
        execution_horizon = max(
            1,
            int(
                rtc.get(
                    "execution_horizon",
                    self.rtc_config.execution_horizon,
                )
            ),
        )

        schedule_value = str(
            rtc.get(
                "prefix_attention_schedule",
                self.rtc_config.prefix_attention_schedule.value,
            )
        ).upper()
        request_config = RTCConfig(
            enabled=bool(rtc.get("enabled", True)),
            prefix_attention_schedule=RTCAttentionSchedule(schedule_value),
            max_guidance_weight=float(
                rtc.get(
                    "max_guidance_weight",
                    self.rtc_config.max_guidance_weight,
                )
            ),
            execution_horizon=execution_horizon,
            debug=bool(rtc.get("debug", self.rtc_config.debug)),
        )
        rtc_processor = GR00TN1RTCProcessor(request_config)

        # Follow original Gr00tPolicy.get_action() batching logic.
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = _unsqueeze_n1_observation(observations)

        normalized_input = self.apply_transforms(observations)

        cache_matches = (
            request_config.enabled
            and previous_sequence_id is not None
            and previous_sequence_id == self._rtc_prev_sequence_id
            and self._rtc_prev_normalized_action is not None
        )
        previous_normalized: Tensor | None = None
        if cache_matches:
            assert self._rtc_prev_normalized_action is not None
            previous_action_index = min(
                previous_action_index,
                int(self._rtc_prev_normalized_action.shape[1]),
            )
            previous_normalized = self._rtc_prev_normalized_action[
                :, previous_action_index:
            ].detach()

        # Original N1 Gr00tPolicy normally uses torch.inference_mode().  RTC
        # cannot do that because it needs autograd with respect to the current
        # action latent.  The backbone is still evaluated under no_grad().
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone_inputs, action_inputs = self.model.prepare_input(
                normalized_input
            )
            with torch.no_grad():
                backbone_outputs = self.model.backbone(backbone_inputs)

            action_head_outputs = get_action_rtc(
                action_head=self.model.action_head,
                backbone_output=backbone_outputs,
                action_input=action_inputs,
                prev_chunk_left_over=previous_normalized,
                inference_delay=inference_delay,
                rtc_processor=rtc_processor,
                execution_horizon=execution_horizon,
            )

        # Keep original N1 model validation behavior where available.
        validate_data = getattr(self.model, "validate_data", None)
        if callable(validate_data):
            validate_data(
                action_head_outputs,
                backbone_outputs,
                is_training=False,
            )

        normalized_action = action_head_outputs["action_pred"].float()
        self._rtc_prev_normalized_action = normalized_action.detach()
        self._rtc_prev_sequence_id = sequence_id

        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = _squeeze_n1_action(unnormalized_action)

        return {
            "actions": unnormalized_action,
            "rtc": {
                "rtc_guidance_applied": previous_normalized is not None,
                "cache_matched": bool(cache_matches),
                "sequence_id": sequence_id,
                "previous_sequence_id": previous_sequence_id,
                "previous_action_index": previous_action_index,
                "inference_delay": inference_delay,
                "execution_horizon": execution_horizon,
                "prefix_attention_schedule": (
                    request_config.prefix_attention_schedule.value
                ),
                "max_guidance_weight": (
                    request_config.max_guidance_weight
                ),
                "debug": (
                    rtc_processor.last_debug
                    if request_config.debug
                    else {}
                ),
            },
        }


class RTCInferenceServer(RobotInferenceServer):
    """Original N1 RobotInferenceServer plus RTC endpoints."""

    def __init__(
        self,
        policy: RTCGr00tPolicy,
        host: str = "*",
        port: int = 5555,
    ) -> None:
        # RobotInferenceServer already registers get_action and
        # get_modality_config using the original N1 service protocol.
        super().__init__(policy, host=host, port=port)
        self.register_endpoint("get_action_rtc", policy.get_action_rtc)
        self.register_endpoint(
            "rtc_capabilities",
            policy.rtc_capabilities,
            requires_input=False,
        )
        self.register_endpoint(
            "reset_rtc",
            policy.reset_rtc,
            requires_input=False,
        )


def _make_g1_example_observation() -> dict[str, Any]:
    """Create an example matching the current G1/BrainCo flat protocol."""
    height, width = 256, 256
    return {
        "video.head_stereo_left": np.random.randint(
            0, 256, (1, height, width, 3), dtype=np.uint8
        ),
        "video.head_stereo_right": np.random.randint(
            0, 256, (1, height, width, 3), dtype=np.uint8
        ),
        "video.wrist_left": np.random.randint(
            0, 256, (1, height, width, 3), dtype=np.uint8
        ),
        "video.wrist_right": np.random.randint(
            0, 256, (1, height, width, 3), dtype=np.uint8
        ),
        "state.left_hand": np.random.rand(1, 6),
        "state.right_hand": np.random.rand(1, 6),
        "state.robot_q_root": np.random.rand(1, 7),
        "state.robot_q_upper": np.random.rand(1, 17),
        "annotation.human.task_description": ["do your thing!"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="nvidia/GR00T-N1-2B",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="unitree_g1_wbc",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the server.",
        default=5555,
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for client mode.",
        default="localhost",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run the server.",
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="Run the client.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps.",
        default=4,
    )

    # RTC specific arguments
    parser.add_argument(
        "--rtc_execution_horizon",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--rtc_max_guidance_weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--rtc_prefix_attention_schedule",
        type=str,
        choices=[item.value for item in RTCAttentionSchedule],
        default=RTCAttentionSchedule.EXP.value,
    )
    parser.add_argument(
        "--rtc_debug",
        action="store_true",
    )
    parser.add_argument(
        "--rtc_client",
        action="store_true",
        help="In --client mode, test get_action_rtc instead of get_action.",
    )
    parser.add_argument(
        "--rtc_inference_delay",
        type=int,
        default=4,
        help="Synthetic delay steps used only by --client --rtc_client.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.server:
        # Same model/config construction path as NVIDIA's original N1
        # inference_service.py.
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule(
                args.rtc_prefix_attention_schedule
            ),
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
            debug=args.rtc_debug,
        )

        policy = RTCGr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            rtc_config=rtc_config,
        )

        # Match the original N1 call style.  The server binds to "*" through
        # RobotInferenceServer's default host; --host remains a client option.
        server = RTCInferenceServer(policy, port=args.port)
        print(
            "GR00T-N1 RTC server configuration: "
            f"port={args.port}, "
            f"schedule={rtc_config.prefix_attention_schedule.value}, "
            f"execution_horizon={rtc_config.execution_horizon}, "
            f"max_guidance_weight={rtc_config.max_guidance_weight}"
        )
        server.run()

    elif args.client:
        # Same client construction path as NVIDIA's original N1
        # inference_service.py.
        policy_client = RobotInferenceClient(
            host=args.host,
            port=args.port,
        )

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        observation = _make_g1_example_observation()
        time_start = time.time()

        if args.rtc_client:
            response = policy_client.call_endpoint(
                "get_action_rtc",
                {
                    "observation": observation,
                    "rtc": {
                        "enabled": True,
                        "sequence_id": 0,
                        "previous_sequence_id": None,
                        "previous_action_index": 0,
                        "inference_delay": args.rtc_inference_delay,
                        "execution_horizon": args.rtc_execution_horizon,
                        "max_guidance_weight": (
                            args.rtc_max_guidance_weight
                        ),
                        "prefix_attention_schedule": (
                            args.rtc_prefix_attention_schedule
                        ),
                        "debug": args.rtc_debug,
                    },
                },
            )
            action = response["actions"]
            print("RTC metadata:", response.get("rtc", {}))
        else:
            action = policy_client.get_action(observation)

        print(
            "Total time taken to get action from server: "
            f"{time.time() - time_start} seconds"
        )
        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")

    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    main()