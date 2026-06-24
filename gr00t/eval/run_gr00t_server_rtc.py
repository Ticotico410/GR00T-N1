#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone RTC server for GR00T-N1.7.

This file depends only on GR00T-N1.7 and its normal runtime dependencies.
It does not import LeRobot.

The standard N1.7 transport remains unchanged:

    PolicyClient.get_action(observation, options={"rtc": ...})
        -> PolicyServer
        -> RTCGr00tPolicy._get_action(...)

RTC guidance is implemented locally and inserted into every N1.7
flow-matching Euler step.  The semantics match the standalone GR00T-N1 RTC
service:

* rtc_execution_horizon
* rtc_max_guidance_weight
* rtc_prefix_attention_schedule
* runtime inference_delay supplied by the robot client

The server stores the previous normalized model action chunk.  The client only
sends sequence IDs and the source-action index, so RTC always operates in the
same normalized action space as the N1.7 action head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import logging
import math
import os
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer


LOGGER = logging.getLogger("run_gr00t_server_rtc")
DEFAULT_MODEL_SERVER_PORT = 5555


class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass(frozen=True)
class RTCConfig:
    """Server-side RTC parameters."""

    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

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


@dataclass(frozen=True)
class ParsedRTCRequest:
    sequence_id: int
    previous_sequence_id: int | None
    previous_action_index: int
    inference_delay: int
    execution_horizon: int
    max_guidance_weight: float
    prefix_attention_schedule: RTCAttentionSchedule


class GR00TN1D7RTCProcessor:
    """Standalone RTC guidance for N1.7's 0 -> 1 flow convention.

    N1.7 starts from Gaussian noise and integrates

        x <- x + dt * v(x, t)

    from t=0 toward t=1.  At time t, the estimated final action is

        x_1 = x_t + (1 - t) * v_t.

    RTC constrains the overlapping prefix of this endpoint estimate to the
    unconsumed part of the previous normalized action chunk.
    """

    def __init__(self, config: RTCConfig) -> None:
        self.config = config

    def denoise_step(
        self,
        *,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int,
        t_cont: float,
        denoise_fn: Any,
        execution_horizon: int | None = None,
    ) -> Tensor:
        if prev_chunk_left_over is None:
            with torch.no_grad():
                return denoise_fn(x_t).detach()

        if x_t.ndim != 3:
            raise ValueError(
                f"x_t must have shape (B,T,A), got {tuple(x_t.shape)}"
            )

        previous = prev_chunk_left_over
        if previous.ndim == 2:
            previous = previous.unsqueeze(0)
        if previous.ndim != 3:
            raise ValueError(
                "prev_chunk_left_over must have shape (T,A) or (B,T,A), got "
                f"{tuple(previous.shape)}"
            )

        batch_size, chunk_size, action_dim = x_t.shape
        previous = previous.to(device=x_t.device, dtype=x_t.dtype)
        if previous.shape[0] == 1 and batch_size > 1:
            previous = previous.expand(batch_size, -1, -1)
        if previous.shape[0] != batch_size:
            raise ValueError(
                "RTC batch mismatch: "
                f"current={batch_size}, previous={previous.shape[0]}"
            )

        padded_previous = torch.zeros_like(x_t)
        copied_steps = min(chunk_size, int(previous.shape[1]))
        copied_dims = min(action_dim, int(previous.shape[2]))
        padded_previous[:, :copied_steps, :copied_dims] = previous[
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
            with torch.no_grad():
                return denoise_fn(x_t).detach()

        inference_delay = max(
            0,
            min(int(inference_delay), overlap_horizon),
        )
        prefix_weights = self.get_prefix_weights(
            inference_delay,
            overlap_horizon,
            chunk_size,
        ).to(device=x_t.device, dtype=x_t.dtype)
        prefix_weights = prefix_weights.view(1, chunk_size, 1)

        # Model parameters are frozen by RTCGr00tPolicy.  Autograd is enabled
        # only for the current action latent so no parameter gradients are kept.
        with torch.enable_grad():
            x_var = x_t.detach().clone().requires_grad_(True)
            base_velocity = denoise_fn(x_var)
            remaining_time = max(0.0, 1.0 - float(t_cont))
            endpoint_estimate = x_var + remaining_time * base_velocity
            endpoint_error = (
                padded_previous - endpoint_estimate
            ) * prefix_weights
            correction = torch.autograd.grad(
                outputs=endpoint_estimate,
                inputs=x_var,
                grad_outputs=endpoint_error.detach(),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        guidance_weight = self._guidance_weight(t_cont)
        guided_velocity = base_velocity + guidance_weight * correction
        return guided_velocity.detach()

    def _guidance_weight(self, t_cont: float) -> float:
        """Bounded guidance gain for the 0 -> 1 flow direction."""
        tau = float(np.clip(t_cont, 0.0, 1.0))
        max_weight = float(self.config.max_guidance_weight)

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

        transition = self._linear_weights(start, end, total)
        if schedule == RTCAttentionSchedule.EXP:
            transition = transition * torch.expm1(transition).div(math.e - 1)

        weights = self._add_trailing_zeros(transition, total, end)
        return self._add_leading_ones(weights, start, total)

    @staticmethod
    def _linear_weights(start: int, end: int, total: int) -> Tensor:
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


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, Tensor) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {
            key: _rec_to_dtype(item, dtype)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_rec_to_dtype(item, dtype) for item in value)
    return value


def _unwrap_model_output(value: Any) -> Tensor:
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError("N1.7 DiT returned an empty output tuple")
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(
            "N1.7 DiT output must be a tensor or tuple whose first item is "
            f"a tensor, got {type(value).__name__}"
        )
    return value


def get_action_rtc(
    *,
    action_head: Any,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    prev_chunk_left_over: Tensor | None,
    inference_delay: int,
    rtc_processor: GR00TN1D7RTCProcessor,
    execution_horizon: int,
) -> BatchFeature:
    """N1.7 action-head inference with standalone RTC guidance."""
    with torch.no_grad():
        features = action_head._encode_features(
            backbone_output,
            action_input,
        )

    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_input.embodiment_id

    batch_size = int(vl_embeds.shape[0])
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
    dt = 1.0 / float(num_steps)

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,),
            fill_value=t_discretized,
            dtype=torch.long,
            device=device,
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
                action_features = action_features + action_head.position_embedding(
                    pos_ids
                ).unsqueeze(0)

            sa_embs = torch.cat((state_features, action_features), dim=1)
            if action_head.config.use_alternate_vl_dit:
                model_output = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=(
                        backbone_output.backbone_attention_mask
                    ),
                )
            else:
                model_output = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            model_output = _unwrap_model_output(model_output)
            pred = action_head.action_decoder(
                model_output,
                embodiment_id,
            )
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

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


class RTCGr00tPolicy(Gr00tPolicy):
    """N1.7 Gr00tPolicy extended with standalone RTC guidance."""

    def __init__(
        self,
        *args: Any,
        rtc_config: RTCConfig,
        denoising_steps: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rtc_config = rtc_config

        action_head = getattr(self.model, "action_head", None)
        required_attributes = (
            "_encode_features",
            "action_horizon",
            "action_dim",
            "num_inference_timesteps",
            "num_timestep_buckets",
        )
        missing = [
            name
            for name in required_attributes
            if action_head is None or not hasattr(action_head, name)
        ]
        if missing:
            raise TypeError(
                "Loaded checkpoint is not a compatible GR00T-N1.7 model; "
                f"action head is missing {missing}"
            )

        if denoising_steps <= 0:
            raise ValueError(
                f"denoising_steps must be positive, got {denoising_steps}"
            )
        action_head.num_inference_timesteps = int(denoising_steps)

        # Only gradients with respect to the current action latent are needed.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

        self._rtc_prev_normalized_action: Tensor | None = None
        self._rtc_prev_sequence_id: int | None = None

    def rtc_capabilities(self) -> dict[str, Any]:
        return {
            "rtc_supported": True,
            "implementation": "standalone.GR00TN1D7RTCProcessor",
            "model_family": "GR00T-N1.7",
            "flow_matching_guidance": True,
            "stateful_normalized_chunk_cache": True,
            "denoising_steps": int(
                self.model.action_head.num_inference_timesteps
            ),
            "execution_horizon": self.rtc_config.execution_horizon,
            "max_guidance_weight": self.rtc_config.max_guidance_weight,
            "prefix_attention_schedule": (
                self.rtc_config.prefix_attention_schedule.value
            ),
            "schedules": [item.value for item in RTCAttentionSchedule],
        }

    def reset_rtc(self) -> dict[str, Any]:
        self._rtc_prev_normalized_action = None
        self._rtc_prev_sequence_id = None
        return {"status": "ok", "rtc_cache_cleared": True}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.reset_rtc()
        return super().reset(options)

    def _parse_rtc_request(
        self,
        options: dict[str, Any] | None,
    ) -> ParsedRTCRequest | None:
        if not options:
            return None

        raw: Any = options.get("rtc", options)
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"options['rtc'] must be a mapping, got {type(raw).__name__}"
            )

        sequence_id_raw = raw.get("sequence_id")
        if sequence_id_raw is None:
            raise ValueError("RTC request is missing sequence_id")

        previous_sequence_id_raw = raw.get("previous_sequence_id")
        schedule_value = str(
            raw.get(
                "prefix_attention_schedule",
                self.rtc_config.prefix_attention_schedule.value,
            )
        ).upper()

        request_config = RTCConfig(
            prefix_attention_schedule=RTCAttentionSchedule(schedule_value),
            max_guidance_weight=float(
                raw.get(
                    "max_guidance_weight",
                    self.rtc_config.max_guidance_weight,
                )
            ),
            execution_horizon=int(
                raw.get(
                    "execution_horizon",
                    self.rtc_config.execution_horizon,
                )
            ),
        )

        return ParsedRTCRequest(
            sequence_id=int(sequence_id_raw),
            previous_sequence_id=(
                int(previous_sequence_id_raw)
                if previous_sequence_id_raw is not None
                else None
            ),
            previous_action_index=max(
                0,
                int(raw.get("previous_action_index", 0)),
            ),
            inference_delay=max(0, int(raw.get("inference_delay", 0))),
            execution_horizon=request_config.execution_horizon,
            max_guidance_weight=request_config.max_guidance_weight,
            prefix_attention_schedule=(
                request_config.prefix_attention_schedule
            ),
        )

    def _get_previous_normalized_chunk(
        self,
        request: ParsedRTCRequest | None,
    ) -> tuple[Tensor | None, bool, str | None, int]:
        if request is None:
            return None, False, "no_rtc_options", 0
        if request.previous_sequence_id is None:
            return None, False, "first_chunk", 0

        cache_matches = (
            self._rtc_prev_normalized_action is not None
            and request.previous_sequence_id == self._rtc_prev_sequence_id
        )
        if not cache_matches:
            return None, False, "previous_sequence_not_cached", 0

        assert self._rtc_prev_normalized_action is not None
        start = min(
            request.previous_action_index,
            int(self._rtc_prev_normalized_action.shape[1]),
        )
        previous = self._rtc_prev_normalized_action[:, start:].detach()
        if previous.shape[1] == 0:
            return None, True, "previous_chunk_exhausted", start
        return previous, True, None, start

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = self._parse_rtc_request(options)
        previous_normalized, cache_hit, skip_reason, previous_index = (
            self._get_previous_normalized_chunk(request)
        )

        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs: list[dict[str, Any]] = []
        states: list[dict[str, np.ndarray]] = []

        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [
                {
                    "type": MessageType.EPISODE_STEP.value,
                    "content": vla_step_data,
                }
            ]
            processed_inputs.append(self.processor(messages))

        collated_inputs: BatchFeature = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(
            collated_inputs,
            dtype=torch.bfloat16,
        )
        if "inputs" not in collated_inputs:
            raise KeyError("N1.7 collator output is missing the 'inputs' key")

        model_inputs = collated_inputs["inputs"]
        with torch.no_grad():
            backbone_inputs, action_inputs = self.model.prepare_input(
                model_inputs
            )
            backbone_outputs = self.model.backbone(backbone_inputs)

        request_config = (
            RTCConfig(
                prefix_attention_schedule=(
                    request.prefix_attention_schedule
                ),
                max_guidance_weight=request.max_guidance_weight,
                execution_horizon=request.execution_horizon,
            )
            if request is not None
            else self.rtc_config
        )
        rtc_processor = GR00TN1D7RTCProcessor(request_config)

        device = next(self.model.parameters()).device
        autocast_device = device.type
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=(autocast_device == "cuda"),
        ):
            action_outputs = get_action_rtc(
                action_head=self.model.action_head,
                backbone_output=backbone_outputs,
                action_input=action_inputs,
                prev_chunk_left_over=previous_normalized,
                inference_delay=(request.inference_delay if request else 0),
                rtc_processor=rtc_processor,
                execution_horizon=request_config.execution_horizon,
            )

        normalized_action = action_outputs["action_pred"].float()
        if request is not None:
            valid_horizon = min(
                len(self.modality_configs["action"].delta_indices),
                int(normalized_action.shape[1]),
            )
            self._rtc_prev_normalized_action = normalized_action[
                :, :valid_horizon
            ].detach().clone()
            self._rtc_prev_sequence_id = request.sequence_id

        batched_states: dict[str, np.ndarray] = {}
        for key in self.modality_configs["state"].modality_keys:
            batched_states[key] = np.stack(
                [state[key] for state in states],
                axis=0,
            )

        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        casted_action = {
            key: value.astype(np.float32)
            for key, value in unnormalized_action.items()
        }

        info = {
            "rtc_request_received": request is not None,
            "rtc_guidance_applied": previous_normalized is not None,
            "rtc_cache_hit": cache_hit,
            "rtc_skip_reason": skip_reason,
            "rtc_sequence_id": request.sequence_id if request else None,
            "rtc_previous_sequence_id": (
                request.previous_sequence_id if request else None
            ),
            "rtc_previous_action_index": previous_index,
            "rtc_inference_delay": request.inference_delay if request else 0,
            "rtc_execution_horizon": request_config.execution_horizon,
            "rtc_max_guidance_weight": request_config.max_guidance_weight,
            "rtc_prefix_attention_schedule": (
                request_config.prefix_attention_schedule.value
            ),
            "rtc_implementation": (
                "standalone.GR00TN1D7RTCProcessor"
            ),
        }
        return casted_action, info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone RTC inference server for GR00T-N1.7."
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run the RTC policy server.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the GR00T-N1.7 checkpoint.",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_MODEL_SERVER_PORT,
    )
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.server:
        raise ValueError("Please specify --server")
    if args.denoising_steps <= 0:
        raise ValueError("--denoising_steps must be positive")
    if args.port <= 0:
        raise ValueError("--port must be positive")
    if args.model_path.startswith("/") and not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model path {args.model_path} does not exist"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    rtc_config = RTCConfig(
        prefix_attention_schedule=RTCAttentionSchedule(
            args.rtc_prefix_attention_schedule
        ),
        max_guidance_weight=args.rtc_max_guidance_weight,
        execution_horizon=args.rtc_execution_horizon,
    )

    print("Starting standalone GR00T-N1.7 RTC inference server...")
    print(f"  Embodiment tag: {embodiment_tag}")
    print(f"  Model path: {args.model_path}")
    print(f"  Device: cuda")
    print(f"  Port: {args.port}")
    print(f"  Denoising steps: {args.denoising_steps}")
    print(f"  RTC execution horizon: {rtc_config.execution_horizon}")
    print(f"  RTC max guidance weight: {rtc_config.max_guidance_weight}")
    print(
        "  RTC prefix attention schedule: "
        f"{rtc_config.prefix_attention_schedule.value}"
    )

    policy = RTCGr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda",
        strict=True,
        rtc_config=rtc_config,
        denoising_steps=args.denoising_steps,
    )

    server = PolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
    )
    server.register_endpoint(
        "rtc_capabilities",
        policy.rtc_capabilities,
        requires_input=False,
    )
    server.register_endpoint(
        "reset_rtc",
        policy.reset_rtc,
        requires_input=False,
    )

    print(
        f"\n✓ Standalone RTC server ready — listening on "
        f"0.0.0.0:{args.port}\n"
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main()