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

"""Visualize deepest VLM-layer text-to-image attention (feeds action head).

For each camera, saves one grid image with every task-language subtoken's
attention heatmap overlaid on the same frame.

Example:
    python gr00t/eval/vl_eval.py \\
        --model-path /path/to/checkpoint-50000 \\
        --dataset-path /path/to/lerobot \\
        --embodiment-tag UNITREE_G1_WBC \\
        --traj-ids 150 --frame 0 \\
        --task "Pick up all scattered cushions and gather them together olderly." \\
        --save-dir /tmp/vl_eval
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from gr00t.eval.open_loop_eval import parse_observation_gr00t


@dataclass
class ArgsConfig:
    """CLI arguments for VLM text-to-image attention visualization."""

    model_path: str
    """Path to the finetuned model checkpoint."""

    dataset_path: str
    """Path to the LeRobot dataset."""

    embodiment_tag: str = "UNITREE_G1_WBC"
    """Embodiment tag (name or value, case-insensitive)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """Trajectory indices to visualize."""

    frame: int = 0
    """Frame index within each trajectory (0-based)."""

    save_dir: str = "/tmp/vl_eval"
    """Output directory for heatmaps and metadata."""

    task: str | None = None
    """Optional language override (replaces dataset task text)."""

    video_backend: str = "pyav"
    """Video decode backend."""

    cameras: list[str] | None = None
    """Optional subset of video modality keys. Default: all configured cameras."""

    overlay_alpha: float = 0.45
    """Heatmap overlay alpha on the base image."""

    device: str | None = None
    """Torch device. Default: cuda if available else cpu."""


def _get_qwen_processor(policy: Gr00tPolicy):
    collator = policy.collate_fn
    if hasattr(collator, "processor"):
        return collator.processor
    raise RuntimeError("Could not locate Qwen3VL processor on policy collator.")


def _get_vl_models(policy: Gr00tPolicy) -> tuple[Any, Any]:
    """Return (Qwen3VLForConditionalGeneration wrapper, Qwen3VLModel inner)."""
    wrapper = policy.model.backbone.model
    inner = wrapper.model if hasattr(wrapper, "model") else wrapper
    return wrapper, inner


def _backbone_output_layer_index(policy: Gr00tPolicy) -> int:
    """Last kept LLM layer index (same layer whose hidden states feed action head)."""
    return len(_get_vl_models(policy)[0].language_model.layers) - 1


def _build_observation(
    traj,
    step_count: int,
    modality_configs: dict[str, Any],
    embodiment_tag: EmbodimentTag,
    task_override: str | None,
) -> dict[str, Any]:
    data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
    obs: dict[str, Any] = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    text = task_override if task_override is not None else data_point.text
    for language_key in modality_configs["language"].modality_keys:
        obs[language_key] = text
    return parse_observation_gr00t(obs, modality_configs)


def _prepare_vlm_batch(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    processed_inputs = []
    for obs in policy._unbatch_observation(observation):
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))
    collated_inputs = policy.collate_fn(processed_inputs)["inputs"]
    device = policy.model.device
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in collated_inputs.items()
    }


def _collect_template_special_ids(tokenizer, model_config) -> set[int]:
    special_ids = {
        model_config.image_token_id,
        model_config.video_token_id,
        model_config.vision_start_token_id,
        model_config.vision_end_token_id,
    }
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    for tok in (
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ):
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            special_ids.add(tok_id)
    return special_ids


def _get_text_token_entries(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    model_config,
) -> list[dict[str, Any]]:
    special_ids = _collect_template_special_ids(tokenizer, model_config)
    entries: list[dict[str, Any]] = []
    for pos, (tok_id, valid) in enumerate(
        zip(input_ids[0].tolist(), attention_mask[0].bool().tolist(), strict=True)
    ):
        if not valid or tok_id in special_ids:
            continue
        token_str = tokenizer.decode([tok_id])
        stripped = token_str.strip()
        if stripped == "" or stripped in ("user", "assistant", "system"):
            continue
        entries.append({"position": pos, "token_id": tok_id, "token": token_str})
    return entries


def _get_image_spans(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    camera_names: list[str],
    image_token_id: int,
) -> list[dict[str, Any]]:
    seq_ids = input_ids[0]
    valid_mask = attention_mask[0].bool()
    image_positions = torch.nonzero((seq_ids == image_token_id) & valid_mask, as_tuple=False)[:, 0]
    image_positions = image_positions.tolist()
    merge_length = spatial_merge_size**2

    spans: list[dict[str, Any]] = []
    cursor = 0
    for cam_idx, (t, h, w) in enumerate(image_grid_thw.detach().cpu().tolist()):
        n_tokens = int(t * h * w // merge_length)
        cam_positions = image_positions[cursor : cursor + n_tokens]
        grid_h, grid_w = int(h // spatial_merge_size), int(w // spatial_merge_size)
        spans.append(
            {
                "camera": camera_names[cam_idx],
                "positions": cam_positions,
                "grid_shape": (grid_h, grid_w),
            }
        )
        cursor += n_tokens
    return spans


def _run_vlm_forward_last_layer_attention(
    vl_wrapper: Any,
    vl_inner: Any,
    vl_inputs: dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    """Run VLM forward and return attention weights from one decoder layer."""
    keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    batch = {k: vl_inputs[k] for k in keys}
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        attn_weights = output[1]
        if attn_weights is not None:
            captured["attn"] = attn_weights.detach()

    handle = vl_wrapper.language_model.layers[layer_idx].self_attn.register_forward_hook(hook)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_grid_thw = batch["image_grid_thw"]

    inputs_embeds = vl_wrapper.get_input_embeddings()(input_ids)
    image_embeds, deepstack_image_embeds = vl_wrapper.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = vl_inner.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    image_mask = image_mask[..., 0]

    position_ids, rope_deltas = vl_inner.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask=attention_mask
    )
    vl_inner.rope_deltas = rope_deltas

    vl_inner.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_image_embeds,
    )
    handle.remove()

    if "attn" not in captured:
        raise RuntimeError(
            "No attention weights captured. Ensure attn_implementation='eager' is set."
        )
    return captured["attn"]


def _text_to_image_attention(
    attn: torch.Tensor,
    text_position: int,
    image_positions: list[int],
) -> np.ndarray:
    """Average over heads: [num_image_tokens]."""
    img_attn = attn[0, :, text_position, image_positions]
    return img_attn.mean(dim=0).float().cpu().numpy()


def _scores_to_heatmap(scores: np.ndarray, grid_shape: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
    heat = scores.reshape(grid_shape)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    resized = F.interpolate(
        torch.from_numpy(heat)[None, None, ...].float(),
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def _overlay_heatmap(image_hwc: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    base = image_hwc.astype(np.float32) / 255.0
    colored = plt.cm.jet(heatmap)[..., :3]
    return np.clip(((1.0 - alpha) * base + alpha * colored) * 255.0, 0, 255).astype(np.uint8)


def _get_display_image(observation: dict[str, Any], camera: str) -> np.ndarray:
    video = observation["video"][camera]
    frame = video[0, 0] if video.ndim == 5 else video[0]
    if frame.shape[0] == 3 and frame.ndim == 3:
        frame = np.transpose(frame, (1, 2, 0))
    return np.asarray(frame)


def _save_camera_token_grid(
    image_hwc: np.ndarray,
    token_entries: list[dict[str, Any]],
    attn: torch.Tensor,
    cam_span: dict[str, Any],
    layer_idx: int,
    save_path: Path,
    alpha: float,
) -> None:
    """One figure per camera: each row is one language token's heatmap."""
    n_rows = len(token_entries)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(5, 3.2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    out_hw = (image_hwc.shape[0], image_hwc.shape[1])
    for ax, entry in zip(axes, token_entries, strict=True):
        scores = _text_to_image_attention(attn, entry["position"], cam_span["positions"])
        heatmap = _scores_to_heatmap(scores, cam_span["grid_shape"], out_hw)
        ax.imshow(_overlay_heatmap(image_hwc, heatmap, alpha=alpha))
        ax.set_ylabel(repr(entry["token"]), fontsize=10, rotation=0, labelpad=40, ha="right")
        ax.set_yticks([])
        ax.set_xticks([])

    fig.suptitle(
        f"{cam_span['camera']} | layer {layer_idx} (action-head backbone)",
        fontsize=13,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_vlm_attention_for_frame(
    policy: Gr00tPolicy,
    observation: dict[str, Any],
    camera_names: list[str],
    layer_idx: int,
    save_step_dir: Path,
    overlay_alpha: float,
) -> dict[str, Any]:
    vl_inputs = _prepare_vlm_batch(policy, observation)
    tokenizer = _get_qwen_processor(policy).tokenizer

    vl_wrapper, vl_inner = _get_vl_models(policy)
    vl_wrapper.set_attn_implementation("eager")

    with torch.inference_mode():
        attn = _run_vlm_forward_last_layer_attention(vl_wrapper, vl_inner, vl_inputs, layer_idx)

    model_config = vl_wrapper.config
    spatial_merge_size = model_config.vision_config.spatial_merge_size
    text_entries = _get_text_token_entries(
        vl_inputs["input_ids"], vl_inputs["attention_mask"], tokenizer, model_config
    )
    image_spans = _get_image_spans(
        vl_inputs["input_ids"],
        vl_inputs["attention_mask"],
        vl_inputs["image_grid_thw"],
        spatial_merge_size,
        camera_names,
        model_config.image_token_id,
    )

    for span in image_spans:
        if span["camera"] not in camera_names:
            continue
        _save_camera_token_grid(
            _get_display_image(observation, span["camera"]),
            text_entries,
            attn,
            span,
            layer_idx,
            save_step_dir / f"{span['camera']}.png",
            overlay_alpha,
        )

    meta = {
        "backbone_layer": layer_idx,
        "cameras": camera_names,
        "text_tokens": text_entries,
        "image_spans": [
            {"camera": s["camera"], "num_tokens": len(s["positions"]), "grid_shape": list(s["grid_shape"])}
            for s in image_spans
        ],
    }
    with (save_step_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def evaluate_trajectory(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    frame: int,
    save_dir: Path,
    task_override: str | None,
    cameras: list[str] | None,
    overlay_alpha: float,
) -> None:
    traj = loader[traj_id]
    if frame < 0 or frame >= len(traj):
        raise ValueError(
            f"Frame {frame} out of range for traj {traj_id} (length {len(traj)})"
        )
    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action", None)

    all_cameras = loader.modality_configs["video"].modality_keys
    camera_names = all_cameras if cameras is None else [c for c in cameras if c in all_cameras]
    if not camera_names:
        raise ValueError(f"No valid cameras selected. Available: {all_cameras}")

    layer_idx = _backbone_output_layer_index(policy)
    logging.info("Using backbone output layer %d for text-to-image attention", layer_idx)
    logging.info("Visualizing traj=%s frame=%s", traj_id, frame)

    save_step_dir = save_dir / f"traj_{traj_id}" / f"frame_{frame}"
    evaluate_vlm_attention_for_frame(
        policy,
        _build_observation(traj, frame, modality_configs, embodiment_tag, task_override),
        camera_names,
        layer_idx,
        save_step_dir,
        overlay_alpha,
    )
    logging.info("Saved to %s", save_step_dir)


def main(args: ArgsConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=args.model_path, device=device)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Dataset length: %s | trajectories: %s", len(loader), args.traj_ids)

    for traj_id in args.traj_ids:
        if traj_id >= len(loader):
            logging.warning("Trajectory ID %s out of range, skipping.", traj_id)
            continue
        evaluate_trajectory(
            policy, loader, traj_id, embodiment_tag,
            args.frame, save_dir, args.task, args.cameras, args.overlay_alpha,
        )


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
