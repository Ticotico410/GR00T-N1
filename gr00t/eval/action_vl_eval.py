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

"""Action-head cross-attention confusion matrices vs VLM backbone embeddings.

Builds heatmaps of how each future action step (query) attends to VL tokens (keys)
in the DiT cross-attention blocks at the last flow-matching denoising step:

  - matrix_text.png   : action steps x language subtokens (text cross-attn block)
  - matrix_image.png  : action steps x cameras (image cross-attn block, summed)
  - matrix_combined.png: text tokens + camera image columns in one view

Spatial image overlays are optional (--save-spatial-heatmaps); for language
grounding on pixels, use vl_eval.py instead.

Example:
    python gr00t/eval/action_vl_eval.py \\
        --model-path /path/to/checkpoint-50000 \\
        --dataset-path /path/to/lerobot \\
        --traj-ids 150 --frame 200 \\
        --save-dir /tmp/action_vl_eval
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from matplotlib import pyplot as plt
import numpy as np
import torch
import tyro

from gr00t.eval.vl_eval import (
    _build_observation,
    _get_image_spans,
    _get_qwen_processor,
    _get_text_token_entries,
    _get_vl_models,
    _prepare_vlm_batch,
)


@dataclass
class ArgsConfig:
    """CLI for action-to-VL cross-attention confusion matrices."""

    model_path: str
    dataset_path: str
    embodiment_tag: str = "UNITREE_G1_WBC"
    traj_ids: list[int] = field(default_factory=lambda: [0])
    frame: int = 0
    save_dir: str = "/tmp/action_vl_eval"
    task: str | None = None
    video_backend: str = "pyav"
    cameras: list[str] | None = None
    action_stride: int = 4
    """Plot every N-th action horizon step as a matrix row."""
    row_normalize: bool = True
    """Re-normalize each row over displayed columns (clearer comparison)."""
    device: str | None = None


def _last_text_cross_block_idx(num_layers: int, attend_text_every_n_blocks: int) -> int:
    last = -1
    for idx in range(num_layers):
        if idx % 2 == 0 and idx % (2 * attend_text_every_n_blocks) == 0:
            last = idx
    if last < 0:
        raise RuntimeError("No text cross-attention block found in DiT.")
    return last


def _last_image_cross_block_idx(num_layers: int, attend_text_every_n_blocks: int) -> int:
    last = -1
    for idx in range(num_layers):
        if idx % 2 == 0 and idx % (2 * attend_text_every_n_blocks) != 0:
            last = idx
    if last < 0:
        raise RuntimeError("No image cross-attention block found in DiT.")
    return last


def _extract_cross_attention_at_block(
    dit: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    image_mask: torch.Tensor,
    backbone_attention_mask: torch.Tensor,
    block_idx: int,
) -> torch.Tensor:
    """Manual QK softmax at one cross-attn block. Returns [T_sa, S_vl] (heads averaged)."""
    temb = dit.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    attend_every = getattr(dit, "attend_text_every_n_blocks", 2)

    image_attention_mask = image_mask & backbone_attention_mask
    non_image_attention_mask = (~image_mask) & backbone_attention_mask

    for idx, block in enumerate(dit.transformer_blocks):
        if idx == block_idx:
            attn_module = block.attn1
            norm_hidden = block.norm1(hidden_states, temb)
            q = attn_module.to_q(norm_hidden)
            k = attn_module.to_k(encoder_hidden_states)

            batch_size, t_sa, _ = q.shape
            seq_len = k.shape[1]
            n_heads = attn_module.heads
            head_dim = attn_module.inner_dim // n_heads

            q = q.view(batch_size, t_sa, n_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            enc_mask = (
                image_attention_mask
                if block_idx % (2 * attend_every) != 0
                else non_image_attention_mask
            )
            scores = scores.masked_fill(~enc_mask[:, None, None, :], float("-inf"))
            return scores.softmax(dim=-1)[0].mean(dim=0).detach().float().cpu()

        if idx % 2 == 1:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=temb,
            )
        else:
            curr_mask = (
                non_image_attention_mask
                if idx % (2 * attend_every) == 0
                else image_attention_mask
            )
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=curr_mask,
                temb=temb,
            )

    raise RuntimeError(f"Block index {block_idx} not reached.")


def _run_action_head_and_extract_attention(
    policy: Gr00tPolicy,
    collated_inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return (action→VL text-block attn, action→VL image-block attn, meta)."""
    model = policy.model
    action_head = model.action_head
    if not action_head.config.use_alternate_vl_dit:
        raise NotImplementedError("Only AlternateVLDiT checkpoints are supported.")

    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
    backbone_inputs, action_inputs = model.prepare_input(collated_inputs)
    backbone_outputs = model.backbone(backbone_inputs)

    image_mask = backbone_outputs["image_mask"]
    backbone_attention_mask = backbone_outputs["backbone_attention_mask"]
    features = action_head._encode_features(backbone_outputs, action_inputs)
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_inputs.embodiment_id

    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    actions = torch.randn(
        size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )

    dit = action_head.model
    num_layers = len(dit.transformer_blocks)
    attend_every = getattr(dit, "attend_text_every_n_blocks", 2)
    image_block = _last_image_cross_block_idx(num_layers, attend_every)
    text_block = _last_text_cross_block_idx(num_layers, attend_every)

    dt = 1.0 / action_head.num_inference_timesteps
    attn_text: torch.Tensor | None = None
    attn_image: torch.Tensor | None = None
    final_timestep: int | None = None

    for step in range(action_head.num_inference_timesteps):
        t_discretized = int((step / action_head.num_inference_timesteps) * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if step == action_head.num_inference_timesteps - 1:
            attn_text = _extract_cross_attention_at_block(
                dit, sa_embs, vl_embeds, timesteps_tensor, image_mask, backbone_attention_mask, text_block
            )
            attn_image = _extract_cross_attention_at_block(
                dit, sa_embs, vl_embeds, timesteps_tensor, image_mask, backbone_attention_mask, image_block
            )
            final_timestep = t_discretized

        model_output = dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=timesteps_tensor,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )
        pred = action_head.action_decoder(model_output, embodiment_id)
        actions = actions + dt * pred[:, -action_head.config.action_horizon :]

    assert attn_text is not None and attn_image is not None
    meta = {
        "dit_image_block_idx": image_block,
        "dit_text_block_idx": text_block,
        "dit_num_layers": num_layers,
        "denoising_step": final_timestep,
        "action_horizon": action_head.config.action_horizon,
        "attend_text_every_n_blocks": attend_every,
    }
    return attn_text, attn_image, meta


def _maybe_row_normalize(matrix: np.ndarray) -> np.ndarray:
    if not matrix.size:
        return matrix
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return matrix / row_sums


def _build_text_matrix(
    attn_text: torch.Tensor,
    text_entries: list[dict[str, Any]],
    action_steps: list[int],
    row_normalize: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    row_labels = [f"t+{s}" for s in action_steps]
    col_labels = [repr(e["token"]) for e in text_entries]
    positions = [e["position"] for e in text_entries]
    matrix = np.stack(
        [attn_text[1 + step, positions].numpy() for step in action_steps],
        axis=0,
    )
    if row_normalize:
        matrix = _maybe_row_normalize(matrix)
    return matrix, row_labels, col_labels


def _build_image_matrix(
    attn_image: torch.Tensor,
    image_spans: list[dict[str, Any]],
    action_steps: list[int],
    row_normalize: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    row_labels = [f"t+{s}" for s in action_steps]
    col_labels = [f"img:{s['camera']}" for s in image_spans]
    matrix = np.stack(
        [
            np.array([attn_image[1 + step, span["positions"]].sum().item() for span in image_spans])
            for step in action_steps
        ],
        axis=0,
    )
    if row_normalize:
        matrix = _maybe_row_normalize(matrix)
    return matrix, row_labels, col_labels


def _build_combined_matrix(
    text_matrix: np.ndarray,
    image_matrix: np.ndarray,
    text_col_labels: list[str],
    image_col_labels: list[str],
    row_labels: list[str],
    row_normalize: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    matrix = np.concatenate([text_matrix, image_matrix], axis=1)
    col_labels = text_col_labels + image_col_labels
    if row_normalize:
        matrix = _maybe_row_normalize(matrix)
    return matrix, row_labels, col_labels


def _save_confusion_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    save_path: Path,
    cmap: str = "Blues",
) -> None:
    n_rows, n_cols = matrix.shape
    fig_w = max(8, 0.45 * n_cols + 2)
    fig_h = max(4, 0.35 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=matrix.max() or 1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("VL embedding (keys)")
    ax.set_ylabel("Action step (queries)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_matrix_csv(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    save_path: Path,
) -> None:
    lines = ["," + ",".join(col_labels)]
    for row_label, row in zip(row_labels, matrix, strict=True):
        vals = ",".join(f"{v:.6f}" for v in row)
        lines.append(f"{row_label},{vals}")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_frame(
    policy: Gr00tPolicy,
    observation: dict[str, Any],
    camera_names: list[str],
    save_step_dir: Path,
    action_stride: int,
    row_normalize: bool,
) -> dict[str, Any]:
    collated_inputs = _prepare_vlm_batch(policy, observation)
    attn_text, attn_image, run_meta = _run_action_head_and_extract_attention(policy, collated_inputs)

    tokenizer = _get_qwen_processor(policy).tokenizer
    vl_wrapper, _ = _get_vl_models(policy)
    model_config = vl_wrapper.config
    spatial_merge_size = model_config.vision_config.spatial_merge_size

    text_entries = _get_text_token_entries(
        collated_inputs["input_ids"],
        collated_inputs["attention_mask"],
        tokenizer,
        model_config,
    )
    image_spans = _get_image_spans(
        collated_inputs["input_ids"],
        collated_inputs["attention_mask"],
        collated_inputs["image_grid_thw"],
        spatial_merge_size,
        camera_names,
        model_config.image_token_id,
    )
    image_spans = [s for s in image_spans if s["camera"] in camera_names]

    action_horizon = run_meta["action_horizon"]
    action_steps = list(range(0, action_horizon, action_stride))

    text_matrix, text_rows, text_cols = _build_text_matrix(
        attn_text, text_entries, action_steps, row_normalize=False
    )
    image_matrix, image_rows, image_cols = _build_image_matrix(
        attn_image, image_spans, action_steps, row_normalize=False
    )

    # Display versions (optionally row-normalized per matrix)
    text_display = _maybe_row_normalize(text_matrix) if row_normalize else text_matrix
    image_display = _maybe_row_normalize(image_matrix) if row_normalize else image_matrix
    combined_display, combined_rows, combined_cols = _build_combined_matrix(
        text_matrix, image_matrix, text_cols, image_cols, text_rows, row_normalize=row_normalize
    )

    norm_note = "row-normalized" if row_normalize else "raw softmax mass"
    _save_confusion_matrix(
        text_display,
        text_rows,
        text_cols,
        f"Action→Text VL cross-attn (DiT block {run_meta['dit_text_block_idx']}, {norm_note})",
        save_step_dir / "matrix_text.png",
        cmap="Purples",
    )
    _save_confusion_matrix(
        image_display,
        image_rows,
        image_cols,
        f"Action→Image VL cross-attn (DiT block {run_meta['dit_image_block_idx']}, {norm_note})",
        save_step_dir / "matrix_image.png",
        cmap="Greens",
    )
    _save_confusion_matrix(
        combined_display,
        combined_rows,
        combined_cols,
        f"Action→VL combined ({norm_note})",
        save_step_dir / "matrix_combined.png",
        cmap="YlOrRd",
    )

    _save_matrix_csv(text_matrix, text_rows, text_cols, save_step_dir / "matrix_text.csv")
    _save_matrix_csv(image_matrix, image_rows, image_cols, save_step_dir / "matrix_image.csv")
    _save_matrix_csv(combined_display, combined_rows, combined_cols, save_step_dir / "matrix_combined.csv")

    meta = {
        **run_meta,
        "row_normalize_display": row_normalize,
        "action_steps": action_steps,
        "text_columns": [{"token": e["token"], "position": e["position"]} for e in text_entries],
        "image_columns": [{"camera": s["camera"], "num_tokens": len(s["positions"])} for s in image_spans],
        "text_matrix_shape": list(text_matrix.shape),
        "image_matrix_shape": list(image_matrix.shape),
        "interpretation": (
            "Rows = future action steps (t+k). Columns = VL keys at DiT cross-attention. "
            "Text matrix uses the last text cross-attn block; image matrix uses the last "
            "image cross-attn block (per-camera attention mass summed over image tokens)."
        ),
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
    action_stride: int,
    row_normalize: bool,
) -> None:
    traj = loader[traj_id]
    if frame < 0 or frame >= len(traj):
        raise ValueError(f"Frame {frame} out of range for traj {traj_id} (length {len(traj)})")

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action", None)

    all_cameras = loader.modality_configs["video"].modality_keys
    camera_names = all_cameras if cameras is None else [c for c in cameras if c in all_cameras]
    if not camera_names:
        raise ValueError(f"No valid cameras selected. Available: {all_cameras}")

    logging.info("Building action-VL matrices for traj=%s frame=%s", traj_id, frame)
    save_step_dir = save_dir / f"traj_{traj_id}" / f"frame_{frame}"
    evaluate_frame(
        policy,
        _build_observation(traj, frame, modality_configs, embodiment_tag, task_override),
        camera_names,
        save_step_dir,
        action_stride,
        row_normalize,
    )
    logging.info("Saved matrices to %s", save_step_dir)


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
            policy,
            loader,
            traj_id,
            embodiment_tag,
            args.frame,
            save_dir,
            args.task,
            args.cameras,
            args.action_stride,
            args.row_normalize,
        )


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
