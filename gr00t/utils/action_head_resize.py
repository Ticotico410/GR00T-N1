# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.action_head.flow_matching_action_head import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    FlowmatchingActionHead,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.transforms import GR00TTransform


def get_max_action_dim_from_transforms(transforms: ComposedModalityTransform) -> Optional[int]:
    for transform in transforms.transforms:
        if isinstance(transform, GR00TTransform):
            return transform.max_action_dim
    return None


def _expand_category_linear_output(
    old_linear: CategorySpecificLinear, new_output_dim: int
) -> CategorySpecificLinear:
    num_categories, hidden_dim, old_output_dim = old_linear.W.shape
    new_linear = CategorySpecificLinear(num_categories, hidden_dim, new_output_dim)
    with torch.no_grad():
        new_linear.W[:, :, :old_output_dim] = old_linear.W
        new_linear.b[:, :old_output_dim] = old_linear.b
    return new_linear


def resize_action_head_action_dim(model: GR00T_N1, new_action_dim: int) -> None:
    """Expand action head I/O dims when finetuning with action_dim > pretrained checkpoint."""
    old_action_dim = model.action_dim
    if old_action_dim == new_action_dim:
        return

    if new_action_dim < old_action_dim:
        raise ValueError(
            f"Cannot shrink action_dim from {old_action_dim} to {new_action_dim}. "
            "Reduce max_action_dim in data_config or use a compatible checkpoint."
        )

    action_head: FlowmatchingActionHead = model.action_head
    device = next(action_head.parameters()).device
    dtype = next(action_head.parameters()).dtype

    old_encoder = action_head.action_encoder
    new_encoder = MultiEmbodimentActionEncoder(
        action_dim=new_action_dim,
        hidden_size=old_encoder.hidden_size,
        num_embodiments=old_encoder.num_embodiments,
    )
    with torch.no_grad():
        new_encoder.W1.W[:, :old_action_dim, :] = old_encoder.W1.W
        new_encoder.W1.b.copy_(old_encoder.W1.b)
        new_encoder.W2.W.copy_(old_encoder.W2.W)
        new_encoder.W2.b.copy_(old_encoder.W2.b)
        new_encoder.W3.W.copy_(old_encoder.W3.W)
        new_encoder.W3.b.copy_(old_encoder.W3.b)

    old_decoder = action_head.action_decoder
    new_decoder = CategorySpecificMLP(
        num_categories=old_decoder.num_categories,
        input_dim=old_decoder.layer1.W.shape[1],
        hidden_dim=old_decoder.layer1.W.shape[2],
        output_dim=new_action_dim,
    )
    new_decoder.layer1.load_state_dict(old_decoder.layer1.state_dict())
    new_decoder.layer2 = _expand_category_linear_output(old_decoder.layer2, new_action_dim)

    action_head.action_encoder = new_encoder.to(device=device, dtype=dtype)
    action_head.action_decoder = new_decoder.to(device=device, dtype=dtype)
    action_head.action_dim = new_action_dim
    action_head.config.action_dim = new_action_dim
    model.config.action_dim = new_action_dim
    model.action_dim = new_action_dim

    print(
        f"Resized action head action_dim: {old_action_dim} -> {new_action_dim} "
        f"(copied first {old_action_dim} dims, initialized remaining dims)"
    )
