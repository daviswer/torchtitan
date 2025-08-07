# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.gptbigcode.model import ModelArgs, Transformer

__all__ = ["Transformer"]

gptbigcode_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "20B": ModelArgs(
        dim=6144,
        max_seq_len=8192,
        n_layers=52,
        n_heads=48,
        n_kv_heads=48,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        rope_theta=500000,
    ),
}
