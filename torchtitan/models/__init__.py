# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import llama3_configs, Transformer as LlamaTransformer
from torchtitan.models.granite import granite3_configs, Transformer as GraniteTransformer

models_config = {
    "llama3": llama3_configs,
    "granite3": granite3_configs,
}

model_name_to_cls = {
    "llama3": LlamaTransformer,
    "granite3": GraniteTransformer,
}

model_name_to_tokenizer = {
    "llama3": "tiktoken",
    "granite3": "tiktoken",
}
