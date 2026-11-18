# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import nn


class DSparkDeepseekV4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        raise NotImplementedError("DeepSeek V4.1 DSpark drafting is not ported to XPU yet.")
