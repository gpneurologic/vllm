# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401

MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]


# Debug: with VLLM_DEBUG_MALLOC_TRIM=1, every vLLM process installs a SIGUSR2
# handler that prints its memory, calls glibc malloc_trim(0) to hand freed
# heap back to the OS, and prints its memory again. Measures how much of a
# process's memory is freed-but-held heap.
def _install_malloc_trim_debug_handler() -> None:
    import os

    if os.environ.get("VLLM_DEBUG_MALLOC_TRIM") != "1":
        return
    import ctypes
    import signal
    import sys

    def _mem() -> str:
        fields = {}
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                key = line.split(":")[0]
                if key in ("RssAnon", "VmSwap"):
                    fields[key] = int(line.split()[1]) // 1024
        return f"anon={fields.get('RssAnon', 0)} MB swap={fields.get('VmSwap', 0)} MB"

    def _handler(signum, frame) -> None:
        try:
            with open(f"/proc/{os.getpid()}/comm") as f:
                name = f.read().strip()
            before = _mem()
            freed = ctypes.CDLL("libc.so.6").malloc_trim(0)
            print(
                f"[malloc-trim] pid={os.getpid()} {name} before: {before} "
                f"after: {_mem()} (malloc_trim returned {freed})",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:  # never let instrumentation break a run
            print(f"[malloc-trim] failed: {e}", file=sys.stderr, flush=True)

    try:
        signal.signal(signal.SIGUSR2, _handler)
    except ValueError:
        pass  # not the main thread


_install_malloc_trim_debug_handler()
