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


# Debug: with VLLM_DEBUG_MEM_REPORT=1, every vLLM process installs a SIGUSR1
# handler that prints where its memory is: private RSS; glibc malloc arenas,
# free bytes and malloc's own large mmaps (malloc_info); a size histogram of
# anonymous memory regions; Python allocations by file (if PYTHONTRACEMALLOC
# is set); and live CPU torch tensors plus the pinned host-memory pool.
def _install_mem_report_debug_handler() -> None:
    import os

    if os.environ.get("VLLM_DEBUG_MEM_REPORT") != "1":
        return
    import signal
    import sys

    def _malloc_info(out) -> None:
        import ctypes
        import re as _re

        libc = ctypes.CDLL("libc.so.6")
        libc.fopen.restype = ctypes.c_void_p
        libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.fclose.argtypes = [ctypes.c_void_p]
        path = f"/tmp/malloc_info_{os.getpid()}.xml"
        fp = libc.fopen(path.encode(), b"w")
        libc.malloc_info(0, fp)
        libc.fclose(fp)
        xml = open(path).read()
        os.unlink(path)
        tail = xml[xml.rfind("</heap>"):]
        def val(pattern):
            m = _re.search(pattern, tail)
            return int(m.group(1)) / 2**20 if m else -1
        arenas = len(_re.findall(r"<heap nr=", xml))
        out.append(
            f"[mem-report]   malloc: {arenas} arenas, arena memory={val(r'<system type=.current. size=.(\d+)')}"
            f" MB (free in bins: fast={val(r'<total type=.fast. count=.\d+. size=.(\d+)'):.0f}"
            f" rest={val(r'<total type=.rest. count=.\d+. size=.(\d+)'):.0f} MB),"
            f" large blocks mmapped by malloc={val(r'<total type=.mmap. count=.\d+. size=.(\d+)'):.0f} MB"
        )

    def _regions(out) -> None:
        buckets = {}
        big = []
        path = ""
        start_end = ""
        for line in open(f"/proc/{os.getpid()}/smaps"):
            f = line.split()
            if f and "-" in f[0] and len(f[0]) > 8 and all(c in "0123456789abcdef-" for c in f[0]):
                start_end = f[0]
                path = f[5] if len(f) > 5 else ""
            elif line.startswith("Rss:") and path in ("", "[heap]"):
                kb = int(f[1])
                a, b = (int(x, 16) for x in start_end.split("-"))
                size_mb = (b - a) / 2**20
                bucket = ("<1MB" if size_mb < 1 else "1-16MB" if size_mb < 16 else
                          "16-64MB" if size_mb < 64 else "64-256MB" if size_mb < 256 else ">=256MB")
                n, r = buckets.get(bucket, (0, 0))
                buckets[bucket] = (n + 1, r + kb)
                big.append((kb, size_mb, path or "anon"))
        out.append("[mem-report]   anonymous regions by size (count, resident MB): " + ", ".join(
            f"{k}: {v[0]} / {v[1] / 1024:.0f}" for k, v in buckets.items()))
        big.sort(reverse=True)
        out.append("[mem-report]   largest resident regions: " + ", ".join(
            f"{kb / 1024:.0f} MB of {sz:.0f} MB {p}" for kb, sz, p in big[:8]))

    def _tensors(out) -> None:
        import gc

        import torch

        seen = {}
        errors = 0
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor):
                    continue
            except Exception:  # e.g. a dead weakref proxy
                continue
            try:
                if obj.device.type != "cpu":
                    continue
                st = obj.untyped_storage()
                key = st.data_ptr()
                if key == 0 or key in seen:
                    continue
                try:
                    pinned = bool(obj.is_pinned())
                except Exception:
                    pinned = None
                seen[key] = (st.nbytes(), pinned, tuple(obj.shape), str(obj.dtype))
            except Exception:
                errors += 1
        tot = sum(v[0] for v in seen.values()) / 2**20
        pinned = sum(v[0] for v in seen.values() if v[1]) / 2**20
        out.append(f"[mem-report]   CPU torch tensors: {len(seen)} storages, {tot:.0f} MB total, "
                   f"{pinned:.0f} MB pinned ({errors} errors); largest:")
        for nbytes, pin, shape, dtype in sorted(seen.values(), key=lambda v: -v[0])[:10]:
            tag = "pinned" if pin else "?pin  " if pin is None else "      "
            out.append(f"[mem-report]     {nbytes / 2**20:8.1f} MB {tag} {dtype} {shape}")
        for getter in ("torch.cuda.host_memory_stats", "torch.xpu.host_memory_stats",
                       "torch._C._host_memory_stats"):
            try:
                mod, _, fn = getter.rpartition(".")
                stats = getattr(eval(mod), fn)()
                keys = [k for k in stats if "allocated_bytes" in k or "reserved_bytes" in k]
                out.append(f"[mem-report]   {getter}: " + ", ".join(
                    f"{k}={stats[k] / 2**20:.0f} MB" for k in sorted(keys)[:6]))
                break
            except Exception:
                continue

    def _handler(signum, frame) -> None:
        out = []
        pid = os.getpid()
        try:
            with open(f"/proc/{pid}/comm") as f:
                name = f.read().strip()
            anon = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        anon = int(line.split()[1]) // 1024
            out.append(f"[mem-report] pid={pid} {name} RssAnon={anon} MB")
        except Exception as e:
            out.append(f"[mem-report] header failed: {e}")
        for part in (_malloc_info, _regions, _tensors):
            try:
                part(out)
            except Exception as e:  # never let instrumentation break a run
                out.append(f"[mem-report]   {part.__name__} failed: {e}")
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                stats = tracemalloc.take_snapshot().statistics("filename")
                out.append(f"[mem-report]   python (tracemalloc) total="
                           f"{sum(st.size for st in stats) / 2**20:.0f} MB")
                for st in stats[:12]:
                    out.append(f"[mem-report]     {st.size / 2**20:7.1f} MB  {st.traceback[0].filename}")
        except Exception as e:
            out.append(f"[mem-report]   tracemalloc failed: {e}")
        print("\n".join(out), file=sys.stderr, flush=True)

    try:
        signal.signal(signal.SIGUSR1, _handler)
    except ValueError:
        pass  # not the main thread


_install_mem_report_debug_handler()
