# XPU peer-to-peer all-reduce

A one-shot all-reduce for Intel GPUs connected only by PCIe, used for small
messages in the tensor-parallel group. It was built and measured on 3x Arc Pro B70
(Battlemage, 32 GB, PCIe Gen5 x16 each, no Xe Link) running Qwen3.8 27B FP8 at
TP=3.

- Python: `vllm/distributed/device_communicators/xpu_p2p_all_reduce.py`
- Kernels (SYCL, loaded with ctypes): `csrc/xpu/p2p_all_reduce.cpp`
- Build: `tools/build_xpu_p2p_all_reduce.sh`
- Test: `tests/distributed/test_xpu_p2p_all_reduce.py`
- Hook: `XpuCommunicator.all_reduce`

## Why

Every XPU all-reduce went through oneCCL, which costs about 21 us per call at
TP=3, whether the message is 10 KB or 245 KB. When vLLM has to zero-pad a message
for oneCCL (its length doesn't split into world_size aligned parts), the cost is
about 40 us. Decode all-reduces are tokens x hidden_size, a few tens of KB, and a
forward pass has about 129 of them, so oneCCL's fixed cost is around 2.7 ms per
step. The data itself crosses PCIe in under a microsecond.

## Enabling

```bash
tools/build_xpu_p2p_all_reduce.sh          # writes _xpu_p2p_all_reduce.so next to the module
export VLLM_XPU_P2P_ALLREDUCE=1            # off by default
export VLLM_XPU_P2P_ALLREDUCE_MAX_BYTES=81920   # optional; larger messages stay on oneCCL
export VLLM_XPU_P2P_ALLREDUCE_LIB=/path/to/_xpu_p2p_all_reduce.so   # optional
```

Only the `tp` group uses it, as on CUDA. Set-up is collective. If any rank fails,
for example because the library is missing or peer mapping fails, every rank
logs a warning and stays on oneCCL. All ranks always take the same path, so they
can't deadlock. The choice per call depends only on dtype, size and
contiguity, which are the same on every rank.

## How it works

Each rank allocates a receive buffer with Level Zero: 2 parities x world slots x
`max_bytes / 4` 8-byte words. Ranks exchange the buffers' IPC handles over the
CPU group. The dma-buf fd inside a handle is duplicated into the receiving
process with `pidfd_getfd`, as oneCCL does with `CCL_ZE_IPC_EXCHANGE=pidfd`, and
then mapped with `zeMemOpenIpcHandle`.

One call enqueues `world` kernels on the current stream:

1. **world - 1 push kernels**, one per peer. Each writes this rank's data into its
   slot in that peer's buffer. Every 8-byte store holds 4 bytes of data plus a
   32-bit call tag ("LL" protocol).
2. **One reduce kernel.** Each work-item polls its words in the local buffer with
   system-scope atomic loads until every peer's tag matches. It then sums all
   ranks' values in rank order in fp32 and writes the result. The output may
   alias the input.

The call counter lives in device memory: `ctl[0]` counts finished calls, and
`ctl[1]` holds the current one. The first push kernel reads `ctl[0]` and writes
`ctl[1]`. Later kernels read `ctl[1]`. The reduce kernel writes `ctl[0]`. No
kernel reads a word it also writes. Because the counter is in device memory,
recorded XPU graphs advance it on every replay. Calls alternate between the two
parities. A rank can begin call n+1 only after receiving every peer's call-n data,
which each peer sends only after finishing call n-1. So a buffer is never
overwritten while a slower rank still reads it. The tag is never 0, so
never-written (zeroed) slots never look fresh.

## B70 findings that shaped the design

| Finding | Consequence |
|---|---|
| No PCIe atomics between GPUs (`atomics_supported = 0`) | No shared flags or counters across GPUs; use the LL tag in every word |
| A data write, then a flag write, can be seen in the wrong order | Again the LL tag: a word is either fresh or not |
| Polling a *peer's* memory over PCIe never completes | Push into the receiver's memory and poll locally |
| Plain or `volatile` local loads see stale cached data | Poll with system-scope atomic loads |
| **One kernel storing to two different peers is ~50x slower** (1,454 us vs 28 us for 983 KB), even with no other traffic | One push kernel per destination |
| Two GPUs writing into one at the same time runs at full speed | One-shot is fine for the receiver |
| The first kernel that touches a peer's memory can stall until that GPU is idle | Warm-up kernel between two host barriers at set-up |
| One write is visible on another GPU after about 1.3 us | Latency floor is low; the cost is mostly kernel launches |

The LL format doubles the bytes on the wire, and one-shot sends world - 1 copies.
That is why large messages stay on oneCCL's ring algorithm.

## Results (3x B70, TP=3)

Microbenchmark, fp16, 3 processes (`test_xpu_p2p_all_reduce.py --bench`). This
oneCCL figure is a direct `dist.all_reduce`, without vLLM's padding:

| Tokens x 5120 fp16 | P2P | oneCCL |
|---|---|---|
| 1 (10 KB) | 12.6 us | 19.3 us (about 40 us when vLLM pads it) |
| 3 (30 KB) | 12.6 us | 17.6 us |
| 6 (60 KB) | 12.7 us | 17.8 us |
| 8 (80 KB) | 15.7 us | 19.3 us |

The P2P path loses from about 12 tokens up (26 us vs 21.7 us at 12), hence the
80 KB default limit.

End to end, vLLM serving Qwen3.8 27B FP8 with MTP (2 draft tokens), temperature 0:

| Load | oneCCL only | With P2P | Change |
|---|---|---|---|
| 1 chat, 1,200 tokens | 72.5 tok/s | 75.2 tok/s | +3.7% |
| 1 chat, 400 tokens | 78.6 tok/s | 80.8 tok/s | +2.8% |
| 2 chats | 140.0 tok/s | 143.7 tok/s | +2.6% |
| 4 chats | 272.7 tok/s | 271.1 tok/s | same (main all-reduces > 80 KB) |

Correctness: the multi-process test checks bit-exact rank-order sums for fp16,
bf16 and fp32, in place and out of place, with batches of 100 calls queued
without synchronization, and 200 XPU-graph replays per dtype, at TP=2 and TP=3.
In vLLM, repeated temperature-0 answers are identical. Against oneCCL, answers
can drift after many tokens, because the sum is rounded differently (one fp32
sum, one rounding).

## Limits and ideas

- Supports fp16, bf16 and fp32, message sizes that are a multiple of 4 bytes, and
  2 to 8 ranks on one node.
- A rank that never arrives makes the poll give up after about 2^32 tries.
  `XpuP2PAllReduce.timeouts()` counts these (it should stay 0).
- Possible improvements: a 16-byte LL word (12 data bytes, so 33% overhead
  instead of 100%), merging the last push into the reduce kernel (one fewer
  launch), and AOT compilation for BMG (no JIT at the first call).
