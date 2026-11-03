// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// One-shot peer-to-peer all-reduce for Intel GPUs that talk over plain PCIe
// (tested on 3x Arc Pro B70). Used by XpuCommunicator for small messages,
// where oneCCL's fixed cost (~21 us per call) dominates.
//
// Design, and why (measured on B70, see docs/design/xpu_p2p_all_reduce.md):
//  * No PCIe atomics between GPUs, so ranks can't sync with flags. Instead each
//    16-byte store carries 12 bytes of data plus a 32-bit call number in its
//    last 4 bytes ("LL" protocol). A word is fresh when its call number
//    matches; no ordering or fences needed. On B70 a 16-byte vector store
//    arrives in one piece (tested: ~300k calls over all GPU pairs, no tearing;
//    two separate 8-byte stores do tear). The receiver polls the upper 8
//    bytes (tag) and then loads the lower 8, both with system-scope atomics.
//  * Data is pushed into the receiver's memory and polled there with local
//    system-scope atomic loads. Polling a peer's memory over PCIe doesn't work.
//  * One kernel storing to two different peers is ~50x slower than storing to
//    one, so each peer gets its own push kernel, except that the last push is
//    done by the kernel that then waits and sums (it stores to one peer only).
//  * The call number lives in device memory, so recorded XPU graphs advance it
//    on every replay. Calls alternate between two buffer sets (parity), so a
//    rank that starts call n+1 never overwrites data a slower rank still reads.
//  * Every rank sums the values in rank order, so all ranks get bit-identical
//    results.
//
// The library is loaded with ctypes; it launches on the caller's sycl::queue
// (torch.xpu.current_stream().sycl_queue).

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

namespace {

constexpr int kMaxWorld = 8;
constexpr int kWG = 256;
constexpr size_t kMaxGroups = 64;
// Safety net only: a rank that never arrives makes the poll give up after
// roughly a minute instead of spinning forever. Counted in ctl[2].
constexpr uint64_t kSpinLimit = 1ull << 32;

// VLLM_XPU_P2P_ALLREDUCE_NO_MERGE=1 keeps every push in its own kernel (the
// pre-merge behaviour), for A/B measurements.
const bool g_no_merge = [] {
  const char *v = std::getenv("VLLM_XPU_P2P_ALLREDUCE_NO_MERGE");
  return v && *v && std::strcmp(v, "0") != 0;
}();

using a64sys = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                sycl::access::address_space::global_space>;
using a64dev = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

enum DType : int { kF16 = 0, kBF16 = 1, kF32 = 2 };

struct State {
  int rank = 0;
  int world = 0;
  size_t max_words = 0;   // max data words (4 bytes) per call
  size_t slot_words = 0;  // uint64 per (parity, source) slot: 2 x ceil(max_words / 3), 4 KB aligned
  size_t buf_bytes = 0;
  uint64_t *recv[kMaxWorld] = {};  // [rank] = own buffer, others = IPC-mapped peers
  bool mapped[kMaxWorld] = {};
  int peer_fd[kMaxWorld];
  uint64_t *ctl = nullptr;  // own device memory: [0] calls done, [1] current call, [2] timeouts,
                            // [kScratch..] fused-norm partials
  ze_context_handle_t zctx = nullptr;
  ze_device_handle_t zdev = nullptr;
  sycl::context *ctx = nullptr;
};

void set_err(char *err, int len, const char *msg, int code = 0) {
  if (err && len > 0) snprintf(err, len, "%s (0x%x)", msg, code);
}

inline uint32_t call_tag(uint64_t n) { return uint32_t(n % 0xFFFFFFFEull) + 1u; }  // never 0

// 32-bit word <-> two (or one) floats
template <int DT>
inline void unpack(uint32_t w, float &a, float &b) {
  if constexpr (DT == kF16) {
    a = float(sycl::bit_cast<sycl::half>(uint16_t(w & 0xFFFF)));
    b = float(sycl::bit_cast<sycl::half>(uint16_t(w >> 16)));
  } else if constexpr (DT == kBF16) {
    a = sycl::bit_cast<float>(w << 16);
    b = sycl::bit_cast<float>(w & 0xFFFF0000u);
  } else {
    a = sycl::bit_cast<float>(w);
    b = 0.f;
  }
}

inline uint16_t f2bf16(float f) {  // round to nearest even, NaN stays NaN
  uint32_t u = sycl::bit_cast<uint32_t>(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) return uint16_t((u >> 16) | 0x40);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

template <int DT>
inline uint32_t pack(float a, float b) {
  if constexpr (DT == kF16) {
    return uint32_t(sycl::bit_cast<uint16_t>(sycl::half(a))) |
           (uint32_t(sycl::bit_cast<uint16_t>(sycl::half(b))) << 16);
  } else if constexpr (DT == kBF16) {
    return uint32_t(f2bf16(a)) | (uint32_t(f2bf16(b)) << 16);
  } else {
    return sycl::bit_cast<uint32_t>(a);
  }
}

struct Ptrs {
  uint64_t *p[kMaxWorld];
};

sycl::nd_range<1> grid(size_t items) {
  size_t groups = std::min(kMaxGroups, std::max<size_t>(1, (items + kWG - 1) / kWG));
  return sycl::nd_range<1>(groups * kWG, kWG);
}

using u32x4 = sycl::vec<uint32_t, 4>;

inline size_t ll_words(size_t nw) { return (nw + 2) / 3; }  // 16-byte LL words for nw data words

// Store LL word j (data words 3j..3j+2 of `in`, zero past the end, then the tag)
// with one 16-byte store.
inline void put_ll(uint64_t *dst, const uint32_t *in, size_t nw, size_t j, uint32_t tag) {
  const size_t b = 3 * j;
  const uint32_t d0 = in[b];
  const uint32_t d1 = b + 1 < nw ? in[b + 1] : 0u;
  const uint32_t d2 = b + 2 < nw ? in[b + 2] : 0u;
  *reinterpret_cast<u32x4 *>(dst + 2 * j) = u32x4(d0, d1, d2, tag);
}

// Push this rank's words into `peer`'s receive buffer. The first push kernel of a
// call computes the call number from ctl[0] and publishes it in ctl[1]; later
// kernels of the same call read ctl[1]. No kernel reads a word it also writes.
void launch_push(sycl::queue &q, const State &s, int peer, bool first, const uint32_t *in, size_t nw) {
  uint64_t *peer_buf = s.recv[peer];
  uint64_t *ctl = s.ctl;
  const int rank = s.rank, world = s.world;
  const size_t slot = s.slot_words;
  const size_t nll = ll_words(nw);
  q.parallel_for(grid(nll), [=](sycl::nd_item<1> it) {
    const size_t g = it.get_global_id(0), G = it.get_global_range(0);
    const uint64_t n = first ? ctl[0] + 1 : ctl[1];
    if (first && g == 0) ctl[1] = n;
    const uint32_t tag = call_tag(n);
    uint64_t *dst = peer_buf + (size_t(n & 1) * world + rank) * slot;
    for (size_t j = g; j < nll; j += G) put_ll(dst, in, nw, j, tag);
  });
}

// Wait for every peer's words of this call, sum in rank order, write out.
// `out` may alias `in`: each work-item reads in[3j..3j+2] before writing them.
// If `last_peer` >= 0 the kernel first pushes this rank's words to that peer (the
// last push of the call), saving one kernel launch. Each work-item does all of
// its push stores before it polls, and the push indices match the reduce
// indices, so nothing in the kernel waits on its own stores. The kernel still
// stores to one peer only (two peers in one kernel is ~50x slower on B70). It
// relies on all its work-groups being resident at once (<= 64 groups of 256),
// so a peer's merged kernel can't be stuck behind its own polling groups.
template <int DT>
void launch_reduce(sycl::queue &q, const State &s, const uint32_t *in, uint32_t *out, size_t nw,
                   int last_peer = -1) {
  uint64_t *own = s.recv[s.rank];
  uint64_t *ctl = s.ctl;
  const int rank = s.rank, world = s.world;
  const size_t slot = s.slot_words;
  const size_t nll = ll_words(nw);
  uint64_t *peer_buf = last_peer >= 0 ? s.recv[last_peer] : nullptr;
  q.parallel_for(grid(nll), [=](sycl::nd_item<1> it) {
    const size_t g = it.get_global_id(0), G = it.get_global_range(0);
    const uint64_t n = ctl[1];
    const uint32_t tag = call_tag(n);
    if (peer_buf) {
      uint64_t *dst = peer_buf + (size_t(n & 1) * world + rank) * slot;
      for (size_t j = g; j < nll; j += G) put_ll(dst, in, nw, j, tag);
    }
    const uint64_t *base = own + size_t(n & 1) * world * slot;
    for (size_t j = g; j < nll; j += G) {
      const size_t b = 3 * j;
      float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int k = 0; k < world; k++) {
        uint32_t w[3];
        if (k == rank) {
          w[0] = in[b];
          w[1] = b + 1 < nw ? in[b + 1] : 0u;
          w[2] = b + 2 < nw ? in[b + 2] : 0u;
        } else {
          uint64_t *p = const_cast<uint64_t *>(base + size_t(k) * slot + 2 * j);
          a64sys hi(p[1]);
          uint64_t h;
          uint64_t spins = 0;
          while (uint32_t((h = hi.load()) >> 32) != tag) {
            if (++spins > kSpinLimit) {
              a64dev(ctl[2]).fetch_add(1);
              break;
            }
          }
          // The whole 16-byte word landed together, so the low half is fresh too.
          const uint64_t l = a64sys(p[0]).load();
          w[0] = uint32_t(l);
          w[1] = uint32_t(l >> 32);
          w[2] = uint32_t(h);
        }
#pragma unroll
        for (int c = 0; c < 3; c++) {
          float a, bb;
          unpack<DT>(w[c], a, bb);
          acc[2 * c] += a;
          acc[2 * c + 1] += bb;
        }
      }
      out[b] = pack<DT>(acc[0], acc[1]);
      if (b + 1 < nw) out[b + 1] = pack<DT>(acc[2], acc[3]);
      if (b + 2 < nw) out[b + 2] = pack<DT>(acc[4], acc[5]);
    }
    if (g == 0) ctl[0] = n;  // nobody in this kernel reads ctl[0]
  });
}

// ---------------------------------------------------------------------------
// All-reduce fused with the residual add and RMSNorm that follow it in most
// transformer layers (vllm_ir fused_add_rms_norm), for a [rows, hidden] input:
//   ar      = all_reduce(in), rounded to the model dtype
//   s       = float(ar) + float(residual)          (fp32)
//   res_out = s rounded to the model dtype
//   y       = s * rsqrt(mean(s^2) + eps)
//   out     = y * (weight + bias), with y first rounded to the model dtype when
//             the weight is in the model dtype (bias 0), as the IR op does.
// Gemma-style norms pass the raw weight with bias 1 (weight.float() + 1).
// One work-group per row: it polls/sums the row's LL words, keeps the sums in
// registers, reduces the sum of squares over the group and writes both outputs.
// LL words that straddle two rows are read by both groups (reads only).
// A row gets up to kFusedMaxGPR work-groups (one group per row left the whole
// poll to a single Xe core and was ~6 us slower). The groups of a row combine
// their sums of squares through tagged partials in ctl[kScratch..]: each group
// stores (call tag << 32 | partial) and polls its row's partials until all carry
// this call's tag, then adds them in group order, so all groups and all ranks
// get the same total. Tags make resets unnecessary.
constexpr int kFusedWG = 256;
constexpr int kFusedMaxGPR = 8;
constexpr int kFusedMaxRows = 32;
constexpr int kFusedMaxLL = 4;  // LL words per work-item
constexpr size_t kScratch = 8;  // ctl[8..] = partials [row][group]
constexpr size_t kCtlWords = kScratch + size_t(kFusedMaxRows) * kFusedMaxGPR;

template <int DT>
inline float round_dt(float v) {
  if constexpr (DT == kF16) return float(sycl::half(v));
  else if constexpr (DT == kBF16) return sycl::bit_cast<float>(uint32_t(f2bf16(v)) << 16);
  else return v;
}

template <int DT>
inline float load_elem(const void *p, size_t e) {
  if constexpr (DT == kF16) return float(static_cast<const sycl::half *>(p)[e]);
  else if constexpr (DT == kBF16) return sycl::bit_cast<float>(uint32_t(static_cast<const uint16_t *>(p)[e]) << 16);
  else return static_cast<const float *>(p)[e];
}

template <int DT>
inline void store_elem(void *p, size_t e, float v) {
  if constexpr (DT == kF16) static_cast<sycl::half *>(p)[e] = sycl::half(v);
  else if constexpr (DT == kBF16) static_cast<uint16_t *>(p)[e] = f2bf16(v);
  else static_cast<float *>(p)[e] = v;
}

template <int DT, int WDT>
void launch_fused_add_rms(sycl::queue &q, const State &s, const uint32_t *in, const void *residual,
                          const void *weight, void *out, void *res_out, size_t rows, size_t hidden,
                          float bias, float eps, int last_peer, int gpr) {
  constexpr int EPW = DT == kF32 ? 1 : 2;  // elements per 4-byte word
  uint64_t *own = s.recv[s.rank];
  uint64_t *ctl = s.ctl;
  const int rank = s.rank, world = s.world;
  const size_t slot = s.slot_words;
  const size_t wpr = hidden / EPW;  // words per row
  const size_t nw = rows * wpr;
  const size_t nll = ll_words(nw);
  const bool round_x = bias == 0.f && WDT != kF32;
  uint64_t *peer_buf = last_peer >= 0 ? s.recv[last_peer] : nullptr;
  const size_t ipr = size_t(gpr) * kFusedWG;  // work-items per row
  q.parallel_for(sycl::nd_range<1>(rows * ipr, kFusedWG), [=](sycl::nd_item<1> it) {
    const size_t g = it.get_global_id(0), G = it.get_global_range(0);
    const uint64_t n = ctl[1];
    const uint32_t tag = call_tag(n);
    if (peer_buf) {  // merged last push, grid-stride over the whole message
      uint64_t *dst = peer_buf + (size_t(n & 1) * world + rank) * slot;
      for (size_t j = g; j < nll; j += G) put_ll(dst, in, nw, j, tag);
    }
    const uint64_t *base = own + size_t(n & 1) * world * slot;
    const size_t grp = it.get_group(0), row = grp / gpr, gi = grp % gpr;
    const size_t lid = it.get_local_id(0), qi = gi * kFusedWG + lid;
    const size_t w_lo = row * wpr, w_hi = w_lo + wpr;  // this row's words
    const size_t j_lo = w_lo / 3, j_hi = (w_hi + 2) / 3;
    float sv[kFusedMaxLL][6];
    float sq = 0.f;
#pragma unroll
    for (int t = 0; t < kFusedMaxLL; t++) {
      const size_t j = j_lo + qi + size_t(t) * ipr;
      if (j >= j_hi) break;
      const size_t b = 3 * j;
      float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int k = 0; k < world; k++) {
        uint32_t w[3];
        if (k == rank) {
          w[0] = in[b];
          w[1] = b + 1 < nw ? in[b + 1] : 0u;
          w[2] = b + 2 < nw ? in[b + 2] : 0u;
        } else {
          uint64_t *p = const_cast<uint64_t *>(base + size_t(k) * slot + 2 * j);
          a64sys hi(p[1]);
          uint64_t h;
          uint64_t spins = 0;
          while (uint32_t((h = hi.load()) >> 32) != tag) {
            if (++spins > kSpinLimit) {
              a64dev(ctl[2]).fetch_add(1);
              break;
            }
          }
          const uint64_t l = a64sys(p[0]).load();
          w[0] = uint32_t(l);
          w[1] = uint32_t(l >> 32);
          w[2] = uint32_t(h);
        }
#pragma unroll
        for (int c = 0; c < 3; c++) {
          float a, bb;
          unpack<DT>(w[c], a, bb);
          acc[2 * c] += a;
          acc[2 * c + 1] += bb;
        }
      }
#pragma unroll
      for (int c = 0; c < 3; c++) {
        const size_t word = b + c;
        const bool mine = word >= w_lo && word < w_hi;
#pragma unroll
        for (int e = 0; e < EPW; e++) {
          float v = 0.f;
          if (mine) {
            const size_t elem = word * EPW + e;  // flat element index
            v = round_dt<DT>(acc[2 * c + e]) + load_elem<DT>(residual, elem);
            sq += v * v;
          }
          sv[t][2 * c + e] = v;
        }
      }
    }
    const float part = sycl::reduce_over_group(it.get_group(), sq, sycl::plus<float>());
    float total = 0.f;
    if (lid == 0) {
      uint64_t *pp = ctl + kScratch + row * gpr;
      a64sys(pp[gi]).store(uint64_t(tag) << 32 | sycl::bit_cast<uint32_t>(part));
      for (int k = 0; k < gpr; k++) {
        a64sys a(pp[k]);
        uint64_t v;
        uint64_t spins = 0;
        while (uint32_t((v = a.load()) >> 32) != tag) {
          if (++spins > kSpinLimit) {
            a64dev(ctl[2]).fetch_add(1);
            break;
          }
        }
        total += sycl::bit_cast<float>(uint32_t(v));
      }
    }
    total = sycl::group_broadcast(it.get_group(), total, 0);
    const float scale = sycl::rsqrt(total / float(hidden) + eps);
#pragma unroll
    for (int t = 0; t < kFusedMaxLL; t++) {
      const size_t j = j_lo + qi + size_t(t) * ipr;
      if (j >= j_hi) break;
      const size_t b = 3 * j;
#pragma unroll
      for (int c = 0; c < 3; c++) {
        const size_t word = b + c;
        if (word < w_lo || word >= w_hi) continue;
#pragma unroll
        for (int e = 0; e < EPW; e++) {
          const size_t elem = word * EPW + e;
          const float v = sv[t][2 * c + e];
          store_elem<DT>(res_out, elem, v);
          float y = v * scale;
          if (round_x) y = round_dt<DT>(y);
          store_elem<DT>(out, elem, y * (load_elem<WDT>(weight, elem - row * hidden) + bias));
        }
      }
    }
    if (g == 0) ctl[0] = n;  // nobody in this kernel reads ctl[0]
  });
}

template <int DT>
void dispatch_fused(int wdtype, sycl::queue &q, const State &s, const uint32_t *in, const void *residual,
                    const void *weight, void *out, void *res_out, size_t rows, size_t hidden, float bias,
                    float eps, int last, int gpr) {
  switch (wdtype) {
    case kF16: launch_fused_add_rms<DT, kF16>(q, s, in, residual, weight, out, res_out, rows, hidden, bias, eps, last, gpr); break;
    case kBF16: launch_fused_add_rms<DT, kBF16>(q, s, in, residual, weight, out, res_out, rows, hidden, bias, eps, last, gpr); break;
    default: launch_fused_add_rms<DT, kF32>(q, s, in, residual, weight, out, res_out, rows, hidden, bias, eps, last, gpr); break;
  }
}

// Enqueue the pushes of one call; returns the peer for the merged last push (or -1).
int enqueue_pushes(sycl::queue &q, const State &s, const uint32_t *in, size_t nw) {
  // Peers in ring order from this rank. With 3+ ranks the last push goes into the
  // reduce kernel. With 2 ranks the only push must stay separate: it is the
  // first push, which sets ctl[1] from ctl[0], and the reduce writes ctl[0].
  const int separate = s.world > 2 && !g_no_merge ? s.world - 2 : s.world - 1;
  for (int d = 1; d <= separate; d++) launch_push(q, s, (s.rank + d) % s.world, d == 1, in, nw);
  return separate < s.world - 1 ? (s.rank + s.world - 1) % s.world : -1;
}

}  // namespace

extern "C" {

// Allocate this rank's receive buffer (2 parities x world slots x
// ceil(max_bytes/12) 16-byte LL words) and control words, zeroed. Returns a handle or nullptr (see err).
void *xpu_p2p_ar_create(void *queue, int rank, int world, uint64_t max_bytes, char *err, int errlen) {
  if (world < 2 || world > kMaxWorld || rank < 0 || rank >= world || max_bytes < 4 || max_bytes % 4) {
    set_err(err, errlen, "bad rank/world/max_bytes");
    return nullptr;
  }
  try {
    sycl::queue &q = *static_cast<sycl::queue *>(queue);
    if (q.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      set_err(err, errlen, "queue is not on the Level Zero backend");
      return nullptr;
    }
    State *s = new State();
    s->rank = rank;
    s->world = world;
    s->max_words = max_bytes / 4;
    // Round each slot up to 4 KB: misaligned slots turn the coalesced 64-byte PCIe
    // writes into partial-line writes, which made the 16-byte LL version ~4x slower.
    s->slot_words = (2 * ll_words(s->max_words) + 511) / 512 * 512;
    s->buf_bytes = 2 * size_t(world) * s->slot_words * sizeof(uint64_t);
    for (int k = 0; k < kMaxWorld; k++) s->peer_fd[k] = -1;
    s->ctx = new sycl::context(q.get_context());
    s->zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*s->ctx);
    s->zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    // Plain Level Zero allocation (not the SYCL USM pool), so its IPC handle
    // covers exactly this buffer.
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void *p = nullptr;
    ze_result_t rc = zeMemAllocDevice(s->zctx, &desc, s->buf_bytes, 4096, s->zdev, &p);
    if (rc != ZE_RESULT_SUCCESS) {
      set_err(err, errlen, "zeMemAllocDevice failed", rc);
      delete s->ctx;
      delete s;
      return nullptr;
    }
    s->recv[rank] = static_cast<uint64_t *>(p);
    s->mapped[rank] = true;
    s->ctl = sycl::malloc_device<uint64_t>(kCtlWords, q);
    q.memset(s->recv[rank], 0, s->buf_bytes);
    q.memset(s->ctl, 0, kCtlWords * sizeof(uint64_t));
    q.wait_and_throw();
    return s;
  } catch (const std::exception &e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

// Fill out[64] with the IPC handle of this rank's receive buffer.
int xpu_p2p_ar_get_ipc_handle(void *state, uint8_t *out) {
  State *s = static_cast<State *>(state);
  ze_ipc_mem_handle_t h;
  ze_result_t rc = zeMemGetIpcHandle(s->zctx, s->recv[s->rank], &h);
  if (rc != ZE_RESULT_SUCCESS) return int(rc);
  memcpy(out, h.data, ZE_MAX_IPC_HANDLE_SIZE);
  return 0;
}

// Map `peer`'s receive buffer. The handle holds a dma-buf fd that is only valid in
// the peer process, so it is duplicated into this process with pidfd_getfd
// first (the same exchange oneCCL uses with CCL_ZE_IPC_EXCHANGE=pidfd).
int xpu_p2p_ar_open_peer(void *state, int peer, int peer_pid, const uint8_t *handle, char *err, int errlen) {
  State *s = static_cast<State *>(state);
  if (peer < 0 || peer >= s->world || peer == s->rank) {
    set_err(err, errlen, "bad peer");
    return -1;
  }
  ze_ipc_mem_handle_t h;
  memcpy(h.data, handle, ZE_MAX_IPC_HANDLE_SIZE);
  int remote_fd;
  memcpy(&remote_fd, h.data, sizeof(int));
  int pidfd = int(syscall(SYS_pidfd_open, peer_pid, 0));
  if (pidfd < 0) {
    set_err(err, errlen, "pidfd_open failed", errno);
    return -1;
  }
  int fd = int(syscall(SYS_pidfd_getfd, pidfd, remote_fd, 0));
  close(pidfd);
  if (fd < 0) {
    set_err(err, errlen, "pidfd_getfd failed", errno);
    return -1;
  }
  memcpy(h.data, &fd, sizeof(int));
  void *p = nullptr;
  ze_result_t rc = zeMemOpenIpcHandle(s->zctx, s->zdev, h, 0, &p);
  if (rc != ZE_RESULT_SUCCESS) {
    close(fd);
    set_err(err, errlen, "zeMemOpenIpcHandle failed", rc);
    return -1;
  }
  s->recv[peer] = static_cast<uint64_t *>(p);
  s->mapped[peer] = true;
  s->peer_fd[peer] = fd;
  return 0;
}

// Write zeros into this rank's slots in every peer's buffer, then wait. Run it
// on all ranks between two host barriers before the first all-reduce: the first
// kernel that touches a peer's memory can stall until that peer GPU is idle,
// which would deadlock against a polling kernel.
int xpu_p2p_ar_warmup(void *state, void *queue, char *err, int errlen) {
  State *s = static_cast<State *>(state);
  try {
    sycl::queue &q = *static_cast<sycl::queue *>(queue);
    for (int k = 0; k < s->world; k++) {
      if (k == s->rank) continue;
      if (!s->mapped[k]) {
        set_err(err, errlen, "peer not mapped");
        return -1;
      }
      for (int par = 0; par < 2; par++) {
        uint64_t *dst = s->recv[k] + (size_t(par) * s->world + s->rank) * s->slot_words;
        size_t nw = s->slot_words;
        q.parallel_for(grid(nw), [=](sycl::nd_item<1> it) {
          for (size_t i = it.get_global_id(0); i < nw; i += it.get_global_range(0)) dst[i] = 0;
        });
      }
    }
    q.wait_and_throw();
    return 0;
  } catch (const std::exception &e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

// Enqueue one all-reduce on `queue` (asynchronous; works under graph capture).
// nbytes must be a multiple of 4 and at most max_bytes; in/out 4-byte aligned.
int xpu_p2p_ar_all_reduce(void *state, void *queue, const void *in, void *out, uint64_t nbytes, int dtype) {
  State *s = static_cast<State *>(state);
  size_t nw = size_t(nbytes / 4);
  if (nbytes % 4 || nw == 0 || nw > s->max_words) return -1;
  sycl::queue &q = *static_cast<sycl::queue *>(queue);
  const uint32_t *I = static_cast<const uint32_t *>(in);
  uint32_t *O = static_cast<uint32_t *>(out);
  if (dtype < kF16 || dtype > kF32) return -2;
  const int last = enqueue_pushes(q, *s, I, nw);
  switch (dtype) {
    case kF16: launch_reduce<kF16>(q, *s, I, O, nw, last); break;
    case kBF16: launch_reduce<kBF16>(q, *s, I, O, nw, last); break;
    default: launch_reduce<kF32>(q, *s, I, O, nw, last); break;
  }
  return 0;
}

// All-reduce of in[rows, hidden] fused with residual add + RMSNorm (see
// launch_fused_add_rms). out and res_out are new [rows, hidden] tensors of the
// same dtype; residual is not modified. weight has `hidden` elements of
// wdtype. Returns -1 for sizes this path can't take (caller falls back).
int xpu_p2p_ar_fused_add_rms_norm(void *state, void *queue, const void *in, const void *residual,
                                  const void *weight, void *out, void *res_out, uint64_t rows,
                                  uint64_t hidden, int dtype, int wdtype, float bias, float eps) {
  State *s = static_cast<State *>(state);
  if (dtype < kF16 || dtype > kF32 || wdtype < kF16 || wdtype > kF32) return -2;
  const size_t esz = dtype == kF32 ? 4 : 2;
  const size_t row_bytes = size_t(hidden) * esz;
  const size_t nw = size_t(rows) * row_bytes / 4;
  if (rows == 0 || hidden == 0 || row_bytes % 4 || nw > s->max_words) return -1;
  // LL words a row's work-items cover (rows don't start on LL word boundaries)
  const size_t ll_row = row_bytes / 4 / 3 + 2;
  const int gpr = int(std::min<size_t>(kFusedMaxGPR, (ll_row + kFusedWG - 1) / kFusedWG));
  // All groups spin on each other, so they must all be resident: keep it small.
  if (rows > size_t(kFusedMaxRows) || rows * gpr > 128 || ll_row > size_t(gpr) * kFusedWG * kFusedMaxLL)
    return -1;
  sycl::queue &q = *static_cast<sycl::queue *>(queue);
  const uint32_t *I = static_cast<const uint32_t *>(in);
  const int last = enqueue_pushes(q, *s, I, nw);
  switch (dtype) {
    case kF16: dispatch_fused<kF16>(wdtype, q, *s, I, residual, weight, out, res_out, rows, hidden, bias, eps, last, gpr); break;
    case kBF16: dispatch_fused<kBF16>(wdtype, q, *s, I, residual, weight, out, res_out, rows, hidden, bias, eps, last, gpr); break;
    default: dispatch_fused<kF32>(wdtype, q, *s, I, residual, weight, out, res_out, rows, hidden, bias, eps, last, gpr); break;
  }
  return 0;
}

// Number of polls that gave up (should stay 0). Synchronous; not for capture.
uint64_t xpu_p2p_ar_timeouts(void *state, void *queue) {
  State *s = static_cast<State *>(state);
  sycl::queue &q = *static_cast<sycl::queue *>(queue);
  uint64_t v = 0;
  q.memcpy(&v, s->ctl + 2, sizeof(v)).wait();
  return v;
}

void xpu_p2p_ar_destroy(void *state) {
  State *s = static_cast<State *>(state);
  if (!s) return;
  for (int k = 0; k < s->world; k++) {
    if (!s->mapped[k]) continue;
    if (k == s->rank) {
      zeMemFree(s->zctx, s->recv[k]);
    } else {
      zeMemCloseIpcHandle(s->zctx, s->recv[k]);
      if (s->peer_fd[k] >= 0) close(s->peer_fd[k]);
    }
  }
  if (s->ctl) sycl::free(s->ctl, *s->ctx);
  delete s->ctx;
  delete s;
}

}  // extern "C"
