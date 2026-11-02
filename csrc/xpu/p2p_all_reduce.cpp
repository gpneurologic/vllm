// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// One-shot peer-to-peer all-reduce for Intel GPUs that talk over plain PCIe
// (tested on 3x Arc Pro B70). Used by XpuCommunicator for small messages,
// where oneCCL's fixed cost (~21 us per call) dominates.
//
// Design, and why (measured on B70, see docs/design/xpu_p2p_all_reduce.md):
//  * No PCIe atomics between GPUs, so ranks can't sync with flags. Instead each
//    8-byte store carries 4 bytes of data plus a 32-bit call number ("LL"
//    protocol). A word is fresh when its call number matches; no ordering or
//    fences needed.
//  * Data is pushed into the receiver's memory and polled there with local
//    system-scope atomic loads. Polling a peer's memory over PCIe doesn't work.
//  * One kernel storing to two different peers is ~50x slower than storing to
//    one, so each peer gets its own push kernel. Then one kernel waits and sums.
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

using a64sys = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                sycl::access::address_space::global_space>;
using a64dev = sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                sycl::access::address_space::global_space>;

enum DType : int { kF16 = 0, kBF16 = 1, kF32 = 2 };

struct State {
  int rank = 0;
  int world = 0;
  size_t slot_words = 0;  // LL words per (parity, source) slot = max data words
  size_t buf_bytes = 0;
  uint64_t *recv[kMaxWorld] = {};  // [rank] = own buffer, others = IPC-mapped peers
  bool mapped[kMaxWorld] = {};
  int peer_fd[kMaxWorld];
  uint64_t *ctl = nullptr;  // own device memory: [0] calls done, [1] current call, [2] timeouts
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

sycl::nd_range<1> grid(size_t nw) {
  size_t groups = std::min(kMaxGroups, std::max<size_t>(1, (nw + kWG - 1) / kWG));
  return sycl::nd_range<1>(groups * kWG, kWG);
}

// Push this rank's words into `peer`'s receive buffer. The first push kernel of a
// call computes the call number from ctl[0] and publishes it in ctl[1]; later
// kernels of the same call read ctl[1]. No kernel reads a word it also writes.
void launch_push(sycl::queue &q, const State &s, int peer, bool first, const uint32_t *in, size_t nw) {
  uint64_t *peer_buf = s.recv[peer];
  uint64_t *ctl = s.ctl;
  const int rank = s.rank, world = s.world;
  const size_t slot = s.slot_words;
  q.parallel_for(grid(nw), [=](sycl::nd_item<1> it) {
    const size_t g = it.get_global_id(0), G = it.get_global_range(0);
    const uint64_t n = first ? ctl[0] + 1 : ctl[1];
    if (first && g == 0) ctl[1] = n;
    const uint64_t tag = uint64_t(call_tag(n)) << 32;
    uint64_t *dst = peer_buf + (size_t(n & 1) * world + rank) * slot;
    for (size_t i = g; i < nw; i += G) dst[i] = tag | in[i];
  });
}

// Wait for every peer's words of this call, sum in rank order, write out.
// `out` may alias `in`: each work-item reads in[i] before writing out[i].
template <int DT>
void launch_reduce(sycl::queue &q, const State &s, const uint32_t *in, uint32_t *out, size_t nw) {
  uint64_t *own = s.recv[s.rank];
  uint64_t *ctl = s.ctl;
  const int rank = s.rank, world = s.world;
  const size_t slot = s.slot_words;
  q.parallel_for(grid(nw), [=](sycl::nd_item<1> it) {
    const size_t g = it.get_global_id(0), G = it.get_global_range(0);
    const uint64_t n = ctl[1];
    const uint32_t tag = call_tag(n);
    const uint64_t *base = own + size_t(n & 1) * world * slot;
    for (size_t i = g; i < nw; i += G) {
      float s0 = 0.f, s1 = 0.f;
      for (int k = 0; k < world; k++) {
        uint32_t w;
        if (k == rank) {
          w = in[i];
        } else {
          a64sys a(*const_cast<uint64_t *>(base + size_t(k) * slot + i));
          uint64_t v;
          uint64_t spins = 0;
          while (uint32_t((v = a.load()) >> 32) != tag) {
            if (++spins > kSpinLimit) {
              a64dev(ctl[2]).fetch_add(1);
              break;
            }
          }
          w = uint32_t(v);
        }
        float a, b;
        unpack<DT>(w, a, b);
        s0 += a;
        s1 += b;
      }
      out[i] = pack<DT>(s0, s1);
    }
    if (g == 0) ctl[0] = n;  // nobody in this kernel reads ctl[0]
  });
}

}  // namespace

extern "C" {

// Allocate this rank's receive buffer (2 parities x world slots x max_bytes/4 LL
// words) and control words, zeroed. Returns a handle or nullptr (see err).
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
    s->slot_words = max_bytes / 4;
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
    s->ctl = sycl::malloc_device<uint64_t>(4, q);
    q.memset(s->recv[rank], 0, s->buf_bytes);
    q.memset(s->ctl, 0, 4 * sizeof(uint64_t));
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
  if (nbytes % 4 || nw == 0 || nw > s->slot_words) return -1;
  sycl::queue &q = *static_cast<sycl::queue *>(queue);
  const uint32_t *I = static_cast<const uint32_t *>(in);
  uint32_t *O = static_cast<uint32_t *>(out);
  bool first = true;
  for (int d = 1; d < s->world; d++) {  // peers in ring order from this rank
    launch_push(q, *s, (s->rank + d) % s->world, first, I, nw);
    first = false;
  }
  switch (dtype) {
    case kF16: launch_reduce<kF16>(q, *s, I, O, nw); break;
    case kBF16: launch_reduce<kBF16>(q, *s, I, O, nw); break;
    case kF32: launch_reduce<kF32>(q, *s, I, O, nw); break;
    default: return -2;
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
