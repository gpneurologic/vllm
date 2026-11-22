// SPDX-License-Identifier: Apache-2.0
// Decode-time MoE GEMV for llama.cpp IQ1_M / IQ2_XXS / IQ2_XS expert weights on
// Intel XPU, for the DeepSeek V4.1 GGUF path in vLLM.
//
// The per-block dot products follow llama.cpp's SYCL backend
// (ggml/src/ggml-sycl/vecdotq.hpp, MIT License, Copyright (c) 2023-2026 The ggml
// authors): activations are quantized to q8_1 (int8 + one scale per 32 values)
// and each 32-value sub-block of a weight row is a handful of dp4a products on
// packed bytes. One 16-lane sub-group computes one output row; lanes take one
// sub-block each and the sub-group sums at the end.
//
// The library is loaded with ctypes and enqueues on the caller's sycl::queue
// (torch.xpu.current_stream().sycl_queue), so it also works under graph capture.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>

#include "ds41_iq_tables.h"

namespace {

constexpr int QK_K = 256;
constexpr int QK8 = 32;
constexpr int SG = 16;      // sub-group size
constexpr int ROWS_PER_WG = 4;  // sub-groups (rows) per work-group
constexpr float IQ1M_DELTA = 0.125f;

struct block_q8_1 {
  sycl::half d;
  sycl::half s;  // d * sum(qs) (unused here)
  int8_t qs[QK8];
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 block");

struct block_iq2_xxs {
  sycl::half d;
  uint16_t qs[QK_K / 8];
};
static_assert(sizeof(block_iq2_xxs) == 66, "iq2_xxs block");

struct block_iq2_xs {
  sycl::half d;
  uint16_t qs[QK_K / 8];
  uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_xs) == 74, "iq2_xs block");

struct block_iq1_m {
  uint8_t qs[QK_K / 8];
  uint8_t qh[QK_K / 16];
  uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq1_m) == 56, "iq1_m block");

// -DDS41_NATIVE_DP4A=0 builds the portable byte-multiply version instead.
// -DDS41_RAW_TILE=0: decode each row right after loading it (the old order).
#ifndef DS41_RAW_TILE
#define DS41_RAW_TILE 1
#endif
#ifndef DS41_NATIVE_DP4A
#define DS41_NATIVE_DP4A 4
#endif
#if defined(__SYCL_DEVICE_ONLY__) && DS41_NATIVE_DP4A > 0
// Intel GPU compiler (IGC) builtin, resolved when the SPIR-V is compiled for
// the device: one hardware DP4A instead of four byte multiplies.
#if DS41_NATIVE_DP4A == 4
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat);
inline int dp4a(int a, int b, int c) { return __builtin_IB_dp4a_ss(c, a, b, false); }
#else
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b);
inline int dp4a(int a, int b, int c) { return __builtin_IB_dp4a_ss(c, a, b); }
#endif
#else
inline int dp4a(int a, int b, int c) {
  const int8_t *va = reinterpret_cast<const int8_t *>(&a);
  const int8_t *vb = reinterpret_cast<const int8_t *>(&b);
  return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}
#endif

inline uint32_t sign_mask4(uint32_t signs) {  // 4 sign bits -> 0xff per negative byte
  const uint32_t bits = (signs & 0x01) | ((signs & 0x02) << 7) | ((signs & 0x04) << 14) |
                        ((signs & 0x08) << 21);
  return bits * 0xff;
}

inline int apply_signs(uint32_t v, uint32_t m) {  // negate the masked bytes
  return static_cast<int>((v ^ m) + (m & 0x01010101));
}

inline int load_i32(const int8_t *p) {
  int v;
  std::memcpy(&v, p, 4);
  return v;
}

// --- one 32-value sub-block (ib32 in 0..7) of a superblock against q8_1 --------
inline float dot_iq2_xxs(const block_iq2_xxs *b, const block_q8_1 *q8b, int ib32,
                         const uint64_t *grid_t, const uint8_t *ksigns) {
  const uint16_t *q2 = b->qs + 4 * ib32;
  const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(q2);
  uint32_t aux32 = q2[2] | (static_cast<uint32_t>(q2[3]) << 16);
  const int8_t *q8 = q8b[ib32].qs;
  int sumi = 0;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint64_t g = grid_t[aux8[l]];
    const uint32_t signs = ksigns[aux32 & 127];
    sumi = dp4a(apply_signs(static_cast<uint32_t>(g), sign_mask4(signs)), load_i32(q8), sumi);
    sumi = dp4a(apply_signs(static_cast<uint32_t>(g >> 32), sign_mask4(signs >> 4)),
                load_i32(q8 + 4), sumi);
    q8 += 8;
    aux32 >>= 7;
  }
  const float d = static_cast<float>(b->d) * (0.5f + aux32) * static_cast<float>(q8b[ib32].d) *
                  0.25f;
  return d * sumi;
}

inline float dot_iq2_xs(const block_iq2_xs *b, const block_q8_1 *q8b, int ib32,
                        const uint64_t *grid_t, const uint8_t *ksigns) {
  const uint16_t *q2 = b->qs + 4 * ib32;
  const int8_t *q8 = q8b[ib32].qs;
  const uint8_t ls1 = b->scales[ib32] & 0xf;
  const uint8_t ls2 = b->scales[ib32] >> 4;
  int sumi[2] = {0, 0};
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint64_t g = grid_t[q2[l] & 511];
    const uint32_t signs = ksigns[q2[l] >> 9];
    int s = sumi[l / 2];
    s = dp4a(apply_signs(static_cast<uint32_t>(g), sign_mask4(signs)), load_i32(q8), s);
    s = dp4a(apply_signs(static_cast<uint32_t>(g >> 32), sign_mask4(signs >> 4)), load_i32(q8 + 4),
             s);
    sumi[l / 2] = s;
    q8 += 8;
  }
  const float d = static_cast<float>(b->d) * static_cast<float>(q8b[ib32].d) * 0.25f;
  return d * ((0.5f + ls1) * sumi[0] + (0.5f + ls2) * sumi[1]);
}

inline float dot_iq1_m(const block_iq1_m *b, const block_q8_1 *q8b, int ib32,
                       const uint32_t *grid_t) {
  int sumi[2] = {0, 0};
  float sumf[2] = {0.f, 0.f};
  const int8_t *q8 = q8b[ib32].qs;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint8_t qh = b->qh[2 * ib32 + l / 2] >> (4 * (l % 2));
    const uint32_t g = grid_t[b->qs[4 * ib32 + l] | ((qh & 7) << 8)];
    const int g0 = static_cast<int>(g & 0x0f0f0f0f);
    const int g1 = static_cast<int>((g >> 4) & 0x0f0f0f0f);
    const int a0 = load_i32(q8 + 8 * l), a1 = load_i32(q8 + 8 * l + 4);
    sumi[l / 2] = dp4a(a1, g1, dp4a(a0, g0, sumi[l / 2]));
    const float delta = (qh & 0x08) ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA;
    const int sumy = dp4a(a1, 0x01010101, dp4a(a0, 0x01010101, 0));
    sumf[l / 2] += delta * sumy;
  }
  uint16_t sc[4];
  std::memcpy(sc, b->scales, 8);
  const uint16_t u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
                     (sc[3] & 0xf000);
  sycl::half hs;
  std::memcpy(&hs, &u, 2);
  const float d = static_cast<float>(hs) * static_cast<float>(q8b[ib32].d);
  const int s1 = (sc[ib32 / 2] >> (6 * (ib32 % 2))) & 0x7;
  const int s2 = (sc[ib32 / 2] >> (6 * (ib32 % 2) + 3)) & 0x7;
  return d * ((sumi[0] + sumf[0]) * (2 * s1 + 1) + (sumi[1] + sumf[1]) * (2 * s2 + 1));
}

// --- kernels -----------------------------------------------------------------
template <typename T>
void quantize_q8_1(sycl::queue &q, const T *x, block_q8_1 *out, int rows, int K) {
  const int nb = K / QK8;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(rows) * nb), [=](sycl::id<1> id) {
    const int i = static_cast<int>(id[0]);
    const T *xr = x + static_cast<size_t>(i) * QK8;  // rows are contiguous, K % 32 == 0
    float v[QK8];
    float amax = 0.f, sum = 0.f;
#pragma unroll
    for (int j = 0; j < QK8; ++j) {
      v[j] = static_cast<float>(xr[j]);
      amax = sycl::fmax(amax, sycl::fabs(v[j]));
      sum += v[j];
    }
    const float d = amax / 127.f;
    const float id_ = d > 0.f ? 1.f / d : 0.f;
    block_q8_1 &o = out[i];
#pragma unroll
    for (int j = 0; j < QK8; ++j) o.qs[j] = static_cast<int8_t>(sycl::round(v[j] * id_));
    o.d = static_cast<sycl::half>(d);
    o.s = static_cast<sycl::half>(sum);
  });
}

// QT: 16 = IQ2_XXS, 17 = IQ2_XS, 29 = IQ1_M (ggml type ids)
template <int QT>
void moe_mmvq(sycl::queue &q, const uint8_t *w, int64_t expert_stride, int64_t row_bytes,
              const int32_t *idx, int e_start, int n_local, const block_q8_1 *xq, int xdiv,
              float *y, int P, int N, int K) {
  const int nsb = K / QK_K;  // superblocks per row
  const int nsub = nsb * 8;  // 32-value sub-blocks per row
  const size_t wg_rows = (N + ROWS_PER_WG - 1) / ROWS_PER_WG;
  const sycl::range<2> global(static_cast<size_t>(P) * wg_rows * ROWS_PER_WG, SG);
  const sycl::range<2> local(ROWS_PER_WG, SG);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<2>(global, local),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
      const int grow = static_cast<int>(it.get_global_id(0));
      const int p = grow / static_cast<int>(wg_rows * ROWS_PER_WG);
      const int row = grow % static_cast<int>(wg_rows * ROWS_PER_WG);
      const int lane = static_cast<int>(it.get_local_id(1));
      if (row >= N) return;
      const int e = idx[p] - e_start;
      float acc = 0.f;
      if (e >= 0 && e < n_local) {
        const uint8_t *wrow = w + static_cast<int64_t>(e) * expert_stride + row * row_bytes;
        const block_q8_1 *xb = xq + static_cast<size_t>(p / xdiv) * (K / QK8);
        for (int sub = lane; sub < nsub; sub += SG) {
          const int sbk = sub >> 3, ib32 = sub & 7;
          const block_q8_1 *q8b = xb + sbk * 8;
          if constexpr (QT == 16) {
            acc += dot_iq2_xxs(reinterpret_cast<const block_iq2_xxs *>(wrow) + sbk, q8b, ib32,
                               iq2xxs_grid, ksigns_iq2xs);
          } else if constexpr (QT == 17) {
            acc += dot_iq2_xs(reinterpret_cast<const block_iq2_xs *>(wrow) + sbk, q8b, ib32,
                              iq2xs_grid, ksigns_iq2xs);
          } else {
            acc += dot_iq1_m(reinterpret_cast<const block_iq1_m *>(wrow) + sbk, q8b, ib32,
                             iq1s_grid_gpu);
          }
        }
      }
      acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
      if (lane == 0) y[static_cast<size_t>(p) * N + row] = acc;
    });
  });
}

// --- grouped variant: one weight read for every pair routed to the same expert ----
// A 32-value weight sub-block unpacked to int8 values (8 x 4 packed) and the
// float factors of its two 16-value halves; IQ1_M also has a delta per 8 values
// (times the activation sum). dot(column) = q8.d * sum_h ws[h] * (sumi_h + sum_l
// delta_l * sumy_l), the same arithmetic as the dot_* functions above.
struct wsub {
  int v[8];
  float ws[2];
  float delta[4];
};

// ksigns_iq2xs[i]: 7 sign bits plus an 8th that makes the count even.
inline uint32_t ksign(uint32_t i) { return i | ((sycl::popcount(i) & 1u) << 7); }

// grid: the IQ2 grid table, from work-group local memory (random gathers from
// global memory made IQ2 rows ~4x slower than IQ1_M ones).
template <int QT>
inline void unpack_sub(const uint8_t *wrow, int sbk, int ib32, wsub &u, const uint64_t *grid) {
  if constexpr (QT == 16) {
    const block_iq2_xxs *b = reinterpret_cast<const block_iq2_xxs *>(wrow) + sbk;
    const uint16_t *q2 = b->qs + 4 * ib32;
    const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(q2);
    uint32_t aux32 = q2[2] | (static_cast<uint32_t>(q2[3]) << 16);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = grid[aux8[l]];
      const uint32_t signs = ksign(aux32 & 127);
      u.v[2 * l] = apply_signs(static_cast<uint32_t>(g), sign_mask4(signs));
      u.v[2 * l + 1] = apply_signs(static_cast<uint32_t>(g >> 32), sign_mask4(signs >> 4));
      aux32 >>= 7;
    }
    u.ws[0] = u.ws[1] = static_cast<float>(b->d) * (0.5f + aux32) * 0.25f;
  } else if constexpr (QT == 17) {
    const block_iq2_xs *b = reinterpret_cast<const block_iq2_xs *>(wrow) + sbk;
    const uint16_t *q2 = b->qs + 4 * ib32;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = grid[q2[l] & 511];
      const uint32_t signs = ksign(q2[l] >> 9);
      u.v[2 * l] = apply_signs(static_cast<uint32_t>(g), sign_mask4(signs));
      u.v[2 * l + 1] = apply_signs(static_cast<uint32_t>(g >> 32), sign_mask4(signs >> 4));
    }
    const float d = static_cast<float>(b->d) * 0.25f;
    u.ws[0] = d * (0.5f + (b->scales[ib32] & 0xf));
    u.ws[1] = d * (0.5f + (b->scales[ib32] >> 4));
  } else {
    const block_iq1_m *b = reinterpret_cast<const block_iq1_m *>(wrow) + sbk;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint8_t qh = b->qh[2 * ib32 + l / 2] >> (4 * (l % 2));
      const uint32_t g = iq1s_grid_gpu[b->qs[4 * ib32 + l] | ((qh & 7) << 8)];
      u.v[2 * l] = static_cast<int>(g & 0x0f0f0f0f);
      u.v[2 * l + 1] = static_cast<int>((g >> 4) & 0x0f0f0f0f);
      u.delta[l] = (qh & 0x08) ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA;
    }
    uint16_t sc[4];
    std::memcpy(sc, b->scales, 8);
    const uint16_t s16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
                         (sc[3] & 0xf000);
    sycl::half hs;
    std::memcpy(&hs, &s16, 2);
    const float d = static_cast<float>(hs);
    u.ws[0] = d * (2 * ((sc[ib32 / 2] >> (6 * (ib32 % 2))) & 0x7) + 1);
    u.ws[1] = d * (2 * ((sc[ib32 / 2] >> (6 * (ib32 % 2) + 3)) & 0x7) + 1);
  }
}

// Raw bytes of one 32-value sub-block, loaded for every row of a tile before
// any of them is decoded, so the loads of all rows are in flight together
// (decoding row by row made each row wait for its own load, then its grid
// lookups: ~87 GB/s for IQ2_XXS rows).
template <int QT> struct rawsub;
template <> struct rawsub<16> { uint16_t q[4]; sycl::half d; };
template <> struct rawsub<17> { uint16_t q[4]; sycl::half d; uint8_t sc; };
// IQ1_M as scalars (byte arrays here were kept in private memory).
template <> struct rawsub<29> { uint32_t qs; uint32_t qh; uint32_t sc_lo, sc_hi; };

template <int QT>
inline void load_raw(const uint8_t *wrow, int sbk, int ib32, rawsub<QT> &r) {
  if constexpr (QT == 16) {
    const block_iq2_xxs *b = reinterpret_cast<const block_iq2_xxs *>(wrow) + sbk;
#pragma unroll
    for (int l = 0; l < 4; ++l) r.q[l] = b->qs[4 * ib32 + l];
    r.d = b->d;
  } else if constexpr (QT == 17) {
    const block_iq2_xs *b = reinterpret_cast<const block_iq2_xs *>(wrow) + sbk;
#pragma unroll
    for (int l = 0; l < 4; ++l) r.q[l] = b->qs[4 * ib32 + l];
    r.d = b->d;
    r.sc = b->scales[ib32];
  } else {
    const block_iq1_m *b = reinterpret_cast<const block_iq1_m *>(wrow) + sbk;
    r.qs = *reinterpret_cast<const uint32_t *>(b->qs + 4 * ib32);  // 4-byte aligned
    r.qh = *reinterpret_cast<const uint16_t *>(b->qh + 2 * ib32);
    r.sc_lo = *reinterpret_cast<const uint32_t *>(b->scales);
    r.sc_hi = *reinterpret_cast<const uint32_t *>(b->scales + 4);
  }
}

// Same result as unpack_sub, from the raw bytes.
template <int QT>
inline void unpack_raw(const rawsub<QT> &r, int ib32, wsub &u, const uint64_t *grid) {
  if constexpr (QT == 16) {
    uint32_t aux32 = r.q[2] | (static_cast<uint32_t>(r.q[3]) << 16);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t a8 = (r.q[l / 2] >> (8 * (l % 2))) & 0xff;
      const uint64_t g = grid[a8];
      const uint32_t signs = ksign(aux32 & 127);
      u.v[2 * l] = apply_signs(static_cast<uint32_t>(g), sign_mask4(signs));
      u.v[2 * l + 1] = apply_signs(static_cast<uint32_t>(g >> 32), sign_mask4(signs >> 4));
      aux32 >>= 7;
    }
    u.ws[0] = u.ws[1] = static_cast<float>(r.d) * (0.5f + aux32) * 0.25f;
  } else if constexpr (QT == 17) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = grid[r.q[l] & 511];
      const uint32_t signs = ksign(r.q[l] >> 9);
      u.v[2 * l] = apply_signs(static_cast<uint32_t>(g), sign_mask4(signs));
      u.v[2 * l + 1] = apply_signs(static_cast<uint32_t>(g >> 32), sign_mask4(signs >> 4));
    }
    const float d = static_cast<float>(r.d) * 0.25f;
    u.ws[0] = d * (0.5f + (r.sc & 0xf));
    u.ws[1] = d * (0.5f + (r.sc >> 4));
  } else {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t qh = r.qh >> (4 * l);
      const uint32_t g = iq1s_grid_gpu[((r.qs >> (8 * l)) & 0xff) | ((qh & 7) << 8)];
      u.v[2 * l] = static_cast<int>(g & 0x0f0f0f0f);
      u.v[2 * l + 1] = static_cast<int>((g >> 4) & 0x0f0f0f0f);
      u.delta[l] = (qh & 0x08) ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA;
    }
    const uint32_t sc[4] = {r.sc_lo & 0xffff, r.sc_lo >> 16, r.sc_hi & 0xffff, r.sc_hi >> 16};
    const uint16_t s16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
                         (sc[3] & 0xf000);
    sycl::half hs;
    std::memcpy(&hs, &s16, 2);
    const float d = static_cast<float>(hs);
    const uint32_t sw = ((ib32 < 4 ? r.sc_lo : r.sc_hi) >> (16 * ((ib32 / 2) & 1))) & 0xffff;
    u.ws[0] = d * (2 * ((sw >> (6 * (ib32 % 2))) & 0x7) + 1);
    u.ws[1] = d * (2 * ((sw >> (6 * (ib32 % 2) + 3)) & 0x7) + 1);
  }
}

template <int QT>
inline float dot_sub(const wsub &u, const block_q8_1 &q8) {
  const int8_t *a = q8.qs;
  float h[2];
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    int si = 0;
    float sf = 0.f;
#pragma unroll
    for (int l = 2 * hh; l < 2 * hh + 2; ++l) {
      const int a0 = load_i32(a + 8 * l), a1 = load_i32(a + 8 * l + 4);
      si = dp4a(u.v[2 * l + 1], a1, dp4a(u.v[2 * l], a0, si));
      if constexpr (QT == 29) sf += u.delta[l] * dp4a(a1, 0x01010101, dp4a(a0, 0x01010101, 0));
    }
    h[hh] = u.ws[hh] * (static_cast<float>(si) + sf);
  }
  return static_cast<float>(q8.d) * (h[0] + h[1]);
}

constexpr int MAXC = 8;  // pairs served by one weight read

// Same result as moe_mmvq. The pair whose rank among equal expert ids is a
// multiple of MAXC leads its group: it collects the next pairs with that
// expert (up to MAXC), reads each weight sub-block once, and writes all their
// rows. The other pairs' work-groups exit. No sorting or host sync: graph-safe.
template <int QT>
void moe_mmvq_grouped(sycl::queue &q, const uint8_t *w, int64_t expert_stride,
                      int64_t row_bytes, const int32_t *idx, int e_start, int n_local,
                      const block_q8_1 *xq, int xdiv, float *y, int P, int N, int K) {
  const int nsb = K / QK_K;
  const int nsub = nsb * 8;
  const int nq8 = K / QK8;
  const size_t wg_rows = (N + ROWS_PER_WG - 1) / ROWS_PER_WG;
  const sycl::range<2> global(static_cast<size_t>(P) * wg_rows * ROWS_PER_WG, SG);
  const sycl::range<2> local(ROWS_PER_WG, SG);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<2>(global, local),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
      const int grow = static_cast<int>(it.get_global_id(0));
      const int p = grow / static_cast<int>(wg_rows * ROWS_PER_WG);
      const int row = grow % static_cast<int>(wg_rows * ROWS_PER_WG);
      const int lane = static_cast<int>(it.get_local_id(1));
      if (row >= N) return;
      const int ep = idx[p];
      const int e = ep - e_start;
      if (e < 0 || e >= n_local) {
        if (lane == 0) y[static_cast<size_t>(p) * N + row] = 0.f;
        return;
      }
      int rank = 0;
      for (int qq = 0; qq < p; ++qq) rank += idx[qq] == ep;
      if (rank % MAXC) return;
      int cols[MAXC];
      int cnt = 0;
      for (int qq = p; qq < P && cnt < MAXC; ++qq)
        if (idx[qq] == ep) cols[cnt++] = qq;
      float acc[MAXC];
#pragma unroll
      for (int c = 0; c < MAXC; ++c) acc[c] = 0.f;
      const uint8_t *wrow = w + static_cast<int64_t>(e) * expert_stride + row * row_bytes;
      for (int sub = lane; sub < nsub; sub += SG) {
        const int sbk = sub >> 3, ib32 = sub & 7;
        wsub u;
        unpack_sub<QT>(wrow, sbk, ib32, u,
                       QT == 16 ? iq2xxs_grid : (QT == 17 ? iq2xs_grid : nullptr));
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
          if (c < cnt)
            acc[c] += dot_sub<QT>(u, xq[static_cast<size_t>(cols[c] / xdiv) * nq8 + sub]);
      }
#pragma unroll
      for (int c = 0; c < MAXC; ++c) {
        if (c >= cnt) break;
        const float s = sycl::reduce_over_group(it.get_sub_group(), acc[c], sycl::plus<float>());
        if (lane == 0) y[static_cast<size_t>(cols[c]) * N + row] = s;
      }
    });
  });
}

// R rows per sub-group: each lane loads its activation sub-blocks once and
// reuses them for R weight rows (the activation reads, ~5.8 KB per row and
// pair, otherwise cost more than the weights). MC > 1: grouped as above.
template <int QT, int R, int MC>
void moe_mmvq_rows(sycl::queue &q, const uint8_t *w, int64_t expert_stride, int64_t row_bytes,
                   const int32_t *idx, int e_start, int n_local, const block_q8_1 *xq,
                   int xdiv, float *y, int P, int N, int K) {
  const int nsub = K / QK_K * 8;
  const int nq8 = K / QK8;
  const int n_rg = (N + R - 1) / R;
  const size_t wg = (n_rg + ROWS_PER_WG - 1) / ROWS_PER_WG;
  const sycl::range<2> global(static_cast<size_t>(P) * wg * ROWS_PER_WG, SG);
  const sycl::range<2> local(ROWS_PER_WG, SG);
  constexpr int NG = QT == 16 ? 256 : (QT == 17 ? 512 : 1);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint64_t, 1> lgrid(sycl::range<1>(NG), h);
    h.parallel_for(sycl::nd_range<2>(global, local),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
      const uint64_t *grid = nullptr;
      if constexpr (NG > 1) {
        const uint64_t *src = QT == 16 ? iq2xxs_grid : iq2xs_grid;
        for (int i = static_cast<int>(it.get_local_linear_id()); i < NG; i += ROWS_PER_WG * SG)
          lgrid[i] = src[i];
        sycl::group_barrier(it.get_group());
        grid = lgrid.get_multi_ptr<sycl::access::decorated::no>().get();
      }
      const int g = static_cast<int>(it.get_global_id(0));
      const int p = g / static_cast<int>(wg * ROWS_PER_WG);
      const int rg = g % static_cast<int>(wg * ROWS_PER_WG);
      const int lane = static_cast<int>(it.get_local_id(1));
      if (rg >= n_rg) return;
      const int row0 = rg * R;
      const int ep = idx[p];
      const int e = ep - e_start;
      if (e < 0 || e >= n_local) {
        for (int r = lane; r < R; r += SG)
          if (row0 + r < N) y[static_cast<size_t>(p) * N + row0 + r] = 0.f;
        return;
      }
      int cols[MC];
      int cnt = 1;
      cols[0] = p;
      if constexpr (MC > 1) {
        int rank = 0;
        for (int qq = 0; qq < p; ++qq) rank += idx[qq] == ep;
        if (rank % MC) return;
        cnt = 0;
        for (int qq = p; qq < P && cnt < MC; ++qq)
          if (idx[qq] == ep) cols[cnt++] = qq;
      }
      float acc[R][MC];
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < MC; ++c) acc[r][c] = 0.f;
      const uint8_t *wbase = w + static_cast<int64_t>(e) * expert_stride + row0 * row_bytes;
#if DS41_RAW_TILE
      if constexpr (MC == 1) {
        if (row0 + R <= N) {  // full tile: no per-row bound checks
          for (int sub = lane; sub < nsub; sub += SG) {
            const int sbk = sub >> 3, ib32 = sub & 7;
            const block_q8_1 qa = xq[static_cast<size_t>(p / xdiv) * nq8 + sub];
            rawsub<QT> rw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) load_raw<QT>(wbase + r * row_bytes, sbk, ib32, rw[r]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
              wsub u;
              unpack_raw<QT>(rw[r], ib32, u, grid);
              acc[r][0] += dot_sub<QT>(u, qa);
            }
          }
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const float s =
                sycl::reduce_over_group(it.get_sub_group(), acc[r][0], sycl::plus<float>());
            if (lane == 0) y[static_cast<size_t>(p) * N + row0 + r] = s;
          }
          return;
        }
      }
#endif
      for (int sub = lane; sub < nsub; sub += SG) {
        const int sbk = sub >> 3, ib32 = sub & 7;
        block_q8_1 qa[MC];
#pragma unroll
        for (int c = 0; c < MC; ++c)
          if (c < cnt) qa[c] = xq[static_cast<size_t>(cols[c] / xdiv) * nq8 + sub];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (row0 + r >= N) break;
          wsub u;
          unpack_sub<QT>(wbase + r * row_bytes, sbk, ib32, u, grid);
#pragma unroll
          for (int c = 0; c < MC; ++c)
            if (c < cnt) acc[r][c] += dot_sub<QT>(u, qa[c]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (row0 + r >= N) break;
#pragma unroll
        for (int c = 0; c < MC; ++c) {
          if (c >= cnt) break;
          const float s =
              sycl::reduce_over_group(it.get_sub_group(), acc[r][c], sycl::plus<float>());
          if (lane == 0) y[static_cast<size_t>(cols[c]) * N + row0 + r] = s;
        }
      }
    });
  });
}

template <int QT>
int mmvq_rows_dispatch(sycl::queue &q, int rows, int grouped, const uint8_t *w, int64_t es,
                       int64_t rb, const int32_t *idx, int e0, int nl, const block_q8_1 *xq,
                       int xdiv, float *y, int P, int N, int K) {
#define DS41_CASE(RR)                                                                        case RR:                                                                                     if (grouped)                                                                                 moe_mmvq_rows<QT, RR, MAXC>(q, w, es, rb, idx, e0, nl, xq, xdiv, y, P, N, K);            else                                                                                         moe_mmvq_rows<QT, RR, 1>(q, w, es, rb, idx, e0, nl, xq, xdiv, y, P, N, K);               return 0;
  switch (rows) {
    DS41_CASE(1)
    DS41_CASE(2)
    DS41_CASE(4)
    DS41_CASE(8)
    DS41_CASE(16)
    default: return -1;
  }
#undef DS41_CASE
}

void set_err(char *err, int len, const char *msg) {
  if (err && len > 0) {
    std::strncpy(err, msg, len - 1);
    err[len - 1] = 0;
  }
}

}  // namespace

extern "C" {

// x: [rows, K] (bf16 if x_bf16 else float32), contiguous; out: rows * K/32 q8_1 blocks.
int ds41_q8_1_quantize(void *queue, const void *x, int x_bf16, void *out, int rows, int K,
                       char *err, int errlen) {
  try {
    sycl::queue &q = *static_cast<sycl::queue *>(queue);
    if (K % QK8) return set_err(err, errlen, "K % 32 != 0"), -1;
    if (x_bf16)
      quantize_q8_1(q, static_cast<const sycl::ext::oneapi::bfloat16 *>(x),
                    static_cast<block_q8_1 *>(out), rows, K);
    else
      quantize_q8_1(q, static_cast<const float *>(x), static_cast<block_q8_1 *>(out), rows, K);
    return 0;
  } catch (const std::exception &e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

// y[p, n] = sum_k W[idx[p] - e_start][n, k] * x[p / xdiv, k] for local experts, else 0.
int ds41_moe_mmvq(void *queue, int qtype, const void *w, int64_t expert_stride,
                  int64_t row_bytes, const int32_t *idx, int e_start, int n_local,
                  const void *xq, int xdiv, float *y, int P, int N, int K, char *err,
                  int errlen) {
  try {
    sycl::queue &q = *static_cast<sycl::queue *>(queue);
    if (K % QK_K) return set_err(err, errlen, "K % 256 != 0"), -1;
    const auto *wb = static_cast<const uint8_t *>(w);
    const auto *xb = static_cast<const block_q8_1 *>(xq);
    switch (qtype) {
      case 16: moe_mmvq<16>(q, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      case 17: moe_mmvq<17>(q, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      case 29: moe_mmvq<29>(q, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      default: return set_err(err, errlen, "unsupported quant type"), -1;
    }
    return 0;
  } catch (const std::exception &e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

// Same contract as ds41_moe_mmvq; pairs that share an expert share its weight reads.
int ds41_moe_mmvq_grouped(void *queue, int qtype, const void *w, int64_t expert_stride,
                          int64_t row_bytes, const int32_t *idx, int e_start, int n_local,
                          const void *xq, int xdiv, float *y, int P, int N, int K, char *err,
                          int errlen) {
  try {
    sycl::queue &q = *static_cast<sycl::queue *>(queue);
    if (K % QK_K) return set_err(err, errlen, "K % 256 != 0"), -1;
    const auto *wb = static_cast<const uint8_t *>(w);
    const auto *xb = static_cast<const block_q8_1 *>(xq);
    switch (qtype) {
      case 16: moe_mmvq_grouped<16>(q, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      case 17: moe_mmvq_grouped<17>(q, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      case 29: moe_mmvq_grouped<29>(q, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      default: return set_err(err, errlen, "unsupported quant type"), -1;
    }
    return 0;
  } catch (const std::exception &e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

// Same contract as ds41_moe_mmvq; rows: output rows per sub-group (1, 2, 4, 8, 16);
// grouped: pairs that share an expert share its weight reads.
int ds41_moe_mmvq2(void *queue, int qtype, const void *w, int64_t expert_stride,
                   int64_t row_bytes, const int32_t *idx, int e_start, int n_local,
                   const void *xq, int xdiv, float *y, int P, int N, int K, int rows,
                   int grouped, char *err, int errlen) {
  try {
    sycl::queue &q = *static_cast<sycl::queue *>(queue);
    if (K % QK_K) return set_err(err, errlen, "K % 256 != 0"), -1;
    const auto *wb = static_cast<const uint8_t *>(w);
    const auto *xb = static_cast<const block_q8_1 *>(xq);
    int rc;
    switch (qtype) {
      case 16: rc = mmvq_rows_dispatch<16>(q, rows, grouped, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      case 17: rc = mmvq_rows_dispatch<17>(q, rows, grouped, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      case 29: rc = mmvq_rows_dispatch<29>(q, rows, grouped, wb, expert_stride, row_bytes, idx, e_start, n_local, xb, xdiv, y, P, N, K); break;
      default: return set_err(err, errlen, "unsupported quant type"), -1;
    }
    if (rc) return set_err(err, errlen, "rows must be 1, 2, 4, 8 or 16"), -1;
    return 0;
  } catch (const std::exception &e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

}  // extern "C"
