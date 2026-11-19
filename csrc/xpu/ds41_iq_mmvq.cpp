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

inline int dp4a(int a, int b, int c) {
  const int8_t *va = reinterpret_cast<const int8_t *>(&a);
  const int8_t *vb = reinterpret_cast<const int8_t *>(&b);
  return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

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

}  // extern "C"
