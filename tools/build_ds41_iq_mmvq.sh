#!/bin/bash
# Build the SYCL library behind vllm/models/deepseek_v41/xpu/iq_mmvq.py.
# Needs the oneAPI DPC++ compiler (icpx), the same major version as the SYCL
# runtime vLLM ships with (libsycl.so.9: oneAPI 2026.1).
#   tools/build_ds41_iq_mmvq.sh [output.so]
# XPU_AOT_DEVICE: ocloc device (default bmg-g31 = Arc Pro B70); "none" = SPIR-V only.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="${DS41_IQ_SRC:-$here/../csrc/xpu/ds41_iq_mmvq.cpp}"
out="${1:-$here/../vllm/models/deepseek_v41/xpu/_ds41_iq_mmvq.so}"
if ! command -v icpx >/dev/null 2>&1; then
  set +u
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  set -u
fi
aot="${XPU_AOT_DEVICE:-bmg-g31}"
if [ "$aot" = "none" ]; then
  targets=(-fsycl-targets=spir64)
else
  targets=(-fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device $aot")
fi
icpx -fsycl "${targets[@]}" -O3 -std=c++17 -fPIC -shared -I"$(dirname "$src")" "$src" -o "$out"
echo "built $out (AOT device: $aot)"
