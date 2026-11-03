#!/bin/bash
# Build the SYCL library behind vllm/distributed/device_communicators/xpu_p2p_all_reduce.py.
# Needs the oneAPI DPC++ compiler (icpx) and the Level Zero headers/loader.
#
#   tools/build_xpu_p2p_all_reduce.sh [output.so]
#
# The default output sits next to the Python module, where it is looked for.
# VLLM_XPU_P2P_ALLREDUCE_LIB=/path/to/lib.so overrides the location at run time.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../csrc/xpu/p2p_all_reduce.cpp"
out="${1:-$here/../vllm/distributed/device_communicators/_xpu_p2p_all_reduce.so}"
if ! command -v icpx >/dev/null 2>&1; then
  set +u  # setvars.sh reads unset variables
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  set -u
fi
# Ahead-of-time compile for the target GPU so the driver doesn't JIT the kernels on the first
# call in every process. The plain SPIR-V image (spir64) is kept as a fallback for other GPUs.
#   XPU_P2P_AOT_DEVICE  ocloc device name (default bmg-g31 = Arc Pro B70); "none" = SPIR-V only
aot_device="${XPU_P2P_AOT_DEVICE:-bmg-g31}"
if [ "$aot_device" = "none" ]; then
  targets=(-fsycl-targets=spir64)
else
  targets=(-fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device $aot_device")
fi
icpx -fsycl "${targets[@]}" -O2 -std=c++17 -fPIC -shared "$src" -o "$out" -lze_loader
echo "built $out (AOT device: $aot_device)"
