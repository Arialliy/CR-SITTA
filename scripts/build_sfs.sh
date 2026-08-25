#!/usr/bin/env bash
set -euo pipefail

CRS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRS_REPOSITORY="$(cd "${CRS_SCRIPT_DIR}/.." && pwd)"
CRS_PYTHON="${1:-${CRS_REPOSITORY}/.conda/bin/python}"
CRS_PYTHON_DIR="$(cd "$(dirname "${CRS_PYTHON}")" && pwd)"

if [[ ! -x "${CRS_PYTHON}" ]]; then
  echo "Python executable not found: ${CRS_PYTHON}" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CRS_PYTHON_DIR}:${CUDA_HOME}/bin:${PATH}"
export CC="${CC:-/usr/bin/gcc-12}"
export CXX="${CXX:-/usr/bin/g++-12}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"

CRS_BUILD_COPY="$(mktemp -d)"
trap 'rm -rf -- "${CRS_BUILD_COPY}"' EXIT
cp -a \
  "${CRS_REPOSITORY}/SFS_MSDeformAttn/ops/README.md" \
  "${CRS_REPOSITORY}/SFS_MSDeformAttn/ops/setup.py" \
  "${CRS_REPOSITORY}/SFS_MSDeformAttn/ops/functions" \
  "${CRS_REPOSITORY}/SFS_MSDeformAttn/ops/modules" \
  "${CRS_REPOSITORY}/SFS_MSDeformAttn/ops/src" \
  "${CRS_BUILD_COPY}/"

cd "${CRS_BUILD_COPY}"
"${CRS_PYTHON}" setup.py build install
cd "${CRS_REPOSITORY}"
"${CRS_PYTHON}" -c \
  'import torch; import MultiScaleDeformableAttention as extension; print(extension.__file__)'
