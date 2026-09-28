#!/usr/bin/env bash
set -euo pipefail

# Build settings can be overridden from the environment, for example:
# SGX_MODE=SIM BUILD_TYPE=Debug ./exec.sh -JFYan --profile
BUILD_DIR="${BUILD_DIR:-build}"
SGX_MODE="${SGX_MODE:-HW}"
SGX_DEBUG="${SGX_DEBUG:-1}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
SGX_SDK="${SGX_SDK:-/opt/intel/sgxsdk}"
JFYAN_MAX_MATERIALIZED_ROWS="${JFYAN_MAX_MATERIALIZED_ROWS:-200000000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ ! -x "${SGX_SDK}/bin/x64/sgx_edger8r" ] || [ ! -x "${SGX_SDK}/bin/x64/sgx_sign" ]; then
  echo "Intel SGX SDK tools were not found under ${SGX_SDK}." >&2
  echo "Run: source /opt/intel/sgxsdk/environment" >&2
  echo "or set SGX_SDK to the installed SDK directory." >&2
  exit 2
fi

# Enable allocation tracking only for memory-profile runs.
MEMORY_PROFILE=OFF
for arg in "$@"; do
  if [ "$arg" = "--memory-profile" ]; then
    MEMORY_PROFILE=ON
    break
  fi
done

cmake -S . -B "${BUILD_DIR}" \
  -DSGX_MODE="${SGX_MODE}" \
  -DSGX_DEBUG="${SGX_DEBUG}" \
  -DSGX_SDK="${SGX_SDK}" \
  -DENABLE_MEMORY_PROFILE="${MEMORY_PROFILE}" \
  -DJFYAN_MAX_MATERIALIZED_ROWS="${JFYAN_MAX_MATERIALIZED_ROWS}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# Dataset paths are relative to this directory.
"${BUILD_DIR}/App/app" "$@"
