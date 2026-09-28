#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/experiments/results}"
THREADS="${THREADS:-16}"
REPEATS="${REPEATS:-1}"
MEMORY_REPEATS="${MEMORY_REPEATS:-1}"
EPC_CONFIG_GB="${EPC_CONFIG_GB:-64}"
TPCH9_TAU="${TPCH9_TAU:-15014300}"
DO_EPSILON="${DO_EPSILON:-4}"
DO_DELTA="${DO_DELTA:-1e-8}"
SCALE="${SCALE:-1}"

JOB13D_DIR="${JOB13D_DIR:-${ROOT_DIR}/experiments/data/job13d}"
JOB13D_SCALE_DIR="${JOB13D_SCALE_DIR:-${ROOT_DIR}/experiments/data/job13d_scales}"
SNAP_DIR="${SNAP_DIR:-${ROOT_DIR}/experiments/data/snap_acyclic_v2}"
SQL18_DIR="${SQL18_DIR:-${ROOT_DIR}/tpcds/sql18_projected/sf_${SCALE}}"
SQL85_DIR="${SQL85_DIR:-${ROOT_DIR}/tpcds/sql85_projected/sf_${SCALE}}"
TPCH9_DIR="${TPCH9_DIR:-${ROOT_DIR}/tpch/sf_0p1}"

mkdir -p "${RESULTS_DIR}"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage: experiments/run_experiments.sh COMMAND

Commands:
  preflight       Verify EPC/SGX and run the built-in tiny JFYan input.
  prepare-job     Download and prepare real IMDb JOB Query 13d.
  prepare-job-scales  Prepare the 20%, 40%, 60%, 80%, and 100% JOB inputs.
  prepare-snap    Download and prepare the controlled SNAP cases.
  prepare-tpch    Generate and project TPC-H Query 9 at scale 0.1.
  job13d-smoke    Run JFYan once on JOB 13d before the long comparison.
  job13d          Run JFYan, ParYan, and ObliYan on JOB 13d.
  job13d-scale    Run one JOB size selected by JOB13D_PERCENT=20|40|60|80|100.
  snap            Run every case listed in the current SNAP suite.
  snap-case       Run one controlled SNAP case selected by SNAP_CASE.
  nonoblivious    Compare JFYan with the non-oblivious JFYan baseline.
  memory          Measure tracked peak memory on representative workloads.
  benchmark       Run TPC-DS Q18/Q85 and TPC-H Q9.
  q18-jfyan       Time JFYan on one Q18 scale (one enclave, REPEATS runs).
  q18-nonobli     Time NonObliJFYan on the same Q18 scale.
  q18-memory      Profile JFYan memory on the same Q18 scale once.
  q85-jfyan       Time JFYan on one Q85 scale (one enclave, REPEATS runs).
  q85-nonobli     Time NonObliJFYan on the same Q85 scale.
  q85-memory      Profile JFYan memory on the same Q85 scale once.
  tpch9-memory    Profile JFYan memory for one public target TPCH9_TAU.
  tpch9-target-simulation  Benchmark deterministic target sizing (NOT DP).

Environment variables:
  THREADS=16 REPEATS=1 MEMORY_REPEATS=1 SCALE=1 EPC_CONFIG_GB=64
  SNAP_CASE=depth_d2_m9_medium
  TPCH9_TAU=15014300 RESULTS_DIR=...
  DO_EPSILON=4 DO_DELTA=1e-8 (target simulation only; no DP guarantee)
  JOB13D_PERCENT=20 JOB13D_DIR=... JOB13D_SCALE_DIR=...
  SNAP_DIR=... SQL18_DIR=... SQL85_DIR=... TPCH9_DIR=...
EOF
}

thread_repeat_list() {
  local count="$1"
  local list=""
  local i
  for i in $(seq 1 "${count}"); do
    if [[ -n "${list}" ]]; then
      list+=","
    fi
    list+="${THREADS}"
  done
  printf '%s' "${list}"
}

run_timing_logged() {
  local label="$1"
  shift
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local log="${RESULTS_DIR}/${label}_t${THREADS}_${stamp}.log"
  local sweep
  sweep="$(thread_repeat_list "${REPEATS}")"
  echo "Writing ${log}"
  {
    system_record
    echo "=== ${label} repeats=${REPEATS} ==="
    "${ROOT_DIR}/exec.sh" "$@" --stage-profile \
      --thread-sweep "${sweep}" -m 1 --print-limit 0
  } 2>&1 | tee "${log}"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing dataset directory: $1" >&2
    exit 2
  fi
}

system_record() {
  local recorded_threads="${1:-${THREADS}}"
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -r)"
  echo "threads=${recorded_threads}"
  echo "epc_config=${EPC_CONFIG_GB} GB"
  echo "measurement_contract=measured-wall-clock"
  echo "source_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "source_worktree_changes=$(git status --porcelain 2>/dev/null | wc -l)"
  lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA)' || true
  dmesg | grep -iE 'sgx: EPC section|sgx.*EPC' | tail -n 5 || true
}

require_snap_case() {
  local case_name="$1"
  require_dir "${SNAP_DIR}"
  if [[ ! "${case_name}" =~ ^[A-Za-z0-9_-]+$ ]] || \
     ! grep -Fq "${case_name}," "${SNAP_DIR}/suite.csv"; then
    echo "Unknown SNAP case '${case_name}' in ${SNAP_DIR}/suite.csv" >&2
    exit 2
  fi
  require_dir "${SNAP_DIR}/${case_name}"
}

run_logged() {
  local label="$1"
  local count="$2"
  shift 2
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local log="${RESULTS_DIR}/${label}_t${THREADS}_${stamp}.log"
  echo "Writing ${log}"
  {
    system_record
    for repeat in $(seq 1 "${count}"); do
      echo "=== ${label} repeat=${repeat}/${count} ==="
      "${ROOT_DIR}/exec.sh" "$@" --profile \
        -t "${THREADS}" -m 1 --print-limit 0
    done
  } 2>&1 | tee "${log}"
}

run_job13d_scale_case() {
  local percent="$1"
  require_dir "${JOB13D_SCALE_DIR}"
  local relative_path
  relative_path="$(awk -F, -v p="${percent}" 'NR > 1 && $2 == p { print $3 }' \
    "${JOB13D_SCALE_DIR}/suite.csv")"
  if [[ -z "${relative_path}" ]] || [[ ! "${relative_path}" =~ ^[A-Za-z0-9_./-]+$ ]]; then
    echo "Missing or invalid JOB 13d ${percent}% path in suite.csv" >&2
    exit 2
  fi
  local case_dir="${JOB13D_SCALE_DIR}/${relative_path}"
  require_dir "${case_dir}"
  run_timing_logged "job13d_pct${percent}_all" \
    --tree-workload "${case_dir}" --all --warmup-jfyan \
    --no-materialize-padding

  local latest_log
  latest_log="$(ls -t "${RESULTS_DIR}/job13d_pct${percent}_all_t${THREADS}_"*.log | head -n 1)"
  local checks
  checks="$(grep -c ' check=OK$' "${latest_log}" || true)"
  if [[ "${checks}" -ne 3 ]]; then
    echo "JOB 13d ${percent}% has ${checks}/3 successful formal checks; stopping." >&2
    exit 3
  fi
  echo "JOB 13d ${percent}% complete: ${latest_log}"
}

case "${1:-}" in
  preflight)
    system_record
    cpuid -1 -l 0x7 | grep -i SGX || true
    cpuid -1 -l 0x12 -s 0x2 || true
    ls -l /dev/sgx_enclave /dev/sgx_provision
    "${ROOT_DIR}/exec.sh" -JFYan --profile --no-materialize-padding \
      -t 2 -m 1 --print-limit 0
    ;;

  prepare-job)
    python3 experiments/prepare_workloads.py job13d --download --force
    ;;

  prepare-job-scales)
    python3 experiments/prepare_workloads.py job13d-scales --download --force
    ;;

  prepare-snap)
    python3 experiments/prepare_workloads.py snap-acyclic \
      --download --out "${SNAP_DIR}" --force
    ;;

  prepare-tpch)
    python3 experiments/prepare_workloads.py tpch9 --force
    ;;

  job13d-smoke)
    require_dir "${JOB13D_DIR}"
    run_logged job13d_jfyan_smoke 1 \
      --tree-workload "${JOB13D_DIR}" -JFYan --no-materialize-padding
    ;;

  job13d)
    require_dir "${JOB13D_DIR}"
    run_logged job13d_all "${REPEATS}" \
      --tree-workload "${JOB13D_DIR}" --all --no-materialize-padding
    ;;

  job13d-scale)
    percent="${JOB13D_PERCENT:-}"
    if [[ ! "${percent}" =~ ^(20|40|60|80|100)$ ]]; then
      echo "Set JOB13D_PERCENT to 20, 40, 60, 80, or 100." >&2
      exit 2
    fi
    run_job13d_scale_case "${percent}"
    ;;

  snap)
    require_dir "${SNAP_DIR}"
    while IFS=, read -r case_name _rest; do
      [[ "${case_name}" == "case" || -z "${case_name}" ]] && continue
      require_dir "${SNAP_DIR}/${case_name}"
      run_timing_logged "snap_${case_name}" \
        --tree-workload "${SNAP_DIR}/${case_name}" --all --no-materialize-padding
    done < "${SNAP_DIR}/suite.csv"
    ;;

  snap-case)
    require_dir "${SNAP_DIR}"
    case_name="${SNAP_CASE:-}"
    if [[ ! "${case_name}" =~ ^[A-Za-z0-9_-]+$ ]] || \
       ! grep -Fq "${case_name}," "${SNAP_DIR}/suite.csv"; then
      echo "Set SNAP_CASE to one case listed in ${SNAP_DIR}/suite.csv" >&2
      exit 2
    fi
    require_dir "${SNAP_DIR}/${case_name}"
    run_timing_logged "snap_${case_name}" \
      --tree-workload "${SNAP_DIR}/${case_name}" --all --no-materialize-padding
    ;;

  nonoblivious)
    require_dir "${JOB13D_DIR}"
    run_logged job13d_jfyan "${REPEATS}" \
      --tree-workload "${JOB13D_DIR}" -JFYan --no-materialize-padding
    run_logged job13d_nonoblivious "${REPEATS}" \
      --tree-workload "${JOB13D_DIR}" -NonObliJFYan --no-materialize-padding
    if [[ -d "${SQL85_DIR}" ]]; then
      run_logged sql85_jfyan "${REPEATS}" \
        --sql85 "${SQL85_DIR}" -JFYan --no-materialize-padding
      run_logged sql85_nonoblivious "${REPEATS}" \
        --sql85 "${SQL85_DIR}" -NonObliJFYan --no-materialize-padding
    fi
    ;;

  memory)
    require_dir "${JOB13D_DIR}"
    require_dir "${SNAP_DIR}/depth_d2_m9_medium"
    require_dir "${TPCH9_DIR}"
    run_logged job13d_memory "${MEMORY_REPEATS}" \
      --tree-workload "${JOB13D_DIR}" --all --memory-profile \
      --no-materialize-padding
    run_logged snap_reference_memory "${MEMORY_REPEATS}" \
      --tree-workload "${SNAP_DIR}/depth_d2_m9_medium" \
      --all --memory-profile --no-materialize-padding
    if [[ -d "${SQL85_DIR}" ]]; then
      run_logged sql85_memory "${MEMORY_REPEATS}" \
        --sql85 "${SQL85_DIR}" --all --memory-profile \
        --no-materialize-padding
    fi
    run_logged tpch9_memory "${MEMORY_REPEATS}" \
      --tpch9 "${TPCH9_DIR}" --all --memory-profile \
      --no-materialize-padding -tau "${TPCH9_TAU}"
    ;;

  benchmark)
    require_dir "${SQL18_DIR}"
    require_dir "${SQL85_DIR}"
    require_dir "${TPCH9_DIR}"
    run_logged tpcds18_all "${REPEATS}" \
      --sql18 "${SQL18_DIR}" --all --no-materialize-padding
    run_logged tpcds85_all "${REPEATS}" \
      --sql85 "${SQL85_DIR}" --all --no-materialize-padding
    run_logged tpch9_all "${REPEATS}" \
      --tpch9 "${TPCH9_DIR}" --all --no-materialize-padding -tau "${TPCH9_TAU}"
    ;;

  q18-jfyan)
    require_dir "${SQL18_DIR}"
    run_timing_logged "tpcds_q18_sf${SCALE}_jfyan" \
      --sql18 "${SQL18_DIR}" -JFYan --no-materialize-padding
    ;;

  q18-nonobli)
    require_dir "${SQL18_DIR}"
    run_timing_logged "tpcds_q18_sf${SCALE}_nonobli" \
      --sql18 "${SQL18_DIR}" -NonObliJFYan --no-materialize-padding
    ;;

  q18-memory)
    require_dir "${SQL18_DIR}"
    run_logged "tpcds_q18_sf${SCALE}_memory" "${MEMORY_REPEATS}" \
      --sql18 "${SQL18_DIR}" -JFYan --memory-profile \
      --no-materialize-padding
    ;;

  q85-jfyan)
    require_dir "${SQL85_DIR}"
    run_timing_logged "tpcds_q85_sf${SCALE}_jfyan" \
      --sql85 "${SQL85_DIR}" -JFYan --no-materialize-padding
    ;;

  q85-nonobli)
    require_dir "${SQL85_DIR}"
    run_timing_logged "tpcds_q85_sf${SCALE}_nonobli" \
      --sql85 "${SQL85_DIR}" -NonObliJFYan --no-materialize-padding
    ;;

  q85-memory)
    require_dir "${SQL85_DIR}"
    run_logged "tpcds_q85_sf${SCALE}_memory" "${MEMORY_REPEATS}" \
      --sql85 "${SQL85_DIR}" -JFYan --memory-profile \
      --no-materialize-padding
    ;;

  tpch9-memory)
    require_dir "${TPCH9_DIR}"
    run_logged "tpch9_lambda${TPCH9_TAU}_memory" "${MEMORY_REPEATS}" \
      --tpch9 "${TPCH9_DIR}" -JFYan --memory-profile \
      --no-materialize-padding -tau "${TPCH9_TAU}"
    ;;

  tpch9-target-simulation)
    require_dir "${TPCH9_DIR}"
    run_logged tpch9_target_simulation_NOT_DP "${REPEATS}" \
      --tpch9 "${TPCH9_DIR}" --all --benchmark-target-simulation \
      --do-epsilon "${DO_EPSILON}" --do-delta "${DO_DELTA}"
    ;;

  -h|--help|help|"")
    usage
    ;;

  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
