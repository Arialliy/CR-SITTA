#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${task_root}/.conda/bin/python"
task_phase="${1:-all}"
task_config="${task_root}/configs/checkpoint_axis_best_pd_v1.yaml"
task_candidate_root="${task_root}/results/checkpoint_axis_v2_parity_candidates/best_miou"
task_gate_receipt="${task_root}/results/checkpoint_axis_v2_parity/best_miou/PARITY_RECEIPT.json"
task_log_root="${task_root}/results/checkpoint_axis_v2_execution_logs_v1"

if [[ ! -x "${task_python}" ]]; then
  echo "Missing project Python: ${task_python}" >&2
  exit 2
fi

case "${task_phase}" in
  all|parity|best_pd) ;;
  *)
    echo "Usage: $0 [all|parity|best_pd]" >&2
    exit 2
    ;;
esac

mkdir -p "${task_log_root}/parity" "${task_log_root}/best_pd"

run_logged() {
  local task_gpu="$1"
  local task_log="$2"
  shift 2
  if [[ -e "${task_log}" ]]; then
    echo "Refusing to overwrite execution log: ${task_log}" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="${task_gpu}" PYTHONHASHSEED=42 \
    "${task_python}" "$@" 2>&1 | tee "${task_log}"
}

verify_gate() {
  CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=42 "${task_python}" -c \
    'from benchmark.checkpoint_axis import load_axis_config, verify_parity_receipt; verify_parity_receipt(config=load_axis_config())'
}

run_clean_role() {
  local task_role="$1"
  local task_log_dir="${task_log_root}/${task_role/best_miou/parity}/clean"
  mkdir -p "${task_log_dir}"
  if [[ "${task_role}" == "best_miou" ]]; then
    (
      run_logged 0 "${task_log_dir}/IRSTD-1K.log" \
        "${task_root}/export_fixed_split_source_axis_v2.py" \
        --dataset IRSTD-1K --checkpoint-role best_miou \
        --axis-config "${task_config}" \
        --output-dir "${task_candidate_root}/clean/IRSTD-1K" --device cuda:0
      run_logged 0 "${task_log_dir}/NUAA-SIRST.log" \
        "${task_root}/export_fixed_split_source_axis_v2.py" \
        --dataset NUAA-SIRST --checkpoint-role best_miou \
        --axis-config "${task_config}" \
        --output-dir "${task_candidate_root}/clean/NUAA-SIRST" --device cuda:0
    ) &
    local task_pid0=$!
    (
      run_logged 1 "${task_log_dir}/NUDT-SIRST.log" \
        "${task_root}/export_fixed_split_source_axis_v2.py" \
        --dataset NUDT-SIRST --checkpoint-role best_miou \
        --axis-config "${task_config}" \
        --output-dir "${task_candidate_root}/clean/NUDT-SIRST" --device cuda:0
    ) &
    local task_pid1=$!
  else
    (
      run_logged 0 "${task_log_dir}/IRSTD-1K.log" \
        "${task_root}/export_fixed_split_source_axis_v2.py" \
        --dataset IRSTD-1K --checkpoint-role best_pd \
        --axis-config "${task_config}" --device cuda:0
      run_logged 0 "${task_log_dir}/NUAA-SIRST.log" \
        "${task_root}/export_fixed_split_source_axis_v2.py" \
        --dataset NUAA-SIRST --checkpoint-role best_pd \
        --axis-config "${task_config}" --device cuda:0
    ) &
    local task_pid0=$!
    (
      run_logged 1 "${task_log_dir}/NUDT-SIRST.log" \
        "${task_root}/export_fixed_split_source_axis_v2.py" \
        --dataset NUDT-SIRST --checkpoint-role best_pd \
        --axis-config "${task_config}" --device cuda:0
    ) &
    local task_pid1=$!
  fi
  local task_status=0
  wait "${task_pid0}" || task_status=$?
  wait "${task_pid1}" || task_status=$?
  return "${task_status}"
}

run_source_role() {
  local task_role="$1"
  local task_log_dir="${task_log_root}/${task_role/best_miou/parity}/source"
  mkdir -p "${task_log_dir}"
  if [[ "${task_role}" == "best_miou" ]]; then
    (
      for task_dataset in IRSTD-1K NUAA-SIRST; do
        run_logged 0 "${task_log_dir}/${task_dataset}.log" \
          "${task_root}/run_source_corruption_checkpoint_axis_v2.py" \
          --dataset "${task_dataset}" --checkpoint-role best_miou \
          --axis-config "${task_config}" \
          --clean-artifact "${task_candidate_root}/clean/${task_dataset}" \
          --output-dir "${task_candidate_root}/source/${task_dataset}" \
          --parity-reference "${task_root}/results/source_corruption_benchmark_fixed_split_v1/${task_dataset}/best_miou" \
          --device cuda:0
      done
    ) &
    local task_pid0=$!
    (
      run_logged 1 "${task_log_dir}/NUDT-SIRST.log" \
        "${task_root}/run_source_corruption_checkpoint_axis_v2.py" \
        --dataset NUDT-SIRST --checkpoint-role best_miou \
        --axis-config "${task_config}" \
        --clean-artifact "${task_candidate_root}/clean/NUDT-SIRST" \
        --output-dir "${task_candidate_root}/source/NUDT-SIRST" \
        --parity-reference "${task_root}/results/source_corruption_benchmark_fixed_split_v1/NUDT-SIRST/best_miou" \
        --device cuda:0
    ) &
    local task_pid1=$!
  else
    (
      for task_dataset in IRSTD-1K NUAA-SIRST; do
        run_logged 0 "${task_log_dir}/${task_dataset}.log" \
          "${task_root}/run_source_corruption_checkpoint_axis_v2.py" \
          --dataset "${task_dataset}" --checkpoint-role best_pd \
          --axis-config "${task_config}" --device cuda:0
      done
    ) &
    local task_pid0=$!
    (
      run_logged 1 "${task_log_dir}/NUDT-SIRST.log" \
        "${task_root}/run_source_corruption_checkpoint_axis_v2.py" \
        --dataset NUDT-SIRST --checkpoint-role best_pd \
        --axis-config "${task_config}" --device cuda:0
    ) &
    local task_pid1=$!
  fi
  local task_status=0
  wait "${task_pid0}" || task_status=$?
  wait "${task_pid1}" || task_status=$?
  return "${task_status}"
}

run_adabn_role() {
  local task_role="$1"
  local task_log_dir="${task_log_root}/${task_role/best_miou/parity}/adabn"
  mkdir -p "${task_log_dir}"
  if [[ "${task_role}" == "best_miou" ]]; then
    (
      for task_dataset in IRSTD-1K NUAA-SIRST; do
        run_logged 0 "${task_log_dir}/${task_dataset}.log" \
          "${task_root}/run_adabn_corruption_checkpoint_axis_v2.py" \
          --checkpoint-role best_miou --parity-only \
          --axis-config "${task_config}" \
          --output-dir "${task_candidate_root}/adabn" \
          --source-artifact-root "${task_candidate_root}/source" \
          --dataset "${task_dataset}" --device cuda:0
      done
    ) &
    local task_pid0=$!
    (
      run_logged 1 "${task_log_dir}/NUDT-SIRST.log" \
        "${task_root}/run_adabn_corruption_checkpoint_axis_v2.py" \
        --checkpoint-role best_miou --parity-only \
        --axis-config "${task_config}" \
        --output-dir "${task_candidate_root}/adabn" \
        --source-artifact-root "${task_candidate_root}/source" \
        --dataset NUDT-SIRST --device cuda:0
    ) &
    local task_pid1=$!
    local task_status=0
    wait "${task_pid0}" || task_status=$?
    wait "${task_pid1}" || task_status=$?
    if [[ "${task_status}" -ne 0 ]]; then
      return "${task_status}"
    fi
    run_logged "" "${task_log_dir}/aggregate.log" \
      "${task_root}/run_adabn_corruption_checkpoint_axis_v2.py" \
      --checkpoint-role best_miou --parity-only \
      --axis-config "${task_config}" \
      --output-dir "${task_candidate_root}/adabn" \
      --source-artifact-root "${task_candidate_root}/source" \
      --aggregate-only --device cpu
  else
    (
      for task_dataset in IRSTD-1K NUAA-SIRST; do
        run_logged 0 "${task_log_dir}/${task_dataset}.log" \
          "${task_root}/run_adabn_corruption_checkpoint_axis_v2.py" \
          --checkpoint-role best_pd --axis-config "${task_config}" \
          --dataset "${task_dataset}" --device cuda:0
      done
    ) &
    local task_pid0=$!
    (
      run_logged 1 "${task_log_dir}/NUDT-SIRST.log" \
        "${task_root}/run_adabn_corruption_checkpoint_axis_v2.py" \
        --checkpoint-role best_pd --axis-config "${task_config}" \
        --dataset NUDT-SIRST --device cuda:0
    ) &
    local task_pid1=$!
    local task_status=0
    wait "${task_pid0}" || task_status=$?
    wait "${task_pid1}" || task_status=$?
    if [[ "${task_status}" -ne 0 ]]; then
      return "${task_status}"
    fi
    run_logged "" "${task_log_dir}/aggregate.log" \
      "${task_root}/run_adabn_corruption_checkpoint_axis_v2.py" \
      --checkpoint-role best_pd --axis-config "${task_config}" \
      --aggregate-only --device cpu
  fi
}

run_parity() {
  if [[ -f "${task_gate_receipt}" ]]; then
    verify_gate
    echo "The frozen global parity gate already exists and is valid."
    return
  fi
  run_clean_role best_miou
  run_source_role best_miou
  run_adabn_role best_miou
  run_logged "" "${task_log_root}/parity/global_verifier.log" \
    "${task_root}/scripts/verify_checkpoint_axis_v2_parity.py" \
    --axis-config "${task_config}" --output "${task_gate_receipt}"
  verify_gate
}

run_best_pd() {
  verify_gate
  run_clean_role best_pd
  run_source_role best_pd
  run_adabn_role best_pd
}

cd "${task_root}"
if [[ "${task_phase}" == "all" || "${task_phase}" == "parity" ]]; then
  run_parity
fi
if [[ "${task_phase}" == "all" || "${task_phase}" == "best_pd" ]]; then
  run_best_pd
fi
