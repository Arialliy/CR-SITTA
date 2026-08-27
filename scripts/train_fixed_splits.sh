#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "${script_dir}/.." && pwd -P)
python_bin="${project_root}/.conda/bin/python"
output_root="${project_root}/results/retraining_fixed_split"

mkdir -p "${output_root}/IRSTD-1K" "${output_root}/NUAA-SIRST" "${output_root}/NUDT-SIRST"

launch_training() {
  local gpu_index="$1"
  local dataset_name="$2"
  local pid_variable="$3"
  local dataset_output="${output_root}/${dataset_name}"
  local resume_args=()
  if [[ -f "${dataset_output}/last.pth.tar" ]]; then
    resume_args=(--resume "${dataset_output}/last.pth.tar")
  fi
  CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" "${project_root}/train_fixed_split.py" \
    --dataset "${dataset_name}" \
    --device cuda:0 \
    --num-workers 12 \
    --output-dir "${dataset_output}" \
    "${resume_args[@]}" \
    >> "${dataset_output}/console.log" 2>&1 &
  printf -v "${pid_variable}" '%s' "$!"
}

launch_training 0 IRSTD-1K irstd_pid
launch_training 1 NUAA-SIRST nuaa_pid
launch_training 2 NUDT-SIRST nudt_pid

status=0
wait "${irstd_pid}" || status=$?
wait "${nuaa_pid}" || status=$?
wait "${nudt_pid}" || status=$?
exit "${status}"
