#!/usr/bin/env bash
set -Eeuo pipefail

task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
task_project=$(cd -- "$task_script_dir/.." && pwd -P)
task_python="$task_project/.conda/bin/python"
task_runner="$task_project/run_adabn_corruption_benchmark.py"
task_protocol="$task_project/configs/adabn_batch_stats_fixed_splits_v1.yaml"
task_root="$task_project/results/adabn/adabn_batch_stats_v1"
task_logs="$task_project/results/adabn/logs/adabn_batch_stats_v1"
task_gpu_a=${ADABN_GPU_A:-${ADABN_GPU_A_UUID:-0}}
task_gpu_b=${ADABN_GPU_B:-${ADABN_GPU_B_UUID:-1}}

cd "$task_project"
mkdir -p "$task_logs"
exec 9>"$task_project/results/adabn/.adabn_batch_stats_v1.launch.lock"
flock -n 9 || {
    echo "Another AdaBN batch-stats-v1 launcher holds the lock" >&2
    exit 1
}
test ! -e "$task_root/COMPLETE.json"

contract_now() {
    CUDA_VISIBLE_DEVICES="" "$task_python" - <<'PY'
import json
import run_adabn_corruption_benchmark as b
path, _protocol = b.load_protocol()
print(json.dumps({
    "protocol_sha256": b.sha256_file(path),
    "repository_code_bundle_sha256": b._repository_contract()["code_bundle_sha256"],
}, sort_keys=True, separators=(",", ":")))
PY
}

task_contract_file="$task_logs/FROZEN_CONTRACT.json"
task_contract=$(contract_now)
if [[ -f "$task_contract_file" ]]; then
    [[ $(<"$task_contract_file") == "$task_contract" ]] || {
        echo "Frozen AdaBN contract differs from the current code/config" >&2
        exit 2
    }
else
    printf '%s\n' "$task_contract" >"$task_contract_file"
fi

verify_frozen_contract() {
    [[ $(contract_now) == "$task_contract" ]] || {
        echo "AdaBN code/config changed after the formal launch started" >&2
        return 2
    }
}

task_jobs_a=(
    "IRSTD-1K clean_S0"
    "IRSTD-1K gaussian_noise_S1"
    "IRSTD-1K gaussian_noise_S3"
    "IRSTD-1K gaussian_noise_S5"
    "IRSTD-1K gaussian_blur_S1"
    "IRSTD-1K gaussian_blur_S3"
    "IRSTD-1K gaussian_blur_S5"
    "IRSTD-1K low_contrast_S1"
    "IRSTD-1K low_contrast_S3"
    "IRSTD-1K low_contrast_S5"
    "IRSTD-1K stripe_noise_S1"
    "IRSTD-1K stripe_noise_S3"
    "NUDT-SIRST clean_S0"
    "NUDT-SIRST gaussian_noise_S1"
    "NUDT-SIRST gaussian_noise_S3"
    "NUDT-SIRST gaussian_noise_S5"
    "NUDT-SIRST gaussian_blur_S1"
    "NUDT-SIRST gaussian_blur_S3"
    "NUDT-SIRST gaussian_blur_S5"
)

task_jobs_b=(
    "IRSTD-1K stripe_noise_S5"
    "NUAA-SIRST clean_S0"
    "NUAA-SIRST gaussian_noise_S1"
    "NUAA-SIRST gaussian_noise_S3"
    "NUAA-SIRST gaussian_noise_S5"
    "NUAA-SIRST gaussian_blur_S1"
    "NUAA-SIRST gaussian_blur_S3"
    "NUAA-SIRST gaussian_blur_S5"
    "NUAA-SIRST low_contrast_S1"
    "NUAA-SIRST low_contrast_S3"
    "NUAA-SIRST low_contrast_S5"
    "NUAA-SIRST stripe_noise_S1"
    "NUAA-SIRST stripe_noise_S3"
    "NUAA-SIRST stripe_noise_S5"
    "NUDT-SIRST low_contrast_S1"
    "NUDT-SIRST low_contrast_S3"
    "NUDT-SIRST low_contrast_S5"
    "NUDT-SIRST stripe_noise_S1"
    "NUDT-SIRST stripe_noise_S3"
    "NUDT-SIRST stripe_noise_S5"
)

run_worker() {
    local task_gpu_uuid=$1
    shift
    (
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="$task_gpu_uuid"
        export PYTHONHASHSEED=42

        "$task_python" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print("visible_device:", torch.cuda.get_device_name(0), flush=True)
print("cuda_rng_devices:", len(torch.cuda.get_rng_state_all()), flush=True)
PY

        local task_job task_dataset task_condition task_destination task_log
        for task_job in "$@"; do
            verify_frozen_contract
            read -r task_dataset task_condition <<<"$task_job"
            task_destination="$task_root/$task_dataset/conditions/$task_condition"
            task_log="$task_logs/${task_dataset}__${task_condition}.log"
            if [[ -f "$task_destination/CONDITION_COMPLETE.json" ]]; then
                echo "SKIP completed: $task_dataset $task_condition"
                continue
            fi
            if [[ -e "$task_destination" ]]; then
                echo "Non-complete final condition directory exists: $task_destination" >&2
                exit 3
            fi
            {
                echo "START $(date -Is) GPU=$task_gpu_uuid DATASET=$task_dataset CONDITION=$task_condition"
                "$task_python" -u "$task_runner" \
                    --protocol "$task_protocol" \
                    --device cuda:0 \
                    --dataset "$task_dataset" \
                    --condition "$task_condition"
                echo "DONE $(date -Is) GPU=$task_gpu_uuid DATASET=$task_dataset CONDITION=$task_condition"
            } 2>&1 | tee -a "$task_log"
        done
    )
}

run_worker "$task_gpu_a" "${task_jobs_a[@]}" >"$task_logs/worker_A.log" 2>&1 &
task_pid_a=$!
run_worker "$task_gpu_b" "${task_jobs_b[@]}" >"$task_logs/worker_B.log" 2>&1 &
task_pid_b=$!

set +e
wait "$task_pid_a"
task_rc_a=$?
wait "$task_pid_b"
task_rc_b=$?
set -e
echo "worker_A=$task_rc_a worker_B=$task_rc_b"
(( task_rc_a == 0 && task_rc_b == 0 ))

verify_frozen_contract
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=42 \
    "$task_python" -u "$task_runner" \
    --protocol "$task_protocol" \
    --aggregate-only \
    2>&1 | tee -a "$task_logs/aggregate.log"

CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=42 "$task_python" - <<'PY'
import run_adabn_corruption_benchmark as b

protocol_path, protocol = b.load_protocol()
root = b._project_path(protocol["outputs"]["formal_root"])
protocol_sha = b.sha256_file(protocol_path)
bundle = b._repository_contract()["code_bundle_sha256"]
verified = {
    dataset: b.verify_dataset_artifact(
        protocol=protocol,
        protocol_sha256=protocol_sha,
        output_root=root,
        dataset_name=dataset,
        repository_code_bundle_sha256=bundle,
    )
    for dataset in b.DATASETS
}
complete = b._load_json(root / "COMPLETE.json")
expected = {
    "complete": True,
    "scope": "global",
    "formal_artifact": True,
    "dataset_count": 3,
    "condition_count_per_dataset": 13,
    "global_dataset_condition_count": 39,
    "protocol_sha256": protocol_sha,
    "repository_code_bundle_sha256": bundle,
    "all_required_gates_passed": True,
}
for key, value in expected.items():
    if complete.get(key) != value:
        raise RuntimeError(f"global COMPLETE mismatch: {key}")
index = root / "global_index"
manifest_path = index / "artifact_manifest.json"
manifest = b._load_json(manifest_path)
if b.sha256_file(manifest_path) != complete["artifact_manifest_sha256"]:
    raise RuntimeError("global manifest SHA256 mismatch")
b._verify_files_mapping(
    index, manifest["files"], allowed_unlisted=("artifact_manifest.json",)
)
for dataset, result in verified.items():
    link = manifest["datasets"][dataset]
    if link["artifact_manifest_sha256"] != result["manifest_sha256"]:
        raise RuntimeError(f"{dataset}: dataset manifest link mismatch")
    if link["completion_sha256"] != result["completion_sha256"]:
        raise RuntimeError(f"{dataset}: dataset completion link mismatch")
stragglers = [
    str(path)
    for path in root.rglob("*")
    if path.is_dir() and (".build-" in path.name or ".incomplete-" in path.name)
]
if stragglers:
    raise RuntimeError(f"stale staging directories remain: {stragglers}")
print("VERIFIED: 3 datasets, 39 formal AdaBN condition shards")
print(root)
PY
