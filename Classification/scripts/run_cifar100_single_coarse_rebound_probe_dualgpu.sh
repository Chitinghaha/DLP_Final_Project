#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_NAMESPACE="${BASE_NAMESPACE:-sequential_single_coarse_rebound_cifar100}"
QUEUE_RUN_ID="${QUEUE_RUN_ID:-dual_gpu_queue_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-results/logs/${BASE_NAMESPACE}/${QUEUE_RUN_ID}}"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"
DRY_RUN="${DRY_RUN:-0}"

SEED="${SEED:-1}"
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar100}"
MAX_K="${MAX_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar}"

mkdir -p "${QUEUE_LOG_DIR}"

log_queue() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

validate_order() {
  local run_id="$1"
  local order="$2"
  python - "$run_id" "$order" "$MAX_K" <<'PY'
import sys
run_id, order_text, max_k = sys.argv[1], sys.argv[2], int(sys.argv[3])
items = [int(item) for item in order_text.split(",") if item.strip()]
if len(items) != max_k:
    raise SystemExit(f"{run_id}: expected {max_k} classes, got {len(items)}")
if len(set(items)) != len(items):
    raise SystemExit(f"{run_id}: duplicate class ids in {items}")
bad = [item for item in items if item < 0 or item > 99]
if bad:
    raise SystemExit(f"{run_id}: class ids out of range 0..99: {bad}")
PY
}

check_no_existing_jobs() {
  local existing
  existing="$(
    pgrep -af 'python .*main_random.py|python .*generate_mask.py|python .*evaluate_cumulative_forgetting.py|bash (./)?scripts/run_incremental_ordered_logged.sh' \
      | grep -v 'pgrep -af' \
      || true
  )"
  if [[ -n "${existing}" ]]; then
    log_queue "ERROR existing CIFAR/torch job detected; refusing to start"
    log_queue "${existing}"
    exit 1
  fi
}

run_one() {
  local gpu="$1"
  local run_id="$2"
  local order="$3"
  local group="$4"
  local order_type="$5"
  local namespace="${BASE_NAMESPACE}/${run_id}"
  local log_dir="results/logs/${BASE_NAMESPACE}/${run_id}"

  validate_order "${run_id}" "${order}"
  log_queue "START gpu=${gpu} run_id=${run_id} group=${group} order_type=${order_type} order=${order}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log_queue "DRY_RUN skip run_id=${run_id}"
    return 0
  fi

  if env \
    RESULT_NAMESPACE="${namespace}" \
    RUN_ID="${run_id}" \
    LOG_DIR="${log_dir}" \
    SEED="${SEED}" \
    GPU="${gpu}" \
    ARCH="${ARCH}" \
    DATASET="${DATASET}" \
    MAX_K="${MAX_K}" \
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS}" \
    MASK_EPOCHS="${MASK_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
    FORGET_ORDER="${order}" \
    ORIGINAL_MODEL="${ORIGINAL_MODEL}" \
    bash ./scripts/run_incremental_ordered_logged.sh; then
    log_queue "DONE gpu=${gpu} run_id=${run_id} status=success"
    return 0
  fi

  log_queue "DONE gpu=${gpu} run_id=${run_id} status=failed"
  return 1
}

worker_gpu0() {
  run_one 0 "large_omni_official_seed1_k5" "15,19,21,31,38" "large_omnivores_and_herbivores" "official"
  run_one 0 "large_omni_random_seed1_k5" "21,15,38,19,31" "large_omnivores_and_herbivores" "random"
  run_one 0 "large_omni_hardfirst_seed1_k5" "19,38,31,15,21" "large_omnivores_and_herbivores" "hardfirst"
  run_one 0 "vehicles1_official_seed1_k5" "8,13,48,58,90" "vehicles_1" "official"
  run_one 0 "vehicles1_random_seed1_k5" "48,8,90,13,58" "vehicles_1" "random"
}

worker_gpu1() {
  run_one 1 "vehicles1_hardfirst_seed1_k5" "13,8,58,90,48" "vehicles_1" "hardfirst"
  run_one 1 "flowers_official_seed1_k5" "54,62,70,82,92" "flowers" "official"
  run_one 1 "flowers_random_seed1_k5" "70,54,92,62,82" "flowers" "random"
  run_one 1 "flowers_hardfirst_seed1_k5" "92,70,62,54,82" "flowers" "hardfirst"
}

log_queue "Queue log directory: ${QUEUE_LOG_DIR}"
log_queue "Settings: DATASET=${DATASET} SEED=${SEED} ARCH=${ARCH} MAX_K=${MAX_K} BATCH_SIZE=${BATCH_SIZE} BASE_NAMESPACE=${BASE_NAMESPACE} DRY_RUN=${DRY_RUN}"
log_queue "Original model: ${ORIGINAL_MODEL}"

if [[ ! -f "${ORIGINAL_MODEL}" ]]; then
  log_queue "ERROR original model not found: ${ORIGINAL_MODEL}"
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  check_no_existing_jobs
fi

worker_gpu0 &
PID0=$!
worker_gpu1 &
PID1=$!

status0=0
status1=0
wait "${PID0}" || status0=$?
wait "${PID1}" || status1=$?

log_queue "Worker gpu0 exit_status=${status0}"
log_queue "Worker gpu1 exit_status=${status1}"

if (( status0 != 0 || status1 != 0 )); then
  log_queue "DONE dual gpu queue status=failed"
  exit 1
fi

log_queue "DONE dual gpu queue status=success"
