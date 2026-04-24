#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-182}"
TRAIN_LR="${TRAIN_LR:-0.1}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-sequential_incremental_ordered}"
QUEUE_RUN_ID="${QUEUE_RUN_ID:-overnight_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-results/logs/${RESULT_NAMESPACE}/${QUEUE_RUN_ID}}"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"

mkdir -p "${QUEUE_LOG_DIR}" "results/logs/original"

log_queue() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

ensure_original_model() {
  local seed="$1"
  local train_gpu="$2"
  local model_path="results/original/cifar10_resnet18_seed${seed}/0model_SA_best.pth.tar"
  local train_log="results/logs/original/train_seed${seed}_${TRAIN_EPOCHS}.log"
  if [[ -f "${model_path}" ]]; then
    log_queue "FOUND original_model seed=${seed} path=${model_path}"
    return
  fi
  log_queue "START train_original seed=${seed} gpu=${train_gpu}"
  python main_train.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --lr "${TRAIN_LR}" \
    --epochs "${TRAIN_EPOCHS}" \
    --save_dir "results/original/cifar10_resnet18_seed${seed}" \
    --gpu "${train_gpu}" \
    --seed "${seed}" \
    --train_seed "${seed}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee -a "${train_log}"
  log_queue "DONE train_original seed=${seed} gpu=${train_gpu}"
}

launch_job() {
  local stage="$1"
  local run_id="$2"
  local gpu="$3"
  local max_k="$4"
  local unlearn_epochs="$5"
  local mask_epochs="$6"
  local order="$7"

  log_queue "START stage=${stage} run_id=${run_id} gpu=${gpu} order=${order}"
  env \
    RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
    RUN_ID="${run_id}" \
    SEED=1 \
    GPU="${gpu}" \
    ARCH="${ARCH}" \
    DATASET="${DATASET}" \
    MAX_K="${max_k}" \
    UNLEARN_EPOCHS="${unlearn_epochs}" \
    MASK_EPOCHS="${mask_epochs}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    FORGET_ORDER="${order}" \
    ORIGINAL_MODEL="results/original/cifar10_resnet18_seed1/0model_SA_best.pth.tar" \
    bash ./scripts/run_incremental_ordered_logged.sh &
  LAST_LAUNCHED_PID=$!
}

wait_for_job() {
  local pid="$1"
  local stage="$2"
  local run_id="$3"
  local gpu="$4"
  local order="$5"
  if wait "${pid}"; then
    log_queue "DONE stage=${stage} run_id=${run_id} gpu=${gpu} order=${order} status=success"
    return 0
  fi
  log_queue "DONE stage=${stage} run_id=${run_id} gpu=${gpu} order=${order} status=failed"
  return 1
}

log_queue "Queue log directory: ${QUEUE_LOG_DIR}"
log_queue "Settings: GPU0=${GPU0} GPU1=${GPU1} ARCH=${ARCH} DATASET=${DATASET} BATCH_SIZE=${BATCH_SIZE} RESULT_NAMESPACE=${RESULT_NAMESPACE}"
log_queue "Baseline comparison run: incremental_newclass_seed1_k9 (existing only, not rerun)"

ensure_original_model 1 "${GPU0}"

SMOKE_INTERLEAVED_ORDER="0,6,1,4,2,7,3,8,5"
SMOKE_CLUSTERED_ORDER="2,3,4,5,6,7,0,1,8"

launch_job "smoke" "smoke_incremental_interleaved_seed1_k2" "${GPU0}" 2 1 1 "${SMOKE_INTERLEAVED_ORDER}"
smoke_pid_0="${LAST_LAUNCHED_PID}"
launch_job "smoke" "smoke_incremental_clustered_animals_seed1_k2" "${GPU1}" 2 1 1 "${SMOKE_CLUSTERED_ORDER}"
smoke_pid_1="${LAST_LAUNCHED_PID}"

smoke_ok=0
wait_for_job "${smoke_pid_0}" "smoke" "smoke_incremental_interleaved_seed1_k2" "${GPU0}" "${SMOKE_INTERLEAVED_ORDER}" || smoke_ok=1
wait_for_job "${smoke_pid_1}" "smoke" "smoke_incremental_clustered_animals_seed1_k2" "${GPU1}" "${SMOKE_CLUSTERED_ORDER}" || smoke_ok=1

if [[ "${smoke_ok}" -ne 0 ]]; then
  log_queue "ABORT full_batch reason=smoke_failed"
  exit 1
fi

launch_job "full" "incremental_interleaved_seed1_k9" "${GPU0}" 9 "${UNLEARN_EPOCHS}" "${MASK_EPOCHS}" "${SMOKE_INTERLEAVED_ORDER}"
full_pid_0="${LAST_LAUNCHED_PID}"
launch_job "full" "incremental_clustered_animals_seed1_k9" "${GPU1}" 9 "${UNLEARN_EPOCHS}" "${MASK_EPOCHS}" "${SMOKE_CLUSTERED_ORDER}"
full_pid_1="${LAST_LAUNCHED_PID}"

full_ok=0
wait_for_job "${full_pid_0}" "full" "incremental_interleaved_seed1_k9" "${GPU0}" "${SMOKE_INTERLEAVED_ORDER}" || full_ok=1
wait_for_job "${full_pid_1}" "full" "incremental_clustered_animals_seed1_k9" "${GPU1}" "${SMOKE_CLUSTERED_ORDER}" || full_ok=1

if [[ "${full_ok}" -ne 0 ]]; then
  log_queue "DONE incremental ordered queue status=failed"
  exit 1
fi

log_queue "DONE incremental ordered queue status=success"
