#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

SEED="${SEED:-1}"
GPU="${GPU:-0}"
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar100}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MAX_K="${MAX_K:-20}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-sequential_incremental_ordered_cifar100_pilot}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-results/original/cifar100_resnet18_seed${SEED}/0model_SA_best.pth.tar}"

CLUSTERED_ORDER="${CLUSTERED_ORDER:-3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,36,50,65,74,80}"
INTERLEAVED_ORDER="${INTERLEAVED_ORDER:-3,15,34,36,42,19,63,50,43,21,64,65,88,31,66,74,97,38,75,80}"
CLUSTERED_RUN_ID="${CLUSTERED_RUN_ID:-incremental_cifar100_clustered_animals_seed1_k20}"
INTERLEAVED_RUN_ID="${INTERLEAVED_RUN_ID:-incremental_cifar100_interleaved_animals_seed1_k20}"
CLUSTERED_START_STEP="${CLUSTERED_START_STEP:-12}"
CLUSTERED_START_STAGE="${CLUSTERED_START_STAGE:-unlearn}"
CLUSTERED_START_MODEL="${CLUSTERED_START_MODEL:-results/unlearn/${RESULT_NAMESPACE}/seed${SEED}/step11_forgot_3_42_43_88_97_15_19_21_31_38_34/RLcheckpoint.pth.tar}"

QUEUE_RUN_ID="${QUEUE_RUN_ID:-ordered_queue_cifar100_resume_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-results/logs/${RESULT_NAMESPACE}/${QUEUE_RUN_ID}}"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"
mkdir -p "${QUEUE_LOG_DIR}" "results/logs/${RESULT_NAMESPACE}"

log_queue() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log_queue "ERROR missing required file: ${path}"
    exit 1
  fi
}

ensure_no_existing_cifar_job() {
  local matches
  matches="$(pgrep -af 'main_random|generate_mask|evaluate_cumulative|torchrun|python.*cifar100' || true)"
  if [[ -n "${matches}" ]]; then
    log_queue "ERROR existing CIFAR/torch job detected; refusing to start"
    log_queue "${matches}"
    exit 1
  fi
}

run_clustered_resume() {
  log_queue "START stage=full_resume order=clustered run_id=${CLUSTERED_RUN_ID} start_step=${CLUSTERED_START_STEP} start_stage=${CLUSTERED_START_STAGE} gpu=${GPU}"
  if env \
    RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
    RUN_ID="${CLUSTERED_RUN_ID}" \
    SEED="${SEED}" \
    GPU="${GPU}" \
    ARCH="${ARCH}" \
    DATASET="${DATASET}" \
    MAX_K="${MAX_K}" \
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS}" \
    MASK_EPOCHS="${MASK_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
    FORGET_ORDER="${CLUSTERED_ORDER}" \
    ORIGINAL_MODEL="${ORIGINAL_MODEL}" \
    RESUME=1 \
    START_STEP="${CLUSTERED_START_STEP}" \
    START_STAGE="${CLUSTERED_START_STAGE}" \
    START_MODEL="${CLUSTERED_START_MODEL}" \
    bash ./scripts/run_incremental_ordered_logged.sh; then
    log_queue "DONE stage=full_resume order=clustered run_id=${CLUSTERED_RUN_ID} status=success"
    return 0
  fi
  log_queue "DONE stage=full_resume order=clustered run_id=${CLUSTERED_RUN_ID} status=failed"
  return 1
}

run_interleaved_full() {
  log_queue "START stage=full order=interleaved run_id=${INTERLEAVED_RUN_ID} max_k=${MAX_K} gpu=${GPU}"
  if env \
    RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
    RUN_ID="${INTERLEAVED_RUN_ID}" \
    SEED="${SEED}" \
    GPU="${GPU}" \
    ARCH="${ARCH}" \
    DATASET="${DATASET}" \
    MAX_K="${MAX_K}" \
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS}" \
    MASK_EPOCHS="${MASK_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
    FORGET_ORDER="${INTERLEAVED_ORDER}" \
    ORIGINAL_MODEL="${ORIGINAL_MODEL}" \
    bash ./scripts/run_incremental_ordered_logged.sh; then
    log_queue "DONE stage=full order=interleaved run_id=${INTERLEAVED_RUN_ID} status=success"
    return 0
  fi
  log_queue "DONE stage=full order=interleaved run_id=${INTERLEAVED_RUN_ID} status=failed"
  return 1
}

run_analysis() {
  log_queue "START analysis script=scripts/generate_cifar100_incremental_ordered_pilot_analysis.py"
  if python scripts/generate_cifar100_incremental_ordered_pilot_analysis.py; then
    log_queue "DONE analysis status=success"
    return 0
  fi
  log_queue "DONE analysis status=failed"
  return 1
}

log_queue "Queue log directory: ${QUEUE_LOG_DIR}"
log_queue "Settings: DATASET=${DATASET} SEED=${SEED} GPU=${GPU} ARCH=${ARCH} BATCH_SIZE=${BATCH_SIZE} RESULT_NAMESPACE=${RESULT_NAMESPACE}"
log_queue "Execution policy: clustered resume -> interleaved full -> analysis"

require_file "${ORIGINAL_MODEL}"
require_file "${CLUSTERED_START_MODEL}"
ensure_no_existing_cifar_job

run_clustered_resume
run_interleaved_full
run_analysis

log_queue "DONE ordered resume queue status=success"
