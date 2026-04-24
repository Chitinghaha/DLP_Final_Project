#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

GPU="${GPU:-1}"
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-182}"
TRAIN_LR="${TRAIN_LR:-0.1}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_K="${MAX_K:-9}"
RESULT_NAMESPACE="${RESULT_NAMESPACE:-sequential_ordered}"
QUEUE_RUN_ID="${QUEUE_RUN_ID:-overnight_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-results/logs/${RESULT_NAMESPACE}/${QUEUE_RUN_ID}}"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"

mkdir -p "${QUEUE_LOG_DIR}" "results/logs/original"

log_queue() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

run_and_log() {
  local label="$1"
  shift
  log_queue "START ${label}"
  "$@" 2>&1 | tee -a "${QUEUE_LOG}"
  log_queue "DONE ${label}"
}

ensure_original_model() {
  local seed="$1"
  local model_path="results/original/cifar10_resnet18_seed${seed}/0model_SA_best.pth.tar"
  local train_log="results/logs/original/train_seed${seed}_${TRAIN_EPOCHS}.log"
  if [[ -f "${model_path}" ]]; then
    log_queue "Found existing original model for seed ${seed}: ${model_path}"
    return
  fi
  log_queue "START train_seed${seed}"
  python main_train.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --lr "${TRAIN_LR}" \
    --epochs "${TRAIN_EPOCHS}" \
    --save_dir "results/original/cifar10_resnet18_seed${seed}" \
    --gpu "${GPU}" \
    --seed "${seed}" \
    --train_seed "${seed}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee -a "${train_log}" | tee -a "${QUEUE_LOG}"
  log_queue "DONE train_seed${seed}"
}

run_smoke() {
  local run_id="$1"
  local seed="$2"
  local order="$3"
  run_and_log "${run_id}" \
    env \
      RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
      RUN_ID="${run_id}" \
      SEED="${seed}" \
      GPU="${GPU}" \
      ARCH="${ARCH}" \
      DATASET="${DATASET}" \
      MAX_K=2 \
      UNLEARN_EPOCHS=1 \
      MASK_EPOCHS=1 \
      BATCH_SIZE="${BATCH_SIZE}" \
      FORGET_ORDER="${order}" \
      ORIGINAL_MODEL="results/original/cifar10_resnet18_seed${seed}/0model_SA_best.pth.tar" \
      bash ./scripts/run_sequential_singleclass_logged.sh
}

run_full() {
  local run_id="$1"
  local seed="$2"
  local order="$3"
  run_and_log "${run_id}" \
    env \
      RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
      RUN_ID="${run_id}" \
      SEED="${seed}" \
      GPU="${GPU}" \
      ARCH="${ARCH}" \
      DATASET="${DATASET}" \
      MAX_K="${MAX_K}" \
      UNLEARN_EPOCHS="${UNLEARN_EPOCHS}" \
      MASK_EPOCHS="${MASK_EPOCHS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      FORGET_ORDER="${order}" \
      ORIGINAL_MODEL="results/original/cifar10_resnet18_seed${seed}/0model_SA_best.pth.tar" \
      bash ./scripts/run_sequential_singleclass_logged.sh
}

log_queue "Queue log directory: ${QUEUE_LOG_DIR}"
log_queue "Settings: GPU=${GPU} ARCH=${ARCH} DATASET=${DATASET} RESULT_NAMESPACE=${RESULT_NAMESPACE}"

ensure_original_model 2

run_smoke "smoke_clustered_animals_seed1_k2" 1 "2,3,4,5,6,7,0,1,8"
run_smoke "smoke_interleaved_seed1_k2" 1 "0,6,1,4,2,7,3,8,5"
run_smoke "smoke_retain_bird_seed1_k2" 1 "0,1,3,4,5,6,7,8,9"

run_full "canonical_seed2_k9" 2 "0,1,2,3,4,5,6,7,8"
run_full "clustered_animals_seed1_k9" 1 "2,3,4,5,6,7,0,1,8"
run_full "clustered_animals_seed2_k9" 2 "2,3,4,5,6,7,0,1,8"
run_full "interleaved_seed1_k9" 1 "0,6,1,4,2,7,3,8,5"
run_full "retain_bird_seed1_k9" 1 "0,1,3,4,5,6,7,8,9"

log_queue "DONE all overnight ordered sequential experiments"
