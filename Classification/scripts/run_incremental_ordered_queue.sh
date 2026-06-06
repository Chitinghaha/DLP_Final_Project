#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

SEED="${SEED:-1}"
GPU="${GPU:-0}"
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar100}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-182}"
TRAIN_LR="${TRAIN_LR:-0.1}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SMOKE_K="${SMOKE_K:-2}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"

default_result_namespace() {
  case "${DATASET}" in
    cifar100)
      echo "sequential_incremental_ordered_cifar100_pilot"
      ;;
    *)
      echo "sequential_incremental_ordered"
      ;;
  esac
}

default_original_model() {
  case "${DATASET}" in
    cifar100)
      echo "results/original/cifar100_resnet18_seed${SEED}/0model_SA_best.pth.tar"
      ;;
    cifar10)
      echo "results/original/cifar10_resnet18_seed${SEED}/0model_SA_best.pth.tar"
      ;;
    *)
      echo "results/original/${DATASET}_${ARCH}_seed${SEED}/0model_SA_best.pth.tar"
      ;;
  esac
}

RESULT_NAMESPACE="${RESULT_NAMESPACE:-$(default_result_namespace)}"
QUEUE_RUN_ID="${QUEUE_RUN_ID:-ordered_queue_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-results/logs/${RESULT_NAMESPACE}/${QUEUE_RUN_ID}}"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue.log"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-$(default_original_model)}"

mkdir -p "${QUEUE_LOG_DIR}" "results/logs/original"

log_queue() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

if [[ "${DATASET}" == "cifar100" ]]; then
  FULL_K="${FULL_K:-20}"
  NORMAL_ORDER="${NORMAL_ORDER:-3,15,19,21,31,34,36,38,42,43,50,63,64,65,66,74,75,80,88,97}"
  CLUSTERED_ORDER="${CLUSTERED_ORDER:-3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,36,50,65,74,80}"
  INTERLEAVED_ORDER="${INTERLEAVED_ORDER:-3,15,34,36,42,19,63,50,43,21,64,65,88,31,66,74,97,38,75,80}"

  SMOKE_RUN_NORMAL="${SMOKE_RUN_NORMAL:-smoke_incremental_cifar100_normal_seed1_k2}"
  SMOKE_RUN_CLUSTERED="${SMOKE_RUN_CLUSTERED:-smoke_incremental_cifar100_clustered_animals_seed1_k2}"
  SMOKE_RUN_INTERLEAVED="${SMOKE_RUN_INTERLEAVED:-smoke_incremental_cifar100_interleaved_animals_seed1_k2}"

  FULL_RUN_NORMAL="${FULL_RUN_NORMAL:-incremental_cifar100_normal_seed1_k20}"
  FULL_RUN_CLUSTERED="${FULL_RUN_CLUSTERED:-incremental_cifar100_clustered_animals_seed1_k20}"
  FULL_RUN_INTERLEAVED="${FULL_RUN_INTERLEAVED:-incremental_cifar100_interleaved_animals_seed1_k20}"
else
  FULL_K="${FULL_K:-9}"
  NORMAL_ORDER="${NORMAL_ORDER:-0,1,2,3,4,5,6,7,8}"
  CLUSTERED_ORDER="${CLUSTERED_ORDER:-2,3,4,5,6,7,0,1,8}"
  INTERLEAVED_ORDER="${INTERLEAVED_ORDER:-0,6,1,4,2,7,3,8,5}"

  SMOKE_RUN_NORMAL="${SMOKE_RUN_NORMAL:-smoke_incremental_${DATASET}_normal_seed${SEED}_k2}"
  SMOKE_RUN_CLUSTERED="${SMOKE_RUN_CLUSTERED:-smoke_incremental_${DATASET}_clustered_seed${SEED}_k2}"
  SMOKE_RUN_INTERLEAVED="${SMOKE_RUN_INTERLEAVED:-smoke_incremental_${DATASET}_interleaved_seed${SEED}_k2}"

  FULL_RUN_NORMAL="${FULL_RUN_NORMAL:-incremental_${DATASET}_normal_seed${SEED}_k${FULL_K}}"
  FULL_RUN_CLUSTERED="${FULL_RUN_CLUSTERED:-incremental_${DATASET}_clustered_seed${SEED}_k${FULL_K}}"
  FULL_RUN_INTERLEAVED="${FULL_RUN_INTERLEAVED:-incremental_${DATASET}_interleaved_seed${SEED}_k${FULL_K}}"
fi

ensure_original_model() {
  local model_path="$1"
  local save_dir
  local checkpoint_path
  local train_curve
  save_dir="$(dirname "${model_path}")"
  checkpoint_path="${save_dir}/0checkpoint.pth.tar"
  train_curve="${save_dir}/0net_train.png"

  if [[ -f "${model_path}" && -f "${checkpoint_path}" && -f "${train_curve}" ]]; then
    log_queue "FOUND original_model path=${model_path}"
    return
  fi

  local train_log
  train_log="results/logs/original/train_${DATASET}_seed${SEED}_${TRAIN_EPOCHS}.log"

  log_queue "START train_original dataset=${DATASET} seed=${SEED} gpu=${GPU} save_dir=${save_dir}"
  python main_train.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --lr "${TRAIN_LR}" \
    --epochs "${TRAIN_EPOCHS}" \
    --save_dir "${save_dir}" \
    --gpu "${GPU}" \
    --seed "${SEED}" \
    --train_seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee -a "${train_log}"
  log_queue "DONE train_original dataset=${DATASET} seed=${SEED} gpu=${GPU}"
}

run_logged_job() {
  local stage="$1"
  local order_name="$2"
  local run_id="$3"
  local max_k="$4"
  local order="$5"

  log_queue "START stage=${stage} order=${order_name} run_id=${run_id} max_k=${max_k} gpu=${GPU}"
  if env \
    RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
    RUN_ID="${run_id}" \
    SEED="${SEED}" \
    GPU="${GPU}" \
    ARCH="${ARCH}" \
    DATASET="${DATASET}" \
    MAX_K="${max_k}" \
    UNLEARN_EPOCHS="${UNLEARN_EPOCHS}" \
    MASK_EPOCHS="${MASK_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
    FORGET_ORDER="${order}" \
    ORIGINAL_MODEL="${ORIGINAL_MODEL}" \
    bash ./scripts/run_incremental_ordered_logged.sh; then
    log_queue "DONE stage=${stage} order=${order_name} run_id=${run_id} status=success"
    return 0
  fi

  log_queue "DONE stage=${stage} order=${order_name} run_id=${run_id} status=failed"
  return 1
}

log_queue "Queue log directory: ${QUEUE_LOG_DIR}"
log_queue "Settings: DATASET=${DATASET} SEED=${SEED} GPU=${GPU} ARCH=${ARCH} BATCH_SIZE=${BATCH_SIZE} RESULT_NAMESPACE=${RESULT_NAMESPACE}"
log_queue "Execution policy: baseline -> smoke(normal, clustered, interleaved) -> full(normal, clustered, interleaved)"

ensure_original_model "${ORIGINAL_MODEL}"

run_logged_job "smoke" "normal" "${SMOKE_RUN_NORMAL}" "${SMOKE_K}" "${NORMAL_ORDER}" || exit 1
run_logged_job "smoke" "clustered" "${SMOKE_RUN_CLUSTERED}" "${SMOKE_K}" "${CLUSTERED_ORDER}" || exit 1
run_logged_job "smoke" "interleaved" "${SMOKE_RUN_INTERLEAVED}" "${SMOKE_K}" "${INTERLEAVED_ORDER}" || exit 1

log_queue "SMOKE_GATE status=passed"

run_logged_job "full" "normal" "${FULL_RUN_NORMAL}" "${FULL_K}" "${NORMAL_ORDER}" || exit 1
run_logged_job "full" "clustered" "${FULL_RUN_CLUSTERED}" "${FULL_K}" "${CLUSTERED_ORDER}" || exit 1
run_logged_job "full" "interleaved" "${FULL_RUN_INTERLEAVED}" "${FULL_K}" "${INTERLEAVED_ORDER}" || exit 1

log_queue "DONE ordered queue status=success"
