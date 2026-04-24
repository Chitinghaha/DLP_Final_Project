#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

SEED="${SEED:-1}"
GPU="${GPU:-1}"
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
MASK_RATIO="${MASK_RATIO:-0.5}"
UNLEARN_LR="${UNLEARN_LR:-0.013}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_K="${MAX_K:-9}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
RUN_ID="${RUN_ID:-incremental_newclass_seed${SEED}_k${MAX_K}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-results/logs/sequential_incremental/${RUN_ID}}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-results/original/cifar10_resnet18_seed${SEED}/0model_SA_best.pth.tar}"

mkdir -p "${LOG_DIR}" \
  "results/masks/sequential_incremental/newclass_mask/seed${SEED}" \
  "results/unlearn/sequential_incremental/newclass_mask/seed${SEED}" \
  "results/eval/sequential_incremental/newclass_mask"

COMMANDS_LOG="${LOG_DIR}/commands.log"
SUMMARY_LOG="${LOG_DIR}/summary.log"
PROGRESS_CSV="${LOG_DIR}/progress.csv"
GPU_CSV="${LOG_DIR}/gpu_monitor.csv"

echo "timestamp,step,stage,forgotten_classes,new_class,status,start_time,end_time,duration_sec,eta_next_stage_sec,eta_remaining_sec,checkpoint_path,log_path" > "${PROGRESS_CSV}"
echo "timestamp,gpu_index,utilization_gpu,memory_used_mb,memory_free_mb,temperature_c,power_draw_w" > "${GPU_CSV}"

log_summary() {
  local message="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${message}" | tee -a "${SUMMARY_LOG}"
}

write_env() {
  {
    echo "date: $(date)"
    echo "hostname: $(hostname)"
    echo "pwd: $(pwd)"
    echo "selected_gpu: ${GPU}"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status:"
    git status --short 2>/dev/null || true
    echo
    echo "python: $(which python)"
    python --version
    echo
    echo "pip freeze:"
    pip freeze
    echo
    echo "nvidia-smi:"
    nvidia-smi || true
    echo
    echo "memory:"
    free -h
    echo
    echo "disk:"
    df -h "${ROOT_DIR}"
  } > "${LOG_DIR}/env.txt" 2>&1
}

monitor_gpu() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw \
      --format=csv,noheader,nounits >> "${GPU_CSV}" 2>/dev/null || true
    sleep "${MONITOR_INTERVAL}"
  done
}

average_duration_for_stage() {
  local stage="$1"
  python - "$PROGRESS_CSV" "$stage" <<'PY'
import csv
import sys
path, stage = sys.argv[1], sys.argv[2]
values = []
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["stage"] == stage and row["status"] == "success" and row["duration_sec"]:
            values.append(float(row["duration_sec"]))
print(round(sum(values) / len(values), 2) if values else "unknown")
PY
}

remaining_eta() {
  local current_step="$1"
  local current_stage="$2"
  python - "$PROGRESS_CSV" "$MAX_K" "$current_step" "$current_stage" <<'PY'
import csv
import sys
path, max_k, step, stage = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
stages = ["mask", "unlearn", "eval"]
durations = {name: [] for name in stages}
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["stage"] in durations and row["status"] == "success" and row["duration_sec"]:
            durations[row["stage"]].append(float(row["duration_sec"]))
averages = {name: (sum(vals) / len(vals) if vals else None) for name, vals in durations.items()}
remaining = stages[stages.index(stage) + 1:]
for _ in range(step + 1, max_k + 1):
    remaining.extend(stages)
total = 0.0
unknown = False
for name in remaining:
    if averages[name] is None:
        unknown = True
    else:
        total += averages[name]
print("unknown" if unknown else round(total, 2))
PY
}

record_progress() {
  local step="$1"
  local stage="$2"
  local forgotten="$3"
  local new_class="$4"
  local status="$5"
  local start="$6"
  local end="$7"
  local duration="$8"
  local checkpoint="$9"
  local log_path="${10}"
  local eta_next
  local eta_remaining
  eta_next="$(average_duration_for_stage "${stage}")"
  eta_remaining="$(remaining_eta "${step}" "${stage}")"
  python - "$PROGRESS_CSV" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$step" "$stage" "$forgotten" "$new_class" "$status" "$start" "$end" "$duration" \
    "$eta_next" "$eta_remaining" "$checkpoint" "$log_path" <<'PY'
import csv
import sys
with open(sys.argv[1], "a", newline="") as handle:
    csv.writer(handle).writerow(sys.argv[2:])
PY
  log_summary "step=${step} stage=${stage} status=${status} duration=${duration}s eta_next_${stage}=${eta_next}s eta_remaining=${eta_remaining}s"
}

run_stage() {
  local step="$1"
  local stage="$2"
  local forgotten="$3"
  local new_class="$4"
  local checkpoint="$5"
  local log_path="$6"
  shift 6
  local start_epoch
  local end_epoch
  local duration
  start_epoch="$(date +%s)"
  log_summary "START step=${step} stage=${stage} forgotten=${forgotten} new_class=${new_class}"
  {
    echo "### $(date '+%Y-%m-%d %H:%M:%S')"
    echo "### Command: $*"
  } >> "${COMMANDS_LOG}"
  if PYTHONPATH="${ROOT_DIR}" PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log_path}"; then
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    record_progress "${step}" "${stage}" "${forgotten}" "${new_class}" "success" "${start_epoch}" "${end_epoch}" "${duration}" "${checkpoint}" "${log_path}"
  else
    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    record_progress "${step}" "${stage}" "${forgotten}" "${new_class}" "failed" "${start_epoch}" "${end_epoch}" "${duration}" "${checkpoint}" "${log_path}"
    exit 1
  fi
}

write_env
monitor_gpu &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

log_summary "Run directory: ${LOG_DIR}"
log_summary "Original model: ${ORIGINAL_MODEL}"
log_summary "Settings: SEED=${SEED} GPU=${GPU} ARCH=${ARCH} DATASET=${DATASET} MAX_K=${MAX_K} UNLEARN_EPOCHS=${UNLEARN_EPOCHS}"

current_model="${ORIGINAL_MODEL}"

for step in $(seq 1 "${MAX_K}"); do
  new_class=$((step - 1))
  forgotten="$(seq -s, 0 "${new_class}")"
  slug="${forgotten//,/_}"
  mask_dir="results/masks/sequential_incremental/newclass_mask/seed${SEED}/step${step}_forget_${new_class}"
  save_dir="results/unlearn/sequential_incremental/newclass_mask/seed${SEED}/step${step}_forgot_${slug}"
  eval_csv="results/eval/sequential_incremental/newclass_mask/seed${SEED}_step${step}_forgot_${slug}.csv"
  mask_log="${LOG_DIR}/step${step}_mask.log"
  unlearn_log="${LOG_DIR}/step${step}_unlearn.log"
  eval_log="${LOG_DIR}/step${step}_eval.log"
  mkdir -p "${mask_dir}" "${save_dir}" "$(dirname "${eval_csv}")"

  run_stage "${step}" "mask" "${forgotten}" "${new_class}" "${mask_dir}/with_${MASK_RATIO}.pt" "${mask_log}" \
    python generate_mask.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --class_to_replace "${new_class}" \
      --model_path "${current_model}" \
      --save_dir "${mask_dir}" \
      --unlearn_lr "${UNLEARN_LR}" \
      --unlearn_epochs "${MASK_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu "${GPU}" \
      --seed "${SEED}"

  run_stage "${step}" "unlearn" "${forgotten}" "${new_class}" "${save_dir}/RLcheckpoint.pth.tar" "${unlearn_log}" \
    python main_random.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --unlearn RL \
      --class_to_replace "${new_class}" \
      --classes_to_replace "${forgotten}" \
      --incremental_forget_only \
      --model_path "${current_model}" \
      --mask_path "${mask_dir}/with_${MASK_RATIO}.pt" \
      --save_dir "${save_dir}" \
      --unlearn_lr "${UNLEARN_LR}" \
      --unlearn_epochs "${UNLEARN_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu "${GPU}" \
      --seed "${SEED}"

  current_model="${save_dir}/RLcheckpoint.pth.tar"

  run_stage "${step}" "eval" "${forgotten}" "${new_class}" "${eval_csv}" "${eval_log}" \
    python scripts/evaluate_cumulative_forgetting.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --model_path "${current_model}" \
      --forgotten_classes "${forgotten}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu "${GPU}" \
      --seed "${SEED}" \
      --output "${eval_csv}"
done

log_summary "DONE all incremental sequential stages"
