# CIFAR-100 Pilot Code Trace

這份文件只 trace `sequential_incremental_ordered_cifar100_pilot` 這條實驗線的程式碼路徑。重點不是重新分析結果，而是說明 CIFAR-100 pilot 從 queue 啟動、每一步執行 `mask -> unlearn -> eval`，最後產生 analysis report 的過程中，程式實際進到哪些檔案、使用哪些參數、輸入輸出放在哪裡。

## 1. 入口與重要註記

CIFAR-100 pilot 的正式結果與分析稿保存在 publish folder：

- 詳細分析報告：[cifar100_incremental_ordered_pilot_detailed_analysis.md](./cifar100_incremental_ordered_pilot_detailed_analysis.md)
- 圖表目錄：[cifar100_incremental_ordered_pilot_detailed_figures/](./cifar100_incremental_ordered_pilot_detailed_figures/)
- 產圖與分析 script：[scripts/generate_cifar100_incremental_ordered_detailed_analysis.py](./scripts/generate_cifar100_incremental_ordered_detailed_analysis.py)

但這條 CIFAR-100 pilot 當時實際跑實驗時，runner 與 evaluator 使用的是工作版：

- Queue runner：`../../Unlearn-Saliency-original/Classification/scripts/run_incremental_ordered_queue.sh`
- Single-run runner：`../../Unlearn-Saliency-original/Classification/scripts/run_incremental_ordered_logged.sh`
- Cumulative evaluator：`../../Unlearn-Saliency-original/Classification/scripts/evaluate_cumulative_forgetting.py`

原因是 publish folder 裡的部分 script 是較早整理出的版本。例如 publish folder 的 `scripts/run_incremental_ordered_queue.sh` 預設仍偏 CIFAR-10 ordered queue；publish folder 的 `scripts/evaluate_cumulative_forgetting.py` 也只支援 CIFAR-10。CIFAR-100 pilot 的 code trace 因此應以 `Unlearn-Saliency-original/Classification/` 中的 runner / evaluator 為準，再搭配 publish folder 中保存的核心實作、報告與圖表。

整體 code path 可以簡化成：

```text
run_incremental_ordered_queue.sh
  -> ensure_original_model()
       -> main_train.py
  -> run_logged_job(smoke/full, order)
       -> run_incremental_ordered_logged.sh
            -> step k:
                 -> generate_mask.py
                 -> main_random.py
                 -> scripts/evaluate_cumulative_forgetting.py
  -> scripts/generate_cifar100_incremental_ordered_detailed_analysis.py
       -> cifar100_incremental_ordered_pilot_detailed_analysis.md
       -> cifar100_incremental_ordered_pilot_detailed_figures/
```

## 2. Queue 層 Code Trace

實驗從工作版 queue runner 啟動：

```text
../../Unlearn-Saliency-original/Classification/scripts/run_incremental_ordered_queue.sh
```

這層負責決定整批 CIFAR-100 pilot 要跑哪些 order、先跑 smoke 還是 full、原始模型是否需要先訓練。

關鍵預設值如下：

```bash
DATASET="${DATASET:-cifar100}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"
ARCH="${ARCH:-resnet18}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-182}"
TRAIN_LR="${TRAIN_LR:-0.1}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
MASK_EPOCHS="${MASK_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SMOKE_K="${SMOKE_K:-2}"
```

當 `DATASET=cifar100` 時，`default_result_namespace()` 會回傳：

```text
sequential_incremental_ordered_cifar100_pilot
```

因此所有 logs、masks、checkpoints、eval CSV 都會放在這個 namespace 底下：

```text
results/logs/sequential_incremental_ordered_cifar100_pilot/
results/masks/sequential_incremental_ordered_cifar100_pilot/
results/unlearn/sequential_incremental_ordered_cifar100_pilot/
results/eval/sequential_incremental_ordered_cifar100_pilot/
```

### 2.1 三條 CIFAR-100 Forget Order

Queue runner 在 `DATASET=cifar100` 時定義 `FULL_K=20`，並設定三條 order：

```text
Normal:
3,15,19,21,31,34,36,38,42,43,50,63,64,65,66,74,75,80,88,97

Clustered:
3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,36,50,65,74,80

Interleaved:
3,15,34,36,42,19,63,50,43,21,64,65,88,31,66,74,97,38,75,80
```

這三條 order 對應的 full run id 是：

```text
incremental_cifar100_normal_seed1_k20
incremental_cifar100_clustered_animals_seed1_k20
incremental_cifar100_interleaved_animals_seed1_k20
```

smoke run id 是：

```text
smoke_incremental_cifar100_normal_seed1_k2
smoke_incremental_cifar100_clustered_animals_seed1_k2
smoke_incremental_cifar100_interleaved_animals_seed1_k2
```

### 2.2 原始模型檢查與訓練

Queue runner 會先呼叫：

```bash
ensure_original_model "${ORIGINAL_MODEL}"
```

其中 CIFAR-100 預設原始模型路徑是：

```text
results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

如果 `0model_SA_best.pth.tar`、`0checkpoint.pth.tar`、`0net_train.png` 都存在，就直接沿用。否則會呼叫：

```bash
python main_train.py \
  --arch "${ARCH}" \
  --dataset "${DATASET}" \
  --lr "${TRAIN_LR}" \
  --epochs "${TRAIN_EPOCHS}" \
  --save_dir "${save_dir}" \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --train_seed "${SEED}" \
  --batch_size "${BATCH_SIZE}"
```

`main_train.py` 的核心路徑是：

```text
main_train.py
  -> arg_parser.parse_args()
  -> utils.setup_model_dataset(args)
       -> dataset.cifar100_dataloaders(...)
       -> model_dict[args.arch](num_classes=100)
  -> trainer.train(...)
  -> trainer.validate(...)
  -> utils.save_checkpoint(...)
```

### 2.3 Smoke Gate 與 Full Runs

Queue runner 的執行策略是：

```text
baseline/original model
  -> smoke normal
  -> smoke clustered
  -> smoke interleaved
  -> if smoke all passed:
       -> full normal
       -> full clustered
       -> full interleaved
```

每次啟動一條 order 都會呼叫：

```bash
run_logged_job stage order_name run_id max_k order
```

`run_logged_job()` 會把環境變數傳給 single-run runner：

```bash
env \
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
  bash ./scripts/run_incremental_ordered_logged.sh
```

## 3. Single Run 層 Code Trace

每一條 order 進入：

```text
../../Unlearn-Saliency-original/Classification/scripts/run_incremental_ordered_logged.sh
```

這個 script 負責單一 run 的所有 step。對 CIFAR-100 full run 而言，`MAX_K=20`，所以會跑 20 個 step。

### 3.1 Run 目錄與紀錄檔

以 Normal full run 為例：

```text
RUN_ID=incremental_cifar100_normal_seed1_k20
RESULT_NAMESPACE=sequential_incremental_ordered_cifar100_pilot
LOG_DIR=results/logs/sequential_incremental_ordered_cifar100_pilot/incremental_cifar100_normal_seed1_k20
```

single-run runner 會建立：

```text
commands.log
summary.log
progress.csv
gpu_monitor.csv
env.txt
step1_mask.log
step1_unlearn.log
step1_eval.log
...
step20_mask.log
step20_unlearn.log
step20_eval.log
```

其中：

- `commands.log` 記錄每個 stage 實際執行的 command。
- `summary.log` 記錄 step / stage 開始、結束與耗時摘要。
- `progress.csv` 記錄 `step`、`stage`、`forgotten_classes`、`new_class`、`status`、`duration_sec`、`checkpoint_path`、`log_path`。
- `gpu_monitor.csv` 由 `nvidia-smi` 定期寫入 GPU utilization / memory / temperature / power。

### 3.2 Step 內部變數

single-run runner 對每個 step 做以下計算：

```bash
new_class="${FORGET_ORDER_ARRAY[$((step - 1))]}"
forgotten="$(join_classes_through_step "${step}")"
slug="${forgotten//,/_}"
```

例如 Normal order：

```text
step1:
  new_class=3
  forgotten=3
  slug=3

step4:
  new_class=21
  forgotten=3,15,19,21
  slug=3_15_19_21

step20:
  new_class=97
  forgotten=3,15,19,21,31,34,36,38,42,43,50,63,64,65,66,74,75,80,88,97
  slug=3_15_19_21_31_34_36_38_42_43_50_63_64_65_66_74_75_80_88_97
```

每個 step 的輸出路徑會依這些變數建立：

```bash
mask_dir="${MASK_ROOT}/seed${SEED}/step${step}_forget_${new_class}"
save_dir="${UNLEARN_ROOT}/seed${SEED}/step${step}_forgot_${slug}"
eval_csv="${EVAL_ROOT}/seed${SEED}_step${step}_forgot_${slug}.csv"
```

每個 step 固定呼叫三個 stage：

```text
mask -> unlearn -> eval
```

## 4. Stage 1: Mask Code Trace

single-run runner 的 mask stage 會呼叫：

```bash
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
```

publish folder 中對應核心檔案：

- [generate_mask.py](./generate_mask.py)
- [utils.py](./utils.py)
- [dataset.py](./dataset.py)

`generate_mask.py` 的主要 trace 是：

```text
generate_mask.py
  -> arg_parser.parse_args()
  -> utils.setup_model_dataset(args)
       -> args.dataset == "cifar100"
       -> dataset.cifar100_dataloaders(...)
       -> class_to_replace = new_class
       -> only_mark=True
       -> replace_class(...)
  -> marked_loader.dataset 中 target < 0 的 samples 成為 forget dataset
  -> load current_model
  -> save_gradient_ratio(unlearn_data_loaders, model, criterion, args)
```

`dataset.py` 的 `replace_class(..., only_mark=True)` 會把該 class 的 target 標成負值。`generate_mask.py` 接著用 `target < 0` 把這些樣本切成 `forget_loader`，並把 target 還原成原 class id。

`save_gradient_ratio()` 的核心邏輯：

```text
for image, target in forget_loader:
    output_clean = model(image)
    loss = -CrossEntropy(output_clean, target)
    loss.backward()
    accumulate abs(gradient) for each parameter

rank all gradient magnitudes
save hard masks for thresholds 0.1, 0.2, ..., 1.0
```

CIFAR-100 pilot 實驗使用的 mask ratio 是 `0.5`，所以後續 unlearn stage 讀：

```text
results/masks/sequential_incremental_ordered_cifar100_pilot/seed1/stepK_forget_C/with_0.5.pt
```

其中 `K` 是 step，`C` 是當步 newly forgotten class。

## 5. Stage 2: Unlearn Code Trace

single-run runner 的 unlearn stage 會呼叫：

```bash
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
```

publish folder 中對應核心檔案：

- [main_random.py](./main_random.py)
- [dataset.py](./dataset.py)
- [utils.py](./utils.py)
- [unlearn/RL.py](./unlearn/RL.py)
- [unlearn/impl.py](./unlearn/impl.py)

### 5.1 Incremental Split

`main_random.py` 看到 `--incremental_forget_only` 後，不使用 marked negative label split，而是進入：

```text
_build_incremental_unlearn_datasets(train_loader_full.dataset, args)
```

這個函式讀兩個參數：

```text
args.class_to_replace     # 當步新 forget class
args.classes_to_replace   # 到目前為止 cumulative forgotten classes
```

然後建立三種 mask：

```text
cumulative_mask = targets in cumulative_classes
forget_mask = targets == current_class
retain_mask = not cumulative_mask
excluded_mask = cumulative_mask and not forget_mask
```

因此在 CIFAR-100 pilot 的正確 incremental data flow 中：

- `forget_dataset` 只包含當步 newly forgotten class。
- `retain_dataset` 只包含還沒被要求 forget 的 classes。
- old forgotten classes 進入 `excluded_mask`，不會進入後續 train loader。

這是整個 CIFAR-100 pilot 最重要的資料流程保證。若 step10 的 cumulative forgotten set 是：

```text
3,15,34,36,42,19,63,50,43,21
```

而當步 `class_to_replace=21`，那麼：

```text
forget_dataset = class 21
retain_dataset = all classes except 3,15,34,36,42,19,63,50,43,21
excluded_old = 3,15,34,36,42,19,63,50,43
```

舊 forgotten classes 不會被重新拿去訓練；它們只會在 eval stage 被檢查。

### 5.2 RL Unlearning

`main_random.py` 載入模型與 mask 後：

```text
unlearn_method = unlearn.get_unlearn_method(args.unlearn)
unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
```

因為 `--unlearn RL`，所以會進入：

```text
unlearn/RL.py -> RL(...)
```

`RL.py` 裡 CIFAR-100 分支的主要行為是：

```text
forget_dataset targets -> random labels in [0, num_classes)
train_dataset = ConcatDataset([forget_dataset, retain_dataset])
for each batch:
    output = model(image)
    loss = CrossEntropy(output, target)
    loss.backward()
    if mask:
        gradient *= mask
    optimizer.step()
    if mask:
        restore masked-out params to theta0
```

也就是：當步 forget class 會被改成 random labels，retain dataset 維持原 label，並透過 saliency mask 限制參數更新區域。

### 5.3 Checkpoint

unlearn 完成後，`main_random.py` 呼叫：

```text
unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
```

實作位置：

```text
unlearn/impl.py
```

輸出 checkpoint：

```text
results/unlearn/sequential_incremental_ordered_cifar100_pilot/seed1/stepK_forgot_<slug>/RLcheckpoint.pth.tar
```

下一個 step 的 `current_model` 會更新成這個 checkpoint。

## 6. Stage 3: Eval 與 Analysis Code Trace

single-run runner 的 eval stage 會呼叫 CIFAR-100 支援版 evaluator：

```text
../../Unlearn-Saliency-original/Classification/scripts/evaluate_cumulative_forgetting.py
```

呼叫形式：

```bash
python scripts/evaluate_cumulative_forgetting.py \
  --arch "${ARCH}" \
  --dataset "${DATASET}" \
  --model_path "${current_model}" \
  --forgotten_classes "${forgotten}" \
  --batch_size "${BATCH_SIZE}" \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  --output "${eval_csv}"
```

CIFAR-100 支援版 evaluator 的 trace：

```text
evaluate_cumulative_forgetting.py
  -> DATASET_CONFIGS["cifar100"]
       -> dataset_cls = CIFAR100
       -> mean/std = CIFAR-100 normalization
       -> num_classes = 100
  -> load_model(...)
       -> model_dict[arch](num_classes=100)
       -> load checkpoint state_dict
  -> build_test_loader(...)
       -> torchvision.datasets.CIFAR100(train=False)
  -> evaluate(...)
       -> compute class_0_accuracy ... class_99_accuracy
       -> forgotten = parse_classes(forgotten_classes)
       -> retain = all 100 classes not in forgotten
       -> forget_accuracy = grouped accuracy over forgotten
       -> UA = 100 - forget_accuracy
       -> retain_accuracy = grouped accuracy over retain
       -> full_test_accuracy = accuracy over full CIFAR-100 test set
  -> write_csv(...)
```

輸出 CSV 路徑規則：

```text
results/eval/sequential_incremental_ordered_cifar100_pilot/seed1_stepK_forgot_<slug>.csv
```

CSV 欄位包含：

```text
model_path
forgotten_classes
forget_accuracy
UA
retain_accuracy
full_test_accuracy
class_0_accuracy
...
class_99_accuracy
```

## 7. Analysis Report Code Trace

完成三條 full runs 後，分析與圖表由 publish folder 內的 generator 產生：

```text
scripts/generate_cifar100_incremental_ordered_detailed_analysis.py
```

這個 script 內部固定定義：

- 20 個 CIFAR-100 pilot class metadata。
- 三條 order 的 `run_id` 與 class sequence。
- `result_namespace=sequential_incremental_ordered_cifar100_pilot`。
- output report path 與 figures path。

核心 trace：

```text
generate_cifar100_incremental_ordered_detailed_analysis.py
  -> ORDER_CONFIGS
       -> normal / clustered / interleaved run_id and order
  -> eval_filename(seed, step, order)
       -> seed1_stepK_forgot_<slug>.csv
  -> read_eval_rows(...)
       -> read results/eval/sequential_incremental_ordered_cifar100_pilot/*.csv
  -> read progress.csv for each run
       -> runtime / stage duration
  -> compute final metrics, newly forgotten accuracy, rebound, retain impact
  -> save figures
  -> write markdown report
```

主要輸出：

```text
cifar100_incremental_ordered_pilot_detailed_analysis.md
cifar100_incremental_ordered_pilot_detailed_figures/
```

這份 publish folder 中的最終報告與 figures 已經是整理後成果；若只要看結論，讀：

```text
cifar100_incremental_ordered_pilot_detailed_analysis.md
```

若要 trace 實驗怎麼從 code 跑出那些結果，讀本文件。

## 8. 一頁版流程圖

```text
Queue:
  run_incremental_ordered_queue.sh
    DATASET=cifar100
    RESULT_NAMESPACE=sequential_incremental_ordered_cifar100_pilot
    FULL_K=20
    orders = normal / clustered / interleaved

Original model:
  ensure_original_model()
    -> main_train.py
       -> utils.setup_model_dataset()
       -> dataset.cifar100_dataloaders()
       -> trainer.train() / trainer.validate()
       -> results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar

Each order:
  run_incremental_ordered_logged.sh
    for step in 1..20:
      new_class = FORGET_ORDER[step - 1]
      forgotten = FORGET_ORDER[:step]
      slug = forgotten joined by _

      mask:
        -> generate_mask.py
        -> dataset marks class_to_replace
        -> save_gradient_ratio()
        -> results/masks/.../with_0.5.pt

      unlearn:
        -> main_random.py --incremental_forget_only --unlearn RL
        -> _build_incremental_unlearn_datasets()
        -> unlearn/RL.py
        -> unlearn/impl.py
        -> results/unlearn/.../RLcheckpoint.pth.tar

      eval:
        -> evaluate_cumulative_forgetting.py
        -> class_0_accuracy ... class_99_accuracy
        -> results/eval/.../seed1_stepK_forgot_<slug>.csv

Analysis:
  generate_cifar100_incremental_ordered_detailed_analysis.py
    -> read eval CSV + progress CSV
    -> cifar100_incremental_ordered_pilot_detailed_analysis.md
    -> cifar100_incremental_ordered_pilot_detailed_figures/
```

## 9. 路徑速查

| 類型 | 路徑 |
| --- | --- |
| 實際 queue runner | `../../Unlearn-Saliency-original/Classification/scripts/run_incremental_ordered_queue.sh` |
| 實際 single-run runner | `../../Unlearn-Saliency-original/Classification/scripts/run_incremental_ordered_logged.sh` |
| 實際 CIFAR-100 evaluator | `../../Unlearn-Saliency-original/Classification/scripts/evaluate_cumulative_forgetting.py` |
| 原始模型訓練入口 | [main_train.py](./main_train.py) |
| mask 入口 | [generate_mask.py](./generate_mask.py) |
| unlearn 入口 | [main_random.py](./main_random.py) |
| dataset split | [dataset.py](./dataset.py) |
| model / loader setup | [utils.py](./utils.py) |
| RL implementation | [unlearn/RL.py](./unlearn/RL.py) |
| checkpoint save/load | [unlearn/impl.py](./unlearn/impl.py) |
| analysis generator | [scripts/generate_cifar100_incremental_ordered_detailed_analysis.py](./scripts/generate_cifar100_incremental_ordered_detailed_analysis.py) |
| final report | [cifar100_incremental_ordered_pilot_detailed_analysis.md](./cifar100_incremental_ordered_pilot_detailed_analysis.md) |
| final figures | [cifar100_incremental_ordered_pilot_detailed_figures/](./cifar100_incremental_ordered_pilot_detailed_figures/) |

`results/...` 路徑是實驗執行時的輸出規則。publish folder 主要保存程式碼、報告與 figures，不一定包含完整大型 `results/`、`data/`、`.venv/`。
