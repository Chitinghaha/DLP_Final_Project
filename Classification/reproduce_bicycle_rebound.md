# 重現 CIFAR-100 Bicycle Rebound

這份文件說明如何只跑一條代表性的 CIFAR-100 `vehicles_1` 實驗，重現 `bicycle / 腳踏車 (8)` 被忘記後又回升的現象。

```text
Run ID: vehicles1_official_seed1_k5
FORGET_ORDER = 8,13,48,58,90
類別順序 = bicycle, bus, motorcycle, pickup_truck, train
```

重點結果是：`bicycle / 腳踏車 (8)` 在 step 1 被忘記後，test accuracy 會先降到接近 0%，但後續繼續忘其他車輛類別時又逐步 rebound。

## 需要的模型參數

這個實驗需要 CIFAR-100 / ResNet-18 / seed 1 的 original baseline model：

```text
0model_SA_best.pth.tar
```

檔案大小約 `86 MB`。可以用雲端硬碟傳給對方。

下載後請放到：

```text
Classification/results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

從 repo root 來看，位置應該像這樣：

```text
DLP_Final_Project_publish_20260424/
└── Classification/
    └── results/
        └── original/
            └── cifar100_resnet18_seed1/
                └── 0model_SA_best.pth.tar
```

如果是從雲端連結下載，可以執行：

```bash
cd /path/to/DLP_Final_Project_publish_20260424/Classification
mkdir -p results/original/cifar100_resnet18_seed1

# 將 MODEL_URL 換成雲端下載連結
wget -O results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar "MODEL_URL"
```

## 建立環境

進到 `Classification/`：

```bash
cd /path/to/DLP_Final_Project_publish_20260424/Classification
bash scripts/setup_venv.sh
source .venv/bin/activate
```

如果本機沒有 CIFAR-100，程式會自動下載。

## 跑單一重現實驗

以下只跑 official `vehicles_1` 順序，也就是：

```text
bicycle -> bus -> motorcycle -> pickup_truck -> train
```

執行：

```bash
RESULT_NAMESPACE=sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5 \
RUN_ID=vehicles1_official_seed1_k5 \
SEED=1 \
GPU=0 \
DATASET=cifar100 \
MAX_K=5 \
BATCH_SIZE=2048 \
UNLEARN_EPOCHS=10 \
MASK_EPOCHS=1 \
FORGET_ORDER=8,13,48,58,90 \
ORIGINAL_MODEL=results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar \
bash scripts/run_incremental_ordered_logged.sh
```

順序說明：

| Step | Newly forgotten class |
| --- | --- |
| 1 | bicycle / 腳踏車 (8) |
| 2 | bus / 公車 (13) |
| 3 | motorcycle / 摩托車 (48) |
| 4 | pickup_truck / 皮卡車 (58) |
| 5 | train / 火車 (90) |

## 檢查 Bicycle Rebound

跑完後，evaluation CSV 會在：

```text
results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/
```

可以用以下指令列出每一步的 `bicycle / class_8_accuracy`：

```bash
python - <<'PY'
import csv
from pathlib import Path

eval_dir = Path("results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5")
files = [
    eval_dir / "seed1_step1_forgot_8.csv",
    eval_dir / "seed1_step2_forgot_8_13.csv",
    eval_dir / "seed1_step3_forgot_8_13_48.csv",
    eval_dir / "seed1_step4_forgot_8_13_48_58.csv",
    eval_dir / "seed1_step5_forgot_8_13_48_58_90.csv",
]

print("step,bicycle_class_8_accuracy,forget_accuracy,retain_accuracy,full_test_accuracy")
for step, path in enumerate(files, start=1):
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    print(
        step,
        row["class_8_accuracy"],
        row["forget_accuracy"],
        row["retain_accuracy"],
        row["full_test_accuracy"],
        sep=",",
    )
PY
```

原始實驗觀察到的趨勢：

| Step | Forgotten so far | bicycle accuracy |
| --- | --- | --- |
| 1 | 8 | 約 0% |
| 2 | 8,13 | 約 22% |
| 3 | 8,13,48 | 約 29% |
| 4 | 8,13,48,58 | 約 41% |
| 5 | 8,13,48,58,90 | 約 43% |

不同硬體或套件版本可能造成小幅差異，但關鍵現象應該一致：`bicycle` 在 step 1 被壓到接近 0%，後續忘其他車輛類別後又回升。

## 輸出位置

跑完後主要檔案會在：

```text
results/logs/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/progress.csv
results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/seed1_step*_forgot_*.csv
results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/seed1/step*_forgot_*/RLcheckpoint.pth.tar
results/masks/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/seed1/step*_forget_*/with_0.5.pt
```

## 補充

- 這份文件只重現一條代表性 run，不會跑完整 9 條 single-coarse probe。
- 原始報告另外也跑了 `vehicles1_random_seed1_k5` 和 `vehicles1_hardfirst_seed1_k5`，兩者也都觀察到 bicycle rebound。
- 報告中的 per-class accuracy 使用 CIFAR-100 test split。
