# 重現 CIFAR-100 Bicycle Rebound

這份文件說明如何重現 CIFAR-100 `vehicles_1` 中 `bicycle / 腳踏車 (8)` 被忘記後又回升的現象。

## 9 條實驗中問題最大的一條

9 條 single-coarse runs 裡，問題最明顯的是：

```text
Run ID: vehicles1_hardfirst_seed1_k5
FORGET_ORDER = 13,8,58,90,48
類別順序 = bus, bicycle, pickup_truck, train, motorcycle
```

選這條的原因：

| 指標 | 數值 |
| --- | --- |
| Final forget acc | 9.6% |
| Final retain acc | 70.3% |
| Final full test acc | 67.3% |
| bicycle forget step accuracy | 1.0% |
| bicycle final accuracy | 48.0% |

也就是說，`bicycle (8)` 在 step 2 被忘記時 accuracy 已經降到 `1.0%`，但 final step 又回到 `48.0%`。這是 9 條實驗中 bicycle final residual 最高的一條，所以最適合拿來展示 rebound 問題。

## 方法一：重新跑 Unlearning 實驗

這個方法會從 CIFAR-100 original model 開始，重新跑 `vehicles1_hardfirst_seed1_k5`。

### 1. 建立環境

進到 `Classification/`：

```bash
cd /path/to/DLP_Final_Project_publish_20260424/Classification
bash scripts/setup_venv.sh
source .venv/bin/activate
```

如果本機沒有 CIFAR-100，程式會自動下載。

### 2. 準備 Original Model

這個實驗需要 CIFAR-100 / ResNet-18 / seed 1 的 original baseline model：

```text
results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

如果要自己重新 train original model，執行：

```bash
python main_train.py \
  --arch resnet18 \
  --dataset cifar100 \
  --lr 0.1 \
  --epochs 182 \
  --save_dir results/original/cifar100_resnet18_seed1 \
  --gpu 0 \
  --seed 1 \
  --train_seed 1 \
  --batch_size 256
```

訓練完成後，應該會產生：

```text
results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
results/original/cifar100_resnet18_seed1/0checkpoint.pth.tar
results/original/cifar100_resnet18_seed1/0net_train.png
```

如果已經有 original model，也可以直接放到：

```text
Classification/results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

### 3. 跑問題最明顯的 hardfirst order

執行：

```bash
RESULT_NAMESPACE=sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5 \
RUN_ID=vehicles1_hardfirst_seed1_k5 \
SEED=1 \
GPU=0 \
DATASET=cifar100 \
MAX_K=5 \
BATCH_SIZE=2048 \
UNLEARN_EPOCHS=10 \
MASK_EPOCHS=1 \
FORGET_ORDER=13,8,58,90,48 \
ORIGINAL_MODEL=results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar \
bash scripts/run_incremental_ordered_logged.sh
```

順序說明：

| Step | Newly forgotten class |
| --- | --- |
| 1 | bus / 公車 (13) |
| 2 | bicycle / 腳踏車 (8) |
| 3 | pickup_truck / 皮卡車 (58) |
| 4 | train / 火車 (90) |
| 5 | motorcycle / 摩托車 (48) |

跑完後，final unlearning checkpoint 會在：

```text
results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1/step5_forgot_13_8_58_90_48/RLcheckpoint.pth.tar
```

## 方法二：使用我提供的 Unlearning 後權重

如果不想重新跑 unlearning，我會用雲端提供這個 final unlearning checkpoint：

```text
RLcheckpoint.pth.tar
```

它對應的是：

```text
vehicles1_hardfirst_seed1_k5
step5_forgot_13_8_58_90_48
```

檔案大小約 `43 MB`。

下載後請放到：

```text
Classification/results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1/step5_forgot_13_8_58_90_48/RLcheckpoint.pth.tar
```

從 repo root 來看，位置應該像這樣：

```text
DLP_Final_Project_publish_20260424/
└── Classification/
    └── results/
        └── unlearn/
            └── sequential_single_coarse_rebound_cifar100/
                └── vehicles1_hardfirst_seed1_k5/
                    └── seed1/
                        └── step5_forgot_13_8_58_90_48/
                            └── RLcheckpoint.pth.tar
```

如果是從雲端連結下載，可以執行：

```bash
cd /path/to/DLP_Final_Project_publish_20260424/Classification
mkdir -p results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1/step5_forgot_13_8_58_90_48

# 將 MODEL_URL 換成雲端下載連結
wget -O results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1/step5_forgot_13_8_58_90_48/RLcheckpoint.pth.tar "MODEL_URL"
```

確認檔案存在：

```bash
ls -lh results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1/step5_forgot_13_8_58_90_48/RLcheckpoint.pth.tar
```

## 評估這個 Unlearning 後權重

如果使用我提供的 `RLcheckpoint.pth.tar`，不需要重跑 unlearning，只要直接 evaluate：

```bash
python scripts/evaluate_cumulative_forgetting.py \
  --arch resnet18 \
  --dataset cifar100 \
  --gpu 0 \
  --seed 1 \
  --batch_size 2048 \
  --model_path results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1/step5_forgot_13_8_58_90_48/RLcheckpoint.pth.tar \
  --forgotten_classes 13,8,58,90,48 \
  --output results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1_step5_forgot_13_8_58_90_48.csv
```

接著印出 bicycle accuracy：

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path("results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_hardfirst_seed1_k5/seed1_step5_forgot_13_8_58_90_48.csv")
with path.open(newline="") as handle:
    row = next(csv.DictReader(handle))

print("forget_accuracy:", row["forget_accuracy"])
print("retain_accuracy:", row["retain_accuracy"])
print("full_test_accuracy:", row["full_test_accuracy"])
print("bicycle_class_8_accuracy:", row["class_8_accuracy"])
PY
```

預期會接近：

```text
forget_accuracy: 9.6
retain_accuracy: 70.3
full_test_accuracy: 67.3
bicycle_class_8_accuracy: 48.0
```

## 補充

- 這份文件選的是 9 條 single-coarse runs 中問題最大的 `vehicles1_hardfirst_seed1_k5`。
- 問題大的原因是：`bicycle (8)` 已經在 step 2 被忘到 `1.0%`，但 final 又回升到 `48.0%`。
- 報告中的 per-class accuracy 使用 CIFAR-100 test split。
