# CIFAR-100 Original Model Training Note

## 目標權重

本文說明下列 CIFAR-100 original baseline 權重是如何產生的：

```text
results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

這個權重是 CIFAR-100 / ResNet-18 / seed 1 的原始分類模型 baseline，不是 unlearning 之後的 checkpoint。後續 CIFAR-100 sequential / incremental unlearning 實驗會以它作為起始模型。

## 產生流程

權重由 queue 腳本自動補訓練產生：

```text
scripts/run_incremental_ordered_queue.sh
```

在 `ensure_original_model()` 中，腳本會檢查下列三個檔案是否都存在：

```text
results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
results/original/cifar100_resnet18_seed1/0checkpoint.pth.tar
results/original/cifar100_resnet18_seed1/0net_train.png
```

如果缺少其中任何一個，腳本就會先執行 `main_train.py` 訓練 original model。這次實際完成訓練的 queue log 是：

```text
results/logs/sequential_incremental_ordered_cifar100_pilot/ordered_queue_cifar100_20260425_152810/queue.log
```

其中紀錄：

```text
[2026-04-25 15:28:11] START train_original dataset=cifar100 seed=1 gpu=0 save_dir=results/original/cifar100_resnet18_seed1
[2026-04-25 16:43:36] DONE train_original dataset=cifar100 seed=1 gpu=0
```

## 實際訓練指令

queue 腳本呼叫的訓練指令等價於：

```bash
cd Unlearn-Saliency-original/Classification
source .venv/bin/activate

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

完整訓練 log 位於：

```text
results/logs/original/train_cifar100_seed1_182.log
```

## 訓練設定

主要設定如下：

| 項目 | 設定 |
| --- | --- |
| Dataset | CIFAR-100 |
| Model | ResNet-18 |
| Seed | 1 |
| Epochs | 182 |
| Batch size | 256 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| LR scheduler | MultiStepLR |
| LR milestones | 91, 136 |
| LR gamma | 0.1 |
| Loss | CrossEntropyLoss |
| GPU | 0 |

資料前處理與切分：

| 項目 | 設定 |
| --- | --- |
| Train augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor |
| Test / validation transform | ToTensor |
| Normalization | 放在 model 的 normalize layer 裡 |
| CIFAR-100 mean | [0.5071, 0.4866, 0.4409] |
| CIFAR-100 std | [0.2673, 0.2564, 0.2762] |
| Training images | 45000 |
| Validation images | 5000 |
| Test images | 10000 |

Validation set 是從 CIFAR-100 training set 中，每個 class 取 10% 樣本形成。其餘 90% 作為 training set。

## Checkpoint 儲存邏輯

訓練流程在 `main_train.py` 中每個 epoch 都會：

1. 用 training loader 訓練一個 epoch。
2. 用 validation loader 計算 validation accuracy。
3. 更新 learning rate scheduler。
4. 儲存目前 epoch 的 checkpoint：

```text
results/original/cifar100_resnet18_seed1/0checkpoint.pth.tar
```

若目前 validation accuracy 高於歷史最佳 `best_sa`，則會把當前 checkpoint 複製成：

```text
results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

因此：

| 檔案 | 意義 |
| --- | --- |
| `0checkpoint.pth.tar` | 最後一次 epoch 的 checkpoint |
| `0model_SA_best.pth.tar` | validation accuracy 最佳 epoch 的 checkpoint |
| `0net_train.png` | train / validation accuracy 曲線 |

## 權重內容確認

直接讀取目標 checkpoint 後，裡面記錄：

```text
keys: ['best_sa', 'epoch', 'optimizer', 'result', 'scheduler', 'state_dict']
epoch: 135
best_sa: 69.8999997314453
```

因此 `0model_SA_best.pth.tar` 對應的是訓練過程中第 135 個 epoch 的最佳 validation accuracy 模型。

訓練最後仍繼續跑到 182 epochs，但最終 epoch 的 validation accuracy 沒有超過第 135 epoch 的最佳值，所以 best model 停留在 epoch 135。

## Log 中的結果摘要

訓練最後幾行顯示：

```text
Performance on the test data set
valid_accuracy 69.820
* best SA = 69.8999997314453, Epoch = 135
```

注意：`main_train.py` 最後印出的「Performance on the test data set」實際仍呼叫 `validate(val_loader, ...)`，所以這裡的 `valid_accuracy 69.820` 是 validation loader 上的結果，不是獨立 test set 的結果。

## 和後續 unlearning 實驗的關係

後續 CIFAR-100 incremental ordered pilot 會把這個 original model 當作起始模型，例如：

```text
--model_path results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

相關實驗 namespace：

```text
results/logs/sequential_incremental_ordered_cifar100_pilot/
results/unlearn/sequential_incremental_ordered_cifar100_pilot/
```

