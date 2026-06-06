# Reproduce CIFAR-100 Bicycle Rebound

This note explains how to reproduce one representative CIFAR-100 `vehicles_1` rebound run:

```text
vehicles1_official_seed1_k5
FORGET_ORDER = 8,13,48,58,90
classes = bicycle, bus, motorcycle, pickup_truck, train
```

The key result is that `bicycle / class 8` is forgotten at step 1, but its test accuracy rebounds after later vehicle classes are forgotten.

## Required Model File

The run starts from the CIFAR-100 ResNet-18 original model:

```text
0model_SA_best.pth.tar
```

Expected file info:

```text
size: about 86 MB
sha256: f57cde233ad7aba1cc0f07aa02d61ccc39b59e83c88b6e5811ba8c8b3a7371b3
```

Share this file through cloud storage, then place it under:

```text
Classification/results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

From the repository root, the final path should look like:

```text
DLP_Final_Project_publish_20260424/
└── Classification/
    └── results/
        └── original/
            └── cifar100_resnet18_seed1/
                └── 0model_SA_best.pth.tar
```

If downloading from a cloud URL:

```bash
cd /path/to/DLP_Final_Project_publish_20260424/Classification
mkdir -p results/original/cifar100_resnet18_seed1

# Replace MODEL_URL with the shared cloud download URL.
wget -O results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar "MODEL_URL"
```

Verify the model file:

```bash
sha256sum results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar
```

The hash should be:

```text
f57cde233ad7aba1cc0f07aa02d61ccc39b59e83c88b6e5811ba8c8b3a7371b3
```

## Environment

From `Classification/`:

```bash
cd /path/to/DLP_Final_Project_publish_20260424/Classification
bash scripts/setup_venv.sh
source .venv/bin/activate
```

CIFAR-100 will be downloaded automatically by the scripts if it is not already present.

## Run One Reproduction Experiment

This runs only the official `vehicles_1` order:

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

Meaning of the order:

| Step | Newly forgotten class |
| --- | --- |
| 1 | bicycle (8) |
| 2 | bus (13) |
| 3 | motorcycle (48) |
| 4 | pickup_truck (58) |
| 5 | train (90) |

## Check The Bicycle Rebound

After the run finishes, the eval CSV files should be under:

```text
results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/
```

Print the step-wise `bicycle / class_8_accuracy`:

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

Expected pattern from the original run:

| Step | Forgotten so far | bicycle accuracy |
| --- | --- | --- |
| 1 | 8 | about 0% |
| 2 | 8,13 | about 22% |
| 3 | 8,13,48 | about 29% |
| 4 | 8,13,48,58 | about 41% |
| 5 | 8,13,48,58,90 | about 43% |

Small numerical differences can happen across hardware or library versions, but the key pattern should remain: bicycle is suppressed at step 1 and rebounds after later vehicle unlearning steps.

## Output Locations

Useful files after the run:

```text
results/logs/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/progress.csv
results/eval/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/seed1_step*_forgot_*.csv
results/unlearn/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/seed1/step*_forgot_*/RLcheckpoint.pth.tar
results/masks/sequential_single_coarse_rebound_cifar100/vehicles1_official_seed1_k5/seed1/step*_forget_*/with_0.5.pt
```

## Notes

- This reproduces only one representative run, not the full 9-run single-coarse probe.
- The original report also ran `vehicles1_random_seed1_k5` and `vehicles1_hardfirst_seed1_k5`; both showed the same bicycle rebound pattern.
- The experiment uses CIFAR-100 test split for the reported per-class accuracy.
