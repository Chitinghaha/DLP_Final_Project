#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR100

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import model_dict
from utils import NormalizeByChannelMeanStd, setup_seed


SEED = 1
REBOUND_THRESHOLD = 10.0
REBOUND_DELTA = 5.0
CONTROL_NAMESPACE = "sequential_vehicle_rebound_cause_probe_cifar100"
REFERENCE_NAMESPACE = "sequential_single_coarse_rebound_cifar100"
VEHICLE_CLASSES = [8, 13, 48, 58, 90, 41, 69, 81, 85, 89]
FLOWER_CLASSES = [54, 62, 70, 82, 92]

RUN_SHORT_NAMES = {
    "vehicles1_official_seed1_k5": "bicycle -> vehicles1",
    "bicycle_then_flowers_seed1_k5": "bicycle -> flowers",
    "bicycle_then_vehicles2_seed1_k5": "bicycle -> vehicles2",
    "motorcycle_then_vehicles_no_bicycle_seed1_k5": "motorcycle early",
    "pickup_then_vehicles_no_bicycle_seed1_k5": "pickup early",
    "bicycle_stronger_then_vehicles1_seed1_k5": "bicycle stronger",
}


CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

ZH_LABELS = {
    8: "腳踏車",
    13: "公車",
    41: "割草機",
    48: "摩托車",
    54: "蘭花",
    58: "皮卡車",
    62: "罌粟花",
    69: "火箭",
    70: "玫瑰",
    81: "路面電車",
    82: "向日葵",
    85: "坦克",
    89: "拖拉機",
    90: "火車",
}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    namespace: str
    order: list[int]
    focus_class: int
    purpose: str
    expected_signal: str


RUNS = [
    RunConfig(
        "vehicles1_official_seed1_k5",
        REFERENCE_NAMESPACE,
        [8, 13, 48, 58, 90],
        8,
        "原始 positive control：bicycle 先忘，再忘 vehicles_1",
        "應重現 bicycle rebound，作為對照基準",
    ),
    RunConfig(
        "bicycle_then_flowers_seed1_k5",
        CONTROL_NAMESPACE,
        [8, 54, 62, 70, 82],
        8,
        "先忘 bicycle，再忘 flowers",
        "若不 rebound，支持 vehicle-specific update 假說",
    ),
    RunConfig(
        "bicycle_then_vehicles2_seed1_k5",
        CONTROL_NAMESPACE,
        [8, 41, 69, 81, 89],
        8,
        "先忘 bicycle，再忘 vehicles_2",
        "若 rebound，支持 shared vehicle representation 假說",
    ),
    RunConfig(
        "motorcycle_then_vehicles_no_bicycle_seed1_k5",
        CONTROL_NAMESPACE,
        [48, 13, 58, 90, 8],
        48,
        "讓 motorcycle 早忘，bicycle 最後忘",
        "若 motorcycle 仍不 rebound，排除單純 exposure 解釋",
    ),
    RunConfig(
        "pickup_then_vehicles_no_bicycle_seed1_k5",
        CONTROL_NAMESPACE,
        [58, 13, 48, 90, 8],
        58,
        "讓 pickup_truck 早忘，bicycle 最後忘",
        "若 pickup_truck 仍不 rebound，排除單純 exposure 解釋",
    ),
    RunConfig(
        "bicycle_stronger_then_vehicles1_seed1_k5",
        CONTROL_NAMESPACE,
        [8, 13, 48, 58, 90],
        8,
        "bicycle 先忘，UNLEARN_EPOCHS=20，再忘 vehicles_1",
        "若 top-5 降低且 rebound 消失，支持 forget strength 是直接原因",
    ),
]


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, target = self.dataset[index]
        return image, target, index


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--data", default="../data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=root / "results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "6_6_cifar100_bicycle_rebound_cause_probe_analysis.md",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "6_6_cifar100_bicycle_rebound_cause_probe_summary.csv",
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=root / "6_6_cifar100_bicycle_rebound_cause_probe_step_trajectory.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=root / "6_6_cifar100_bicycle_rebound_cause_probe_figures",
    )
    return parser.parse_args()


def label(class_id: int) -> str:
    zh = ZH_LABELS.get(class_id)
    en = CIFAR100_FINE_LABELS[class_id]
    return f"{zh} / {en} ({class_id})" if zh else f"{en} ({class_id})"


def short_name(run_id: str) -> str:
    return RUN_SHORT_NAMES.get(run_id, run_id)


def md_img(figures_dir: Path, filename: str) -> str:
    return f"![{filename}]({figures_dir.name}/{filename})"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def forgotten_slug(order: list[int], step: int) -> str:
    return "_".join(str(class_id) for class_id in order[:step])


def eval_path(root: Path, run: RunConfig, step: int) -> Path:
    return root / "results" / "eval" / run.namespace / run.run_id / f"seed{SEED}_step{step}_forgot_{forgotten_slug(run.order, step)}.csv"


def checkpoint_path(root: Path, run: RunConfig, step: int) -> Path:
    return (
        root
        / "results"
        / "unlearn"
        / run.namespace
        / run.run_id
        / f"seed{SEED}"
        / f"step{step}_forgot_{forgotten_slug(run.order, step)}"
        / "RLcheckpoint.pth.tar"
    )


def read_eval_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def fnum(row: dict | None, key: str) -> float | None:
    if row is None or row.get(key) in (None, ""):
        return None
    return float(row[key])


def load_model(path: Path, arch: str, device: torch.device):
    model = model_dict[arch](num_classes=100)
    model.normalize = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4866, 0.4409],
        std=[0.2673, 0.2564, 0.2762],
    )
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_loader(data_root: str, class_ids: list[int], batch_size: int):
    dataset = CIFAR100(
        data_root,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True,
    )
    wanted = set(class_ids)
    indices = [idx for idx, target in enumerate(dataset.targets) if target in wanted]
    subset = IndexedSubset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_checkpoint(model, loader, device: torch.device) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    features: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(torch.flatten(output.detach(), 1).cpu())

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        for images, targets, indices in loader:
            captured.clear()
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_ids = torch.topk(probs, k=5, dim=1)
            preds = top_ids[:, 0]
            target_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
            masked_logits = logits.clone()
            masked_logits.scatter_(1, targets.view(-1, 1), -float("inf"))
            max_other_logits, _ = masked_logits.max(dim=1)
            margins = target_logits - max_other_logits
            batch_features = captured[-1].numpy()
            features.append(batch_features)
            targets_out.append(targets.cpu().numpy())
            for row_idx in range(targets.size(0)):
                target = int(targets[row_idx].item())
                pred = int(preds[row_idx].item())
                top5 = [int(item) for item in top_ids[row_idx].cpu().tolist()]
                rows.append(
                    {
                        "test_index": int(indices[row_idx].item()),
                        "true_class_id": target,
                        "pred_class_id": pred,
                        "correct": pred == target,
                        "top5_contains_true": target in top5,
                        "margin": float(margins[row_idx].item()),
                    }
                )
    handle.remove()
    return rows, np.concatenate(features, axis=0), np.concatenate(targets_out, axis=0)


def summarize_focus(rows: list[dict], focus_class: int) -> dict:
    selected = [row for row in rows if row["true_class_id"] == focus_class]
    if not selected:
        return {"top1": None, "top5": None, "mean_margin": None}
    return {
        "top1": 100.0 * sum(row["correct"] for row in selected) / len(selected),
        "top5": 100.0 * sum(row["top5_contains_true"] for row in selected) / len(selected),
        "mean_margin": float(np.mean([row["margin"] for row in selected])),
    }


def recovered_count(at_forget: list[dict], final: list[dict], focus_class: int) -> int | None:
    before = {
        row["test_index"]: row
        for row in at_forget
        if row["true_class_id"] == focus_class
    }
    after = {
        row["test_index"]: row
        for row in final
        if row["true_class_id"] == focus_class
    }
    if not before or not after:
        return None
    return sum((not before[idx]["correct"]) and after[idx]["correct"] for idx in before if idx in after)


def nearest_vehicle_distance(features: np.ndarray, targets: np.ndarray, focus_class: int) -> float | None:
    if focus_class != 8 or not np.any(targets == focus_class):
        return None
    centroids = {
        class_id: features[targets == class_id].mean(axis=0)
        for class_id in VEHICLE_CLASSES
        if np.any(targets == class_id)
    }
    if focus_class not in centroids:
        return None
    bicycle = centroids[focus_class]
    distances = [
        float(np.linalg.norm(bicycle - centroid))
        for class_id, centroid in centroids.items()
        if class_id != focus_class
    ]
    return None if not distances else min(distances)


def collect_checkpoint_metrics(args: argparse.Namespace, run: RunConfig, device: torch.device) -> dict:
    focus_loader = build_loader(args.data, sorted(set([run.focus_class] + VEHICLE_CLASSES + FLOWER_CLASSES)), args.batch_size)
    forget_step = run.order.index(run.focus_class) + 1
    paths = {
        "forget": checkpoint_path(args.root, run, forget_step),
        "final": checkpoint_path(args.root, run, 5),
    }
    if not all(path.exists() for path in paths.values()):
        return {
            "checkpoint_metrics_available": False,
            "forget_top5": None,
            "forget_margin": None,
            "final_top5": None,
            "final_margin": None,
            "recovered_samples": None,
            "forget_nearest_vehicle_distance": None,
            "final_nearest_vehicle_distance": None,
        }

    model = load_model(paths["forget"], args.arch, device)
    forget_rows, forget_features, forget_targets = evaluate_checkpoint(model, focus_loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = load_model(paths["final"], args.arch, device)
    final_rows, final_features, final_targets = evaluate_checkpoint(model, focus_loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    forget_summary = summarize_focus(forget_rows, run.focus_class)
    final_summary = summarize_focus(final_rows, run.focus_class)
    return {
        "checkpoint_metrics_available": True,
        "forget_top5": forget_summary["top5"],
        "forget_margin": forget_summary["mean_margin"],
        "final_top5": final_summary["top5"],
        "final_margin": final_summary["mean_margin"],
        "recovered_samples": recovered_count(forget_rows, final_rows, run.focus_class),
        "forget_nearest_vehicle_distance": nearest_vehicle_distance(forget_features, forget_targets, run.focus_class),
        "final_nearest_vehicle_distance": nearest_vehicle_distance(final_features, final_targets, run.focus_class),
    }


def collect_run(args: argparse.Namespace, run: RunConfig, device: torch.device) -> dict:
    forget_step = run.order.index(run.focus_class) + 1
    eval_rows = [read_eval_row(eval_path(args.root, run, step)) for step in range(1, 6)]
    at_forget = eval_rows[forget_step - 1]
    final = eval_rows[4]
    focus_key = f"class_{run.focus_class}_accuracy"
    focus_at_forget = fnum(at_forget, focus_key)
    focus_final = fnum(final, focus_key)
    later_values = [
        fnum(row, focus_key)
        for row in eval_rows[forget_step:]
        if row is not None and fnum(row, focus_key) is not None
    ]
    max_later = None if not later_values else max(later_values)
    significant = (
        focus_at_forget is not None
        and max_later is not None
        and max_later >= REBOUND_THRESHOLD
        and (max_later - focus_at_forget) >= REBOUND_DELTA
    )
    out = {
        "run_id": run.run_id,
        "namespace": run.namespace,
        "order": ",".join(str(item) for item in run.order),
        "focus_class_id": run.focus_class,
        "focus_class": label(run.focus_class),
        "purpose": run.purpose,
        "expected_signal": run.expected_signal,
        "forget_step": forget_step,
        "eval_complete": all(row is not None for row in eval_rows),
        "focus_at_forget": focus_at_forget,
        "focus_max_later": max_later,
        "focus_final": focus_final,
        "significant_rebound": significant,
        "forget_accuracy_final": fnum(final, "forget_accuracy"),
        "retain_accuracy_final": fnum(final, "retain_accuracy"),
        "full_test_accuracy_final": fnum(final, "full_test_accuracy"),
    }
    if out["eval_complete"]:
        out.update(collect_checkpoint_metrics(args, run, device))
    else:
        out.update(
            {
                "checkpoint_metrics_available": False,
                "forget_top5": None,
                "forget_margin": None,
                "final_top5": None,
                "final_margin": None,
                "recovered_samples": None,
                "forget_nearest_vehicle_distance": None,
                "final_nearest_vehicle_distance": None,
            }
        )
    return out


def collect_step_trajectory(args: argparse.Namespace) -> list[dict]:
    rows = []
    for run in RUNS:
        focus_key = f"class_{run.focus_class}_accuracy"
        for step in range(1, 6):
            row = read_eval_row(eval_path(args.root, run, step))
            rows.append(
                {
                    "run_id": run.run_id,
                    "run_short_name": short_name(run.run_id),
                    "namespace": run.namespace,
                    "step": step,
                    "forgotten_classes": ",".join(str(item) for item in run.order[:step]),
                    "new_class_id": run.order[step - 1],
                    "new_class": label(run.order[step - 1]),
                    "focus_class_id": run.focus_class,
                    "focus_class": label(run.focus_class),
                    "focus_accuracy": fnum(row, focus_key),
                    "forget_accuracy": fnum(row, "forget_accuracy"),
                    "retain_accuracy": fnum(row, "retain_accuracy"),
                    "full_test_accuracy": fnum(row, "full_test_accuracy"),
                    "eval_available": row is not None,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def status_text(row: dict) -> str:
    if not row["eval_complete"]:
        return "pending"
    return "rebound" if row["significant_rebound"] else "no rebound"


def interpretation(rows: list[dict]) -> list[str]:
    by_id = {row["run_id"]: row for row in rows}
    lines = []
    flowers = by_id.get("bicycle_then_flowers_seed1_k5")
    vehicles2 = by_id.get("bicycle_then_vehicles2_seed1_k5")
    stronger = by_id.get("bicycle_stronger_then_vehicles1_seed1_k5")
    moto = by_id.get("motorcycle_then_vehicles_no_bicycle_seed1_k5")
    pickup = by_id.get("pickup_then_vehicles_no_bicycle_seed1_k5")

    if flowers and vehicles2 and flowers["eval_complete"] and vehicles2["eval_complete"]:
        if (not flowers["significant_rebound"]) and vehicles2["significant_rebound"]:
            lines.append("`bicycle_then_flowers` 不 rebound、`bicycle_then_vehicles2` rebound：支持 vehicle-specific shared representation。")
        elif flowers["significant_rebound"]:
            lines.append("`bicycle_then_flowers` 也 rebound：需要考慮 general sequential drift，而不是只看 vehicle-specific update。")
        else:
            lines.append("`bicycle_then_flowers` 和 `bicycle_then_vehicles2` 都不 rebound：原本 vehicles_1 的特定順序或類別組合可能更關鍵。")
    else:
        lines.append("vehicle-specific 控制實驗尚未全部完成，暫時不能判定 flowers vs vehicles_2 差異。")

    if moto and pickup and moto["eval_complete"] and pickup["eval_complete"]:
        if (not moto["significant_rebound"]) and (not pickup["significant_rebound"]):
            lines.append("motorcycle / pickup_truck 早忘仍不 rebound：支持 bicycle 特殊性不是單純 exposure。")
        else:
            lines.append("motorcycle 或 pickup_truck 早忘後也 rebound：需要修正「只有 bicycle 特殊」的結論。")
    else:
        lines.append("early-forget 控制實驗尚未全部完成，暫時不能完全排除 exposure effect。")

    if stronger and stronger["eval_complete"]:
        if stronger["forget_top5"] is not None and stronger["forget_top5"] <= 5.0 and not stronger["significant_rebound"]:
            lines.append("stronger forget 讓 bicycle top-5 接近 0 且 rebound 消失：支持 forget strength 是直接原因。")
        elif stronger["significant_rebound"]:
            lines.append("stronger forget 後 bicycle 仍 rebound：表示後續 vehicle boundary / feature drift 很強。")
        else:
            lines.append("stronger forget 後未達 significant rebound，但需搭配 top-5 / margin 判讀是否真的壓深。")
    else:
        lines.append("forget-strength 實驗尚未完成，暫時不能判定加強 bicycle forgetting 是否能消除 rebound。")
    return lines


def trajectory_for(trajectory_rows: list[dict], run_id: str) -> list[dict]:
    return [row for row in trajectory_rows if row["run_id"] == run_id]


def values_for(trajectory_rows: list[dict], run_id: str, key: str) -> list[float | None]:
    return [row[key] for row in trajectory_for(trajectory_rows, run_id)]


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_focus_accuracy_trajectory(trajectory_rows: list[dict], path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    for run in RUNS:
        ys = values_for(trajectory_rows, run.run_id, "focus_accuracy")
        plt.plot(range(1, 6), ys, marker="o", linewidth=2, label=short_name(run.run_id))
    plt.axhline(REBOUND_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="10% rebound threshold")
    plt.xlabel("Unlearning step")
    plt.ylabel("Focus class top-1 accuracy (%)")
    plt.title("Focus class accuracy trajectory")
    plt.xticks(range(1, 6))
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    save_figure(path)


def plot_bicycle_control_comparison(trajectory_rows: list[dict], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_id in [
        "vehicles1_official_seed1_k5",
        "bicycle_then_flowers_seed1_k5",
        "bicycle_then_vehicles2_seed1_k5",
    ]:
        ys = values_for(trajectory_rows, run_id, "focus_accuracy")
        plt.plot(range(1, 6), ys, marker="o", linewidth=2.4, label=short_name(run_id))
    plt.xlabel("Unlearning step")
    plt.ylabel("Bicycle top-1 accuracy (%)")
    plt.title("Bicycle rebound under different later classes")
    plt.xticks(range(1, 6))
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(path)


def plot_early_forget_negative_controls(trajectory_rows: list[dict], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_id in [
        "motorcycle_then_vehicles_no_bicycle_seed1_k5",
        "pickup_then_vehicles_no_bicycle_seed1_k5",
    ]:
        ys = values_for(trajectory_rows, run_id, "focus_accuracy")
        plt.plot(range(1, 6), ys, marker="o", linewidth=2.4, label=short_name(run_id))
    plt.axhline(REBOUND_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Unlearning step")
    plt.ylabel("Early-forgotten class top-1 accuracy (%)")
    plt.title("Early-forget negative controls")
    plt.xticks(range(1, 6))
    plt.ylim(bottom=0, top=15)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(path)


def plot_stronger_forget_check(rows: list[dict], trajectory_rows: list[dict], path: Path) -> None:
    by_id = {row["run_id"]: row for row in rows}
    run_ids = ["vehicles1_official_seed1_k5", "bicycle_stronger_then_vehicles1_seed1_k5"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for run_id in run_ids:
        axes[0].plot(
            range(1, 6),
            values_for(trajectory_rows, run_id, "focus_accuracy"),
            marker="o",
            linewidth=2.4,
            label=short_name(run_id),
        )
    axes[0].set_title("Top-1 trajectory")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Bicycle top-1 (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    x = np.arange(len(run_ids))
    axes[1].bar(x - 0.18, [by_id[r]["forget_top5"] for r in run_ids], width=0.36, label="at forget")
    axes[1].bar(x + 0.18, [by_id[r]["final_top5"] for r in run_ids], width=0.36, label="final")
    axes[1].set_title("Top-5 signal")
    axes[1].set_xticks(x, [short_name(r) for r in run_ids], rotation=18, ha="right")
    axes[1].set_ylabel("Top-5 contains true (%)")
    axes[1].legend(fontsize=8)

    axes[2].bar(x - 0.18, [by_id[r]["forget_margin"] for r in run_ids], width=0.36, label="at forget")
    axes[2].bar(x + 0.18, [by_id[r]["final_margin"] for r in run_ids], width=0.36, label="final")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Mean true-label margin")
    axes[2].set_xticks(x, [short_name(r) for r in run_ids], rotation=18, ha="right")
    axes[2].set_ylabel("Logit margin")
    axes[2].legend(fontsize=8)
    save_figure(path)


def plot_top5_margin_recovered_summary(rows: list[dict], path: Path) -> None:
    labels = [short_name(row["run_id"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(x, [row["forget_top5"] or 0 for row in rows], color="#4c78a8")
    axes[0].set_ylabel("Top-5 at forget (%)")
    axes[0].set_title("Soft vs hard forgetting diagnostics")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, [row["forget_margin"] or 0 for row in rows], color="#f58518")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Margin at forget")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, [row["recovered_samples"] or 0 for row in rows], color="#54a24b")
    axes[2].set_ylabel("Recovered samples")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    save_figure(path)


def plot_feature_distance_shift(rows: list[dict], path: Path) -> None:
    bike_rows = [row for row in rows if row["focus_class_id"] == 8 and row["forget_nearest_vehicle_distance"] is not None]
    labels = [short_name(row["run_id"]) for row in bike_rows]
    x = np.arange(len(bike_rows))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.18, [row["forget_nearest_vehicle_distance"] for row in bike_rows], width=0.36, label="at forget")
    plt.bar(x + 0.18, [row["final_nearest_vehicle_distance"] for row in bike_rows], width=0.36, label="final")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Nearest vehicle centroid distance")
    plt.title("Bicycle feature distance shift")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_figure(path)


def plot_final_metrics_by_control_run(rows: list[dict], path: Path) -> None:
    labels = [short_name(row["run_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar(x - width, [row["forget_accuracy_final"] or 0 for row in rows], width=width, label="forget acc")
    plt.bar(x, [row["retain_accuracy_final"] or 0 for row in rows], width=width, label="retain acc")
    plt.bar(x + width, [row["full_test_accuracy_final"] or 0 for row in rows], width=width, label="full test acc")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Final metrics by run")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_figure(path)


def make_figures(rows: list[dict], trajectory_rows: list[dict], figures_dir: Path) -> list[str]:
    figures = [
        ("focus_accuracy_trajectory_by_run.png", lambda p: plot_focus_accuracy_trajectory(trajectory_rows, p)),
        ("bicycle_control_comparison.png", lambda p: plot_bicycle_control_comparison(trajectory_rows, p)),
        ("early_forget_negative_controls.png", lambda p: plot_early_forget_negative_controls(trajectory_rows, p)),
        ("stronger_forget_check.png", lambda p: plot_stronger_forget_check(rows, trajectory_rows, p)),
        ("top5_margin_recovered_summary.png", lambda p: plot_top5_margin_recovered_summary(rows, p)),
        ("feature_distance_shift.png", lambda p: plot_feature_distance_shift(rows, p)),
        ("final_metrics_by_control_run.png", lambda p: plot_final_metrics_by_control_run(rows, p)),
    ]
    written = []
    for filename, plotter in figures:
        plotter(figures_dir / filename)
        written.append(filename)
    return written


def build_report(rows: list[dict], trajectory_rows: list[dict], figures_dir: Path) -> str:
    by_id = {row["run_id"]: row for row in rows}
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["run_id"],
                row["focus_class"],
                row["forget_step"],
                status_text(row),
                pct(row["focus_at_forget"]),
                pct(row["focus_max_later"]),
                pct(row["focus_final"]),
                pct(row["forget_top5"]),
                fmt(row["forget_margin"]),
                row["recovered_samples"] if row["recovered_samples"] is not None else "n/a",
                fmt(row["forget_nearest_vehicle_distance"]),
                fmt(row["final_nearest_vehicle_distance"]),
            ]
        )

    final_metric_rows = [
        [
            row["run_id"],
            status_text(row),
            pct(row["forget_accuracy_final"]),
            pct(row["retain_accuracy_final"]),
            pct(row["full_test_accuracy_final"]),
        ]
        for row in rows
    ]
    trajectory_table_rows = [
        [
            row["run_short_name"],
            row["step"],
            row["new_class"],
            pct(row["focus_accuracy"]),
            pct(row["forget_accuracy"]),
            pct(row["retain_accuracy"]),
            pct(row["full_test_accuracy"]),
        ]
        for row in trajectory_rows
    ]
    bike_control_rows = [
        [
            short_name(run_id),
            pct(by_id[run_id]["focus_at_forget"]),
            pct(by_id[run_id]["focus_max_later"]),
            pct(by_id[run_id]["focus_final"]),
            pct(by_id[run_id]["forget_top5"]),
            fmt(by_id[run_id]["forget_margin"]),
            by_id[run_id]["recovered_samples"],
        ]
        for run_id in [
            "vehicles1_official_seed1_k5",
            "bicycle_then_flowers_seed1_k5",
            "bicycle_then_vehicles2_seed1_k5",
        ]
    ]
    negative_control_rows = [
        [
            short_name(run_id),
            by_id[run_id]["focus_class"],
            pct(by_id[run_id]["focus_at_forget"]),
            pct(by_id[run_id]["focus_max_later"]),
            pct(by_id[run_id]["focus_final"]),
            pct(by_id[run_id]["forget_top5"]),
            fmt(by_id[run_id]["forget_margin"]),
            by_id[run_id]["recovered_samples"],
        ]
        for run_id in [
            "motorcycle_then_vehicles_no_bicycle_seed1_k5",
            "pickup_then_vehicles_no_bicycle_seed1_k5",
        ]
    ]
    stronger_rows = [
        [
            short_name(run_id),
            pct(by_id[run_id]["focus_at_forget"]),
            pct(by_id[run_id]["focus_max_later"]),
            pct(by_id[run_id]["focus_final"]),
            pct(by_id[run_id]["forget_top5"]),
            pct(by_id[run_id]["final_top5"]),
            fmt(by_id[run_id]["forget_margin"]),
            fmt(by_id[run_id]["final_margin"]),
        ]
        for run_id in [
            "vehicles1_official_seed1_k5",
            "bicycle_stronger_then_vehicles1_seed1_k5",
        ]
    ]
    feature_rows = [
        [
            short_name(row["run_id"]),
            fmt(row["forget_nearest_vehicle_distance"]),
            fmt(row["final_nearest_vehicle_distance"]),
        ]
        for row in rows
        if row["focus_class_id"] == 8 and row["forget_nearest_vehicle_distance"] is not None
    ]

    sections = [
        "# 6_6 CIFAR-100 Bicycle Rebound Cause Probe Analysis",
        "",
        "## 實驗問題與判讀標準",
        "",
        "這份報告彙整 5 條控制實驗與 1 條原始 positive control，目標是判斷 `bicycle / 腳踏車 (8)` 的 post-forget rebound 比較像哪一種原因：vehicle-specific shared representation、一般 sequential drift、單純早忘 exposure，或 forgetting strength 不足。",
        "",
        f"本報告使用同一個 significant rebound 定義：later top-1 accuracy 至少 `{REBOUND_THRESHOLD:.0f}%`，且比 forget-step top-1 高至少 `{REBOUND_DELTA:.0f}pp`。所有 accuracy 都來自 CIFAR-100 test split。",
        "",
        markdown_table(
            ["Run ID", "Order", "Purpose", "Expected signal"],
            [[row["run_id"], row["order"], row["purpose"], row["expected_signal"]] for row in rows],
        ),
        "",
        "**本章結論：**這組控制實驗可以直接分辨三件事：flowers 後是否也 rebound、其他車輛類別早忘是否也 rebound、以及單純增加 `UNLEARN_EPOCHS` 是否能讓 bicycle 被忘得更深。",
        "",
        "## 整體結果總覽",
        "",
        md_img(figures_dir, "focus_accuracy_trajectory_by_run.png"),
        "",
        markdown_table(
            [
                "Run ID",
                "Focus class",
                "Forget step",
                "Status",
                "At forget top-1",
                "Max later top-1",
                "Final top-1",
                "At forget top-5",
                "At forget margin",
                "Recovered samples",
                "Forget nearest vehicle dist",
                "Final nearest vehicle dist",
            ],
            table_rows,
        ),
        "",
        md_img(figures_dir, "final_metrics_by_control_run.png"),
        "",
        markdown_table(
            ["Run ID", "Status", "Final forget acc", "Final retain acc", "Final full test acc"],
            final_metric_rows,
        ),
        "",
        "**本章結論：**6 條 run 裡，只有 bicycle 作為 focus class 時出現 rebound；motorcycle 與 pickup 作為 early-forgotten focus class 時都沒有 rebound。final retain accuracy 都維持在約 `69.9%~70.3%`，代表這些現象不是整個模型崩壞造成。",
        "",
        "## Bicycle 後續接 flowers vs vehicles_2",
        "",
        md_img(figures_dir, "bicycle_control_comparison.png"),
        "",
        markdown_table(
            ["Run", "At forget top-1", "Max later top-1", "Final top-1", "At forget top-5", "At forget margin", "Recovered samples"],
            bike_control_rows,
        ),
        "",
        "`bicycle_then_flowers` 的 bicycle 從 `0.0%` 回到 final `36.0%`，`bicycle_then_vehicles2` 從 `0.0%` 回到 final `27.0%`。這兩條都符合 significant rebound。原本 `vehicles1_official` 從 `0.0%` 回到 `43.0%`，仍是最高，但 flowers 也能造成明顯回升。",
        "",
        "**本章結論：**bicycle rebound 不能被簡化成「只有後續忘 vehicle 類別才會發生」。flowers 後也回到 `36.0%`，所以更合理的解釋是 bicycle 本身被忘得較淺，後續 sequential update 即使不是 vehicle-specific，也可能讓 decision boundary 回漂。",
        "",
        "## Early-forget negative controls：motorcycle / pickup",
        "",
        md_img(figures_dir, "early_forget_negative_controls.png"),
        "",
        markdown_table(
            ["Run", "Focus class", "At forget top-1", "Max later top-1", "Final top-1", "At forget top-5", "At forget margin", "Recovered samples"],
            negative_control_rows,
        ),
        "",
        "motorcycle 早忘後 top-1 從 `0.0%` 到 final 仍是 `0.0%`，pickup_truck 早忘後也是 `0.0%`。兩者 forget-step top-5 都是 `0.0%`，margin 分別是 `-8.56` 與 `-6.29`，比 bicycle 的 `-2.35` 更負很多。",
        "",
        "**本章結論：**「比較早被忘、有更多 later steps」不是 rebound 的充分條件。motorcycle / pickup 都早忘但完全不 rebound，支持 bicycle 的特殊性在於 soft-forgotten signal，而不是 exposure 長短。",
        "",
        "## Stronger forgetting 是否有效",
        "",
        md_img(figures_dir, "stronger_forget_check.png"),
        "",
        markdown_table(
            ["Run", "At forget top-1", "Max later top-1", "Final top-1", "At forget top-5", "Final top-5", "At forget margin", "Final margin"],
            stronger_rows,
        ),
        "",
        "`bicycle_stronger_then_vehicles1` 使用 `UNLEARN_EPOCHS=20`，但 bicycle forget-step top-1 反而是 `14.0%`，top-5 是 `80.0%`，比原始 official run 的 top-1 `0.0%`、top-5 `35.0%` 更不像 deeper forgetting。final bicycle top-1 仍有 `52.0%`。",
        "",
        "**本章結論：**單純把 unlearn epochs 從 `10` 加到 `20` 沒有讓 bicycle 忘得更深，反而保留更強 top-k signal。若要測試 forgetting strength，下一步應該改 `unlearn_lr`、mask threshold、loss weighting 或更直接的 class suppression，而不是只加 epoch。",
        "",
        "## Top-5、margin、recovered samples 分析",
        "",
        md_img(figures_dir, "top5_margin_recovered_summary.png"),
        "",
        markdown_table(
            [
                "Run ID",
                "Focus class",
                "Status",
                "At forget top-1",
                "At forget top-5",
                "At forget margin",
                "Recovered samples",
            ],
            [
                [
                    row["run_id"],
                    row["focus_class"],
                    status_text(row),
                    pct(row["focus_at_forget"]),
                    pct(row["forget_top5"]),
                    fmt(row["forget_margin"]),
                    row["recovered_samples"] if row["recovered_samples"] is not None else "n/a",
                ]
                for row in rows
            ],
        ),
        "",
        "原始 bicycle official run 在 forget step 已經 top-1 `0.0%`，但 top-5 還有 `35.0%`；flowers / vehicles2 run 也是同一個 step1 checkpoint，所以 top-5 同樣是 `35.0%`。相對地，motorcycle / pickup 的 forget-step top-5 都是 `0.0%`，且 recovered samples 都是 `0`。",
        "",
        "**本章結論：**bicycle rebound 的關鍵證據是 soft-forgotten：top-1 已被壓到 `0.0%`，但 top-5 仍有 `35.0%`，margin 也只到 `-2.35`。motorcycle / pickup 則是 hard-forgotten，top-5 `0.0%` 且 margin 更負，因此後續 update 很難把它們拉回 top-1。",
        "",
        "## Feature centroid distance 分析",
        "",
        md_img(figures_dir, "feature_distance_shift.png"),
        "",
        markdown_table(
            ["Run", "Forget-step nearest vehicle dist", "Final nearest vehicle dist"],
            feature_rows,
        ),
        "",
        "bicycle official run 的 nearest vehicle centroid distance 從 forget step `11.69` 降到 final `4.86`；bicycle_then_vehicles2 從 `11.69` 降到 `4.27`；bicycle_then_flowers final 則是 `9.45`，靠近幅度較小但仍然 rebound。stronger run 從 `13.44` 降到 `4.59`。",
        "",
        "**本章結論：**feature distance 支持「後續 steps 會改變 bicycle 的 representation / boundary 狀態」，尤其 vehicles1 / vehicles2 / stronger run final 都靠近 vehicle centroid。不過 flowers run final distance 仍較遠卻也 rebound，因此 feature centroid distance 不能單獨解釋全部現象，必須和 top-5、margin 一起看。",
        "",
        "## Step-by-step trajectory 明細",
        "",
        markdown_table(
            ["Run", "Step", "New class", "Focus acc", "Forget acc", "Retain acc", "Full test acc"],
            trajectory_table_rows,
        ),
        "",
        "**本章結論：**逐步數據顯示 bicycle rebound 在 step2 就出現：official `22.0%`、flowers `36.0%`、vehicles2 `18.0%`。motorcycle / pickup 則從 step1 到 step5 都維持 `0.0%`。",
        "",
        "## 最終結論與限制",
        "",
        "目前控制實驗支持以下判斷：",
        "",
    ]
    sections.extend(f"- {line}" for line in interpretation(rows))
    sections.extend(
        [
            "- bicycle rebound 的主因目前更像是 bicycle 本身的 soft-forgotten / boundary sensitivity，而不是純 vehicle-specific shared representation。",
            "- flowers 後也 rebound，代表 general sequential drift 必須納入解釋。",
            "- motorcycle / pickup 早忘後不 rebound，代表 exposure 不是充分條件。",
            "- `UNLEARN_EPOCHS=20` 沒有形成 stronger forgetting，因此不能用這條 run 證明「更強 forgetting 會消除 rebound」。",
            "",
            "**本章結論：**目前最可信的說法是：bicycle 在 forget step 被壓掉 top-1，但仍保留 top-5 candidate signal 與較不負的 margin；後續 sequential updates 會讓邊界回漂，因此 bicycle 比其他 early-forgotten classes 更容易回到 top-1。這是 seed=1 的機制分析，不應誇大成跨 seed 結論。",
            "",
            "## Notes",
            "",
            f"- Significant rebound 定義：later top-1 >= `{REBOUND_THRESHOLD:.0f}%` 且比 forget-step top-1 高至少 `{REBOUND_DELTA:.0f}pp`。",
            "- Top-5、margin、recovered samples、feature distance 使用 CIFAR-100 test split 與對應 checkpoints 計算。",
            "- Feature centroid distance 只比較交通工具相關類別，不代表 CIFAR-100 全部 100 類的 feature geometry。",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    rows = [collect_run(args, run, device) for run in RUNS]
    trajectory_rows = collect_step_trajectory(args)
    figure_files = make_figures(rows, trajectory_rows, args.figures_dir)
    write_csv(args.summary_csv, rows)
    write_csv(args.trajectory_csv, trajectory_rows)
    args.report_path.write_text(build_report(rows, trajectory_rows, args.figures_dir), encoding="utf-8")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.trajectory_csv}")
    for filename in figure_files:
        print(f"Wrote {args.figures_dir / filename}")


if __name__ == "__main__":
    main()
