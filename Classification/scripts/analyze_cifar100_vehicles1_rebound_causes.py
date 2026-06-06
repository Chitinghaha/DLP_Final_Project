#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.datasets import CIFAR100

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import model_dict
from utils import NormalizeByChannelMeanStd, setup_seed


BASE_NAMESPACE = "sequential_single_coarse_rebound_cifar100"
SEED = 1
MASK_RATIO = "0.5"
VEHICLES1_CLASSES = [8, 13, 48, 58, 90]
VEHICLES2_CLASSES = [41, 69, 81, 85, 89]
PCA_CLASSES = VEHICLES1_CLASSES + VEHICLES2_CLASSES


CIFAR100_FINE_LABELS = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

ZH_LABELS = {
    8: "腳踏車",
    13: "公車",
    41: "割草機",
    48: "摩托車",
    58: "皮卡車",
    69: "火箭",
    81: "路面電車",
    85: "坦克",
    89: "拖拉機",
    90: "火車",
}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    order_type: str
    order: list[int]


RUNS = [
    RunConfig("vehicles1_official_seed1_k5", "official", [8, 13, 48, 58, 90]),
    RunConfig("vehicles1_random_seed1_k5", "random", [48, 8, 90, 13, 58]),
    RunConfig("vehicles1_hardfirst_seed1_k5", "hardfirst", [13, 8, 58, 90, 48]),
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
    parser.add_argument("--base-namespace", default=BASE_NAMESPACE)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--data", default="../data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=root / "results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "6_6_cifar100_vehicles1_rebound_cause_analysis.md",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "6_6_cifar100_vehicles1_rebound_cause_figures",
    )
    parser.add_argument(
        "--per-sample-csv",
        type=Path,
        default=root / "6_6_cifar100_vehicles1_rebound_per_sample_predictions.csv",
    )
    parser.add_argument(
        "--class-summary-csv",
        type=Path,
        default=root / "6_6_cifar100_vehicles1_rebound_class_summary.csv",
    )
    return parser.parse_args()


def label(class_id: int) -> str:
    zh = ZH_LABELS.get(class_id)
    en = CIFAR100_FINE_LABELS[class_id]
    return f"{zh} / {en} ({class_id})" if zh else f"{en} ({class_id})"


def short_label(class_id: int) -> str:
    return f"{CIFAR100_FINE_LABELS[class_id]} ({class_id})"


def order_zh(order_type: str) -> str:
    return {"official": "官方順序", "random": "隨機順序", "hardfirst": "困難優先"}[order_type]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def step_checkpoint_path(root: Path, base_namespace: str, run: RunConfig, step: int) -> Path:
    forgotten = "_".join(str(class_id) for class_id in run.order[:step])
    return (
        root
        / "results"
        / "unlearn"
        / base_namespace
        / run.run_id
        / f"seed{SEED}"
        / f"step{step}_forgot_{forgotten}"
        / "RLcheckpoint.pth.tar"
    )


def mask_path(root: Path, base_namespace: str, run: RunConfig, step: int) -> Path:
    new_class = run.order[step - 1]
    return (
        root
        / "results"
        / "masks"
        / base_namespace
        / run.run_id
        / f"seed{SEED}"
        / f"step{step}_forget_{new_class}"
        / f"with_{MASK_RATIO}.pt"
    )


def checkpoints(root: Path, base_namespace: str, run: RunConfig, baseline: Path) -> list[tuple[int, str, Path]]:
    items = [(0, "baseline", baseline)]
    for step in range(1, len(run.order) + 1):
        items.append((step, f"step{step}", step_checkpoint_path(root, base_namespace, run, step)))
    return items


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
    indices = [idx for idx, target in enumerate(dataset.targets) if target in set(class_ids)]
    subset = IndexedSubset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, dataset


def evaluate_checkpoint(model, loader, device: torch.device) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    features: list[np.ndarray] = []
    feature_targets: list[np.ndarray] = []
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
            max_other_logits, max_other_ids = masked_logits.max(dim=1)
            margins = target_logits - max_other_logits

            batch_features = captured[-1].numpy()
            features.append(batch_features)
            feature_targets.append(targets.cpu().numpy())
            fc_weight = model.fc.weight.detach()
            fc_bias = model.fc.bias.detach() if model.fc.bias is not None else torch.zeros(fc_weight.size(0), device=device)
            target_weights = fc_weight[targets]
            true_feature_dots = torch.sum(captured[-1].to(device) * target_weights, dim=1)
            true_biases = fc_bias[targets]

            for row_idx in range(targets.size(0)):
                target = int(targets[row_idx].item())
                pred = int(preds[row_idx].item())
                best_other = int(max_other_ids[row_idx].item())
                top5 = [int(item) for item in top_ids[row_idx].cpu().tolist()]
                top5_probs = [float(item) for item in top_probs[row_idx].cpu().tolist()]
                true_rank = next((idx + 1 for idx, item in enumerate(top5) if item == target), None)
                rows.append(
                    {
                        "test_index": int(indices[row_idx].item()),
                        "true_class_id": target,
                        "true_class": label(target),
                        "pred_class_id": pred,
                        "pred_class": label(pred),
                        "correct": pred == target,
                        "true_logit": float(target_logits[row_idx].item()),
                        "true_feature_dot": float(true_feature_dots[row_idx].item()),
                        "true_bias": float(true_biases[row_idx].item()),
                        "top1_logit": float(logits[row_idx, pred].item()),
                        "top1_confidence": float(top_probs[row_idx, 0].item()),
                        "best_other_class_id": best_other,
                        "best_other_class": label(best_other),
                        "best_other_logit": float(max_other_logits[row_idx].item()),
                        "margin": float(margins[row_idx].item()),
                        "top5_contains_true": true_rank is not None,
                        "true_rank_in_top5": "" if true_rank is None else true_rank,
                        "top5_classes": ",".join(str(item) for item in top5),
                        "top5_probabilities": ",".join(f"{item:.6f}" for item in top5_probs),
                    }
                )
    handle.remove()
    return rows, np.concatenate(features, axis=0), np.concatenate(feature_targets, axis=0)


def summarize_class(rows: list[dict], class_id: int) -> dict:
    selected = [row for row in rows if row["true_class_id"] == class_id]
    if not selected:
        return {
            "accuracy": None,
            "top5_accuracy": None,
            "mean_margin": None,
            "median_margin": None,
            "mean_true_logit": None,
            "mean_true_feature_dot": None,
            "mean_true_bias": None,
            "mean_best_other_logit": None,
            "mean_top1_confidence": None,
        }
    return {
        "accuracy": 100.0 * sum(row["correct"] for row in selected) / len(selected),
        "top5_accuracy": 100.0 * sum(row["top5_contains_true"] for row in selected) / len(selected),
        "mean_margin": float(np.mean([row["margin"] for row in selected])),
        "median_margin": float(np.median([row["margin"] for row in selected])),
        "mean_true_logit": float(np.mean([row["true_logit"] for row in selected])),
        "mean_true_feature_dot": float(np.mean([row["true_feature_dot"] for row in selected])),
        "mean_true_bias": float(np.mean([row["true_bias"] for row in selected])),
        "mean_best_other_logit": float(np.mean([row["best_other_logit"] for row in selected])),
        "mean_top1_confidence": float(np.mean([row["top1_confidence"] for row in selected])),
    }


def collect_analysis(args: argparse.Namespace, device: torch.device) -> dict:
    loader, _dataset = build_loader(args.data, PCA_CLASSES, args.batch_size)
    per_sample_rows: list[dict] = []
    class_summary_rows: list[dict] = []
    feature_sets: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    eval_by_run_step: dict[tuple[str, int], list[dict]] = {}

    for run in RUNS:
        for step, step_label, checkpoint_path in checkpoints(args.root, args.base_namespace, run, args.baseline_model):
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            model = load_model(checkpoint_path, args.arch, device)
            rows, features, targets = evaluate_checkpoint(model, loader, device)
            eval_by_run_step[(run.run_id, step)] = rows
            feature_sets[(run.run_id, step)] = (features, targets)
            for row in rows:
                if row["true_class_id"] in VEHICLES1_CLASSES:
                    per_sample_rows.append(
                        {
                            "run_id": run.run_id,
                            "order_type": run.order_type,
                            "step": step,
                            "step_label": step_label,
                            "checkpoint_path": str(checkpoint_path),
                            **row,
                        }
                    )
            for class_id in VEHICLES1_CLASSES:
                summary = summarize_class(rows, class_id)
                class_summary_rows.append(
                    {
                        "run_id": run.run_id,
                        "order_type": run.order_type,
                        "step": step,
                        "step_label": step_label,
                        "class_id": class_id,
                        "class": label(class_id),
                        **summary,
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "per_sample_rows": per_sample_rows,
        "class_summary_rows": class_summary_rows,
        "feature_sets": feature_sets,
        "eval_by_run_step": eval_by_run_step,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_clean_figure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.png"):
        old.unlink()


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def class_summary_lookup(class_rows: list[dict]) -> dict[tuple[str, int, int], dict]:
    return {(row["run_id"], int(row["step"]), int(row["class_id"])): row for row in class_rows}


def plot_bicycle_margin(class_rows: list[dict], figure_dir: Path) -> None:
    lookup = class_summary_lookup(class_rows)
    plt.figure(figsize=(8.5, 5))
    for run in RUNS:
        steps = list(range(0, 6))
        margins = [lookup[(run.run_id, step, 8)]["mean_margin"] for step in steps]
        accuracies = [lookup[(run.run_id, step, 8)]["accuracy"] for step in steps]
        plt.plot(steps, margins, marker="o", linewidth=2, label=f"{run.order_type} margin")
        for step, margin, acc in zip(steps, margins, accuracies):
            if step == run.order.index(8) + 1 or step == 5:
                plt.text(step, margin, f"{acc:.0f}%", fontsize=7, ha="center", va="bottom")
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.xticks(range(0, 6))
    plt.xlabel("Step")
    plt.ylabel("Mean true-vs-best-other logit margin")
    plt.title("Bicycle margin trajectory")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    savefig(figure_dir / "bicycle_margin_trajectory.png")


def plot_topk(class_rows: list[dict], figure_dir: Path) -> None:
    lookup = class_summary_lookup(class_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for axis, run in zip(axes, RUNS):
        steps = list(range(0, 6))
        top1 = [lookup[(run.run_id, step, 8)]["accuracy"] for step in steps]
        top5 = [lookup[(run.run_id, step, 8)]["top5_accuracy"] for step in steps]
        axis.plot(steps, top1, marker="o", label="top-1")
        axis.plot(steps, top5, marker="o", label="top-5")
        axis.axvline(run.order.index(8) + 1, color="#888888", linestyle="--", linewidth=1)
        axis.set_title(run.order_type)
        axis.set_xticks(steps)
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Bicycle test accuracy (%)")
    axes[-1].legend()
    fig.suptitle("Bicycle top-1 vs top-5 recovery")
    savefig(figure_dir / "bicycle_topk_recovery.png")


def plot_confusion(eval_by_run_step: dict[tuple[str, int], list[dict]], figure_dir: Path) -> None:
    cols = VEHICLES1_CLASSES + [-1]
    col_labels = [short_label(class_id) for class_id in VEHICLES1_CLASSES] + ["other"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, run in zip(axes, RUNS):
        rows = eval_by_run_step[(run.run_id, 5)]
        matrix = np.zeros((len(VEHICLES1_CLASSES), len(cols)), dtype=int)
        for row in rows:
            if row["true_class_id"] not in VEHICLES1_CLASSES:
                continue
            r = VEHICLES1_CLASSES.index(row["true_class_id"])
            pred = row["pred_class_id"]
            c = VEHICLES1_CLASSES.index(pred) if pred in VEHICLES1_CLASSES else len(cols) - 1
            matrix[r, c] += 1
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        axis.set_title(run.order_type)
        axis.set_xticks(range(len(cols)), col_labels, rotation=35, ha="right", fontsize=7)
        axis.set_yticks(range(len(VEHICLES1_CLASSES)), [short_label(c) for c in VEHICLES1_CLASSES], fontsize=8)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                axis.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="test samples")
    fig.suptitle("vehicles_1 final confusion: predicted vehicles_1 class or other")
    savefig(figure_dir / "vehicles1_final_confusion.png")


def plot_pca(feature_sets: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], figure_dir: Path) -> dict:
    run = RUNS[0]
    bicycle_forget_step = run.order.index(8) + 1
    timeline_steps = [("baseline", 0), ("bicycle forget step", bicycle_forget_step), ("final", 5)]
    timeline_features = [feature_sets[(run.run_id, step)][0] for _label, step in timeline_steps]
    timeline_targets = [feature_sets[(run.run_id, step)][1] for _label, step in timeline_steps]
    combined = np.vstack(timeline_features)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(combined)
    split_points = np.cumsum([len(features) for features in timeline_features])
    coord_slices = np.split(coords, split_points[:-1])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(PCA_CLASSES)))
    for axis, (step_label, _step), step_coords, targets in zip(axes, timeline_steps, coord_slices, timeline_targets):
        for color, class_id in zip(colors, PCA_CLASSES):
            mask = targets == class_id
            axis.scatter(
                step_coords[mask, 0],
                step_coords[mask, 1],
                s=10,
                alpha=0.58,
                color=color,
                label=short_label(class_id),
            )
        axis.set_title(step_label)
        axis.set_xlabel("PC1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("PC2")
    axes[-1].legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Penultimate feature PCA timeline, official run")
    savefig(figure_dir / "feature_pca_official_bicycle_timeline.png")

    centroid_rows = []
    for run in RUNS:
        bicycle_forget_step = run.order.index(8) + 1
        for step_label, step in [("baseline", 0), ("bicycle forget step", bicycle_forget_step), ("final", 5)]:
            features, targets = feature_sets[(run.run_id, step)]
            centroids = {
                class_id: features[targets == class_id].mean(axis=0)
                for class_id in PCA_CLASSES
                if np.any(targets == class_id)
            }
            bicycle = centroids[8]
            for class_id in PCA_CLASSES:
                if class_id == 8:
                    continue
                centroid_rows.append(
                    {
                        "run_id": run.run_id,
                        "order_type": run.order_type,
                        "step": step_label,
                        "class_id": class_id,
                        "class": label(class_id),
                        "distance_from_bicycle": float(np.linalg.norm(bicycle - centroids[class_id])),
                    }
                )
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "centroid_rows": centroid_rows,
    }


def mask_jaccard(mask_a: dict[str, torch.Tensor], mask_b: dict[str, torch.Tensor]) -> float | None:
    keys = sorted(set(mask_a).intersection(mask_b))
    if not keys:
        return None
    inter = 0
    union = 0
    for key in keys:
        a = mask_a[key].bool().view(-1)
        b = mask_b[key].bool().view(-1)
        inter += int((a & b).sum().item())
        union += int((a | b).sum().item())
    return None if union == 0 else inter / union


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.norm(a).item() * torch.norm(b).item())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a.flatten().float(), b.flatten().float()).item() / denom)


def classifier_weight_rows(root: Path, base_namespace: str, baseline_model: Path) -> list[dict]:
    baseline = load_state_dict(baseline_model)
    baseline_w = baseline["fc.weight"].float()
    rows = []
    for run in RUNS:
        final_sd = load_state_dict(step_checkpoint_path(root, base_namespace, run, 5))
        final_w = final_sd["fc.weight"].float()
        final_b = final_sd["fc.bias"].float()
        for class_id in VEHICLES1_CLASSES:
            forget_step = run.order.index(class_id) + 1
            forget_sd = load_state_dict(step_checkpoint_path(root, base_namespace, run, forget_step))
            forget_w = forget_sd["fc.weight"].float()
            forget_b = forget_sd["fc.bias"].float()
            rows.append(
                {
                    "run_id": run.run_id,
                    "order_type": run.order_type,
                    "class_id": class_id,
                    "class": label(class_id),
                    "forget_step": forget_step,
                    "cosine_with_bicycle_final": cosine(final_w[class_id], final_w[8]),
                    "weight_delta_forget_to_final": float(torch.norm(final_w[class_id] - forget_w[class_id]).item()),
                    "weight_delta_baseline_to_final": float(torch.norm(final_w[class_id] - baseline_w[class_id]).item()),
                    "bias_delta_forget_to_final": float((final_b[class_id] - forget_b[class_id]).item()),
                }
            )
    return rows


def state_delta(prev_path: Path, current_path: Path) -> dict[str, float]:
    prev = load_state_dict(prev_path)
    current = load_state_dict(current_path)
    groups = {
        "classifier_fc": lambda key: key.startswith("fc."),
        "layer4": lambda key: key.startswith("layer4."),
        "backbone_other": lambda key: not key.startswith("fc.") and not key.startswith("layer4.") and not key.startswith("normalize."),
    }
    out = {}
    for name, predicate in groups.items():
        sq_sum = 0.0
        for key, value in current.items():
            if key in prev and predicate(key) and torch.is_floating_point(value):
                diff = value.float() - prev[key].float()
                sq_sum += float(torch.sum(diff * diff).item())
        out[name] = math.sqrt(sq_sum)
    return out


def mask_and_delta_rows(root: Path, base_namespace: str, baseline_model: Path) -> list[dict]:
    rows = []
    for run in RUNS:
        first_mask = None
        previous_mask = None
        previous_ckpt = baseline_model
        for step in range(1, 6):
            mpath = mask_path(root, base_namespace, run, step)
            cpath = step_checkpoint_path(root, base_namespace, run, step)
            current_mask = torch.load(mpath, map_location="cpu") if mpath.exists() else None
            delta = state_delta(previous_ckpt, cpath)
            rows.append(
                {
                    "run_id": run.run_id,
                    "order_type": run.order_type,
                    "step": step,
                    "new_class_id": run.order[step - 1],
                    "new_class": label(run.order[step - 1]),
                    "mask_path": str(mpath),
                    "mask_exists": current_mask is not None,
                    "jaccard_with_step1": None if first_mask is None or current_mask is None else mask_jaccard(first_mask, current_mask),
                    "jaccard_with_previous": None if previous_mask is None or current_mask is None else mask_jaccard(previous_mask, current_mask),
                    **delta,
                }
            )
            if step == 1:
                first_mask = current_mask
            previous_mask = current_mask
            previous_ckpt = cpath
    return rows


def forgotten_mask_overlap_rows(root: Path, base_namespace: str) -> tuple[list[dict], list[dict], list[dict]]:
    detail_rows = []
    per_run_summary_rows = []
    aggregate_rows = []
    for run in RUNS:
        masks = {}
        for step in range(1, 6):
            mpath = mask_path(root, base_namespace, run, step)
            masks[step] = torch.load(mpath, map_location="cpu") if mpath.exists() else None

        for class_id in VEHICLES1_CLASSES:
            forget_step = run.order.index(class_id) + 1
            base_mask = masks[forget_step]
            values = []
            later_steps = []
            for later_step in range(forget_step + 1, 6):
                later_class_id = run.order[later_step - 1]
                later_mask = masks[later_step]
                jaccard = None if base_mask is None or later_mask is None else mask_jaccard(base_mask, later_mask)
                if jaccard is not None:
                    values.append(jaccard)
                later_steps.append(later_step)
                detail_rows.append(
                    {
                        "run_id": run.run_id,
                        "order_type": run.order_type,
                        "base_class_id": class_id,
                        "base_class": label(class_id),
                        "forget_step": forget_step,
                        "later_step": later_step,
                        "later_class_id": later_class_id,
                        "later_class": label(later_class_id),
                        "jaccard": jaccard,
                    }
                )
            per_run_summary_rows.append(
                {
                    "run_id": run.run_id,
                    "order_type": run.order_type,
                    "class_id": class_id,
                    "class": label(class_id),
                    "forget_step": forget_step,
                    "later_step_count": len(later_steps),
                    "average_later_jaccard": None if not values else float(np.mean(values)),
                    "max_later_jaccard": None if not values else float(np.max(values)),
                    "later_steps": ",".join(str(step) for step in later_steps) if later_steps else "none",
                }
            )

    for class_id in VEHICLES1_CLASSES:
        values = [
            row["jaccard"]
            for row in detail_rows
            if row["base_class_id"] == class_id and row["jaccard"] is not None
        ]
        aggregate_rows.append(
            {
                "class_id": class_id,
                "class": label(class_id),
                "pair_count": len(values),
                "average_later_jaccard": None if not values else float(np.mean(values)),
                "max_later_jaccard": None if not values else float(np.max(values)),
            }
        )
    return detail_rows, per_run_summary_rows, aggregate_rows


def plot_mask_delta(mask_rows: list[dict], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in RUNS:
        rows = [row for row in mask_rows if row["run_id"] == run.run_id]
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, [row["jaccard_with_step1"] or 0.0 for row in rows], marker="o", label=run.order_type)
        axes[1].plot(steps, [row["classifier_fc"] for row in rows], marker="o", label=run.order_type)
    axes[0].set_title("Mask Jaccard with step1")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Jaccard")
    axes[1].set_title("Classifier head update L2")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("L2 delta")
    for axis in axes:
        axis.set_xticks(range(1, 6))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    savefig(figure_dir / "mask_overlap_and_fc_delta.png")


def plot_forgotten_mask_overlap(aggregate_rows: list[dict], per_run_summary_rows: list[dict], figure_dir: Path) -> None:
    labels = [CIFAR100_FINE_LABELS[row["class_id"]] for row in aggregate_rows]
    values = [row["average_later_jaccard"] or 0.0 for row in aggregate_rows]
    x = np.arange(len(aggregate_rows))
    colors = ["#d62728" if row["class_id"] == 8 else "#4c78a8" for row in aggregate_rows]
    plt.figure(figsize=(9.2, 4.8))
    plt.bar(x, values, color=colors, alpha=0.85, label="class average")
    offsets = {"official": -0.18, "random": 0.0, "hardfirst": 0.18}
    markers = {"official": "o", "random": "s", "hardfirst": "^"}
    for row in per_run_summary_rows:
        if row["average_later_jaccard"] is None:
            continue
        class_index = VEHICLES1_CLASSES.index(row["class_id"])
        plt.scatter(
            class_index + offsets[row["order_type"]],
            row["average_later_jaccard"],
            marker=markers[row["order_type"]],
            color="#222222",
            s=34,
            alpha=0.8,
        )
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Average Jaccard with later masks")
    plt.title("Forgotten-class mask overlap with later vehicles_1 masks")
    plt.ylim(0, max(values + [0.55]) * 1.12)
    plt.grid(axis="y", alpha=0.25)
    savefig(figure_dir / "mask_overlap_by_forgotten_class.png")


def soft_vs_hard_rows(class_rows: list[dict]) -> list[dict]:
    lookup = class_summary_lookup(class_rows)
    rows = []
    for run in RUNS:
        for class_id in VEHICLES1_CLASSES:
            forget_step = run.order.index(class_id) + 1
            at_forget = lookup[(run.run_id, forget_step, class_id)]
            final = lookup[(run.run_id, 5, class_id)]
            top1 = at_forget["accuracy"]
            top5 = at_forget["top5_accuracy"]
            if top1 <= 2.0 and top5 >= 30.0:
                forget_type = "soft-forgotten"
            elif top1 <= 2.0 and top5 <= 5.0:
                forget_type = "hard-forgotten"
            else:
                forget_type = "not-yet-forgotten"
            rows.append(
                {
                    "run_id": run.run_id,
                    "order_type": run.order_type,
                    "class_id": class_id,
                    "class": label(class_id),
                    "forget_step": forget_step,
                    "later_steps": 5 - forget_step,
                    "forget_type": forget_type,
                    "forget_top1": top1,
                    "forget_top5": top5,
                    "forget_mean_margin": at_forget["mean_margin"],
                    "forget_median_margin": at_forget["median_margin"],
                    "final_top1": final["accuracy"],
                    "final_top5": final["top5_accuracy"],
                    "final_mean_margin": final["mean_margin"],
                    "margin_gain": final["mean_margin"] - at_forget["mean_margin"],
                    "top5_gain": final["top5_accuracy"] - at_forget["top5_accuracy"],
                }
            )
    return rows


def plot_soft_vs_hard(soft_rows: list[dict], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for axis, run in zip(axes, RUNS):
        rows = [row for row in soft_rows if row["run_id"] == run.run_id]
        labels = [CIFAR100_FINE_LABELS[row["class_id"]] for row in rows]
        x = np.arange(len(rows))
        width = 0.36
        axis.bar(x - width / 2, [row["forget_top5"] for row in rows], width, label="top-5 at forget")
        axis.bar(x + width / 2, [row["final_top1"] for row in rows], width, label="final top-1")
        axis.set_title(run.order_type)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Soft vs hard forgetting: top-5 signal at forget vs final top-1")
    savefig(figure_dir / "soft_vs_hard_forgetting.png")


def logit_decomposition_rows(class_rows: list[dict]) -> list[dict]:
    lookup = class_summary_lookup(class_rows)
    rows = []
    for run in RUNS:
        for class_id in VEHICLES1_CLASSES:
            forget_step = run.order.index(class_id) + 1
            at_forget = lookup[(run.run_id, forget_step, class_id)]
            final = lookup[(run.run_id, 5, class_id)]
            rows.append(
                {
                    "run_id": run.run_id,
                    "order_type": run.order_type,
                    "class_id": class_id,
                    "class": label(class_id),
                    "forget_step": forget_step,
                    "true_logit_gain": final["mean_true_logit"] - at_forget["mean_true_logit"],
                    "feature_dot_gain": final["mean_true_feature_dot"] - at_forget["mean_true_feature_dot"],
                    "bias_gain": final["mean_true_bias"] - at_forget["mean_true_bias"],
                    "best_other_logit_gain": final["mean_best_other_logit"] - at_forget["mean_best_other_logit"],
                    "margin_gain": final["mean_margin"] - at_forget["mean_margin"],
                    "final_accuracy": final["accuracy"],
                }
            )
    return rows


def plot_logit_decomposition(decomp_rows: list[dict], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, run in zip(axes, RUNS):
        rows = [row for row in decomp_rows if row["run_id"] == run.run_id]
        labels = [CIFAR100_FINE_LABELS[row["class_id"]] for row in rows]
        x = np.arange(len(rows))
        width = 0.28
        axis.bar(x - width, [row["feature_dot_gain"] for row in rows], width, label="feature dot gain")
        axis.bar(x, [row["bias_gain"] for row in rows], width, label="bias gain")
        axis.bar(x + width, [-row["best_other_logit_gain"] for row in rows], width, label="-best other gain")
        axis.axhline(0, color="#555555", linewidth=1)
        axis.set_title(run.order_type)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Logit contribution change, forget -> final")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Forget-to-final margin components")
    savefig(figure_dir / "logit_decomposition_forget_to_final.png")


def generate_figures(
    analysis: dict,
    mask_rows: list[dict],
    forgotten_mask_per_run_rows: list[dict],
    forgotten_mask_aggregate_rows: list[dict],
    figure_dir: Path,
) -> dict:
    ensure_clean_figure_dir(figure_dir)
    soft_rows = soft_vs_hard_rows(analysis["class_summary_rows"])
    decomp_rows = logit_decomposition_rows(analysis["class_summary_rows"])
    plot_bicycle_margin(analysis["class_summary_rows"], figure_dir)
    plot_topk(analysis["class_summary_rows"], figure_dir)
    plot_confusion(analysis["eval_by_run_step"], figure_dir)
    plot_soft_vs_hard(soft_rows, figure_dir)
    plot_logit_decomposition(decomp_rows, figure_dir)
    pca_info = plot_pca(analysis["feature_sets"], figure_dir)
    plot_mask_delta(mask_rows, figure_dir)
    plot_forgotten_mask_overlap(forgotten_mask_aggregate_rows, forgotten_mask_per_run_rows, figure_dir)
    return pca_info


def top_predictions_for_bicycle(eval_by_run_step: dict[tuple[str, int], list[dict]]) -> list[list[object]]:
    rows = []
    for run in RUNS:
        forget_step = run.order.index(8) + 1
        for step in [forget_step, 5]:
            selected = [row for row in eval_by_run_step[(run.run_id, step)] if row["true_class_id"] == 8]
            counts: dict[int, int] = {}
            for row in selected:
                counts[row["pred_class_id"]] = counts.get(row["pred_class_id"], 0) + 1
            top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
            rows.append(
                [
                    order_zh(run.order_type),
                    step,
                    "forget step" if step == forget_step else "final",
                    ", ".join(f"{short_label(class_id)}: {count}" for class_id, count in top),
                ]
            )
    return rows


def recovery_stats(per_sample_rows: list[dict]) -> list[list[object]]:
    rows = []
    for run in RUNS:
        forget_step = run.order.index(8) + 1
        at_forget = {
            row["test_index"]: row
            for row in per_sample_rows
            if row["run_id"] == run.run_id and row["step"] == forget_step and row["true_class_id"] == 8
        }
        final = {
            row["test_index"]: row
            for row in per_sample_rows
            if row["run_id"] == run.run_id and row["step"] == 5 and row["true_class_id"] == 8
        }
        recovered = [
            idx
            for idx in at_forget
            if (not at_forget[idx]["correct"]) and final[idx]["correct"]
        ]
        stayed_correct = [
            idx
            for idx in at_forget
            if at_forget[idx]["correct"] and final[idx]["correct"]
        ]
        rows.append(
            [
                order_zh(run.order_type),
                forget_step,
                len(recovered),
                len(stayed_correct),
                fmt(float(np.mean([final[idx]["margin"] - at_forget[idx]["margin"] for idx in recovered])) if recovered else None, 2),
                fmt(float(np.mean([final[idx]["true_logit"] - at_forget[idx]["true_logit"] for idx in recovered])) if recovered else None, 2),
            ]
        )
    return rows


def soft_vs_hard_table(soft_rows: list[dict]) -> list[list[object]]:
    rows = []
    for row in soft_rows:
        rows.append(
            [
                order_zh(row["order_type"]),
                row["class"],
                row["forget_step"],
                row["later_steps"],
                row["forget_type"],
                pct(row["forget_top1"]),
                pct(row["forget_top5"]),
                fmt(row["forget_mean_margin"], 2),
                fmt(row["forget_median_margin"], 2),
                pct(row["final_top1"]),
                pct(row["final_top5"]),
                fmt(row["margin_gain"], 2),
            ]
        )
    return rows


def recovered_sample_detail_rows(per_sample_rows: list[dict]) -> list[list[object]]:
    rows = []
    for run in RUNS:
        forget_step = run.order.index(8) + 1
        at_forget = {
            row["test_index"]: row
            for row in per_sample_rows
            if row["run_id"] == run.run_id and row["step"] == forget_step and row["true_class_id"] == 8
        }
        final = {
            row["test_index"]: row
            for row in per_sample_rows
            if row["run_id"] == run.run_id and row["step"] == 5 and row["true_class_id"] == 8
        }
        groups = {
            "recovered": [idx for idx in at_forget if (not at_forget[idx]["correct"]) and final[idx]["correct"]],
            "not recovered": [idx for idx in at_forget if (not at_forget[idx]["correct"]) and (not final[idx]["correct"])],
        }
        for group_name, indices in groups.items():
            if not indices:
                rows.append([order_zh(run.order_type), group_name, 0, "n/a", "n/a", "n/a", "n/a", "n/a"])
                continue
            top5_rate = 100.0 * sum(at_forget[idx]["top5_contains_true"] for idx in indices) / len(indices)
            ranks = [
                int(at_forget[idx]["true_rank_in_top5"])
                for idx in indices
                if at_forget[idx]["true_rank_in_top5"] not in ("", None)
            ]
            pred_counts: dict[int, int] = {}
            for idx in indices:
                pred = at_forget[idx]["pred_class_id"]
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            top_preds = sorted(pred_counts.items(), key=lambda item: item[1], reverse=True)[:3]
            rows.append(
                [
                    order_zh(run.order_type),
                    group_name,
                    len(indices),
                    pct(top5_rate),
                    fmt(float(np.mean(ranks)) if ranks else None, 2),
                    fmt(float(np.mean([at_forget[idx]["margin"] for idx in indices])), 2),
                    fmt(float(np.mean([final[idx]["margin"] - at_forget[idx]["margin"] for idx in indices])), 2),
                    ", ".join(f"{short_label(class_id)}: {count}" for class_id, count in top_preds),
                ]
            )
    return rows


def logit_decomposition_table(decomp_rows: list[dict]) -> list[list[object]]:
    rows = []
    for row in decomp_rows:
        rows.append(
            [
                order_zh(row["order_type"]),
                row["class"],
                row["forget_step"],
                fmt(row["true_logit_gain"], 2),
                fmt(row["feature_dot_gain"], 2),
                fmt(row["bias_gain"], 2),
                fmt(row["best_other_logit_gain"], 2),
                fmt(row["margin_gain"], 2),
                pct(row["final_accuracy"]),
            ]
        )
    return rows


def classifier_weight_table(weight_rows: list[dict]) -> list[list[object]]:
    rows = []
    for row in weight_rows:
        rows.append(
            [
                order_zh(row["order_type"]),
                row["class"],
                row["forget_step"],
                fmt(row["cosine_with_bicycle_final"], 3),
                fmt(row["weight_delta_forget_to_final"], 3),
                fmt(row["weight_delta_baseline_to_final"], 3),
                fmt(row["bias_delta_forget_to_final"], 3),
            ]
        )
    return rows


def class_summary_table(class_rows: list[dict], class_id: int) -> list[list[object]]:
    rows = []
    lookup = class_summary_lookup(class_rows)
    for run in RUNS:
        for step in range(0, 6):
            item = lookup[(run.run_id, step, class_id)]
            rows.append(
                [
                    order_zh(run.order_type),
                    step,
                    pct(item["accuracy"]),
                    pct(item["top5_accuracy"]),
                    fmt(item["mean_margin"], 2),
                    fmt(item["mean_true_logit"], 2),
                    fmt(item["mean_top1_confidence"], 3),
                ]
            )
    return rows


def centroid_table(centroid_rows: list[dict]) -> list[list[object]]:
    rows = []
    for run in RUNS:
        for step in ["baseline", "bicycle forget step", "final"]:
            subset = [
                row
                for row in centroid_rows
                if row["run_id"] == run.run_id
                and row["step"] == step
                and row["class_id"] in VEHICLES1_CLASSES + VEHICLES2_CLASSES
            ]
            nearest = sorted(subset, key=lambda row: row["distance_from_bicycle"])[:5]
            rows.append(
                [
                    order_zh(run.order_type),
                    step,
                    ", ".join(f"{short_label(row['class_id'])}: {row['distance_from_bicycle']:.2f}" for row in nearest),
                ]
            )
    return rows


def mask_table(mask_rows: list[dict]) -> list[list[object]]:
    rows = []
    for row in mask_rows:
        rows.append(
            [
                order_zh(row["order_type"]),
                row["step"],
                row["new_class"],
                "yes" if row["mask_exists"] else "no",
                fmt(row["jaccard_with_step1"], 3),
                fmt(row["jaccard_with_previous"], 3),
                fmt(row["classifier_fc"], 3),
                fmt(row["layer4"], 3),
                fmt(row["backbone_other"], 3),
            ]
        )
    return rows


def forgotten_mask_detail_table(detail_rows: list[dict], base_class_id: int | None = None) -> list[list[object]]:
    rows = []
    selected = detail_rows if base_class_id is None else [row for row in detail_rows if row["base_class_id"] == base_class_id]
    for row in selected:
        rows.append(
            [
                order_zh(row["order_type"]),
                row["base_class"],
                row["forget_step"],
                row["later_step"],
                row["later_class"],
                fmt(row["jaccard"], 3),
            ]
        )
    return rows


def forgotten_mask_summary_table(summary_rows: list[dict]) -> list[list[object]]:
    rows = []
    for row in summary_rows:
        rows.append(
            [
                order_zh(row["order_type"]),
                row["class"],
                row["forget_step"],
                row["later_step_count"],
                row["later_steps"],
                fmt(row["average_later_jaccard"], 3),
                fmt(row["max_later_jaccard"], 3),
            ]
        )
    return rows


def forgotten_mask_aggregate_table(aggregate_rows: list[dict]) -> list[list[object]]:
    return [
        [
            row["class"],
            row["pair_count"],
            fmt(row["average_later_jaccard"], 3),
            fmt(row["max_later_jaccard"], 3),
        ]
        for row in aggregate_rows
    ]


def forgotten_mask_conclusion(aggregate_rows: list[dict]) -> str:
    values = {
        row["class_id"]: row["average_later_jaccard"]
        for row in aggregate_rows
        if row["average_later_jaccard"] is not None
    }
    bicycle_avg = values.get(8)
    other_values = [value for class_id, value in values.items() if class_id != 8]
    if bicycle_avg is None or not other_values:
        return "目前缺少足夠 later-mask pairs，不能用固定 forgotten-class mask overlap 判斷 bicycle 是否特別容易被後續 masks 影響。"
    other_avg = float(np.mean(other_values))
    other_max = float(np.max(other_values))
    diff = bicycle_avg - other_avg
    if bicycle_avg >= other_max + 0.02:
        return (
            f"`bicycle` 的平均 later-mask Jaccard 是 `{bicycle_avg:.3f}`，高於其他類別平均 `{other_avg:.3f}`，"
            f"也高於其他類別最高平均 `{other_max:.3f}`；這支持 mask overlap 是 bicycle rebound 的重要原因之一。"
        )
    return (
        f"`bicycle` 的平均 later-mask Jaccard 是 `{bicycle_avg:.3f}`，其他類別平均是 `{other_avg:.3f}`，"
        f"其他類別最高平均是 `{other_max:.3f}`。bicycle 並沒有明顯高於其他 vehicles_1 類別，"
        "因此 mask overlap 目前只能當輔助證據；更主要的解釋仍是 bicycle 在 forget step 後保留較強 top-k signal 與 feature-space vehicle representation。"
    )


def fig_link(report_path: Path, figure_dir: Path, filename: str) -> str:
    path = figure_dir / filename
    try:
        return str(path.relative_to(report_path.parent))
    except ValueError:
        return str(path)


def build_report(
    args: argparse.Namespace,
    analysis: dict,
    pca_info: dict,
    mask_rows: list[dict],
    forgotten_mask_detail_rows: list[dict],
    forgotten_mask_per_run_rows: list[dict],
    forgotten_mask_aggregate_rows: list[dict],
    weight_rows: list[dict],
) -> str:
    per_sample_count = len(analysis["per_sample_rows"])
    expected = len(RUNS) * 6 * len(VEHICLES1_CLASSES) * 100
    soft_rows = soft_vs_hard_rows(analysis["class_summary_rows"])
    decomp_rows = logit_decomposition_rows(analysis["class_summary_rows"])
    sections = [
        "# 6_6 CIFAR-100 vehicles_1 / bicycle Rebound 成因分析",
        "",
        "## 分析設定",
        "",
        f"- 使用 namespace：`{args.base_namespace}`。",
        "- 不重新跑 unlearning，只讀取已完成的 `vehicles_1` 三條 runs 的 checkpoints。",
        "- 準確率、logit、prediction tracking 都使用 CIFAR-100 test split。",
        f"- 分析 checkpoints：每條 run 的 step0 baseline + step1-step5，共 `{len(RUNS) * 6}` 個模型狀態。",
        f"- per-sample rows：`{per_sample_count}`，預期 `{expected}`；每條 run、每一步、vehicles_1 五類各 100 張 test images。",
        f"- mask overlap 使用 unlearn 實際預設 ratio：`with_{MASK_RATIO}.pt`。",
        "",
        "## Evidence Summary",
        "",
        "目前證據比較支持：`bicycle / 腳踏車 (8)` 的 rebound 不是評估資料錯置，也不是 immediate forgetting lag，而是後續 sequential unlearning 調整 shared vehicle representation 或 classifier boundary 時，讓 bicycle 的 logit margin 局部回升。這仍是 evidence-supported hypothesis，還不是完全因果證明。",
        "",
        "**快速結論：**最有說服力的數字是：`bicycle` 被忘當下 top-1 只有 `0/2/1%`，但 top-5 仍有 `35/34/42%`；其他 vehicles_1 類別被忘後 top-5 大多是 `0%`。所以 bicycle 不是沒有被忘，而是被軟性壓低，後續更新較容易把它拉回 top-1。",
        "",
        markdown_table(
            ["Hypothesis", "判斷", "主要證據"],
            [
                ["評估假象", "不支持", "evaluation script 使用 CIFAR-100 `train=False` test split；per-sample tracking 直接追蹤同一批 test images。"],
                ["forgetting lag", "不支持", "bicycle 在被忘記當步 top-1 accuracy 已降到 0% 到 2%，後續才回升。"],
                ["少數樣本偶然翻正", "部分不支持", "recovery 是數十張 test samples，而不是 1-2 張；需看 recovery table。"],
                ["decision boundary 回漂", "支持", "bicycle mean margin 在 forget step 後往 final 方向回升，且 final top-1 accuracy 回到 40% 以上。"],
                ["shared vehicle features", "支持", "feature PCA / centroid 顯示 bicycle 仍和交通工具 representation 同區域；top-5 recovery 若高於 top-1，表示 representation 未完全消失。"],
                ["mask/update interference", "待驗證但合理", "後續 step masks 與 classifier/head update 仍會改寫共享參數；mask overlap 和 delta table 提供初步證據。"],
            ],
        ),
        "",
        "## 為什麼 bicycle 被影響最多",
        "",
        "現有數據顯示，`bicycle (8)` 和其他 vehicles_1 類別最大的差異不是 exposure，而是忘記後的殘餘訊號強度。`bicycle` 在 forget step 的 top-1 幾乎歸零，但 top-5 仍有 `34%~42%`；相反地，多數其他車輛類別一旦被忘，top-1 和 top-5 都接近 `0%`。這代表 bicycle 比較像 `soft-forgotten`：分類邊界暫時被壓下去，但 representation 或候選 ranking 沒有完全消失。",
        "",
        f"![Soft vs hard forgetting]({fig_link(args.report_path, args.figure_dir, 'soft_vs_hard_forgetting.png')})",
        "",
        markdown_table(
            [
                "順序",
                "Class",
                "Forget step",
                "Later steps",
                "Type",
                "Forget top-1",
                "Forget top-5",
                "Forget mean margin",
                "Forget median margin",
                "Final top-1",
                "Final top-5",
                "Margin gain",
            ],
            soft_vs_hard_table(soft_rows),
        ),
        "",
        "**數據說明：**`Forget top-5` 表示該類別被忘當下是否仍在模型前 5 名候選內；`Margin gain` 表示從 forget step 到 final，true label 相對最佳競爭類別的 logit 是否變好。",
        "",
        "**結論：**`bicycle` 三次 forget step 都是 soft-forgotten：top-1 `0/2/1%`，top-5 `35/34/42%`，final top-1 回到 `43/40/48%`。相反地，`bus` 在 hardfirst step1、`motorcycle` 在 random step1 被忘時 top-5 都是 `0%`，final top-1 仍是 `0%`；因此 rebound 的關鍵不是早忘，而是 forget 後是否還保留 top-k 訊號。",
        "",
        "### Recovered vs non-recovered bicycle samples",
        "",
        markdown_table(
            ["順序", "Group", "Samples", "Top-5 contains bicycle at forget", "Mean top-5 rank at forget", "Mean margin at forget", "Mean margin gain", "Top wrong preds at forget"],
            recovered_sample_detail_rows(analysis["per_sample_rows"]),
        ),
        "",
        "**數據說明：**`recovered` 是 forget step 判錯、final 又判回 bicycle 的 test images；`Top-5 contains bicycle at forget` 越高，代表忘記當下仍保留 bicycle 候選訊號。",
        "",
        "**結論：**bicycle rebound 是一批樣本整體跨回 decision boundary：三條 run 有 `43/39/47` 張 recovered samples。recovered samples 在 forget step 的平均 margin 是 `-1.64/-1.19/-1.20`，比未 recovery 的 `-2.88/-2.34/-2.88` 更接近 0，所以它們本來就比較容易被後續更新拉回。",
        "",
        "### Logit decomposition：margin 回升從哪裡來",
        "",
        f"![Logit decomposition]({fig_link(args.report_path, args.figure_dir, 'logit_decomposition_forget_to_final.png')})",
        "",
        markdown_table(
            ["順序", "Class", "Forget step", "True-logit gain", "Feature-dot gain", "Bias gain", "Best-other logit gain", "Margin gain", "Final acc"],
            logit_decomposition_table(decomp_rows),
        ),
        "",
        "**數據說明：**`margin gain = true-logit gain - best-other-logit gain`；`feature-dot gain` 代表 feature 與該 class head 的內積變化，越大表示 representation 對該 class 更有利。",
        "",
        "**結論：**其他車輛類別不是完全沒有被後續更新影響，而是被壓得太深。例如 `motorcycle` 在 official 的 margin gain 有 `+4.62`，但 forget mean margin 從 `-9.32` 只回到約 `-4.69`，final acc 仍是 `0%`；bicycle 的 forget mean margin 只有 `-2.35/-1.84/-2.05`，回升 `+1.42/+0.88/+1.40` 就足以讓 final acc 到 `43/40/48%`。",
        "",
        "### Classifier head weight comparison",
        "",
        markdown_table(
            ["順序", "Class", "Forget step", "Final cosine with bicycle weight", "Weight delta forget->final", "Weight delta baseline->final", "Bias delta forget->final"],
            classifier_weight_table(weight_rows),
        ),
        "",
        "**數據說明：**`Weight delta forget->final` 看該 class classifier head 從被忘當下到 final 是否被大幅改動；`Cosine with bicycle weight` 看其他 vehicle head 是否和 bicycle head 方向相近。",
        "",
        "**結論：**bicycle rebound 不像是單純 classifier head 把 bicycle 權重改回來；bicycle 的 `weight delta forget->final` 只有 `0.086/0.039/0.048`，但 final top-5 卻回到 `79/84/86%`。這比較支持 shared feature 或整體 boundary drift，而不是 bicycle head 大幅改回。",
        "",
        "## Per-Sample Recovery",
        "",
        markdown_table(
            ["順序", "Bicycle forget step", "Forget wrong -> final correct", "Forget correct -> final correct", "Recovered mean margin gain", "Recovered mean true-logit gain"],
            recovery_stats(analysis["per_sample_rows"]),
        ),
        "",
        "**數據說明：**這張表只看 bicycle test images，統計同一張圖片是否從 forget step 的錯誤預測，變成 final 的正確預測。",
        "",
        "**結論：**三條 run 分別有 `43/39/47` 張 bicycle 圖片從錯誤變正確，占 100 張 bicycle test images 的 `39%~47%`。這是穩定的 sample group recovery，不是 1-2 張圖片造成的統計雜訊。",
        "",
        "## Bicycle Logit / Top-k Trajectory",
        "",
        f"![Bicycle margin trajectory]({fig_link(args.report_path, args.figure_dir, 'bicycle_margin_trajectory.png')})",
        "",
        f"![Bicycle top-k recovery]({fig_link(args.report_path, args.figure_dir, 'bicycle_topk_recovery.png')})",
        "",
        markdown_table(
            ["順序", "Step", "Top-1 acc", "Top-5 acc", "Mean margin", "Mean true logit", "Mean top1 confidence"],
            class_summary_table(analysis["class_summary_rows"], 8),
        ),
        "",
        "**數據說明：**Top-1 是正式 accuracy；Top-5 代表 bicycle 是否仍在候選答案中。Mean margin 越接近 0，表示越接近重新變成 top-1。",
        "",
        "**結論：**bicycle 被忘後仍保留候選訊號：forget step top-5 是 `35/34/42%`，後續最高回到 `85/84/86%`，final top-1 是 `43/40/48%`。這直接支持「bicycle 是 soft-forgotten，因此容易被後續更新拉回」的解釋。",
        "",
        "## Confusion Matrix",
        "",
        f"![vehicles_1 final confusion]({fig_link(args.report_path, args.figure_dir, 'vehicles1_final_confusion.png')})",
        "",
        markdown_table(
            ["順序", "Step", "階段", "Bicycle test images top predicted labels"],
            top_predictions_for_bicycle(analysis["eval_by_run_step"]),
        ),
        "",
        "**數據說明：**這裡看 bicycle 圖片在 forget step 被錯分到哪裡，以及 final 是否回到 bicycle。",
        "",
        "**結論：**final residual 幾乎全由 bicycle 貢獻：final 時 bicycle 正確 `43/40/48` 張，但 `bus/motorcycle/pickup_truck/train` final top-1 幾乎都是 `0` 張。這表示問題是 bicycle-specific，不是整個 vehicles_1 都忘失敗。",
        "",
        "## Feature Centroid / PCA",
        "",
        "這一節不是只看 final checkpoint，而是檢查 bicycle 的 penultimate feature representation 在 unlearning 前後是否真的被移走。三個時間點分別是：`baseline / step0`、`bicycle forget step`、`final / step5`。",
        "",
        f"![Feature PCA official bicycle timeline]({fig_link(args.report_path, args.figure_dir, 'feature_pca_official_bicycle_timeline.png')})",
        "",
        f"- PCA explained variance ratio：PC1 `{pca_info['explained_variance'][0]:.3f}`，PC2 `{pca_info['explained_variance'][1]:.3f}`。",
        "- PCA 圖先以 official run 顯示，因為 official 中 bicycle 是 step1，時間點最直觀。",
        "- Centroid table 則列出三條 run 的 `baseline / bicycle forget step / final`。",
        "",
        markdown_table(
            ["順序", "Step", "Nearest centroids to bicycle among vehicles_1 + vehicles_2"],
            centroid_table(pca_info["centroid_rows"]),
        ),
        "",
        "**數據說明：**PCA / centroid 使用 ResNet penultimate feature；centroid distance 越小，表示 feature 空間越接近。這裡只比較交通工具相關類別：vehicles_1 與 vehicles_2，不是 CIFAR-100 全部 100 類。",
        "",
        "**結論：**不能只用 final PCA 說 bicycle 靠近其他車輛；要看 trajectory。`bicycle forget step` 時，最近 vehicle centroid 距離分別是 official `11.69`、random `3.52`、hardfirst `10.09`，final 則縮到 `4.86/4.76/5.00`。這表示 feature 空間確實受到後續 vehicle steps 影響，但是否「完全沒被移走」不能只靠 centroid 證明。比較穩健的解釋是：bicycle 被忘當下 top-1 已降到 `0~2%`，但 top-5 仍有 `34~42%`，代表它仍保留部分 vehicle / candidate signal；後續 boundary 或 shared representation drift 足以把這批 soft-forgotten samples 拉回 top-1。這比單純說「feature 被重新學回來」更符合目前 top-5、margin 與 centroid trajectory 的共同證據。",
        "",
        "## Mask Overlap / Parameter Delta",
        "",
        f"![Mask overlap and FC delta]({fig_link(args.report_path, args.figure_dir, 'mask_overlap_and_fc_delta.png')})",
        "",
        markdown_table(
            ["順序", "Step", "New class", "Mask", "Jaccard vs step1", "Jaccard vs previous", "FC delta", "Layer4 delta", "Other backbone delta"],
            mask_table(mask_rows),
        ),
        "",
        "**數據說明：**Jaccard 越高表示 mask 選到的參數區域越重疊；parameter delta 表示每一步 unlearning 對 classifier head / backbone 的改動幅度。",
        "",
        "**結論：**後續 steps 仍持續改動共享參數：每步 layer4 delta 約 `0.868~1.502`，FC delta 約 `0.109~0.179`，mask 與 step1 的 Jaccard 約 `0.464~0.534`。因此舊 forgotten class 即使沒有再進入訓練，也可能被共享參數更新間接推動。",
        "",
        "### 固定 forgotten-class mask 的後續重疊比較",
        "",
        "上表的 `Jaccard vs step1` 是每條 run 的 step1 mask 當基準；如果 step1 不是 bicycle，就不能直接回答 bicycle mask 是否特別容易和後續 masks 重疊。所以下表改成：每個 class 被忘記那一步的 mask，固定拿來和該 run 的後續 steps masks 比較。",
        "",
        f"![Forgotten-class mask overlap]({fig_link(args.report_path, args.figure_dir, 'mask_overlap_by_forgotten_class.png')})",
        "",
        markdown_table(
            ["Class", "Later mask pairs", "Average later Jaccard", "Max later Jaccard"],
            forgotten_mask_aggregate_table(forgotten_mask_aggregate_rows),
        ),
        "",
        markdown_table(
            ["順序", "Class", "Forget step", "Later step count", "Later steps", "Avg later Jaccard", "Max later Jaccard"],
            forgotten_mask_summary_table(forgotten_mask_per_run_rows),
        ),
        "",
        "**Bicycle mask vs later masks：**",
        "",
        markdown_table(
            ["順序", "Base class", "Forget step", "Later step", "Later class", "Jaccard"],
            forgotten_mask_detail_table(forgotten_mask_detail_rows, base_class_id=8),
        ),
        "",
        "**數據說明：**這裡的每個 Jaccard 都是「某 class 被忘當下的 mask」和「後續某一步 unlearning mask」的重疊，不再依賴 step1 是否剛好是 bicycle。",
        "",
        f"**結論：**{forgotten_mask_conclusion(forgotten_mask_aggregate_rows)}",
        "",
        "## 控制實驗設計",
        "",
        "已新增 launcher：`scripts/run_cifar100_bicycle_rebound_cause_controls.sh`。它預設 `DRY_RUN=1`，只檢查 order 和輸出 queue log，不會直接開跑 unlearning。若要正式執行，可用 `DRY_RUN=0 bash scripts/run_cifar100_bicycle_rebound_cause_controls.sh`。",
        "",
        markdown_table(
            ["Run ID", "Forget order", "Purpose", "Expected interpretation"],
            [
                ["bicycle_then_flowers_seed1_k5", "8,54,62,70,82", "先忘 bicycle，再忘非交通工具 flower classes", "若不 rebound，支持 vehicle-related steps 才會拉回 bicycle。"],
                ["bicycle_then_vehicles2_seed1_k5", "8,41,69,81,89", "先忘 bicycle，再忘 vehicles_2", "若 rebound，支持交通工具 shared representation。"],
                ["motorcycle_then_vehicles_no_bicycle_seed1_k5", "48,13,58,90,8", "讓 motorcycle 早忘，bicycle 最後忘", "若 motorcycle 仍不 rebound，表示早忘不是充分條件。"],
                ["pickup_then_vehicles_no_bicycle_seed1_k5", "58,13,48,90,8", "讓 pickup_truck 早忘，bicycle 最後忘", "若 pickup 仍不 rebound，表示原本不是 exposure 不足。"],
            ],
        ),
        "",
        "**結論：**現有數據已支持 soft-forgotten + shared representation 是最合理解釋：bicycle forget top-5 `35/34/42%`、recovered samples `43/39/47`、final top-1 `43/40/48%`。但是否真的是 vehicle-specific causal effect，仍需要 `bicycle_then_flowers` 和 `bicycle_then_vehicles2` 控制實驗確認。",
        "",
        "## 結論",
        "",
        "1. 目前最合理的解釋是 decision boundary / shared representation rebound，而不是 test data 評估錯誤或當步忘不掉。",
        "2. `bicycle (8)` 和其他 vehicles_1 類別的關鍵差異是 soft-forgotten pattern：forget step top-1 幾乎歸零，但 top-5 仍保留 34% 到 42%，其他車輛類別多半 top-1/top-5 都接近 0%。",
        "3. 因為 bicycle 在 forget step 後仍保留候選 ranking 和較不極端的 negative margin，後續 boundary drift 比較容易把一批 test samples 推回 top-1；其他類別即使有 margin gain，也仍離 decision boundary 太遠。",
        "4. exposure 不是充分原因：`bus` 與 `motorcycle` 都有早忘案例，但沒有 rebound；因此 bicycle 的特殊性更可能來自 suppression 強度、feature geometry 或 shared vehicle representation。",
        "5. 下一步若要做 causal test，最小控制實驗是 `bicycle_then_flowers` 與 `bicycle_then_vehicles2`：前者若不 rebound、後者若 rebound，就能更直接支持 vehicle shared features 假說。",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    analysis = collect_analysis(args, device)
    mask_rows = mask_and_delta_rows(args.root, args.base_namespace, args.baseline_model)
    forgotten_mask_detail_rows, forgotten_mask_per_run_rows, forgotten_mask_aggregate_rows = forgotten_mask_overlap_rows(
        args.root,
        args.base_namespace,
    )
    weight_rows = classifier_weight_rows(args.root, args.base_namespace, args.baseline_model)
    write_csv(args.per_sample_csv, analysis["per_sample_rows"])
    write_csv(args.class_summary_csv, analysis["class_summary_rows"])
    pca_info = generate_figures(
        analysis,
        mask_rows,
        forgotten_mask_per_run_rows,
        forgotten_mask_aggregate_rows,
        args.figure_dir,
    )
    args.report_path.write_text(
        build_report(
            args,
            analysis,
            pca_info,
            mask_rows,
            forgotten_mask_detail_rows,
            forgotten_mask_per_run_rows,
            forgotten_mask_aggregate_rows,
            weight_rows,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.per_sample_csv} ({len(analysis['per_sample_rows'])} rows)")
    print(f"Wrote {args.class_summary_csv} ({len(analysis['class_summary_rows'])} rows)")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
