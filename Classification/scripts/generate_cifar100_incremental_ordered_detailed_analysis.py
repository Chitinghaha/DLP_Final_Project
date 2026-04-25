#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PILOT_CLASSES = [
    {"id": 3, "name": "bear", "zh": "熊", "group": "large_carnivores"},
    {"id": 42, "name": "leopard", "zh": "豹", "group": "large_carnivores"},
    {"id": 43, "name": "lion", "zh": "獅子", "group": "large_carnivores"},
    {"id": 88, "name": "tiger", "zh": "老虎", "group": "large_carnivores"},
    {"id": 97, "name": "wolf", "zh": "狼", "group": "large_carnivores"},
    {"id": 15, "name": "camel", "zh": "駱駝", "group": "large_omnivores_and_herbivores"},
    {"id": 19, "name": "cattle", "zh": "牛", "group": "large_omnivores_and_herbivores"},
    {"id": 21, "name": "chimpanzee", "zh": "黑猩猩", "group": "large_omnivores_and_herbivores"},
    {"id": 31, "name": "elephant", "zh": "大象", "group": "large_omnivores_and_herbivores"},
    {"id": 38, "name": "kangaroo", "zh": "袋鼠", "group": "large_omnivores_and_herbivores"},
    {"id": 34, "name": "fox", "zh": "狐狸", "group": "medium_mammals"},
    {"id": 63, "name": "porcupine", "zh": "豪豬", "group": "medium_mammals"},
    {"id": 64, "name": "possum", "zh": "負鼠", "group": "medium_mammals"},
    {"id": 66, "name": "raccoon", "zh": "浣熊", "group": "medium_mammals"},
    {"id": 75, "name": "skunk", "zh": "臭鼬", "group": "medium_mammals"},
    {"id": 36, "name": "hamster", "zh": "倉鼠", "group": "small_mammals"},
    {"id": 50, "name": "mouse", "zh": "老鼠", "group": "small_mammals"},
    {"id": 65, "name": "rabbit", "zh": "兔子", "group": "small_mammals"},
    {"id": 74, "name": "shrew", "zh": "鼩鼱", "group": "small_mammals"},
    {"id": 80, "name": "squirrel", "zh": "松鼠", "group": "small_mammals"},
]

CLASS_META = {item["id"]: item for item in PILOT_CLASSES}
GROUP_LABELS = {
    "large_carnivores": "large carnivores",
    "large_omnivores_and_herbivores": "large omnivores/herbivores",
    "medium_mammals": "medium mammals",
    "small_mammals": "small mammals",
}
GROUP_COLORS = {
    "large_carnivores": "#E45756",
    "large_omnivores_and_herbivores": "#72B7B2",
    "medium_mammals": "#54A24B",
    "small_mammals": "#F58518",
}
ORDER_COLORS = {
    "normal": "#4C78A8",
    "clustered": "#F58518",
    "interleaved": "#54A24B",
}

ORDER_CONFIGS = {
    "normal": {
        "label": "Normal",
        "run_id": "incremental_cifar100_normal_seed1_k20",
        "smoke_run_id": "smoke_incremental_cifar100_normal_seed1_k2",
        "order": [3, 15, 19, 21, 31, 34, 36, 38, 42, 43, 50, 63, 64, 65, 66, 74, 75, 80, 88, 97],
    },
    "clustered": {
        "label": "Clustered",
        "run_id": "incremental_cifar100_clustered_animals_seed1_k20",
        "smoke_run_id": "smoke_incremental_cifar100_clustered_animals_seed1_k2",
        "order": [3, 42, 43, 88, 97, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 36, 50, 65, 74, 80],
    },
    "interleaved": {
        "label": "Interleaved",
        "run_id": "incremental_cifar100_interleaved_animals_seed1_k20",
        "smoke_run_id": "smoke_incremental_cifar100_interleaved_animals_seed1_k2",
        "order": [3, 15, 34, 36, 42, 19, 63, 50, 43, 21, 64, 65, 88, 31, 66, 74, 97, 38, 75, 80],
    },
}

METRICS = ["forget_accuracy", "UA", "retain_accuracy", "full_test_accuracy"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--result-namespace",
        default="sequential_incremental_ordered_cifar100_pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "cifar100_incremental_ordered_pilot_detailed_figures",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "cifar100_incremental_ordered_pilot_detailed_analysis.md",
    )
    return parser.parse_args()


def eval_filename(seed: int, step: int, order: list[int]) -> str:
    forgotten = "_".join(str(class_id) for class_id in order[:step])
    return f"seed{seed}_step{step}_forgot_{forgotten}.csv"


def read_eval_rows(eval_root: Path, namespace: str, seed: int, order: list[int]) -> list[dict]:
    rows = []
    for step in range(1, len(order) + 1):
        path = eval_root / namespace / eval_filename(seed, step, order)
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["step"] = step
        rows.append(row)
    return rows


def read_progress_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str | float | int | None, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def metric_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    return [int(row["step"]) for row in rows], [fnum(row[key]) for row in rows]


def class_accuracy(row: dict, class_id: int) -> float:
    return fnum(row[f"class_{class_id}_accuracy"])


def class_series(rows: list[dict], class_id: int) -> tuple[list[int], list[float]]:
    return [int(row["step"]) for row in rows], [class_accuracy(row, class_id) for row in rows]


def newly_forgotten_values(rows: list[dict], order: list[int]) -> list[float]:
    values = []
    for step, class_id in enumerate(order, start=1):
        row = rows[step - 1]
        values.append(class_accuracy(row, class_id))
    return values


def total_duration(rows: list[dict]) -> float:
    return sum(fnum(row.get("duration_sec")) for row in rows if row.get("status") == "success")


def duration_by_stage(rows: list[dict]) -> dict[str, float]:
    result = {"mask": 0.0, "unlearn": 0.0, "eval": 0.0}
    for row in rows:
        if row.get("status") == "success" and row.get("stage") in result:
            result[row["stage"]] += fnum(row.get("duration_sec"))
    return result


def duration_by_step(rows: list[dict]) -> tuple[list[int], dict[str, list[float]]]:
    steps = sorted({int(row["step"]) for row in rows if row.get("step")})
    by_stage = {"mask": [], "unlearn": [], "eval": []}
    for step in steps:
        step_rows = [
            row for row in rows
            if row.get("step") and int(row["step"]) == step and row.get("status") == "success"
        ]
        lookup = {row["stage"]: row for row in step_rows}
        for stage in by_stage:
            by_stage[stage].append(fnum(lookup.get(stage, {}).get("duration_sec")))
    return steps, by_stage


def lag_items(rows: list[dict], order: list[int]) -> list[dict]:
    items = []
    for step, class_id in enumerate(order, start=1):
        acc = class_accuracy(rows[step - 1], class_id)
        meta = CLASS_META[class_id]
        items.append(
            {
                "step": step,
                "class_id": class_id,
                "class_name": meta["name"],
                "group": meta["group"],
                "accuracy": acc,
                "is_failure": acc > 10.0,
            }
        )
    return items


def final_residual_items(rows: list[dict], order: list[int]) -> list[dict]:
    final = rows[-1]
    items = []
    for class_id in order:
        meta = CLASS_META[class_id]
        acc = class_accuracy(final, class_id)
        items.append(
            {
                "class_id": class_id,
                "class_name": meta["name"],
                "group": meta["group"],
                "accuracy": acc,
                "is_residual": acc > 10.0,
            }
        )
    return items


def post_forget_max(rows: list[dict], order: list[int]) -> dict[int, float]:
    result = {}
    for forget_step, class_id in enumerate(order, start=1):
        values = [class_accuracy(row, class_id) for row in rows[forget_step - 1 :]]
        result[class_id] = max(values)
    return result


def post_forget_values(rows: list[dict], order: list[int], class_id: int) -> list[float]:
    forget_step = order.index(class_id) + 1
    return [class_accuracy(row, class_id) for row in rows[forget_step - 1 :]]


def post_forget_nonzero_stats(rows: list[dict], order: list[int]) -> list[dict]:
    stats = []
    for class_id in order:
        values = post_forget_values(rows, order, class_id)
        later_values = values[1:]
        meta = CLASS_META[class_id]
        stats.append(
            {
                "class_id": class_id,
                "class_name": meta["name"],
                "group": meta["group"],
                "forget_step": order.index(class_id) + 1,
                "nonzero_count": sum(value > 0 for value in values),
                "later_nonzero_count": sum(value > 0 for value in later_values),
                "observed_steps": len(values),
                "max_accuracy": max(values) if values else 0.0,
                "final_accuracy": values[-1] if values else 0.0,
                "trajectory": values,
            }
        )
    return stats


def trajectory_after_forget(rows: list[dict], order: list[int], class_id: int) -> tuple[list[int], list[float]]:
    forget_step = order.index(class_id) + 1
    steps = list(range(forget_step, len(order) + 1))
    values = [class_accuracy(rows[step - 1], class_id) for step in steps]
    return steps, values


def smoke_status(rows: list[dict]) -> str:
    if not rows:
        return "missing"
    if any(row.get("status") == "failed" for row in rows):
        return "failed"
    return "success"


def style_ax(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)


def add_value_labels(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=8, padding=2)


def save_metric_plot(
    output_path: Path,
    title: str,
    ylabel: str,
    metric_key: str,
    order_rows: dict[str, list[dict]],
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    markers = {"normal": "o", "clustered": "s", "interleaved": "^"}
    for key, config in ORDER_CONFIGS.items():
        x, y = metric_series(order_rows[key], metric_key)
        ax.plot(x, y, marker=markers[key], linewidth=2.2, label=config["label"], color=ORDER_COLORS[key])
    ax.set_xticks(range(1, 21))
    style_ax(ax, title, ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_final_metrics_bar(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    labels = [ORDER_CONFIGS[key]["label"] for key in ORDER_CONFIGS]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    metric_labels = {
        "forget_accuracy": "Forget acc",
        "UA": "UA",
        "retain_accuracy": "Retain acc",
        "full_test_accuracy": "Full test acc",
    }
    for offset, metric in zip(offsets, METRICS):
        values = [fnum(order_rows[key][-1][metric]) for key in ORDER_CONFIGS]
        ax.bar(x + offset * width, values, width, label=metric_labels[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Metrics by Order", fontsize=12, weight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    add_value_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_newly_forgotten_plot(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    markers = {"normal": "o", "clustered": "s", "interleaved": "^"}
    for key, config in ORDER_CONFIGS.items():
        values = newly_forgotten_values(order_rows[key], config["order"])
        ax.plot(range(1, 21), values, marker=markers[key], linewidth=2.2, label=config["label"], color=ORDER_COLORS[key])
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8, label="10% failure guide")
    ax.set_xticks(range(1, 21))
    style_ax(ax, "Newly Forgotten Class Accuracy at Each Step", "Accuracy (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_newly_forgotten_heatmap(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    data = np.array([newly_forgotten_values(order_rows[key], ORDER_CONFIGS[key]["order"]) for key in ORDER_CONFIGS])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(10, data.max()))
    ax.set_yticks(range(3))
    ax.set_yticklabels([ORDER_CONFIGS[key]["label"] for key in ORDER_CONFIGS])
    ax.set_xticks(range(20))
    ax.set_xticklabels(range(1, 21))
    ax.set_xlabel("Step")
    ax.set_title("Immediate Forgetting Lag Heatmap", fontsize=12, weight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Newly forgotten class accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_class_heatmap(output_path: Path, title: str, rows: list[dict], order: list[int]) -> None:
    data = np.array([[class_accuracy(row, class_id) for row in rows] for class_id in order])
    labels = [f"{CLASS_META[class_id]['name']} ({class_id})" for class_id in order]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(20))
    ax.set_xticklabels(range(1, 21))
    ax.set_yticks(range(20))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Step")
    ax.set_title(title, fontsize=12, weight="bold")
    for y, class_id in enumerate(order):
        forget_step = order.index(class_id)
        ax.scatter(forget_step, y, marker="x", color="white", s=45, linewidths=1.4)
    fig.colorbar(im, ax=ax, label="Class accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_selected_class_plot(output_path: Path, title: str, rows: list[dict], order: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for class_id in order:
        meta = CLASS_META[class_id]
        x, y = class_series(rows, class_id)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.7,
            color=GROUP_COLORS[meta["group"]],
            label=f"{meta['name']} ({class_id}, forget@{order.index(class_id) + 1})",
        )
        ax.axvline(order.index(class_id) + 1, color=GROUP_COLORS[meta["group"]], linestyle="--", linewidth=0.8, alpha=0.13)
    ax.axhline(10, color="crimson", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(1, 21))
    style_ax(ax, title, "Accuracy (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_final_forgotten_accuracy(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    pilot_order = [item["id"] for item in PILOT_CLASSES]
    x = np.arange(len(pilot_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, key in enumerate(ORDER_CONFIGS):
        final = order_rows[key][-1]
        values = [class_accuracy(final, class_id) for class_id in pilot_order]
        ax.bar(x + (idx - 1) * width, values, width, label=ORDER_CONFIGS[key]["label"], color=ORDER_COLORS[key])
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1.1, alpha=0.8, label="10% guide")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLASS_META[c]['name']}\n({c})" for c in pilot_order], rotation=0, fontsize=8)
    ax.set_ylabel("Final accuracy (%)")
    ax.set_title("Final Accuracy of the 20 Forgotten Classes", fontsize=12, weight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_final_residual_over_10(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    residuals = []
    for key, config in ORDER_CONFIGS.items():
        for item in final_residual_items(order_rows[key], config["order"]):
            if item["is_residual"]:
                residuals.append((config["label"], f"{item['class_name']} ({item['class_id']})", item["accuracy"], ORDER_COLORS[key]))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if residuals:
        labels = [f"{order}\n{cls}" for order, cls, _, _ in residuals]
        values = [value for _, _, value, _ in residuals]
        colors = [color for *_, color in residuals]
        ax.bar(labels, values, color=colors)
        ax.axhline(10, color="crimson", linestyle="--", linewidth=1.1, alpha=0.8)
        add_value_labels(ax)
    else:
        ax.text(0.5, 0.5, "No final residual > 10%", ha="center", va="center", fontsize=14)
        ax.set_xticks([])
    ax.set_ylabel("Final accuracy (%)")
    ax.set_title("Final Residual / Rebound Classes Above 10%", fontsize=12, weight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_max_post_forget_heatmap(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    pilot_order = [item["id"] for item in PILOT_CLASSES]
    data = []
    for key, config in ORDER_CONFIGS.items():
        max_map = post_forget_max(order_rows[key], config["order"])
        data.append([max_map[class_id] for class_id in pilot_order])
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(13, 4.3))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(20, data.max()))
    ax.set_yticks(range(3))
    ax.set_yticklabels([ORDER_CONFIGS[key]["label"] for key in ORDER_CONFIGS])
    ax.set_xticks(range(len(pilot_order)))
    ax.set_xticklabels([f"{CLASS_META[c]['name']}\n({c})" for c in pilot_order], fontsize=8)
    ax.set_title("Maximum Post-Forget Accuracy by Class", fontsize=12, weight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Max accuracy after its forget step (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_post_forget_rebound_heatmap(
    output_path: Path,
    title: str,
    rows: list[dict],
    order: list[int],
) -> None:
    data = np.full((len(order), len(order)), np.nan)
    for row_idx, class_id in enumerate(order):
        forget_step = order.index(class_id)
        for step_idx in range(forget_step, len(order)):
            data[row_idx, step_idx] = class_accuracy(rows[step_idx], class_id)

    cmap = plt.cm.magma.copy()
    cmap.set_bad("#E6E6E6")
    valid_max = np.nanmax(data) if np.isfinite(data).any() else 1.0
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=max(20.0, valid_max))
    ax.set_xticks(range(20))
    ax.set_xticklabels(range(1, 21))
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"{CLASS_META[c]['name']} ({c})" for c in order], fontsize=8)
    ax.set_xlabel("Step")
    ax.set_title(title, fontsize=12, weight="bold")
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if np.isfinite(value) and value > 0:
                color = "white" if value > 8 else "black"
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="Post-forget class accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_post_forget_nonzero_count(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    pilot_order = [item["id"] for item in PILOT_CLASSES]
    x = np.arange(len(pilot_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, key in enumerate(ORDER_CONFIGS):
        stats = {item["class_id"]: item for item in post_forget_nonzero_stats(order_rows[key], ORDER_CONFIGS[key]["order"])}
        values = [stats[class_id]["nonzero_count"] for class_id in pilot_order]
        ax.bar(x + (idx - 1) * width, values, width, label=ORDER_CONFIGS[key]["label"], color=ORDER_COLORS[key])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLASS_META[c]['name']}\n({c})" for c in pilot_order], fontsize=8)
    ax.set_ylabel("Non-zero post-forget steps")
    ax.set_title("How Often Each Forgotten Class Becomes Non-Zero After Forgetting", fontsize=12, weight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_post_forget_max_accuracy(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    pilot_order = [item["id"] for item in PILOT_CLASSES]
    data = []
    for key, config in ORDER_CONFIGS.items():
        stats = {item["class_id"]: item for item in post_forget_nonzero_stats(order_rows[key], config["order"])}
        data.append([stats[class_id]["max_accuracy"] for class_id in pilot_order])
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(20.0, data.max()))
    ax.set_yticks(range(3))
    ax.set_yticklabels([ORDER_CONFIGS[key]["label"] for key in ORDER_CONFIGS])
    ax.set_xticks(range(len(pilot_order)))
    ax.set_xticklabels([f"{CLASS_META[c]['name']}\n({c})" for c in pilot_order], fontsize=8)
    ax.set_title("Maximum Non-Zero Post-Forget Accuracy", fontsize=12, weight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if value > 0:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Max accuracy after its forget step (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_top_rebound_trajectories(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, key in zip(axes, ORDER_CONFIGS):
        config = ORDER_CONFIGS[key]
        stats = sorted(
            post_forget_nonzero_stats(order_rows[key], config["order"]),
            key=lambda item: (item["max_accuracy"], item["nonzero_count"]),
            reverse=True,
        )[:5]
        for item in stats:
            class_id = item["class_id"]
            steps, values = trajectory_after_forget(order_rows[key], config["order"], class_id)
            ax.plot(steps, values, marker="o", linewidth=1.9, label=f"{CLASS_META[class_id]['name']} ({class_id})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(10, color="crimson", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(f"{config['label']} top rebound trajectories", loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlabel("Step")
    axes[-1].set_xticks(range(1, 21))
    fig.suptitle("Top Non-Zero Post-Forget Rebound Trajectories", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_rebound_trajectory(output_path: Path, class_id: int, order_rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    meta = CLASS_META[class_id]
    for key, config in ORDER_CONFIGS.items():
        steps, values = trajectory_after_forget(order_rows[key], config["order"], class_id)
        ax.plot(steps, values, marker="o", linewidth=2.2, label=f"{config['label']} (forget@{config['order'].index(class_id) + 1})", color=ORDER_COLORS[key])
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1.1, alpha=0.8, label="10% guide")
    ax.set_xticks(range(1, 21))
    style_ax(ax, f"{meta['name'].title()} ({class_id}) Post-Forget Trajectory", "Accuracy (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_group_mean_plot(output_path: Path, title: str, rows: list[dict], order: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for group, label in GROUP_LABELS.items():
        class_ids = [class_id for class_id in order if CLASS_META[class_id]["group"] == group]
        values = []
        for row in rows:
            values.append(np.mean([class_accuracy(row, class_id) for class_id in class_ids]))
        ax.plot(range(1, 21), values, marker="o", linewidth=2.0, color=GROUP_COLORS[group], label=label)
    ax.axhline(10, color="crimson", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(1, 21))
    style_ax(ax, title, "Mean accuracy of group pilot classes (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_order_group_timeline(output_path: Path) -> None:
    group_index = {group: idx for idx, group in enumerate(GROUP_LABELS)}
    fig, axes = plt.subplots(3, 1, figsize=(12, 5.8), sharex=True)
    for ax, key in zip(axes, ORDER_CONFIGS):
        config = ORDER_CONFIGS[key]
        y = [group_index[CLASS_META[class_id]["group"]] for class_id in config["order"]]
        colors = [GROUP_COLORS[CLASS_META[class_id]["group"]] for class_id in config["order"]]
        ax.scatter(range(1, 21), y, s=120, color=colors)
        for step, class_id in enumerate(config["order"], start=1):
            ax.text(step, y[step - 1], str(class_id), color="white", ha="center", va="center", fontsize=7, weight="bold")
        ax.set_yticks(list(group_index.values()))
        ax.set_yticklabels([GROUP_LABELS[group] for group in GROUP_LABELS], fontsize=8)
        ax.set_title(config["label"], loc="left", fontsize=11, weight="bold")
        ax.grid(True, axis="x", alpha=0.18)
    axes[-1].set_xticks(range(1, 21))
    axes[-1].set_xlabel("Forget step")
    fig.suptitle("Order Design: Semantic Group Timeline", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_retain_forget_tradeoff(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for key, config in ORDER_CONFIGS.items():
        forget = [fnum(row["forget_accuracy"]) for row in order_rows[key]]
        retain = [fnum(row["retain_accuracy"]) for row in order_rows[key]]
        ax.plot(forget, retain, marker="o", linewidth=2.0, color=ORDER_COLORS[key], label=config["label"])
        for step, (x, y) in enumerate(zip(forget, retain), start=1):
            if step in {1, 5, 10, 15, 20}:
                ax.text(x, y, str(step), fontsize=8)
    ax.set_xlabel("Cumulative forget accuracy (%)")
    ax.set_ylabel("Retain accuracy (%)")
    ax.set_title("Retain-vs-Forget Tradeoff Across Steps", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_runtime_total(output_path: Path, order_progress: dict[str, list[dict]]) -> None:
    labels = [ORDER_CONFIGS[key]["label"] for key in ORDER_CONFIGS]
    values = [total_duration(order_progress[key]) / 60.0 for key in ORDER_CONFIGS]
    colors = [ORDER_COLORS[key] for key in ORDER_CONFIGS]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Runtime (min)")
    ax.set_title("Total Runtime by Order", fontsize=12, weight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    add_value_labels(ax, fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_runtime_stacked_by_step(output_path: Path, order_progress: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    stage_colors = {"mask": "#4C78A8", "unlearn": "#F58518", "eval": "#54A24B"}
    for ax, key in zip(axes, ORDER_CONFIGS):
        steps, stages = duration_by_step(order_progress[key])
        bottom = np.zeros(len(steps))
        for stage in ["mask", "unlearn", "eval"]:
            values = np.array(stages[stage])
            ax.bar(steps, values, bottom=bottom, color=stage_colors[stage], label=stage)
            bottom += values
        ax.set_title(ORDER_CONFIGS[key]["label"], loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("sec")
        ax.grid(True, axis="y", alpha=0.22)
    axes[-1].set_xlabel("Step")
    axes[-1].set_xticks(range(1, 21))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Runtime Breakdown by Step", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_runtime_stage_breakdown(output_path: Path, order_progress: dict[str, list[dict]]) -> None:
    labels = [ORDER_CONFIGS[key]["label"] for key in ORDER_CONFIGS]
    x = np.arange(len(labels))
    width = 0.25
    stage_colors = {"mask": "#4C78A8", "unlearn": "#F58518", "eval": "#54A24B"}
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for idx, stage in enumerate(["mask", "unlearn", "eval"]):
        values = [duration_by_stage(order_progress[key])[stage] / 60.0 for key in ORDER_CONFIGS]
        ax.bar(x + (idx - 1) * width, values, width, color=stage_colors[stage], label=stage)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (min)")
    ax.set_title("Runtime by Stage", fontsize=12, weight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)
    add_value_labels(ax, fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fig_link(output_dir: Path, filename: str) -> str:
    return f"./{output_dir.name}/{filename}"


def class_label(class_id: int) -> str:
    meta = CLASS_META[class_id]
    return f"{meta['name']} / {meta['zh']} ({class_id})"


def inline_class_list(class_ids: list[int]) -> str:
    return ", ".join(class_label(class_id) for class_id in class_ids)


def final_metric_table(order_rows: dict[str, list[dict]], order_progress: dict[str, list[dict]]) -> list[list[str]]:
    rows = []
    for key, config in ORDER_CONFIGS.items():
        final = order_rows[key][-1]
        rows.append(
            [
                config["label"],
                f"{fnum(final['forget_accuracy']):.2f}",
                f"{fnum(final['UA']):.2f}",
                f"{fnum(final['retain_accuracy']):.2f}",
                f"{fnum(final['full_test_accuracy']):.2f}",
                f"{total_duration(order_progress[key]) / 60.0:.1f}",
            ]
        )
    return rows


def lag_summary_table(order_rows: dict[str, list[dict]]) -> list[list[str]]:
    rows = []
    for key, config in ORDER_CONFIGS.items():
        items = lag_items(order_rows[key], config["order"])
        failures = [item for item in items if item["is_failure"]]
        worst = max(items, key=lambda item: item["accuracy"])
        rows.append(
            [
                config["label"],
                str(len(failures)),
                class_label(worst["class_id"]),
                f"{worst['accuracy']:.1f}",
            ]
        )
    return rows


def residual_summary_table(order_rows: dict[str, list[dict]]) -> list[list[str]]:
    rows = []
    for key, config in ORDER_CONFIGS.items():
        residuals = [item for item in final_residual_items(order_rows[key], config["order"]) if item["is_residual"]]
        if not residuals:
            rows.append([config["label"], "0", "-", "-"])
        for item in residuals:
            rows.append(
                [
                    config["label"],
                    "1",
                    class_label(item["class_id"]),
                    f"{item['accuracy']:.1f}",
                ]
            )
    return rows


def top_post_forget_table(order_rows: dict[str, list[dict]]) -> list[list[str]]:
    rows = []
    for key, config in ORDER_CONFIGS.items():
        max_map = post_forget_max(order_rows[key], config["order"])
        top_items = sorted(max_map.items(), key=lambda item: item[1], reverse=True)[:5]
        for class_id, value in top_items:
            rows.append([config["label"], class_label(class_id), f"{value:.1f}"])
    return rows


def nonzero_rebound_summary_table(order_rows: dict[str, list[dict]]) -> list[list[str]]:
    rows = []
    for key, config in ORDER_CONFIGS.items():
        stats = post_forget_nonzero_stats(order_rows[key], config["order"])
        classes_with_nonzero = [item for item in stats if item["nonzero_count"] > 0]
        total_nonzero_cells = sum(item["nonzero_count"] for item in stats)
        top_by_count = max(stats, key=lambda item: (item["nonzero_count"], item["max_accuracy"]))
        top_by_max = max(stats, key=lambda item: (item["max_accuracy"], item["nonzero_count"]))
        rows.append(
            [
                config["label"],
                f"{len(classes_with_nonzero)} / 20",
                str(total_nonzero_cells),
                f"{class_label(top_by_count['class_id'])}: {top_by_count['nonzero_count']} steps",
                f"{class_label(top_by_max['class_id'])}: {top_by_max['max_accuracy']:.1f}%",
            ]
        )
    return rows


def top_nonzero_rebound_table(order_rows: dict[str, list[dict]], limit: int = 18) -> list[list[str]]:
    rows = []
    for key, config in ORDER_CONFIGS.items():
        stats = sorted(
            [item for item in post_forget_nonzero_stats(order_rows[key], config["order"]) if item["nonzero_count"] > 0],
            key=lambda item: (item["max_accuracy"], item["nonzero_count"]),
            reverse=True,
        )[:limit]
        for item in stats:
            rows.append(
                [
                    config["label"],
                    class_label(item["class_id"]),
                    str(item["forget_step"]),
                    f"{item['nonzero_count']} / {item['observed_steps']}",
                    f"{item['max_accuracy']:.1f}",
                    f"{item['final_accuracy']:.1f}",
                ]
            )
    return rows


def smoke_table(smoke_progress: dict[str, list[dict]]) -> list[list[str]]:
    return [
        [config["label"], config["smoke_run_id"], smoke_status(smoke_progress[key])]
        for key, config in ORDER_CONFIGS.items()
    ]


def write_report(
    report_path: Path,
    output_dir: Path,
    order_rows: dict[str, list[dict]],
    order_progress: dict[str, list[dict]],
    smoke_progress: dict[str, list[dict]],
) -> None:
    normal_c21 = " -> ".join(f"{v:g}" for _, v in zip(*trajectory_after_forget(order_rows["normal"], ORDER_CONFIGS["normal"]["order"], 21)))
    clustered_c21 = " -> ".join(f"{v:g}" for _, v in zip(*trajectory_after_forget(order_rows["clustered"], ORDER_CONFIGS["clustered"]["order"], 21)))
    interleaved_c21 = " -> ".join(f"{v:g}" for _, v in zip(*trajectory_after_forget(order_rows["interleaved"], ORDER_CONFIGS["interleaved"]["order"], 21)))

    report = f"""# CIFAR-100 Sequential Incremental Ordered Pilot 詳細分析報告

## 0. 大綱

1. Executive summary：先給結論，這次沒有 immediate forgetting lag，但有少數 final residual / rebound。
2. Experiment setup：說明 CIFAR-100、ResNet-18、seed 1、20-class animal pilot 與正確 incremental data flow。
3. Baseline comparison：比較 CIFAR-10 與 CIFAR-100 baseline accuracy，說明為什麼 CIFAR-100 更適合 failure hunting。
4. Ordered comparison：比較 Normal、Clustered、Interleaved 三種順序設計。
5. Aggregate metrics：用 final metrics 和 step-wise curves 檢查整體 forgetting / retain tradeoff。
6. Immediate lag analysis：用 newly forgotten class 當步 accuracy 判斷是否有當步遺忘失效。
7. Residual / rebound analysis：檢查 old forgotten classes 後續是否回升，重點分析 class 21。
8. Class-wise / group-wise trajectories：看 20 個 pilot classes 與 4 個 semantic groups 的變化。
9. Runtime and smoke gate：確認 smoke-first queue 成功與 runtime 成本。
10. Conclusion：整理本次 CIFAR-100 pilot 對後續實驗設計的意義。

## 1. Executive Summary

本次 CIFAR-100 ordered pilot 的主要結論是：**沒有觀察到 immediate forgetting lag 型遺忘失效，但觀察到少數 old forgotten class 的 final residual / rebound 訊號**。

若採用原先定義的 failure 標準，也就是 newly forgotten class 在當步 accuracy `>10%`，三條 order 都沒有 failure。`Normal / Clustered / Interleaved` 的 lag steps 都是 `0`，worst newly forgotten class accuracy 都只有 `2.0%`。這表示當某個 class 被指定 forget 的那一步，SalUn / RL unlearning 幾乎都能立即把該 class 壓到接近 0。

但若用更嚴格的 final per-class 檢查，`class 21` 在三條 order 最後仍高於或接近 `10%`：`Normal=17.0%`、`Clustered=11.0%`、`Interleaved=11.0%`。因此本次真正看到的不是「當步忘不掉」，而是「已忘 class 在後續 sequential steps 中被 shared representation / decision boundary 漂移帶回一點」的 rebound / residual 現象。

## 2. Experiment Setup

本次實驗使用 `CIFAR-100 / ResNet-18 / seed 1`，以 20 個 animal fine classes 作為 pilot subset。選擇 CIFAR-100 的原因是 fine-grained classes 較多，動物類別之間也更容易共享特徵，因此比 CIFAR-10 更容易放大 sequential unlearning 中的邊界漂移或 class rebound。

CIFAR-100 和 CIFAR-10 最大的差異不是影像大小，而是 label granularity。兩者都是 `32x32` 彩色影像，但 CIFAR-10 只有 10 個 broad classes，CIFAR-100 則有 100 個 fine classes，且官方同時提供 20 個 coarse superclasses。這代表 CIFAR-100 的單一 fine class 每類測試樣本較少、類別邊界較細，也更常出現「不同 fine classes 共享局部外觀或語義特徵」的情況。對 unlearning 來說，這會讓 class-wise forgetting 更有挑戰，因為模型可能不是只用單一 class-specific neuron 記住某個類別，而是把多個相近類別放在共享 backbone representation 裡一起判別。

本次 20-class pilot 刻意選擇 animal-heavy subset，而不是平均抽樣 CIFAR-100。這樣設計的用意是提高類別間共享特徵的比例，例如毛色、身形、臉部輪廓、四足姿態、自然背景等。若 sequential unlearning 會出現 failure，這種高混淆 subset 比隨機 class subset 更容易暴露「當步忘不乾淨」或「舊 class 後續回升」。

Pilot subset 依語義群整理如下，括號中的數字是 CIFAR-100 fine-class id：

{markdown_table(["Group", "Fine classes"], [
    ["large carnivores", inline_class_list([3, 42, 43, 88, 97])],
    ["large omnivores/herbivores", inline_class_list([15, 19, 21, 31, 38])],
    ["medium mammals", inline_class_list([34, 63, 64, 66, 75])],
    ["small mammals", inline_class_list([36, 50, 65, 74, 80])],
])}

需要特別注意的是，這裡的 group 是為了實驗分析與 order design 使用的語義分組，並不是把模型訓練成 coarse classifier。實際 evaluation 仍然使用 CIFAR-100 的 100-way fine-class accuracy；因此 `chimpanzee / 黑猩猩 (21)` 的 residual 代表模型仍能在 100-way fine labels 中部分辨識 class 21，而不是只辨識到某個 animal coarse group。

正確 incremental data flow 固定如下：

- 每一步 `forget loader` 只包含當步 newly forgotten class。
- `retain loader` 只包含尚未被指定 forget 的 classes。
- 舊 forgotten classes 永久排除在後續 train loaders 外。
- 舊 forgotten classes 只會出現在 cumulative evaluation，不會回到 training。

這個設定很重要，因為它讓後續 class accuracy 的回升不能被解讀成資料重新訓練造成的 recovery，而比較像是 shared model parameters 被後續 unlearning 間接改動後的 rebound。

## 3. Baseline Comparison

在目前 `ResNet-18 / seed 1` 設定下，CIFAR-10 baseline 的 full test accuracy 為 `94.52%`，CIFAR-100 baseline 的 full test accuracy 為 `70.47%`；CIFAR-100 約低 `24.05` 個百分點。這個差距代表 CIFAR-100 本身分類難度明顯更高，也更適合用來測試 sequential forgetting 是否會出現脆弱點。

目前不建議為了追求更高 CIFAR-100 baseline 而中止重訓，因為這次目標是 failure hunting / lag observation，不是刷新 CIFAR-100 accuracy。`70.47%` 已足以表示模型具備有效分類能力，而 final forgetting metrics 也能清楚顯示 unlearning 是否真正壓低目標類別。

## 4. Ordered Comparison

三條 ordered variants 使用同一組 20 classes，只改 forget order。

`Normal`

```text
{",".join(str(x) for x in ORDER_CONFIGS["normal"]["order"])}
```

`Clustered`

```text
{",".join(str(x) for x in ORDER_CONFIGS["clustered"]["order"])}
```

`Interleaved`

```text
{",".join(str(x) for x in ORDER_CONFIGS["interleaved"]["order"])}
```

![Order Group Timeline]({fig_link(output_dir, "order_group_timeline.png")})

這張圖把每一步忘掉的 class 依 semantic group 標出。`Clustered` 會連續處理同一語義群，因此同群 shared features 受到連續壓力；`Interleaved` 則在不同語義群間切換，較適合觀察 heterogeneous steps 是否會讓已忘 class 回彈。`Normal` 則作為固定順序 baseline，讓我們確認這組 subset 在一般排列下是否已經容易失效。

## 5. Aggregate Metrics

{markdown_table(["Order", "Forget acc", "UA", "Retain acc", "Full test acc", "Runtime (min)"], final_metric_table(order_rows, order_progress))}

![Final Metrics Grouped Bar]({fig_link(output_dir, "final_metrics_grouped_bar.png")})

這張 grouped bar 圖濃縮 final endpoint。三條 order 的 final forget accuracy 都非常低：`Normal=2.20%`、`Clustered=1.20%`、`Interleaved=1.40%`，對應 UA 都高於 `97%`。這表示從 cumulative forgetting 的角度看，三條路線最後都成功把 20 個 forgotten classes 壓到很低。

Retain accuracy 則維持在約 `71.38% ~ 71.65%`，三條 order 差距很小。這點很重要，因為它說明 final forgetting 並不是靠整個模型崩壞達成；retain side 仍然保有和 CIFAR-100 baseline 相近的有效分類能力。

![Forget Accuracy vs Step]({fig_link(output_dir, "forget_accuracy_vs_step.png")})

`forget_accuracy` 是 cumulative forgotten set 的平均 accuracy。這張圖主要看「已經被指定忘掉的集合」是否隨 step 增加而維持低值。三條 order 整體都維持在很低區間，表示 cumulative forgetting 沒有隨著 step 變多而明顯失控。

![UA vs Step]({fig_link(output_dir, "ua_vs_step.png")})

UA 是 `100 - forget_accuracy`。因為 forget accuracy 很低，所以 UA 幾乎全程維持高值。這張圖和 forget accuracy 是互補視角，能直觀看出三條 order 的 forgetting side 都很強。

![Retain Accuracy vs Step]({fig_link(output_dir, "retain_accuracy_vs_step.png")})

retain accuracy 用來檢查 unlearning 是否犧牲太多 non-forgotten classes。三條曲線都沒有一路崩掉，代表方法在多步忘卻下仍保留相當的 retain 能力。這使得後面看到的 class 21 residual 更像局部 rebound，而不是整體模型不穩定。

![Full Test Accuracy vs Step]({fig_link(output_dir, "full_test_accuracy_vs_step.png")})

full test accuracy 會隨著 forgotten set 增加而下降，這是合理現象，因為 evaluation 仍包含已被刻意遺忘的 classes。三條 order 的 final full test accuracy 幾乎一致，表示 order 對 final endpoint 的影響不大，差異主要會出現在 class-wise trajectory 與 residual pattern。

![Retain vs Forget Tradeoff]({fig_link(output_dir, "retain_vs_forget_tradeoff.png")})

這張圖把每一步放在 retain accuracy 與 cumulative forget accuracy 的二維平面上。理想方向是左上角：forget accuracy 低、retain accuracy 高。本次三條 order 都集中在低 forget accuracy 區間，顯示主要 tradeoff 不是「能不能忘」，而是少數 class 在後續 step 是否會局部回升。

## 6. Immediate Forgetting Lag Analysis

本節使用原先定義的 failure 標準：newly forgotten class 在當步 evaluation accuracy `>10%`，視為 immediate forgetting lag / 遺忘失效。

{markdown_table(["Order", "Lag steps (>10%)", "Worst class", "Worst newly forgotten acc"], lag_summary_table(order_rows))}

![Newly Forgotten Class Accuracy vs Step]({fig_link(output_dir, "newly_forgotten_class_accuracy_vs_step.png")})

這張圖只看「當步新指定 forget 的 class」在當步 evaluation 的 accuracy。三條線都遠低於 `10%` guide line，最大值只有 step 1 的 `bear / 熊 (3)=2.0%`。因此，若把 failure 定義為 immediate forgetting lag，本次 CIFAR-100 pilot 並沒有出現失效。

![Newly Forgotten Accuracy Heatmap]({fig_link(output_dir, "newly_forgotten_accuracy_heatmap.png")})

heatmap 更直接地顯示每條 order、每一步的新忘 class accuracy。圖中幾乎全是 `0` 或接近 `0`，說明單步施力對 newly forgotten class 非常有效。這也呼應先前 CIFAR-10 single-class forgetting 的觀察：單一類別被指定忘掉時，方法通常能很快壓低該 class。

## 7. Final Residual / Rebound Analysis

Immediate lag 沒有發生，但 final per-class 檢查揭露了另一種更細的問題：old forgotten class 在後續 steps 中可能回升。這種現象不是 recovery data flow，因為舊 forgotten classes 沒有被放回 training；比較合理的解讀是後續 unlearning 其他 classes 時，同一套 shared backbone / decision boundary 發生漂移，讓少數舊 class 的判別能力局部恢復。

{markdown_table(["Order", "Residual count", "Class", "Final acc"], residual_summary_table(order_rows))}

![Final Residual Over 10]({fig_link(output_dir, "final_residual_over_10.png")})

這張圖只列 final accuracy 高於 `10%` 的 forgotten classes。主要殘留集中在 `class 21 (chimpanzee / 黑猩猩)`：Normal 最後回到 `17.0%`，Clustered 與 Interleaved 都是 `11.0%`。這不是大規模 forgetting failure，因為 cumulative forget accuracy 仍只有 `1.2% ~ 2.2%`；但它是一個值得報告的 residual signal。

![Final Forgotten Class Accuracy by Order]({fig_link(output_dir, "final_forgotten_class_accuracy_by_order.png")})

這張圖把 20 個 pilot classes 在 final checkpoint 的 accuracy 全部列出。大部分 class 都停在 `0% ~ 2%` 左右，只有少數 class 接近或超過 `10%`。因此 final forgetting 整體很好，但不是所有 class 都被同等程度地壓到完全 0。

![Max Post Forget Accuracy Heatmap]({fig_link(output_dir, "max_post_forget_accuracy_heatmap.png")})

這張 heatmap 看每個 class 在「被忘掉之後」曾經回升到的最大 accuracy，因此比 final-only 更敏感。它能回答：某個 class 即使最後不高，中間是否曾 rebound。圖中可以看到 `class 21` 與部分早期 class 曾出現較高 post-forget value，表示 sequential updates 對 old forgotten classes 的影響不是單調下降。

### 7.1 Non-zero Post-Forget Rebound

如果把 post-forget rebound 的門檻從 `>10%` 放寬成「只要不是 `0%` 就算有殘留訊號」，觀察會更細。這個標準比較敏感，因此不應直接等同於嚴重 failure；它比較適合用來看 forgotten class 在後續 steps 中是否完全維持 0，或是否有任何局部回升。

{markdown_table(["Order", "Classes with non-zero post-forget", "Total non-zero cells", "Most frequent non-zero", "Largest rebound"], nonzero_rebound_summary_table(order_rows))}

![Normal Post-Forget Rebound Heatmap]({fig_link(output_dir, "normal_post_forget_rebound_heatmap.png")})

Normal 的 non-zero heatmap 顯示，雖然 newly forgotten class 當步幾乎都能被壓到 `0%`，但部分早期 forgotten classes 在後續 steps 會零星回升。最明顯的是 `chimpanzee / 黑猩猩 (21)`，它不只是偶發非零，而是在後段多次維持非零並超過 `10%`；`bear / 熊 (3)` 和 `elephant / 大象 (31)` 也曾在中途回升，但最後沒有形成同樣明顯的 final residual。

![Clustered Post-Forget Rebound Heatmap]({fig_link(output_dir, "clustered_post_forget_rebound_heatmap.png")})

Clustered 的 heatmap 可以看到同語義群連續 forget 後，多數 class 被壓得很低，但 early forgotten class 仍可能在後續 steps 中短暫非零。這說明 clustered order 雖然 final forget accuracy 最好，仍不是把每個 class 永久鎖在 `0%`；它比較像是降低 rebound 幅度，而不是完全消除 rebound。

![Interleaved Post-Forget Rebound Heatmap]({fig_link(output_dir, "interleaved_post_forget_rebound_heatmap.png")})

Interleaved 的 non-zero pattern 較分散，因為它在不同語義群間切換。即使如此，final residual 仍主要集中在 `chimpanzee / 黑猩猩 (21)`，而不是所有 class 都普遍回升。這表示 heterogeneous order 沒有造成大規模遺忘崩潰，但會讓某些 shared-feature class 在後續 step 出現小幅殘留。

![Post-Forget Non-Zero Count by Class]({fig_link(output_dir, "post_forget_nonzero_count_by_class.png")})

這張圖統計每個 class 在被忘之後有多少個 observation steps accuracy 不是 `0%`。它回答的是「哪個 class 最常回升」，不是「哪個 class 回升最大」。因此它比 final residual 更能揭露長尾小幅 rebound。若某 class non-zero count 高但 final 不高，代表它有中途震盪；若 non-zero count 高且 final 也高，才是比較值得警覺的 residual。

![Post-Forget Max Accuracy by Class]({fig_link(output_dir, "post_forget_max_accuracy_by_class.png")})

這張圖則看每個 class 在被忘後曾經達到的最大 accuracy。和 non-zero count 搭配看，可以區分兩種情況：一種是頻繁但幅度小的殘留，另一種是次數少但幅度大的 rebound。本次 `class 21` 同時具備後段持續非零與較高 max value，因此比單純中途跳動的 class 更重要。

![Top Non-Zero Rebound Trajectories]({fig_link(output_dir, "top_rebound_trajectories_by_order.png")})

這張圖把每條 order 中 post-forget 最大值較高的 classes 抽出來畫軌跡。它比 heatmap 更容易看出「回升是在某幾個 step 突然發生，還是逐步累積」。Normal 的 `class 21` 呈現較明顯的逐步回升；`class 3` 則比較像中途震盪，最後又降下來。

{markdown_table(["Order", "Class", "Forget step", "Non-zero / observed", "Max acc", "Final acc"], top_nonzero_rebound_table(order_rows, limit=6))}

這張表把 non-zero post-forget 訊號列成數字。用這個標準時，`class 21` 不是唯一有 rebound 的 class；`class 3` 和 `class 31` 也有中途非零或短暫回升。不過 `class 21` 是唯一在三條 order 的 final checkpoint 都仍高於 `10%` 的 class，所以它仍是最重要的 residual case。

![Class 21 Rebound Trajectory]({fig_link(output_dir, "class21_rebound_trajectory.png")})

`class 21 (chimpanzee / 黑猩猩)` 是本次最重要的 rebound case。三條 order 中，它在被忘當下都接近 `0%`，但後續逐步回升。Normal 軌跡為：

```text
{normal_c21}
```

Clustered 軌跡為：

```text
{clustered_c21}
```

Interleaved 軌跡為：

```text
{interleaved_c21}
```

這個模式支持一個比較謹慎的結論：方法對當步 class 的直接 unlearning 很有效，但 sequential setting 下，舊 forgotten class 仍可能被後續 decision boundary drift 間接影響。這是「rebound residual」，不是「當步忘不掉」。

### 7.2 Why Post-Forget Rebound Happens

Post-forget rebound 比較合理的解讀不是資料回灌，也不是 immediate forgetting failure，而是 sequential unlearning 在共享參數空間中反覆更新後，讓少數舊 class 的 decision margin 局部回升。這裡仍應把原因寫成 evidence-supported hypothesis，而不是已經被直接證明的 causal claim；若要進一步證明，需要追加 logit margin、confusion matrix、feature embedding 或 per-step gradient/update overlap 分析。

第一個機制是 shared mammal features。本次 subset 刻意選 animal-heavy classes，許多類別會共用毛髮紋理、臉部輪廓、四足或直立姿態、自然背景、近景動物構圖等特徵。Unlearning 某一個 fine class 時，方法可以把該 class 的當步 accuracy 壓到接近 `0%`，但 shared backbone 中仍可能保留服務其他 mammal classes 的特徵。後續 steps 繼續更新這些共享特徵時，舊 forgotten class 可能重新落回可被正確分類的區域。

第二個機制是 no post-forget constraint。正確 incremental data flow 會把舊 forgotten classes 永久排除在後續 train loaders 外，這能排除 recovery training 的解釋；但它同時也代表後續 optimization 沒有直接約束「已忘 class 必須維持低 accuracy」。因此當後續 step 為了新 forget class 與 retain set 調整參數時，只要沒有再次看到舊 forgotten examples，就不會直接懲罰舊 class 的 logit margin 回升。

第三個機制是 decision boundary drift。每一步 RL-SalUn 都會在 forget objective 和 retain objective 之間重新平衡分類邊界。即使舊 class 的 training samples 不再出現，和它相近的 retain / newly forgotten classes 仍會推動 classifier head 與 backbone representation。若這些更新讓舊 class 對原本 label 的 logit 相對於競爭 labels 上升，per-class accuracy 就可能從 `0%` 回到非零，甚至在少數 cases 超過 `10%`。

第四個機制是 incremental mask/update interference。每一步 mask 與 unlearning update 主要針對當步 newly forgotten class 產生，並不是針對所有歷史 forgotten classes 做全域約束。後續 step 的 mask 可能落在不同參數區域，也可能為了 retain performance 改寫部分先前被壓制的 shared features。這不代表前一步 unlearning 失效，而是表示 sequential local updates 之間可能互相干擾，使舊 class 的抑制效果不是單調不可逆。

第五個機制是 order / exposure effect。越早被忘掉的 class，越長時間暴露在後續 unlearning steps 的邊界漂移之下，因此更容易出現中途 rebound。`class 3 (bear / 熊)` 三條 order 都在 step 1 被忘，non-zero count 很高，正符合 exposure effect；但它 final residual 不如 `class 21` 明顯，表示 exposure 只是一個條件，不是充分原因。真正值得警覺的是 exposure、shared features、以及後段 margin 回升同時出現的 class。

這個解讀也能說明為什麼 `class 21 (chimpanzee / 黑猩猩)` 是本次最重要的 rebound case。它不是單純「很難忘」，因為它在 Normal 的 forget step 4、Clustered 的 forget step 8、Interleaved 的 forget step 10 當下都掉到 `0%`。更準確的說法是：class 21 被忘掉後，比較容易被後續 sequential updates 局部帶回。

`class 21` 在本 subset 中像一個 semantic bridge。它被歸在 `large_omnivores_and_herbivores`，但視覺上同時可能和大型哺乳類、小型/中型哺乳類、靈長類臉部與肢體姿態、毛髮質地、自然背景等特徵共享 representation。當後續 steps 繼續 unlearn elephant / 大象、kangaroo / 袋鼠、fox / 狐狸、porcupine / 豪豬、possum / 負鼠、raccoon / 浣熊、skunk / 臭鼬、small mammals 等類別時，模型可能反覆調整與 mammal recognition 相關的共享方向，間接讓 class 21 的 logit margin 回升。

三條 order 的差異也支持這個假設。Normal 中 class 21 在 step 4 就被忘，後續還有 16 個 animal classes 會推動 shared representation，因此它從 step 11 後多次超過 `10%`，最後停在 `17%`。Clustered 中 class 21 到 step 8 才被忘，而且同群大型雜食/草食動物被連續處理，可能較集中地壓低相關 shared features，所以 final residual 降到 `11%`，但沒有完全消失。Interleaved 中 class 21 到 step 10 才被忘，後續 exposure 更短，final 也是 `11%`；這說明 heterogeneous order 沒有造成大規模失效，但仍可能留下局部 residual。

最後需要注意 interpretation guardrail。CIFAR-100 每個 fine class 的 test samples 約為 100 張，所以 `1%` per-class accuracy 大約只對應 1 張圖；non-zero post-forget cell 很適合用來看細微 rebound 痕跡，但不應直接等同嚴重 failure。本報告比較保守地把 `>10%` final residual 視為較值得警覺的訊號，因為它代表約 10 張以上測試圖在 final checkpoint 仍被正確辨識。以這個標準看，本次核心結論仍是：immediate lag 沒有發生，post-forget rebound 是少數 class 的局部 residual，而且目前原因分析仍需後續 logit / margin / confusion evidence 驗證。

![Class 3 Rebound Trajectory]({fig_link(output_dir, "class3_rebound_trajectory.png")})

`class 3 (bear / 熊)` 是三條 order 的第一個 forget class，因此它暴露在最多後續 unlearning steps 之下。它不像 class 21 一樣 final residual 明顯，但中間也曾在部分 order 回升到 `10%` 以上。這張圖說明早期 forgotten class 更容易受到後續多步更新累積影響。

![Class 31 Rebound Trajectory]({fig_link(output_dir, "class31_rebound_trajectory.png")})

`class 31 (elephant / 大象)` 則是 Normal 中較明顯的中途 rebound case。它在 Normal 後段曾達到 `13%`，但 final 回到 `10%`，沒有超過 final residual threshold。這個例子提醒我們：若只看 final checkpoint，會漏掉一些中途回升；若只看 non-zero rebound，又可能把短暫震盪和持續 failure 混在一起。因此報告中需要同時呈現 immediate lag、non-zero rebound、final residual 三種層次。

## 8. Class-Wise Trajectories

![Normal Selected Class Accuracy]({fig_link(output_dir, "normal_selected_class_accuracy.png")})

Normal trajectory 顯示，大多數 class 在被忘之後立即掉到接近 0。比較值得注意的是早期 forgotten classes 後續仍可能有小幅震盪，尤其 `class 21` 在後段回升較明顯。這表示 Normal 沒有 immediate lag，但有少數 old class residual。

![Clustered Selected Class Accuracy]({fig_link(output_dir, "clustered_selected_class_accuracy.png")})

Clustered trajectory 的特色是同群 classes 連續被忘，shared semantic features 會被連續壓制。final forgetting 最好的是 Clustered，forget accuracy 只有 `1.20%`、UA `98.80%`。不過它仍未完全消除 class 21 residual，表示 clustered pressure 有幫助但不是保證所有 class final 皆為 0。

![Interleaved Selected Class Accuracy]({fig_link(output_dir, "interleaved_selected_class_accuracy.png")})

Interleaved 在不同 groups 之間切換，原本預期可能更容易拖長 immediate lag；但結果顯示 newly forgotten class 當步仍能被快速壓低。它的 final residual 也主要集中在 class 21，沒有形成多 class 大規模失效。

![Normal Class Accuracy Heatmap]({fig_link(output_dir, "normal_class_accuracy_heatmap.png")})

Normal heatmap 用顏色顯示 20 個 pilot classes 在 20 steps 中的 accuracy。白色叉號代表該 class 被指定 forget 的 step。大部分 class 在叉號後顏色迅速變暗，代表 accuracy 被壓低；少數 class 在後段變亮，代表 rebound。

![Clustered Class Accuracy Heatmap]({fig_link(output_dir, "clustered_class_accuracy_heatmap.png")})

Clustered heatmap 更能看出語義群連續 forget 的效果。同群連續施壓後，多數 class 在被忘後維持低值。這支持 clustered order 對 cumulative forgetting 較穩的觀察，但 class 21 的殘留仍顯示 shared representation 不是完全不可逆地被刪除。

![Interleaved Class Accuracy Heatmap]({fig_link(output_dir, "interleaved_class_accuracy_heatmap.png")})

Interleaved heatmap 則用來檢查 heterogeneous order 是否導致更明顯 rebound。結果看起來沒有導致 immediate lag，也沒有造成 final 大規模失敗；但 class 21 和 class 3 的局部回升仍提醒我們，order 會改變 rebound 出現的位置與幅度。

## 9. Group-Wise Trajectories

![Normal Group Mean Accuracy]({fig_link(output_dir, "group_mean_accuracy_vs_step_normal.png")})

這張圖把 20 classes 依 coarse group 聚合，觀察各 group 的平均 accuracy。Normal 下不同 groups 被逐步拉低，顯示 forgetting pressure 不是只作用於單一群。若某 group 在後段回升，代表該群 shared features 可能被後續 retain / forget boundary 間接恢復。

![Clustered Group Mean Accuracy]({fig_link(output_dir, "group_mean_accuracy_vs_step_clustered.png")})

Clustered 的 group mean 最能呈現「同語義群連續施壓」效果。某個 group 被連續 forget 時，其平均 accuracy 會快速下降。這也是 Clustered final forget accuracy 最低的可能原因之一。

![Interleaved Group Mean Accuracy]({fig_link(output_dir, "group_mean_accuracy_vs_step_interleaved.png")})

Interleaved 的 group mean 較分散，因為每個 group 的 forget pressure 被拆開。即使如此，newly forgotten class 仍能被有效壓低，代表目前超參下 immediate forgetting 能力足夠強，沒有被 heterogeneous order 明顯破壞。

## 10. Runtime and Smoke Gate

Smoke gate 結果如下：

{markdown_table(["Order", "Smoke run ID", "Status"], smoke_table(smoke_progress))}

![Runtime Total by Order]({fig_link(output_dir, "runtime_total_by_order.png")})

總 runtime 顯示 Clustered 明顯較久，主要是因為該 run 曾 resume，中間包含較長的 wall-clock 成本。Normal 和 Interleaved runtime 接近，表示 full 20-step pilot 在目前 `BATCH_SIZE=2048` 下成本可控。

![Runtime Stacked by Step]({fig_link(output_dir, "runtime_stacked_by_step.png")})

stacked runtime 圖顯示每一步主要成本來自 unlearn stage，mask 和 eval 相對很短。這對後續擴大實驗很有用：若要跑更多 seeds 或更多 classes，最佳化重點應該放在 unlearn training，而不是 evaluation。

![Runtime Stage Breakdown]({fig_link(output_dir, "runtime_stage_breakdown.png")})

stage breakdown 進一步確認 unlearn 是主成本。由於 eval 已經使用較大 batch size，後續如果要縮短總時間，最有效的策略會是減少 unlearn epochs、調整 data loading，或只對特定 checkpoint 做更密集 eval。

## 11. Additional Tables

### Top post-forget maximum accuracy

{markdown_table(["Order", "Class", "Max post-forget acc"], top_post_forget_table(order_rows))}

這張表列出每條 order 中被忘後曾經回升最高的 classes。它比 immediate lag 更敏感，因為它抓的是後續 rebound，而不是當步 forgetting。從這張表可以看出，本次 failure signal 主要不是 newly forgotten class 忘不掉，而是少數 old forgotten classes 在後續被間接帶回。

## 12. Conclusion

本次 CIFAR-100 20-step ordered pilot 沒有達到原先期待的 immediate forgetting lag failure：三條 order 的 newly forgotten class 當步 accuracy 全部低於 `10%`，而且 worst case 只有 `2.0%`。這說明在目前 `ResNet-18 / seed 1 / RL-SalUn` 設定下，單步 class forgetting 對 CIFAR-100 animal classes 仍然非常有效。

但本次仍然提供了有價值的 failure hunting 訊號：`class 21` 在三條 order 中都出現 final residual，Normal 最明顯。這表示 sequential unlearning 的風險不一定表現在「當步忘不掉」，也可能表現在「已忘 class 在後續更新中局部 rebound」。因此後續報告應把 failure 分成兩類：immediate lag failure 與 post-forget rebound / residual failure。

若後續要更明顯地放大 failure，可以考慮更長的 sequence、更高混淆的 class subset、更多 seeds，或調整 unlearning strength；但在這次 pilot 中，最準確的結論是：**final cumulative forgetting 很好，immediate lag 沒有發生，少數 old forgotten classes 有可觀察的 rebound residual。**
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_root = root / "results/eval"
    logs_root = root / "results/logs"
    order_rows = {}
    order_progress = {}
    smoke_progress = {}

    for key, config in ORDER_CONFIGS.items():
        order_rows[key] = read_eval_rows(eval_root, args.result_namespace, args.seed, config["order"])
        order_progress[key] = read_progress_rows(logs_root / args.result_namespace / config["run_id"] / "progress.csv")
        smoke_progress[key] = read_progress_rows(logs_root / args.result_namespace / config["smoke_run_id"] / "progress.csv")

    save_final_metrics_bar(output_dir / "final_metrics_grouped_bar.png", order_rows)
    save_metric_plot(output_dir / "forget_accuracy_vs_step.png", "Cumulative Forget Accuracy vs Step", "Forget accuracy (%)", "forget_accuracy", order_rows)
    save_metric_plot(output_dir / "ua_vs_step.png", "UA vs Step", "UA (%)", "UA", order_rows)
    save_metric_plot(output_dir / "retain_accuracy_vs_step.png", "Retain Accuracy vs Step", "Retain accuracy (%)", "retain_accuracy", order_rows)
    save_metric_plot(output_dir / "full_test_accuracy_vs_step.png", "Full Test Accuracy vs Step", "Full test accuracy (%)", "full_test_accuracy", order_rows)
    save_newly_forgotten_plot(output_dir / "newly_forgotten_class_accuracy_vs_step.png", order_rows)
    save_newly_forgotten_heatmap(output_dir / "newly_forgotten_accuracy_heatmap.png", order_rows)

    for key, config in ORDER_CONFIGS.items():
        save_selected_class_plot(output_dir / f"{key}_selected_class_accuracy.png", f"{config['label']} Selected-Class Trajectories", order_rows[key], config["order"])
        save_class_heatmap(output_dir / f"{key}_class_accuracy_heatmap.png", f"{config['label']} Class Accuracy Heatmap", order_rows[key], config["order"])
        save_post_forget_rebound_heatmap(output_dir / f"{key}_post_forget_rebound_heatmap.png", f"{config['label']} Non-Zero Post-Forget Rebound Heatmap", order_rows[key], config["order"])
        save_group_mean_plot(output_dir / f"group_mean_accuracy_vs_step_{key}.png", f"{config['label']} Group Mean Accuracy vs Step", order_rows[key], config["order"])

    save_final_forgotten_accuracy(output_dir / "final_forgotten_class_accuracy_by_order.png", order_rows)
    save_final_residual_over_10(output_dir / "final_residual_over_10.png", order_rows)
    save_max_post_forget_heatmap(output_dir / "max_post_forget_accuracy_heatmap.png", order_rows)
    save_post_forget_nonzero_count(output_dir / "post_forget_nonzero_count_by_class.png", order_rows)
    save_post_forget_max_accuracy(output_dir / "post_forget_max_accuracy_by_class.png", order_rows)
    save_top_rebound_trajectories(output_dir / "top_rebound_trajectories_by_order.png", order_rows)
    save_rebound_trajectory(output_dir / "class21_rebound_trajectory.png", 21, order_rows)
    save_rebound_trajectory(output_dir / "class3_rebound_trajectory.png", 3, order_rows)
    save_rebound_trajectory(output_dir / "class31_rebound_trajectory.png", 31, order_rows)
    save_order_group_timeline(output_dir / "order_group_timeline.png")
    save_retain_forget_tradeoff(output_dir / "retain_vs_forget_tradeoff.png", order_rows)
    save_runtime_total(output_dir / "runtime_total_by_order.png", order_progress)
    save_runtime_stacked_by_step(output_dir / "runtime_stacked_by_step.png", order_progress)
    save_runtime_stage_breakdown(output_dir / "runtime_stage_breakdown.png", order_progress)

    write_report(args.report_path, output_dir, order_rows, order_progress, smoke_progress)

    print(f"Wrote report: {args.report_path}")
    print(f"Wrote figures: {output_dir}")


if __name__ == "__main__":
    main()
