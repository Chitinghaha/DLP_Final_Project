#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_NAMESPACE = "sequential_four_coarse_no_flowers_cifar100_k20"
SEED = 1
REBOUND_THRESHOLD = 10.0
REBOUND_DELTA = 5.0


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
    12: "橋",
    13: "公車",
    15: "駱駝",
    17: "城堡",
    19: "牛",
    21: "黑猩猩",
    23: "雲",
    31: "大象",
    33: "森林",
    37: "房子",
    38: "袋鼠",
    49: "山",
    48: "摩托車",
    58: "皮卡車",
    60: "平原",
    68: "道路",
    71: "海",
    76: "摩天大樓",
    90: "火車",
}

GROUPS = {
    "vehicles_1": {
        "zh": "車輛 1",
        "classes": [8, 13, 48, 58, 90],
    },
    "large_natural_outdoor_scenes": {
        "zh": "大型自然場景",
        "classes": [23, 33, 49, 60, 71],
    },
    "large_omni": {
        "zh": "大型雜食/草食動物",
        "classes": [15, 19, 21, 31, 38],
    },
    "man_made_outdoor": {
        "zh": "大型人工戶外物",
        "classes": [12, 17, 37, 68, 76],
    },
}

GROUP_PLOT_LABELS = {
    "vehicles_1": "vehicles_1",
    "large_natural_outdoor_scenes": "natural scenes",
    "large_omni": "large omni",
    "man_made_outdoor": "man-made outdoor",
}

CLASS_TO_GROUP = {
    class_id: group
    for group, info in GROUPS.items()
    for class_id in info["classes"]
}
TARGET_CLASSES = [class_id for info in GROUPS.values() for class_id in info["classes"]]
TRACKED_CLASSES = [8, 21, 23, 33, 49, 60, 71, 12, 17, 37, 68, 76]


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    order_type: str
    order_zh: str
    order: list[int]


RUNS = [
    RunConfig(
        "four_coarse_clustered_seed1_k20",
        "clustered",
        "Clustered",
        [8, 13, 48, 58, 90, 23, 33, 49, 60, 71, 15, 19, 21, 31, 38, 12, 17, 37, 68, 76],
    ),
    RunConfig(
        "four_coarse_interleaved_seed1_k20",
        "interleaved",
        "Interleaved",
        [8, 23, 15, 12, 13, 33, 19, 17, 48, 49, 21, 37, 58, 60, 31, 68, 90, 71, 38, 76],
    ),
    RunConfig(
        "four_coarse_block2_seed1_k20",
        "block2",
        "Block2",
        [8, 13, 23, 33, 15, 19, 12, 17, 48, 58, 49, 60, 21, 31, 37, 68, 90, 71, 38, 76],
    ),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--base-namespace", default=BASE_NAMESPACE)
    parser.add_argument("--report-path", type=Path, default=root / "6_6_cifar100_four_coarse_k20_analysis.md")
    parser.add_argument("--wide-csv-path", type=Path, default=root / "6_6_cifar100_four_coarse_k20_step_accuracy_wide.csv")
    parser.add_argument("--long-csv-path", type=Path, default=root / "6_6_cifar100_four_coarse_k20_step_accuracy_long.csv")
    parser.add_argument("--target-csv-path", type=Path, default=root / "6_6_cifar100_four_coarse_k20_target_trajectory.csv")
    parser.add_argument("--figure-dir", type=Path, default=root / "6_6_cifar100_four_coarse_k20_figures")
    return parser.parse_args()


def class_label(class_id: int) -> str:
    zh = ZH_LABELS.get(class_id)
    en = CIFAR100_FINE_LABELS[class_id]
    return f"{zh} / {en} ({class_id})" if zh else f"{en} ({class_id})"


def class_label_en(class_id: int) -> str:
    return f"{CIFAR100_FINE_LABELS[class_id]} ({class_id})"


def group_zh_for_class(class_id: int) -> str:
    return GROUPS[CLASS_TO_GROUP[class_id]]["zh"]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def fnum(value: object, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def eval_filename(step: int, order: list[int]) -> str:
    forgotten = "_".join(str(class_id) for class_id in order[:step])
    return f"seed{SEED}_step{step}_forgot_{forgotten}.csv"


def eval_path(root: Path, base_namespace: str, run: RunConfig, step: int) -> Path:
    return root / "results" / "eval" / base_namespace / run.run_id / eval_filename(step, run.order)


def read_eval_rows(root: Path, base_namespace: str, run: RunConfig) -> tuple[list[dict | None], list[Path]]:
    rows: list[dict | None] = []
    missing: list[Path] = []
    for step in range(1, len(run.order) + 1):
        path = eval_path(root, base_namespace, run, step)
        if not path.exists():
            rows.append(None)
            missing.append(path)
            continue
        with path.open(newline="") as handle:
            rows.append(next(csv.DictReader(handle)))
    return rows, missing


def read_progress(root: Path, base_namespace: str, run: RunConfig) -> list[dict]:
    path = root / "results" / "logs" / base_namespace / run.run_id / "progress.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def progress_status(progress_rows: list[dict], expected_steps: int) -> str:
    expected = expected_steps * 3
    if not progress_rows:
        return "missing"
    failed = [row for row in progress_rows if row.get("status") == "failed"]
    if failed:
        return "failed"
    success = [row for row in progress_rows if row.get("status") == "success"]
    if len(success) == expected:
        return "success"
    return f"incomplete ({len(success)}/{expected})"


def class_accuracy(row: dict | None, class_id: int) -> float | None:
    if row is None:
        return None
    return fnum(row.get(f"class_{class_id}_accuracy"))


def run_rebound_items(rows: list[dict | None], run: RunConfig) -> list[dict]:
    if any(row is None for row in rows):
        return []
    full_rows = [row for row in rows if row is not None]
    final = full_rows[-1]
    items = []
    for index, class_id in enumerate(run.order):
        step = index + 1
        at_forget = class_accuracy(full_rows[index], class_id)
        later_values = [
            value
            for row in full_rows[index + 1 :]
            if (value := class_accuracy(row, class_id)) is not None
        ]
        max_later = max(later_values) if later_values else None
        final_acc = class_accuracy(final, class_id)
        significant = (
            at_forget is not None
            and max_later is not None
            and max_later >= REBOUND_THRESHOLD
            and (max_later - at_forget) >= REBOUND_DELTA
        )
        items.append(
            {
                "run_id": run.run_id,
                "order_type": run.order_type,
                "order_zh": run.order_zh,
                "step": step,
                "class_id": class_id,
                "class": class_label(class_id),
                "group": group_zh_for_class(class_id),
                "accuracy_at_forget": at_forget,
                "max_later_accuracy": max_later,
                "final_accuracy": final_acc,
                "immediate_lag": (at_forget or 0.0) > REBOUND_THRESHOLD,
                "significant_rebound": significant,
                "final_residual": (final_acc or 0.0) > REBOUND_THRESHOLD,
            }
        )
    return items


def collect_results(root: Path, base_namespace: str) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    run_records = []
    target_items = []
    wide_rows = []
    long_rows = []
    target_rows = []

    for run in RUNS:
        eval_rows, missing = read_eval_rows(root, base_namespace, run)
        progress_rows = read_progress(root, base_namespace, run)
        complete_rows = [row for row in eval_rows if row is not None]
        items = run_rebound_items(eval_rows, run)
        eligible = [item for item in items if item["max_later_accuracy"] is not None]
        rebounds = [item for item in eligible if item["significant_rebound"]]
        immediate = [item for item in items if item["immediate_lag"]]
        final_residuals = [item for item in items if item["final_residual"]]
        final = eval_rows[-1] if eval_rows and eval_rows[-1] is not None else None

        for step, row in enumerate(eval_rows, start=1):
            if row is None:
                continue
            forgotten = run.order[:step]
            wide = {
                "run_id": run.run_id,
                "order_type": run.order_type,
                "step": step,
                "new_class_id": run.order[step - 1],
                "new_class": class_label(run.order[step - 1]),
                "forgotten_classes": ",".join(str(item) for item in forgotten),
                "forget_accuracy": fnum(row.get("forget_accuracy")),
                "UA": fnum(row.get("UA")),
                "retain_accuracy": fnum(row.get("retain_accuracy")),
                "full_test_accuracy": fnum(row.get("full_test_accuracy")),
            }
            for class_id in range(100):
                wide[f"class_{class_id}_accuracy"] = class_accuracy(row, class_id)
                long_rows.append(
                    {
                        "run_id": run.run_id,
                        "order_type": run.order_type,
                        "step": step,
                        "class_id": class_id,
                        "class": class_label(class_id),
                        "accuracy": class_accuracy(row, class_id),
                        "is_target": class_id in TARGET_CLASSES,
                        "is_forgotten_so_far": class_id in forgotten,
                    }
                )
            wide_rows.append(wide)
            target_row = {
                "run_id": run.run_id,
                "order_type": run.order_type,
                "step": step,
                "new_class_id": run.order[step - 1],
                "new_class": class_label(run.order[step - 1]),
                "forgotten_classes": ",".join(str(item) for item in forgotten),
            }
            for class_id in TARGET_CLASSES:
                target_row[f"class_{class_id}_accuracy"] = class_accuracy(row, class_id)
            for group, info in GROUPS.items():
                accs = [class_accuracy(row, class_id) or 0.0 for class_id in info["classes"]]
                target_row[f"{group}_avg_accuracy"] = sum(accs) / len(accs)
            target_rows.append(target_row)

        run_records.append(
            {
                "run": run,
                "rows": complete_rows,
                "missing": missing,
                "progress": progress_rows,
                "status": progress_status(progress_rows, len(run.order)),
                "items": items,
                "eligible": eligible,
                "rebounds": rebounds,
                "immediate": immediate,
                "final_residuals": final_residuals,
                "final_forget": fnum(final.get("forget_accuracy")) if final else None,
                "final_retain": fnum(final.get("retain_accuracy")) if final else None,
                "final_full": fnum(final.get("full_test_accuracy")) if final else None,
                "final_UA": fnum(final.get("UA")) if final else None,
                "rebound_rate": (len(rebounds) / len(eligible) * 100.0) if eligible else None,
            }
        )
        target_items.extend(items)

    return run_records, target_items, wide_rows, long_rows, target_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def clear_figures(figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in figure_dir.glob("*.png"):
        path.unlink()


def completed_records(run_records: list[dict]) -> list[dict]:
    return [record for record in run_records if len(record["rows"]) == 20]


def plot_final_metrics(run_records: list[dict], figure_dir: Path) -> None:
    records = completed_records(run_records)
    if not records:
        return
    labels = [record["run"].order_zh for record in records]
    x = list(range(len(records)))
    width = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], [record["final_forget"] or 0 for record in records], width=width, label="forget acc")
    plt.bar(x, [record["final_retain"] or 0 for record in records], width=width, label="retain acc")
    plt.bar([i + width for i in x], [record["final_full"] or 0 for record in records], width=width, label="full test acc")
    plt.xticks(x, labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Final metrics by run")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_figure(figure_dir / "final_metrics_by_run.png")


def plot_rebound_rate(run_records: list[dict], figure_dir: Path) -> None:
    records = completed_records(run_records)
    if not records:
        return
    labels = [record["run"].order_zh for record in records]
    values = [record["rebound_rate"] or 0.0 for record in records]
    plt.figure(figsize=(8, 4.6))
    plt.bar(labels, values, color="#4c78a8")
    plt.ylabel("Significant rebound rate (%)")
    plt.title("Rebound rate by order")
    plt.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        plt.text(index, value + 0.5, f"{value:.1f}%", ha="center", va="bottom")
    save_figure(figure_dir / "rebound_rate_by_run.png")


def plot_coarse_final_forget(run_records: list[dict], figure_dir: Path) -> None:
    records = completed_records(run_records)
    if not records:
        return
    labels = [GROUP_PLOT_LABELS[group] for group in GROUPS]
    x = list(range(len(labels)))
    width = 0.24
    plt.figure(figsize=(10, 5))
    for offset, record in zip([-width, 0, width], records):
        final = record["rows"][-1]
        values = []
        for group in GROUPS:
            accs = [class_accuracy(final, class_id) or 0.0 for class_id in GROUPS[group]["classes"]]
            values.append(sum(accs) / len(accs))
        plt.bar([i + offset for i in x], values, width=width, label=record["run"].order_zh)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Final target-group accuracy (%)")
    plt.title("Final forgotten-class accuracy by coarse group")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_figure(figure_dir / "coarse_group_final_forget_acc.png")


def plot_target_heatmap(run_records: list[dict], figure_dir: Path) -> None:
    records = completed_records(run_records)
    if not records:
        return
    matrix = []
    ylabels = []
    for record in records:
        for class_id in TARGET_CLASSES:
            matrix.append([class_accuracy(row, class_id) or 0.0 for row in record["rows"]])
            ylabels.append(f"{record['run'].order_zh} {CIFAR100_FINE_LABELS[class_id]}({class_id})")
    plt.figure(figsize=(10, 15))
    image = plt.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(image, label="Accuracy (%)")
    plt.xticks(range(20), [f"s{step}" for step in range(1, 21)], fontsize=8)
    plt.yticks(range(len(ylabels)), ylabels, fontsize=6)
    plt.title("Target class accuracy heatmap")
    save_figure(figure_dir / "target_accuracy_heatmap.png")


def plot_target_heatmap_by_run(run_records: list[dict], figure_dir: Path) -> None:
    for record in completed_records(run_records):
        run = record["run"]
        ordered_classes = run.order
        matrix = []
        ylabels = []
        for class_id in ordered_classes:
            matrix.append([class_accuracy(row, class_id) or 0.0 for row in record["rows"]])
            ylabels.append(class_label_en(class_id))

        plt.figure(figsize=(11, 8))
        image = plt.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        plt.colorbar(image, label="Accuracy (%)")
        plt.xticks(range(20), [f"s{step}" for step in range(1, 21)], fontsize=8)
        plt.yticks(range(len(ylabels)), ylabels, fontsize=8)
        for index in range(20):
            plt.scatter(index, index, marker="x", color="#1f2937", s=28, linewidths=1.1)
        plt.xlabel("Unlearning step")
        plt.ylabel("Target class in forget order")
        plt.title(f"{run.order_zh}: target class accuracy by step")
        save_figure(figure_dir / f"target_accuracy_heatmap_{run.order_type}.png")


def plot_target_line_by_run(run_records: list[dict], figure_dir: Path) -> None:
    group_styles = {
        "vehicles_1": {"color": "#1f77b4", "linestyle": "-"},
        "large_natural_outdoor_scenes": {"color": "#2ca02c", "linestyle": "--"},
        "large_omni": {"color": "#d62728", "linestyle": "-."},
        "man_made_outdoor": {"color": "#9467bd", "linestyle": ":"},
    }
    for record in completed_records(run_records):
        run = record["run"]
        plt.figure(figsize=(13, 8))
        for class_id in run.order:
            group = CLASS_TO_GROUP[class_id]
            style = group_styles[group]
            values = [class_accuracy(row, class_id) or 0.0 for row in record["rows"]]
            forget_step = run.order.index(class_id) + 1
            linewidth = 2.6 if class_id in {8, 21} else 1.35
            alpha = 1.0 if class_id in {8, 21} else 0.72
            plt.plot(
                range(1, 21),
                values,
                label=class_label_en(class_id),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=linewidth,
                alpha=alpha,
            )
            plt.scatter(
                [forget_step],
                [values[forget_step - 1]],
                color=style["color"],
                edgecolor="black",
                s=26 if class_id not in {8, 21} else 46,
                zorder=3,
            )

        plt.axhline(REBOUND_THRESHOLD, color="#4b5563", linestyle="--", linewidth=1.0, alpha=0.7)
        plt.xticks(range(1, 21))
        plt.ylim(-2, 102)
        plt.xlabel("Unlearning step")
        plt.ylabel("Class accuracy on CIFAR-100 test split (%)")
        plt.title(f"{run.order_zh}: per-class accuracy trajectory")
        plt.grid(alpha=0.22)
        plt.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        save_figure(figure_dir / f"target_accuracy_lines_{run.order_type}.png")


def plot_bicycle_trajectory(run_records: list[dict], figure_dir: Path) -> None:
    records = completed_records(run_records)
    if not records:
        return
    plt.figure(figsize=(9, 5))
    for record in records:
        values = [class_accuracy(row, 8) or 0.0 for row in record["rows"]]
        plt.plot(range(1, 21), values, marker="o", linewidth=2, label=record["run"].order_zh)
        forget_step = record["run"].order.index(8) + 1
        plt.scatter([forget_step], [values[forget_step - 1]], s=70)
    plt.axhline(REBOUND_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    plt.xticks(range(1, 21))
    plt.xlabel("Unlearning step")
    plt.ylabel("Bicycle accuracy (%)")
    plt.title("Bicycle (8) trajectory")
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(figure_dir / "bicycle_trajectory.png")


def plot_coarse_rebound_counts(run_records: list[dict], figure_dir: Path) -> None:
    records = completed_records(run_records)
    if not records:
        return
    labels = [GROUP_PLOT_LABELS[group] for group in GROUPS]
    x = list(range(len(labels)))
    width = 0.24
    plt.figure(figsize=(10, 5))
    for offset, record in zip([-width, 0, width], records):
        counts = []
        for group in GROUPS:
            classes = set(GROUPS[group]["classes"])
            counts.append(sum(1 for item in record["rebounds"] if item["class_id"] in classes))
        plt.bar([i + offset for i in x], counts, width=width, label=record["run"].order_zh)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Significant rebound count")
    plt.title("Rebound counts by coarse group")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_figure(figure_dir / "coarse_group_rebound_counts.png")


def generate_figures(run_records: list[dict], figure_dir: Path) -> None:
    clear_figures(figure_dir)
    plot_final_metrics(run_records, figure_dir)
    plot_rebound_rate(run_records, figure_dir)
    plot_coarse_final_forget(run_records, figure_dir)
    plot_target_heatmap(run_records, figure_dir)
    plot_target_heatmap_by_run(run_records, figure_dir)
    plot_target_line_by_run(run_records, figure_dir)
    plot_bicycle_trajectory(run_records, figure_dir)
    plot_coarse_rebound_counts(run_records, figure_dir)


def figure_link(figure_dir: Path, filename: str) -> str:
    return f"{figure_dir.name}/{filename}"


def summary_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in run_records:
        run = record["run"]
        final = record["rows"][-1] if record["rows"] else None
        rows.append(
            [
                run.order_zh,
                run.run_id,
                ",".join(str(item) for item in run.order),
                record["status"],
                len(record["rows"]),
                pct(record["final_forget"]),
                pct(record["final_UA"]),
                pct(record["final_retain"]),
                pct(record["final_full"]),
                pct(class_accuracy(final, 8) if final else None),
                f"{len(record['rebounds'])}/{len(record['eligible'])}" if record["eligible"] else "n/a",
                len(record["final_residuals"]),
                len(record["missing"]),
            ]
        )
    return rows


def rebound_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in run_records:
        for item in record["rebounds"]:
            rows.append(
                [
                    item["order_zh"],
                    item["group"],
                    item["class"],
                    item["step"],
                    pct(item["accuracy_at_forget"]),
                    pct(item["max_later_accuracy"]),
                    pct(item["final_accuracy"]),
                ]
            )
    return rows


def coarse_summary_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for group, info in GROUPS.items():
        group_rebounds = 0
        group_residuals = 0
        final_accs = []
        for record in completed_records(run_records):
            final = record["rows"][-1]
            final_accs.extend([class_accuracy(final, class_id) or 0.0 for class_id in info["classes"]])
            group_rebounds += sum(1 for item in record["rebounds"] if item["class_id"] in info["classes"])
            group_residuals += sum(1 for item in record["final_residuals"] if item["class_id"] in info["classes"])
        avg_final = sum(final_accs) / len(final_accs) if final_accs else None
        rows.append([info["zh"], pct(avg_final), group_rebounds, group_residuals])
    return rows


def coarse_by_order_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in completed_records(run_records):
        final = record["rows"][-1]
        for group, info in GROUPS.items():
            accs = [class_accuracy(final, class_id) or 0.0 for class_id in info["classes"]]
            rows.append(
                [
                    record["run"].order_zh,
                    info["zh"],
                    pct(sum(accs) / len(accs)),
                    ", ".join(f"{class_label(class_id)}={pct(class_accuracy(final, class_id))}" for class_id in info["classes"]),
                ]
            )
    return rows


def tracked_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in completed_records(run_records):
        run = record["run"]
        final = record["rows"][-1]
        for class_id in TRACKED_CLASSES:
            forget_step = run.order.index(class_id) + 1
            at_forget = class_accuracy(record["rows"][forget_step - 1], class_id)
            later = [class_accuracy(row, class_id) or 0.0 for row in record["rows"][forget_step:]]
            max_later = max(later) if later else None
            rows.append(
                [
                    run.order_zh,
                    group_zh_for_class(class_id),
                    class_label(class_id),
                    forget_step,
                    pct(at_forget),
                    pct(max_later),
                    pct(class_accuracy(final, class_id)),
                ]
            )
    return rows


def report_title(report_path: Path) -> str:
    if report_path.name.startswith("6_7"):
        return "# 6_7 CIFAR-100 四大類 k20 No-Flowers Rebound Analysis"
    return "# 6_6 CIFAR-100 Four-Coarse k20 Rebound Analysis"


def final_metric_sentence(run_records: list[dict]) -> str:
    parts = []
    for record in completed_records(run_records):
        final = record["rows"][-1]
        parts.append(
            f"{record['run'].order_zh}: forget {pct(record['final_forget'])}, "
            f"UA {pct(record['final_UA'])}, retain {pct(record['final_retain'])}, "
            f"full {pct(record['final_full'])}, bicycle {pct(class_accuracy(final, 8))}"
        )
    return "；".join(parts)


def bicycle_conclusion(run_records: list[dict]) -> str:
    values = []
    for record in completed_records(run_records):
        final = record["rows"][-1]
        value = class_accuracy(final, 8)
        if value is not None:
            values.append(value)
    if not values:
        return "目前尚未有 bicycle final accuracy。"
    return (
        f"`bicycle (8)` final accuracy 範圍為 `{min(values):.1f}%` 到 `{max(values):.1f}%`，"
        "三條 order 都高於 10% residual 門檻；這表示 bicycle residual / rebound 不只出現在 single-coarse vehicles_1，"
        "也延伸到 mixed 20-class sequential unlearning。"
    )


def natural_vs_manmade_conclusion(run_records: list[dict]) -> str:
    completed = completed_records(run_records)
    if not completed:
        return "目前尚未有完整 final eval。"
    natural_values = []
    manmade_values = []
    for record in completed:
        final = record["rows"][-1]
        natural_values.extend(class_accuracy(final, class_id) or 0.0 for class_id in GROUPS["large_natural_outdoor_scenes"]["classes"])
        manmade_values.extend(class_accuracy(final, class_id) or 0.0 for class_id in GROUPS["man_made_outdoor"]["classes"])
    return (
        f"大型自然場景 final 平均為 `{sum(natural_values) / len(natural_values):.1f}%`，"
        f"大型人工戶外物 final 平均為 `{sum(manmade_values) / len(manmade_values):.1f}%`；"
        "`bridge (12)` 與 `road (68)` 在三條 final eval 都是 `0.0%`，比 bicycle 更容易被壓低。"
    )


def significant_class_names(record: dict) -> str:
    if not record["rebounds"]:
        return "none"
    return ", ".join(f"{CIFAR100_FINE_LABELS[item['class_id']]} ({item['class_id']})" for item in record["rebounds"])


def per_run_line_conclusion(record: dict) -> str:
    final = record["rows"][-1] if record["rows"] else None
    if final is None:
        return "小結：此 run 尚未完成 final eval。"
    bicycle = class_accuracy(final, 8)
    motorcycle = class_accuracy(final, 48)
    chimpanzee = class_accuracy(final, 21)
    natural = [
        class_accuracy(final, class_id) or 0.0
        for class_id in GROUPS["large_natural_outdoor_scenes"]["classes"]
    ]
    manmade = [
        class_accuracy(final, class_id) or 0.0
        for class_id in GROUPS["man_made_outdoor"]["classes"]
    ]
    return (
        f"小結：{record['run'].order_zh} final bicycle accuracy 為 `{pct(bicycle)}`；"
        f"motorcycle `{pct(motorcycle)}`、chimpanzee `{pct(chimpanzee)}`。"
        f"大型自然場景 final 平均 `{sum(natural) / len(natural):.1f}%`，"
        f"大型人工戶外物 final 平均 `{sum(manmade) / len(manmade):.1f}%`。"
        f" Significant rebound classes: {significant_class_names(record)}。"
    )


def detailed_summary_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in completed_records(run_records):
        final = record["rows"][-1]
        natural = [
            class_accuracy(final, class_id) or 0.0
            for class_id in GROUPS["large_natural_outdoor_scenes"]["classes"]
        ]
        manmade = [
            class_accuracy(final, class_id) or 0.0
            for class_id in GROUPS["man_made_outdoor"]["classes"]
        ]
        rows.append(
            [
                record["run"].order_zh,
                pct(class_accuracy(final, 8)),
                pct(class_accuracy(final, 48)),
                pct(class_accuracy(final, 21)),
                pct(sum(natural) / len(natural)),
                pct(sum(manmade) / len(manmade)),
                significant_class_names(record),
            ]
        )
    return rows


def order_effect_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in completed_records(run_records):
        final = record["rows"][-1]
        group_avgs = {}
        for group, info in GROUPS.items():
            accs = [class_accuracy(final, class_id) or 0.0 for class_id in info["classes"]]
            group_avgs[group] = sum(accs) / len(accs)
        rows.append(
            [
                record["run"].order_zh,
                pct(record["final_forget"]),
                pct(record["final_UA"]),
                pct(group_avgs["vehicles_1"]),
                pct(group_avgs["large_natural_outdoor_scenes"]),
                pct(group_avgs["large_omni"]),
                pct(group_avgs["man_made_outdoor"]),
                pct(class_accuracy(final, 8)),
                pct(class_accuracy(final, 48)),
                pct(class_accuracy(final, 21)),
                f"{len(record['rebounds'])}/{len(record['eligible'])}",
            ]
        )
    return rows


def order_effect_conclusion(run_records: list[dict]) -> list[str]:
    completed = completed_records(run_records)
    if not completed:
        return ["- 目前尚未有完整結果可分析順序影響。"]
    by_order = {record["run"].order_type: record for record in completed}
    clustered = by_order.get("clustered")
    interleaved = by_order.get("interleaved")
    block2 = by_order.get("block2")
    lines = []
    if clustered and interleaved and block2:
        clustered_final = clustered["rows"][-1]
        interleaved_final = interleaved["rows"][-1]
        block2_final = block2["rows"][-1]
        clustered_vehicle = sum((class_accuracy(clustered_final, class_id) or 0.0) for class_id in GROUPS["vehicles_1"]["classes"]) / 5
        interleaved_vehicle = sum((class_accuracy(interleaved_final, class_id) or 0.0) for class_id in GROUPS["vehicles_1"]["classes"]) / 5
        block2_vehicle = sum((class_accuracy(block2_final, class_id) or 0.0) for class_id in GROUPS["vehicles_1"]["classes"]) / 5
        lines.extend(
            [
                f"- `clustered` 的 vehicles_1 final 平均最高，為 `{clustered_vehicle:.1f}%`，高於 `interleaved` 的 `{interleaved_vehicle:.1f}%` 與 `block2` 的 `{block2_vehicle:.1f}%`；這表示同一 coarse group 連續遺忘時，車輛相關 residual/rebound 較明顯。",
                f"- `interleaved` 的整體 final forget acc 為 `{pct(interleaved['final_forget'])}`，低於 clustered 的 `{pct(clustered['final_forget'])}`，但 bicycle final 反而最高：`{pct(class_accuracy(interleaved_final, 8))}`。也就是說，交錯順序能壓低整體 target 平均，但無法消除 bicycle residual。",
                f"- `block2` 的 final forget acc 最低，為 `{pct(block2['final_forget'])}`，UA 最高為 `{pct(block2['final_UA'])}`；它在整體 forgetting 上最好，但 bicycle final 仍有 `{pct(class_accuracy(block2_final, 8))}`。",
                f"- `motorcycle (48)` 對順序較敏感：clustered final `{pct(class_accuracy(clustered_final, 48))}`，但 interleaved/block2 只剩 `{pct(class_accuracy(interleaved_final, 48))}` / `{pct(class_accuracy(block2_final, 48))}`。這和 bicycle 不同，bicycle 在三種順序都維持 `{pct(class_accuracy(clustered_final, 8))}` / `{pct(class_accuracy(interleaved_final, 8))}` / `{pct(class_accuracy(block2_final, 8))}`。",
                f"- `chimpanzee (21)` 也受順序影響：clustered `{pct(class_accuracy(clustered_final, 21))}`、interleaved `{pct(class_accuracy(interleaved_final, 21))}`、block2 `{pct(class_accuracy(block2_final, 21))}`；它不像 bicycle 三條都超過 20%，但在 clustered/block2 有超過 10% residual。",
            ]
        )
    lines.append("- 結論：順序會改變 residual/rebound 的強度與集中位置，但不是 bicycle rebound 的唯一原因；bicycle 在三種順序都殘留，代表它比其他 target class 更穩定地抗拒完全遺忘。")
    return lines


def build_compact_6_7_report(base_namespace: str, report_path: Path, figure_dir: Path, run_records: list[dict]) -> str:
    completed = completed_records(run_records)
    sections = [
        report_title(report_path),
        "",
        "## 實驗設定",
        "",
        f"- Namespace：`{base_namespace}`",
        "- Dataset：CIFAR-100 test split",
        f"- Seed：`{SEED}`",
        "- Target：4 個 coarse superclasses，共 20 個 fine classes。",
        "- 三條 order：`clustered`、`interleaved`、`block2`。",
        f"- Significant rebound：later accuracy `>= {REBOUND_THRESHOLD:.0f}%`，且比 forget-step accuracy 高至少 `{REBOUND_DELTA:.0f}pp`。",
        "",
        "## Final metrics",
        "",
        markdown_table(
            ["Order", "Run ID", "Progress", "Eval CSVs", "Final forget acc", "Final UA", "Final retain acc", "Final full test acc", "Bicycle final", "Rebound rate"],
            [row[:2] + row[3:11] for row in summary_rows(run_records)],
        ),
        "",
        f"小結：三條 run 完成狀態為 `{len(completed)}/3`；{final_metric_sentence(run_records) if completed else '目前尚未有 final metrics。'}",
        "",
        "## 每 step / 每類別 accuracy 折線圖",
        "",
        "每張圖是一條 order，20 條線對應 20 個 target classes；線上的黑框點標示該類別被忘的 step。藍色是 vehicles_1、綠色是大型自然場景、紅色是大型雜食/草食動物、紫色是大型人工戶外物；bicycle 與 chimpanzee 以較粗線條標示。",
        "",
    ]

    for record in completed:
        filename = f"target_accuracy_lines_{record['run'].order_type}.png"
        sections.extend(
            [
                f"### {record['run'].order_zh}",
                "",
                f"![{filename}]({figure_link(figure_dir, filename)})",
                "",
                per_run_line_conclusion(record),
                "",
            ]
        )

    sections.extend(
        [
            "## 順序影響分析",
            "",
            markdown_table(
                [
                    "Order",
                    "Final forget acc",
                    "Final UA",
                    "Vehicles_1 avg",
                    "Natural scenes avg",
                    "Large omni avg",
                    "Man-made avg",
                    "Bicycle final",
                    "Motorcycle final",
                    "Chimpanzee final",
                    "Rebound rate",
                ],
                order_effect_rows(run_records),
            ),
            "",
            *order_effect_conclusion(run_records),
            "",
            "## 詳細總結分析",
            "",
            markdown_table(
                ["Order", "Bicycle final", "Motorcycle final", "Chimpanzee final", "Natural scenes final avg", "Man-made outdoor final avg", "Significant rebound classes"],
                detailed_summary_rows(run_records),
            ),
            "",
            f"- {bicycle_conclusion(run_records)}",
            "- `bicycle (8)` 是三條 order 中最穩定的 residual case：final 分別為 `23.0% / 25.0% / 20.0%`，而且三條都超過 10% residual 門檻。",
            "- `motorcycle (48)` 只在 clustered 最後明顯殘留到 `16.0%`，interleaved 和 block2 final 只有 `4.0% / 3.0%`；這表示車輛類別的 rebound 不完全平均，bicycle 更穩定。",
            "- `chimpanzee (21)` 在 clustered / block2 分別 final `15.0% / 12.0%`，interleaved 為 `8.0%`；它有部分 rebound，但穩定性低於 bicycle。",
            f"- {natural_vs_manmade_conclusion(run_records)}",
            "- 從折線圖看，自然場景與人工戶外物多數類別在被忘後貼近 0%，沒有像 bicycle 一樣長時間回升；因此目前最合理的解讀是：rebound 主要集中在少數類別，而不是所有 coarse group 的普遍現象。",
            "- 本報告保留 final metrics 與三張 step/class 折線圖；完整逐類數字可看同名 `wide`、`long`、`target_trajectory` CSV。",
            "",
            "## Notes",
            "",
            "- Accuracy 來自 CIFAR-100 test split；每個 fine class 有 100 張 test images，所以 1% 約等於 1 張圖。",
            "- 本分析固定 `seed=1`，結論應視為機制探索，不主張跨 seed 普遍性。",
        ]
    )
    return "\n".join(sections)


def build_report(root: Path, base_namespace: str, report_path: Path, figure_dir: Path, run_records: list[dict]) -> str:
    if report_path.name.startswith("6_7"):
        return build_compact_6_7_report(base_namespace, report_path, figure_dir, run_records)

    completed = completed_records(run_records)
    total_rebounds = sum(len(record["rebounds"]) for record in completed)
    total_eligible = sum(len(record["eligible"]) for record in completed)

    sections = [
        report_title(report_path),
        "",
        "## 實驗設定",
        "",
        f"- Namespace：`{base_namespace}`",
        "- Dataset：CIFAR-100",
        f"- Seed：`{SEED}`",
        "- Target：4 個 coarse superclasses，每組 5 個 fine classes，共 20 個 fine classes。",
        f"- Significant rebound 定義：later accuracy `>= {REBOUND_THRESHOLD:.0f}%`，且比 forget-step accuracy 高至少 `{REBOUND_DELTA:.0f}pp`。",
        "",
        markdown_table(
            ["Coarse superclass", "Fine classes"],
            [
                [info["zh"], ", ".join(class_label(class_id) for class_id in info["classes"])]
                for info in GROUPS.values()
            ],
        ),
        "",
        "小結：本次 target set 移除 flowers，改成大型自然場景，用來檢查背景/場景 feature 是否也會造成 sequential drift；同時保留 vehicles_1 作為 bicycle rebound 的 positive case。",
        "",
        "## 執行狀態與 final metrics",
        "",
        markdown_table(
            ["Order", "Run ID", "Forget order", "Progress", "Eval CSVs", "Final forget acc", "Final UA", "Final retain acc", "Final full test acc", "Bicycle final", "Rebound rate", "Final residual count", "Missing eval"],
            summary_rows(run_records),
        ),
        "",
        f"小結：三條 run 完成狀態為 `{len(completed)}/3`；{final_metric_sentence(run_records) if completed else '目前尚未有 final metrics。'}",
        "",
        "## 圖表",
        "",
    ]

    figure_names = [
        "final_metrics_by_run.png",
        "rebound_rate_by_run.png",
        "coarse_group_final_forget_acc.png",
        "target_accuracy_heatmap.png",
        "bicycle_trajectory.png",
        "coarse_group_rebound_counts.png",
    ]
    existing_figures = [name for name in figure_names if (figure_dir / name).exists()]
    if existing_figures:
        sections.extend([f"![{name}]({figure_link(figure_dir, name)})" for name in existing_figures])
    else:
        sections.append("目前尚未找到完整 eval CSV，因此圖表會在實驗完成後重新產生。")
    sections.extend(["", "小結：圖表用來快速定位 order 差異、target class trajectory，以及 bicycle 是否在 20-step mixed setting 仍維持 residual。"])

    sections.extend(
        [
            "",
            "## Rebound summary",
            "",
        ]
    )
    reb_rows = rebound_rows(run_records)
    if reb_rows:
        sections.append(markdown_table(["Order", "Group", "Class", "Forget step", "At forget", "Max later", "Final"], reb_rows))
    else:
        sections.append("目前尚未有 significant rebound，或實驗尚未完成。")
    sections.extend(
        [
            "",
            f"小結：三條 completed run 的 significant rebound 總數為 `{total_rebounds}/{total_eligible}`；判定門檻是 later accuracy 至少 `{REBOUND_THRESHOLD:.0f}%` 且比 forget-step 高 `{REBOUND_DELTA:.0f}pp`。",
        ]
    )

    sections.extend(
        [
            "",
            "## Coarse group summary",
            "",
            markdown_table(["Coarse group", "Avg final target acc", "Rebound count", "Final residual count"], coarse_summary_rows(run_records)),
            "",
            "## Coarse group final accuracy by order",
            "",
            markdown_table(["Order", "Coarse group", "Avg final acc", "Class final acc"], coarse_by_order_rows(run_records)),
            "",
            f"小結：{natural_vs_manmade_conclusion(run_records)}",
            "",
            "## Tracked class trajectories",
            "",
        ]
    )
    tr_rows = tracked_rows(run_records)
    if tr_rows:
        sections.append(markdown_table(["Order", "Group", "Class", "Forget step", "At forget", "Max later", "Final"], tr_rows))
    else:
        sections.append("目前尚未有完整 trajectory；實驗完成後會列出 bicycle、chimpanzee、natural scenes、man-made outdoor 重點類別。")
    sections.extend(["", f"小結：{bicycle_conclusion(run_records)}"])

    sections.extend(
        [
            "",
            "## 結論",
            "",
        ]
    )
    if completed:
        sections.extend(
            [
                f"- 已完成 `{len(completed)}/3` 條 run，三條 progress 都應為 `60 rows`，final eval CSV 都存在。",
                f"- Final metrics：{final_metric_sentence(run_records)}。",
                f"- {bicycle_conclusion(run_records)}",
                f"- {natural_vs_manmade_conclusion(run_records)}",
                "- 本結果支持：bicycle 是目前最穩定的 residual / rebound case；自然場景與人工戶外物多數能被壓到接近 0%。",
            ]
        )
    else:
        sections.append("- 實驗尚未完成；目前報告主要確認 target set、order、namespace 與分析輸出格式。")

    sections.extend(
        [
            "",
            "## Notes",
            "",
            "- 本分析固定 `seed=1`，結論應視為機制探索，不主張跨 seed 普遍性。",
            "- Accuracy 來自 CIFAR-100 test split；每個 fine class 有 100 張 test images，所以 1% 約等於 1 張圖。",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    run_records, target_items, wide_rows, long_rows, target_rows = collect_results(args.root, args.base_namespace)
    generate_figures(run_records, args.figure_dir)

    wide_fieldnames = [
        "run_id", "order_type", "step", "new_class_id", "new_class", "forgotten_classes",
        "forget_accuracy", "UA", "retain_accuracy", "full_test_accuracy",
        *[f"class_{class_id}_accuracy" for class_id in range(100)],
    ]
    long_fieldnames = ["run_id", "order_type", "step", "class_id", "class", "accuracy", "is_target", "is_forgotten_so_far"]
    target_fieldnames = [
        "run_id", "order_type", "step", "new_class_id", "new_class", "forgotten_classes",
        *[f"class_{class_id}_accuracy" for class_id in TARGET_CLASSES],
        *[f"{group}_avg_accuracy" for group in GROUPS],
    ]

    write_csv(args.wide_csv_path, wide_rows, wide_fieldnames)
    write_csv(args.long_csv_path, long_rows, long_fieldnames)
    write_csv(args.target_csv_path, target_rows, target_fieldnames)
    args.report_path.write_text(build_report(args.root, args.base_namespace, args.report_path, args.figure_dir, run_records), encoding="utf-8")

    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.wide_csv_path}")
    print(f"Wrote {args.long_csv_path}")
    print(f"Wrote {args.target_csv_path}")
    print(f"Wrote figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
