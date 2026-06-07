#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_NAMESPACE = "sequential_single_coarse_rebound_cifar100"
SEED = 1
REBOUND_THRESHOLD = 10.0
REBOUND_DELTA = 5.0
VEHICLES1_CLASSES = [8, 13, 48, 58, 90]


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
    15: "駱駝",
    19: "牛",
    21: "黑猩猩",
    31: "大象",
    38: "袋鼠",
    48: "摩托車",
    54: "蘭花",
    58: "皮卡車",
    62: "罌粟花",
    70: "玫瑰",
    82: "向日葵",
    90: "火車",
    92: "鬱金香",
}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    group: str
    group_zh: str
    order_type: str
    order: list[int]


RUNS = [
    RunConfig("large_omni_official_seed1_k5", "large_omnivores_and_herbivores", "大型雜食/草食動物", "official", [15, 19, 21, 31, 38]),
    RunConfig("large_omni_random_seed1_k5", "large_omnivores_and_herbivores", "大型雜食/草食動物", "random", [21, 15, 38, 19, 31]),
    RunConfig("large_omni_hardfirst_seed1_k5", "large_omnivores_and_herbivores", "大型雜食/草食動物", "hardfirst", [19, 38, 31, 15, 21]),
    RunConfig("vehicles1_official_seed1_k5", "vehicles_1", "車輛 1", "official", [8, 13, 48, 58, 90]),
    RunConfig("vehicles1_random_seed1_k5", "vehicles_1", "車輛 1", "random", [48, 8, 90, 13, 58]),
    RunConfig("vehicles1_hardfirst_seed1_k5", "vehicles_1", "車輛 1", "hardfirst", [13, 8, 58, 90, 48]),
    RunConfig("flowers_official_seed1_k5", "flowers", "花卉", "official", [54, 62, 70, 82, 92]),
    RunConfig("flowers_random_seed1_k5", "flowers", "花卉", "random", [70, 54, 92, 62, 82]),
    RunConfig("flowers_hardfirst_seed1_k5", "flowers", "花卉", "hardfirst", [92, 70, 62, 54, 82]),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--base-namespace", default=BASE_NAMESPACE)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "6_6_cifar100_single_coarse_rebound_analysis.md",
    )
    parser.add_argument(
        "--wide-csv-path",
        type=Path,
        default=root / "6_6_cifar100_single_coarse_rebound_step_accuracy_wide.csv",
    )
    parser.add_argument(
        "--long-csv-path",
        type=Path,
        default=root / "6_6_cifar100_single_coarse_rebound_step_accuracy_long.csv",
    )
    parser.add_argument(
        "--target-csv-path",
        type=Path,
        default=root / "6_6_cifar100_single_coarse_rebound_target_trajectory.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "6_6_cifar100_single_coarse_rebound_figures",
    )
    return parser.parse_args()


def eval_filename(step: int, order: list[int]) -> str:
    forgotten = "_".join(str(class_id) for class_id in order[:step])
    return f"seed{SEED}_step{step}_forgot_{forgotten}.csv"


def read_eval_rows(root: Path, base_namespace: str, run: RunConfig) -> tuple[list[dict], list[Path]]:
    eval_root = root / "results" / "eval" / base_namespace / run.run_id
    rows: list[dict] = []
    missing: list[Path] = []
    for step in range(1, len(run.order) + 1):
        path = eval_root / eval_filename(step, run.order)
        if not path.exists():
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


def fnum(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def class_label(class_id: int) -> str:
    zh = ZH_LABELS.get(class_id)
    en = CIFAR100_FINE_LABELS[class_id]
    return f"{zh} / {en} ({class_id})" if zh else f"{en} ({class_id})"


def class_label_en(class_id: int) -> str:
    return f"{CIFAR100_FINE_LABELS[class_id]} ({class_id})"


def class_accuracy(row: dict, class_id: int) -> float:
    return fnum(row.get(f"class_{class_id}_accuracy"))


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


def run_rebound_items(rows: list[dict], run: RunConfig) -> list[dict]:
    items = []
    if len(rows) != len(run.order):
        return items

    final = rows[-1]
    for idx, class_id in enumerate(run.order):
        step = idx + 1
        at_forget = class_accuracy(rows[idx], class_id)
        later = [class_accuracy(row, class_id) for row in rows[idx + 1 :]]
        max_later = max(later) if later else None
        final_acc = class_accuracy(final, class_id)
        significant = (
            max_later is not None
            and max_later >= REBOUND_THRESHOLD
            and (max_later - at_forget) >= REBOUND_DELTA
        )
        items.append(
            {
                "run_id": run.run_id,
                "group": run.group_zh,
                "order_type": run.order_type,
                "step": step,
                "class_id": class_id,
                "class": class_label(class_id),
                "accuracy_at_forget": at_forget,
                "max_later_accuracy": max_later,
                "final_accuracy": final_acc,
                "immediate_lag": at_forget > REBOUND_THRESHOLD,
                "significant_rebound": significant,
                "final_residual": final_acc > REBOUND_THRESHOLD,
            }
        )
    return items


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def bool_zh(value: bool) -> str:
    return "是" if value else "否"


def run_short_name(run: RunConfig) -> str:
    group_short = {
        "large_omnivores_and_herbivores": "large",
        "vehicles_1": "vehicles",
        "flowers": "flowers",
    }[run.group]
    order_short = {
        "official": "off",
        "random": "rand",
        "hardfirst": "hard",
    }[run.order_type]
    return f"{group_short}-{order_short}"


def order_zh(order_type: str) -> str:
    return {
        "official": "官方順序",
        "random": "隨機順序",
        "hardfirst": "困難優先",
    }[order_type]


def order_en(order_type: str) -> str:
    return {
        "official": "official",
        "random": "random",
        "hardfirst": "hardfirst",
    }[order_type]


def collect_results(root: Path, base_namespace: str) -> tuple[list[dict], list[dict], list[dict]]:
    run_records = []
    target_items = []
    step_records = []

    for run in RUNS:
        rows, missing = read_eval_rows(root, base_namespace, run)
        progress = read_progress(root, base_namespace, run)
        items = run_rebound_items(rows, run)
        eligible = [item for item in items if item["max_later_accuracy"] is not None]
        rebounds = [item for item in eligible if item["significant_rebound"]]
        immediate = [item for item in items if item["immediate_lag"]]
        final_residuals = [item for item in items if item["final_residual"]]
        final_row = rows[-1] if rows else {}

        for step, row in enumerate(rows, start=1):
            step_records.append(
                {
                    "run": run,
                    "step": step,
                    "row": row,
                    "new_class_id": run.order[step - 1],
                    "new_class": class_label(run.order[step - 1]),
                    "forget_accuracy": fnum(row.get("forget_accuracy"), None),
                    "retain_accuracy": fnum(row.get("retain_accuracy"), None),
                    "full_test_accuracy": fnum(row.get("full_test_accuracy"), None),
                    "UA": fnum(row.get("UA"), None),
                }
            )

        run_records.append(
            {
                "run": run,
                "rows": rows,
                "missing": missing,
                "progress": progress,
                "status": progress_status(progress, len(run.order)),
                "items": items,
                "eligible": eligible,
                "rebounds": rebounds,
                "immediate": immediate,
                "final_residuals": final_residuals,
                "final_forget": fnum(final_row.get("forget_accuracy"), None) if final_row else None,
                "final_retain": fnum(final_row.get("retain_accuracy"), None) if final_row else None,
                "final_full": fnum(final_row.get("full_test_accuracy"), None) if final_row else None,
                "rebound_rate": (len(rebounds) / len(eligible) * 100.0) if eligible else None,
            }
        )
        target_items.extend(items)

    return run_records, target_items, step_records


def make_figure_dir(figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for old_png in figure_dir.glob("*.png"):
        old_png.unlink()


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_final_metrics(run_records: list[dict], figure_dir: Path) -> None:
    labels = [run_short_name(record["run"]) for record in run_records]
    x = list(range(len(labels)))
    width = 0.25
    metrics = [
        ("Final forget", [record["final_forget"] or 0.0 for record in run_records], -width),
        ("Final retain", [record["final_retain"] or 0.0 for record in run_records], 0),
        ("Full test", [record["final_full"] or 0.0 for record in run_records], width),
    ]
    plt.figure(figsize=(12, 5.5))
    for label, values, offset in metrics:
        plt.bar([i + offset for i in x], values, width=width, label=label)
    plt.axhline(REBOUND_THRESHOLD, color="#555555", linestyle="--", linewidth=1, label="10% threshold")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 80)
    plt.title("Final metrics by run")
    plt.legend(ncol=4, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(figure_dir / "final_metrics_by_run.png")


def plot_rebound_rate(run_records: list[dict], figure_dir: Path) -> None:
    labels = [run_short_name(record["run"]) for record in run_records]
    values = [record["rebound_rate"] or 0.0 for record in run_records]
    colors = ["#577590" if record["run"].group != "vehicles_1" else "#f3722c" for record in run_records]
    plt.figure(figsize=(11, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Significant rebound rate (%)")
    plt.ylim(0, 35)
    plt.title("Post-forget rebound rate by run")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        plt.text(index, value + 1.0, f"{value:.0f}%", ha="center", va="bottom", fontsize=8)
    save_current_figure(figure_dir / "rebound_rate_by_run.png")


def plot_target_accuracy_heatmap(run_records: list[dict], figure_dir: Path) -> None:
    matrix = []
    ylabels = []
    for record in run_records:
        run = record["run"]
        rows = record["rows"]
        for class_id in run.order:
            matrix.append([class_accuracy(row, class_id) for row in rows])
            ylabels.append(f"{run_short_name(run)} {CIFAR100_FINE_LABELS[class_id]}({class_id})")

    plt.figure(figsize=(8.5, 14))
    image = plt.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(image, label="Accuracy (%)")
    plt.xticks(range(5), ["step1", "step2", "step3", "step4", "step5"])
    plt.yticks(range(len(ylabels)), ylabels, fontsize=7)
    plt.title("Target class accuracy over sequential unlearning steps")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            if value >= 10:
                plt.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=6, color="black")
    save_current_figure(figure_dir / "target_accuracy_heatmap.png")


def plot_class_trajectory(
    run_records: list[dict],
    figure_dir: Path,
    class_id: int,
    group: str,
    filename: str,
    title: str,
) -> None:
    selected = [record for record in run_records if record["run"].group == group]
    plt.figure(figsize=(8.5, 5))
    for record in selected:
        run = record["run"]
        rows = record["rows"]
        values = [class_accuracy(row, class_id) for row in rows]
        forget_step = run.order.index(class_id) + 1 if class_id in run.order else None
        plt.plot(range(1, len(values) + 1), values, marker="o", linewidth=2, label=order_en(run.order_type))
        if forget_step is not None:
            plt.scatter([forget_step], [values[forget_step - 1]], s=70)
    plt.axhline(REBOUND_THRESHOLD, color="#555555", linestyle="--", linewidth=1, label="10% threshold")
    plt.xticks(range(1, 6))
    plt.xlabel("Unlearning step")
    plt.ylabel("Class accuracy (%)")
    plt.ylim(0, max(55, max([class_accuracy(row, class_id) for record in selected for row in record["rows"]] + [0]) + 8))
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    save_current_figure(figure_dir / filename)


def plot_vehicles1_target_trajectories(run_records: list[dict], figure_dir: Path) -> None:
    records = [record for record in run_records if record["run"].group == "vehicles_1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for axis, record in zip(axes, records):
        run = record["run"]
        rows = record["rows"]
        for class_id in VEHICLES1_CLASSES:
            values = [class_accuracy(row, class_id) for row in rows]
            linewidth = 3 if class_id == 8 else 1.8
            marker = "o" if class_id == 8 else "."
            axis.plot(
                range(1, len(values) + 1),
                values,
                marker=marker,
                linewidth=linewidth,
                label=class_label_en(class_id),
            )
        axis.axhline(REBOUND_THRESHOLD, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(order_en(run.order_type))
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("Unlearning step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Class accuracy (%)")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("vehicles_1 target class trajectories")
    save_current_figure(figure_dir / "vehicles1_target_trajectories.png")


def plot_coarse_group_summary(run_records: list[dict], figure_dir: Path) -> None:
    groups = []
    for run in RUNS:
        if run.group not in groups:
            groups.append(run.group)
    labels = [group.replace("_", "\n") for group in groups]
    avg_forget = []
    avg_retain = []
    rebound_counts = []
    residual_counts = []
    for group in groups:
        records = [record for record in run_records if record["run"].group == group]
        avg_forget.append(sum(record["final_forget"] or 0.0 for record in records) / len(records))
        avg_retain.append(sum(record["final_retain"] or 0.0 for record in records) / len(records))
        rebound_counts.append(sum(len(record["rebounds"]) for record in records))
        residual_counts.append(sum(len(record["final_residuals"]) for record in records))

    x = list(range(len(groups)))
    plt.figure(figsize=(9, 5.5))
    plt.bar([i - 0.18 for i in x], avg_forget, width=0.18, label="Avg final forget acc")
    plt.bar([i for i in x], avg_retain, width=0.18, label="Avg final retain acc")
    plt.bar([i + 0.18 for i in x], rebound_counts, width=0.18, label="Rebound count")
    plt.bar([i + 0.36 for i in x], residual_counts, width=0.18, label="Final residual count")
    plt.xticks([i + 0.09 for i in x], labels)
    plt.ylabel("Accuracy (%) or count")
    plt.title("Coarse superclass summary")
    plt.ylim(0, 80)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(figure_dir / "coarse_group_summary.png")


def generate_figures(run_records: list[dict], figure_dir: Path) -> list[Path]:
    make_figure_dir(figure_dir)
    plot_final_metrics(run_records, figure_dir)
    plot_rebound_rate(run_records, figure_dir)
    plot_target_accuracy_heatmap(run_records, figure_dir)
    plot_class_trajectory(
        run_records,
        figure_dir,
        class_id=8,
        group="vehicles_1",
        filename="vehicles1_bicycle_rebound.png",
        title="Bicycle (8) rebound in vehicles_1 runs",
    )
    plot_vehicles1_target_trajectories(run_records, figure_dir)
    plot_coarse_group_summary(run_records, figure_dir)
    return sorted(figure_dir.glob("*.png"))


def figure_link(report_path: Path, figure_path: Path) -> str:
    try:
        rel = figure_path.relative_to(report_path.parent)
    except ValueError:
        rel = figure_path
    return str(rel)


def group_summary_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    groups = []
    for run in RUNS:
        if run.group not in groups:
            groups.append(run.group)
    for group in groups:
        records = [record for record in run_records if record["run"].group == group]
        group_zh = records[0]["run"].group_zh
        avg_forget = sum(record["final_forget"] or 0.0 for record in records) / len(records)
        avg_retain = sum(record["final_retain"] or 0.0 for record in records) / len(records)
        total_rebounds = sum(len(record["rebounds"]) for record in records)
        total_eligible = sum(len(record["eligible"]) for record in records)
        total_residuals = sum(len(record["final_residuals"]) for record in records)
        rows.append(
            [
                group_zh,
                len(records),
                pct(avg_forget),
                pct(avg_retain),
                f"{total_rebounds}/{total_eligible}",
                total_residuals,
            ]
        )
    return rows


def order_summary_rows(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for order_type in ["official", "random", "hardfirst"]:
        records = [record for record in run_records if record["run"].order_type == order_type]
        avg_forget = sum(record["final_forget"] or 0.0 for record in records) / len(records)
        avg_retain = sum(record["final_retain"] or 0.0 for record in records) / len(records)
        total_rebounds = sum(len(record["rebounds"]) for record in records)
        total_eligible = sum(len(record["eligible"]) for record in records)
        rows.append([order_zh(order_type), pct(avg_forget), pct(avg_retain), f"{total_rebounds}/{total_eligible}"])
    return rows


def vehicles1_step_rows(record: dict) -> list[list[object]]:
    run = record["run"]
    rows = []
    for step, eval_row in enumerate(record["rows"], start=1):
        new_class = run.order[step - 1]
        forgotten = run.order[:step]
        rows.append(
            [
                step,
                class_label(new_class),
                ",".join(str(class_id) for class_id in forgotten),
                pct(fnum(eval_row.get("forget_accuracy"), None)),
                pct(fnum(eval_row.get("retain_accuracy"), None)),
                *[pct(class_accuracy(eval_row, class_id)) for class_id in VEHICLES1_CLASSES],
            ]
        )
    return rows


def vehicles1_rebound_diagnostics(run_records: list[dict]) -> list[list[object]]:
    rows = []
    for record in [item for item in run_records if item["run"].group == "vehicles_1"]:
        run = record["run"]
        eval_rows = record["rows"]
        for class_id in VEHICLES1_CLASSES:
            forget_step = run.order.index(class_id) + 1
            at_forget = class_accuracy(eval_rows[forget_step - 1], class_id)
            later_values = [class_accuracy(row, class_id) for row in eval_rows[forget_step:]]
            max_later = max(later_values) if later_values else None
            final_acc = class_accuracy(eval_rows[-1], class_id)
            rows.append(
                [
                    order_zh(run.order_type),
                    class_label(class_id),
                    forget_step,
                    pct(at_forget),
                    pct(max_later),
                    pct(final_acc),
                    bool_zh(
                        max_later is not None
                        and max_later >= REBOUND_THRESHOLD
                        and (max_later - at_forget) >= REBOUND_DELTA
                    ),
                ]
            )
    return rows


def build_report(root: Path, base_namespace: str, report_path: Path, figure_dir: Path) -> str:
    run_records, target_items, _step_records = collect_results(root, base_namespace)

    run_summaries = []
    rebound_rows = []
    immediate_rows = []
    final_residual_rows = []

    for record in run_records:
        run = record["run"]
        rows = record["rows"]
        run_summaries.append(
            [
                run.group_zh,
                order_zh(run.order_type),
                run.run_id,
                ",".join(str(class_id) for class_id in run.order),
                record["status"],
                len(rows),
                pct(record["final_forget"]),
                pct(record["final_retain"]),
                pct(record["final_full"]),
                f"{len(record['rebounds'])}/{len(record['eligible'])}" if record["eligible"] else "n/a",
                len(record["final_residuals"]),
                len(record["missing"]),
            ]
        )

        for item in record["rebounds"]:
            rebound_rows.append(
                [
                    item["group"],
                    order_zh(item["order_type"]),
                    item["class"],
                    item["step"],
                    pct(item["accuracy_at_forget"]),
                    pct(item["max_later_accuracy"]),
                    pct(item["final_accuracy"]),
                ]
            )
        for item in record["immediate"]:
            immediate_rows.append(
                [
                    item["group"],
                    order_zh(item["order_type"]),
                    item["class"],
                    item["step"],
                    pct(item["accuracy_at_forget"]),
                ]
            )
        for item in record["final_residuals"]:
            final_residual_rows.append(
                [
                    item["group"],
                    order_zh(item["order_type"]),
                    item["class"],
                    pct(item["final_accuracy"]),
                ]
            )

    completed_runs = sum(1 for record in run_records if record["status"] == "success")
    total_rebounds = sum(len(record["rebounds"]) for record in run_records)
    total_final_residuals = sum(len(record["final_residuals"]) for record in run_records)
    total_eligible = sum(len(record["eligible"]) for record in run_records)
    total_immediate = sum(len(record["immediate"]) for record in run_records)
    vehicles1_records = [record for record in run_records if record["run"].group == "vehicles_1"]
    figure_paths = generate_figures(run_records, figure_dir)
    figures = {path.name: figure_link(report_path, path) for path in figure_paths}

    sections = [
        "# 6_6 CIFAR-100 Single-Coarse Rebound 實驗結果分析",
        "",
        "## 實驗設定",
        "",
        f"- Namespace：`{base_namespace}`",
        f"- Dataset：CIFAR-100，固定 `seed={SEED}`。",
        "- 每條 run 只選一個 CIFAR-100 coarse superclass，依序忘記其中 5 個 fine classes。",
        "- 本次比較三個 coarse superclass：大型雜食/草食動物、車輛 1、花卉。",
        "- 每個 coarse superclass 各跑三種順序：官方順序、隨機順序、困難優先。",
        f"- Significant post-forget rebound 定義：`max_later_accuracy >= {REBOUND_THRESHOLD:.0f}%`，且比 `accuracy_at_forget` 至少高 `{REBOUND_DELTA:.0f}pp`。",
        "- 本報告只分析已完成結果，不重新訓練、不重新 unlearn。",
        "",
        "### 實驗參數設定細節",
        "",
        "本實驗用 `scripts/run_cifar100_single_coarse_rebound_probe_dualgpu.sh` 啟動 9 條 sequential runs；每條 run 再呼叫 `scripts/run_incremental_ordered_logged.sh`，依序執行 `mask -> unlearn -> eval` 三個 stage。每一步都從前一步輸出的 `RLcheckpoint.pth.tar` 接著做下一個 class，因此這是 sequential unlearning，不是 5 個 class 各自從原始模型獨立 unlearn。",
        "",
        markdown_table(
            ["類別", "參數", "設定值"],
            [
                ["Dataset", "`DATASET`", "`cifar100`"],
                ["Model", "`ARCH`", "`resnet18`"],
                ["Initial checkpoint", "`ORIGINAL_MODEL`", "`results/original/cifar100_resnet18_seed1/0model_SA_best.pth.tar`"],
                ["Random seed", "`SEED`", "`1`"],
                ["Steps per run", "`MAX_K`", "`5`"],
                ["Batch size", "`BATCH_SIZE`", "`2048`"],
                ["Mask ratio", "`MASK_RATIO`", "`0.5`，輸出檔名為 `with_0.5.pt`"],
                ["Mask generation epochs", "`MASK_EPOCHS`", "`1`"],
                ["Unlearn method", "`--unlearn`", "`RL`"],
                ["Unlearn epochs", "`UNLEARN_EPOCHS`", "`10`"],
                ["Unlearn learning rate", "`UNLEARN_LR`", "`0.013`"],
                ["Result namespace", "`BASE_NAMESPACE`", "`sequential_single_coarse_rebound_cifar100`"],
                ["GPU scheduling", "`GPU`", "dual-GPU queue：GPU 0 跑 large_omni 與部分 vehicles_1，GPU 1 跑其餘 vehicles_1 與 flowers"],
            ],
        ),
        "",
        "每一步的資料切分如下。假設第 `t` 步的新忘記類別是 `c_t`，累積 forgotten classes 是 `[c_1, ..., c_t]`：mask stage 用當前模型和 `c_t` 產生 saliency mask；unlearn stage 設定 `--class_to_replace c_t`、`--classes_to_replace c_1,...,c_t` 與 `--incremental_forget_only`。因此 `forget loader` 只含當步新類別 `c_t`，`retain loader` 排除所有累積 forgotten classes，舊 forgotten classes 不會被放回 retain training；CIFAR-100 每個 fine class 在 train split 有 500 張，所以每一步 forget set 約 500 張，retain set 會隨累積 forgotten classes 增加而遞減。",
        "",
        "evaluation stage 使用 CIFAR-100 test split (`train=False`) 逐類計算 accuracy，並輸出 `results/eval/<namespace>/<run_id>/seed1_step{t}_forgot_<classes>.csv`。`forget_accuracy` 是累積 forgotten classes 的整體 test accuracy，`retain_accuracy` 是其餘 95 到 99 個 classes 的 test accuracy，`full_test_accuracy` 是全部 100 類 test accuracy。post-forget rebound 只統計「已被忘記且後面還有 later steps 可觀察」的 class；若某 class 在最後一步才被忘，沒有 later step，就不能計入 rebound rate，只能看 final residual。",
        "",
        "## 整體結果",
        "",
        f"- 9 條 run 全部完成：`{completed_runs}/{len(RUNS)}`。",
        f"- old forgotten class 可觀察 later steps 的樣本共有 `{total_eligible}` 個，其中 significant rebound 為 `{total_rebounds}` 個。",
        f"- final residual，也就是最後仍高於 10% 的已忘類別，共 `{total_final_residuals}` 個。",
        f"- immediate forgetting lag 為 `{total_immediate}` 個，表示每個 newly forgotten class 在剛被忘記當步都已經壓到 10% 以下或等於附近。",
        "",
        "關鍵結論很集中：這批 single-coarse probe 中，明顯 rebound 只出現在 `vehicles_1`，而且三種順序都是同一個 fine class：`bicycle / 腳踏車 (8)`。大型雜食/草食動物與花卉組沒有出現 significant rebound，因此目前結果不支持「所有 old forgotten class 都普遍 rebound」；比較合理的說法是，rebound 可能和特定 coarse superclass、特定 fine class、或該類別與 retained classes 的共享特徵有關。",
        "",
        "## 圖表總覽",
        "",
        f"![Final metrics by run]({figures['final_metrics_by_run.png']})",
        "",
        f"![Rebound rate by run]({figures['rebound_rate_by_run.png']})",
        "",
        f"![Target accuracy heatmap]({figures['target_accuracy_heatmap.png']})",
        "",
        f"![Coarse group summary]({figures['coarse_group_summary.png']})",
        "",
        "## 9 條 Run Final Metrics",
        "",
        markdown_table(
            [
                "Coarse superclass",
                "順序",
                "Run ID",
                "Forget order",
                "Progress",
                "Eval CSVs",
                "Final forget acc",
                "Final retain acc",
                "Final full test acc",
                "Rebound rate",
                "Final residual count",
                "Missing eval",
            ],
            run_summaries,
        ),
        "",
        "## Coarse Superclass 比較",
        "",
        markdown_table(
            ["Coarse superclass", "Runs", "平均 final forget acc", "平均 final retain acc", "Rebound count", "Final residual count"],
            group_summary_rows(run_records),
        ),
        "",
        "分組結果顯示，`vehicles_1` 的平均 final forget accuracy 約 8% 到 10% 附近，明顯高於另外兩組，且所有 rebound 與 final residual 都集中在這一組。`large_omni` 的 final forget acc 幾乎歸零，`flowers` 則完全歸零，代表同樣的 unlearning 流程對不同 coarse superclass 的後續殘留行為並不一致。",
        "",
        "## 順序比較",
        "",
        markdown_table(
            ["順序", "平均 final forget acc", "平均 final retain acc", "Rebound count"],
            order_summary_rows(run_records),
        ),
        "",
        "順序本身沒有改變主要結論：`vehicles_1` 在官方、隨機、困難優先三種順序中都出現 `bicycle (8)` rebound；`large_omni` 與 `flowers` 在三種順序中都沒有 significant rebound。不過順序仍然有意義，因為只有較早被忘記的 class 才有足夠 later steps 可以觀察 rebound；如果某個 class 排在最後，就無法測量 post-forget rebound，只能看 final residual。",
        "",
        "## Significant Post-Forget Rebound",
        "",
        markdown_table(
            ["Coarse superclass", "順序", "Class", "Forget step", "At forget", "Max later", "Final"],
            rebound_rows or [["none", "-", "-", "-", "-", "-", "-"]],
        ),
        "",
        "這三個 rebound row 都是 `bicycle / 腳踏車 (8)`。它在三種順序中剛被忘記時 accuracy 都被壓低到 0% 到 2%，但後續 step 又回升到 41% 到 48%，而且 final accuracy 仍維持 40% 到 48%。這代表它不是 immediate forgetting 沒做好，而是比較像 old forgotten class 在後續 unlearning 其他車輛類別時重新恢復辨識能力。",
        "",
        f"![Bicycle rebound]({figures['vehicles1_bicycle_rebound.png']})",
        "",
        "## vehicles_1 詳細分析：每一步每個車輛類別準確率",
        "",
        "以下表格只列 `vehicles_1` 的五個 target classes，因為這一組是本次唯一出現 rebound 的 coarse superclass。完整 CIFAR-100 100 個 class 在每一步的 accuracy 仍保留於 `6_6_cifar100_single_coarse_rebound_step_accuracy_wide.csv` 與 `6_6_cifar100_single_coarse_rebound_step_accuracy_long.csv`。",
        "",
        f"![vehicles_1 target trajectories]({figures['vehicles1_target_trajectories.png']})",
        "",
        "### vehicles_1 official：8,13,48,58,90",
        "",
        markdown_table(
            ["Step", "Newly forgotten", "Forgotten so far", "Forget acc", "Retain acc", "bicycle (8)", "bus (13)", "motorcycle (48)", "pickup_truck (58)", "train (90)"],
            vehicles1_step_rows(next(record for record in vehicles1_records if record["run"].order_type == "official")),
        ),
        "",
        "### vehicles_1 random：48,8,90,13,58",
        "",
        markdown_table(
            ["Step", "Newly forgotten", "Forgotten so far", "Forget acc", "Retain acc", "bicycle (8)", "bus (13)", "motorcycle (48)", "pickup_truck (58)", "train (90)"],
            vehicles1_step_rows(next(record for record in vehicles1_records if record["run"].order_type == "random")),
        ),
        "",
        "### vehicles_1 hardfirst：13,8,58,90,48",
        "",
        markdown_table(
            ["Step", "Newly forgotten", "Forgotten so far", "Forget acc", "Retain acc", "bicycle (8)", "bus (13)", "motorcycle (48)", "pickup_truck (58)", "train (90)"],
            vehicles1_step_rows(next(record for record in vehicles1_records if record["run"].order_type == "hardfirst")),
        ),
        "",
        "### vehicles_1 rebound 診斷表",
        "",
        markdown_table(
            ["順序", "Class", "Forget step", "At forget", "Max later", "Final", "Significant rebound"],
            vehicles1_rebound_diagnostics(run_records),
        ),
        "",
        "從逐步表格看，`bicycle / 腳踏車 (8)` 的行為和其他車輛類別不同。它在剛被忘記時確實被壓到 0% 到 2%，但後續每當繼續忘其他 vehicles_1 類別時又逐步回升；official 從 0% 回到 22%、29%、41%、43%，random 從 2% 回到 29%、41%、40%，hardfirst 從 1% 回到 28%、30%、48%。相反地，`bus`、`motorcycle`、`pickup_truck`、`train` 一旦被忘記後大多維持在 0%，沒有同樣的回升軌跡。",
        "",
        "可能原因可以分成三點。第一，這不是單純的 forgetting lag：`bicycle` 在被忘記當步已經降到 0% 到 2%，後續才回升。第二，這不是整個 vehicles_1 都忘不好：其他已忘車輛類別大多維持 0%，final residual 幾乎完全由 `bicycle` 貢獻。第三，這比較像 shared representation 被後續步驟重新調整：`bicycle` 和其他 vehicles_1 類別共享輪子、道路背景、交通工具輪廓等低階/中階視覺特徵，但它和 `bus/train/pickup_truck` 的外觀差異又足夠大，因此後續忘大型車輛時可能沒有直接把 bicycle feature 一起壓掉，反而讓它從 shared vehicle features 中被拉回來。",
        "",
        "也因此，vehicles_1 的 final forget accuracy 偏高主要不是五個車輛類別都忘不好，而是被 `bicycle (8)` 單一類別拉高。這點很重要：若只看 final forget acc 8% 到 10%，可能會誤以為整個 vehicles_1 都有殘留；但逐類檢查後可見 final residual 幾乎完全由 bicycle 貢獻。",
        "",
        "## vehicles_1 成因分析延伸報告",
        "",
        "更細的 checkpoint-level 診斷已整理在 `6_6_cifar100_vehicles1_rebound_cause_analysis.md`。該報告不重新跑 unlearning，而是讀取三條 `vehicles_1` runs 的 step0-step5 checkpoints，追蹤 CIFAR-100 test split 中每張 vehicles_1 圖片的 prediction、logit margin、top-k、confusion、penultimate feature、mask overlap 與 parameter delta。",
        "",
        "目前延伸分析支持的解釋是：`bicycle (8)` 的 rebound 不是 test data 評估錯置，也不是 immediate forgetting lag；比較像後續 sequential unlearning 其他車輛類別時，shared vehicle representation 或 classifier decision boundary 被重新調整，使一批 bicycle test samples 的 true-label margin 回升。",
        "",
        "## 未發生 Rebound 的案例",
        "",
        "### Immediate Forgetting Lag",
        "",
        markdown_table(
            ["Coarse superclass", "順序", "Class", "Forget step", "At forget"],
            immediate_rows or [["none", "-", "-", "-", "-"]],
        ),
        "",
        "這裡沒有任何 row，表示 newly forgotten class 在當步沒有明顯殘留。換句話說，這次觀察到的 `bicycle (8)` 問題不是因為第一時間忘不掉，而是因為後面再忘其他類別時發生 rebound。",
        "",
        "### Final Residual",
        "",
        markdown_table(
            ["Coarse superclass", "順序", "Class", "Final accuracy"],
            final_residual_rows or [["none", "-", "-", "-"]],
        ),
        "",
        "final residual 也完全對應到 `bicycle (8)`，表示它不只短暫 rebound，而是一路殘留到 k=5 final evaluation。",
        "",
        "## 結論",
        "",
        "1. 這批 single-coarse 實驗沒有支持「old forgotten class rebound 普遍發生在所有類別」；9 條 run 中只有 3 個 significant rebound row。",
        "2. rebound 高度集中在 `vehicles_1` 的 `bicycle / 腳踏車 (8)`，且三種順序都重現，表示這是一個穩定而值得追查的案例。",
        "3. `large_omni` 與 `flowers` 幾乎沒有 rebound，說明同樣是 coarse superclass 內 sequential unlearning，不同語義群的 residual 行為差異很大。",
        "4. 順序不是這次是否 rebound 的主要決定因素，但會影響可觀察 later steps 的長度，因此之後若要量化 rebound rate，應避免把重要 class 放在最後一步。",
        "5. 本實驗固定 `seed=1`，所以目前結論應寫成初步 evidence；若要主張普遍性，需要補 seed 或補更多 coarse superclass。",
        "",
        "## 建議下一步",
        "",
        "- 以 `vehicles_1` 為 positive case，追蹤 `bicycle (8)` 在後續每一步的 confusion matrix 或 logits，看它是否被重新推回正確類別。",
        "- 增加 3 到 5 個 coarse superclass，尤其是其他人工物與交通相關類別，檢查 rebound 是否是 vehicle-like classes 的特性。",
        "- 若資源允許，再補不同 seed；若不考慮 seed，至少要增加不同 coarse superclass 來支撐「不是單一群組偶然現象」。",
        "",
    ]
    return "\n".join(sections)


def forgotten_classes_for_step(row: dict) -> set[int]:
    text = row.get("forgotten_classes") or ""
    return {int(item) for item in text.split(",") if item.strip()}


def export_accuracy_csvs(
    root: Path,
    base_namespace: str,
    wide_csv_path: Path,
    long_csv_path: Path,
    target_csv_path: Path,
) -> tuple[int, int, int]:
    wide_rows = []
    long_rows = []
    target_rows = []

    for run in RUNS:
        rows, _missing = read_eval_rows(root, base_namespace, run)
        for step, row in enumerate(rows, start=1):
            forgotten = forgotten_classes_for_step(row)
            new_class = run.order[step - 1]
            wide_row = {
                "group": run.group,
                "group_zh": run.group_zh,
                "order_type": run.order_type,
                "run_id": run.run_id,
                "step": step,
                "new_class_id": new_class,
                "new_class": class_label(new_class),
                "forgotten_classes": row.get("forgotten_classes", ""),
                "forget_accuracy": row.get("forget_accuracy", ""),
                "UA": row.get("UA", ""),
                "retain_accuracy": row.get("retain_accuracy", ""),
                "full_test_accuracy": row.get("full_test_accuracy", ""),
            }
            for class_id in range(100):
                acc = class_accuracy(row, class_id)
                wide_row[f"class_{class_id}_accuracy"] = acc
                long_rows.append(
                    {
                        "group": run.group,
                        "group_zh": run.group_zh,
                        "order_type": run.order_type,
                        "run_id": run.run_id,
                        "step": step,
                        "new_class_id": new_class,
                        "new_class": class_label(new_class),
                        "class_id": class_id,
                        "class_name": CIFAR100_FINE_LABELS[class_id],
                        "class_label": class_label(class_id),
                        "accuracy": acc,
                        "is_forgotten_so_far": class_id in forgotten,
                        "is_newly_forgotten": class_id == new_class,
                        "is_target_class": class_id in run.order,
                    }
                )
            wide_rows.append(wide_row)

        for item in run_rebound_items(rows, run):
            target_rows.append(
                {
                    "group": run.group,
                    "group_zh": run.group_zh,
                    "order_type": run.order_type,
                    "run_id": run.run_id,
                    "class_id": item["class_id"],
                    "class": item["class"],
                    "forget_step": item["step"],
                    "accuracy_at_forget": item["accuracy_at_forget"],
                    "max_later_accuracy": "" if item["max_later_accuracy"] is None else item["max_later_accuracy"],
                    "final_accuracy": item["final_accuracy"],
                    "immediate_lag": item["immediate_lag"],
                    "significant_rebound": item["significant_rebound"],
                    "final_residual": item["final_residual"],
                }
            )

    if wide_rows:
        with wide_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(wide_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wide_rows)

    if long_rows:
        with long_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
            writer.writeheader()
            writer.writerows(long_rows)

    if target_rows:
        with target_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(target_rows[0].keys()))
            writer.writeheader()
            writer.writerows(target_rows)

    return len(wide_rows), len(long_rows), len(target_rows)


def main() -> None:
    args = parse_args()
    report = build_report(args.root, args.base_namespace, args.report_path, args.figure_dir)
    args.report_path.write_text(report, encoding="utf-8")
    wide_count, long_count, target_count = export_accuracy_csvs(
        args.root,
        args.base_namespace,
        args.wide_csv_path,
        args.long_csv_path,
        args.target_csv_path,
    )
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.wide_csv_path} ({wide_count} rows)")
    print(f"Wrote {args.long_csv_path} ({long_count} rows)")
    print(f"Wrote {args.target_csv_path} ({target_count} rows)")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
