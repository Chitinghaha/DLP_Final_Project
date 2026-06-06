#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PILOT_CLASSES = [
    {"id": 3, "name": "bear", "group": "large_carnivores"},
    {"id": 42, "name": "leopard", "group": "large_carnivores"},
    {"id": 43, "name": "lion", "group": "large_carnivores"},
    {"id": 88, "name": "tiger", "group": "large_carnivores"},
    {"id": 97, "name": "wolf", "group": "large_carnivores"},
    {"id": 15, "name": "camel", "group": "large_omnivores_and_herbivores"},
    {"id": 19, "name": "cattle", "group": "large_omnivores_and_herbivores"},
    {"id": 21, "name": "chimpanzee", "group": "large_omnivores_and_herbivores"},
    {"id": 31, "name": "elephant", "group": "large_omnivores_and_herbivores"},
    {"id": 38, "name": "kangaroo", "group": "large_omnivores_and_herbivores"},
    {"id": 34, "name": "fox", "group": "medium_mammals"},
    {"id": 63, "name": "porcupine", "group": "medium_mammals"},
    {"id": 64, "name": "possum", "group": "medium_mammals"},
    {"id": 66, "name": "raccoon", "group": "medium_mammals"},
    {"id": 75, "name": "skunk", "group": "medium_mammals"},
    {"id": 36, "name": "hamster", "group": "small_mammals"},
    {"id": 50, "name": "mouse", "group": "small_mammals"},
    {"id": 65, "name": "rabbit", "group": "small_mammals"},
    {"id": 74, "name": "shrew", "group": "small_mammals"},
    {"id": 80, "name": "squirrel", "group": "small_mammals"},
]

CLASS_META = {item["id"]: item for item in PILOT_CLASSES}
GROUP_COLORS = {
    "large_carnivores": "#E45756",
    "large_omnivores_and_herbivores": "#72B7B2",
    "medium_mammals": "#54A24B",
    "small_mammals": "#F58518",
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
        default=root / "cifar100_incremental_ordered_pilot_analysis_figures",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "cifar100_incremental_ordered_pilot_analysis.md",
    )
    return parser.parse_args()


def eval_filename(seed: int, step: int, order: list[int]) -> str:
    forgotten = "_".join(str(class_id) for class_id in order[:step])
    return f"seed{seed}_step{step}_forgot_{forgotten}.csv"


def read_eval_rows(eval_root: Path, namespace: str, seed: int, order: list[int]) -> list[dict]:
    rows = []
    for step in range(1, len(order) + 1):
        filename = eval_filename(seed, step, order)
        path = eval_root / namespace / filename
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["step"] = step
        rows.append(row)
    return rows


def read_progress_rows(progress_csv: Path) -> list[dict]:
    with progress_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    return [int(row["step"]) for row in rows], [float(row[key]) for row in rows]


def class_series(rows: list[dict], class_id: int) -> tuple[list[int], list[float]]:
    return metric_series(rows, f"class_{class_id}_accuracy")


def newly_forgotten_series(rows: list[dict], order: list[int]) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    for step, class_id in enumerate(order, start=1):
        row = next(row for row in rows if int(row["step"]) == step)
        steps.append(step)
        values.append(float(row[f"class_{class_id}_accuracy"]))
    return steps, values


def runtime_by_step(rows: list[dict]) -> tuple[list[int], dict[str, list[float]]]:
    steps = sorted({int(row["step"]) for row in rows})
    stages = {"mask": [], "unlearn": [], "eval": []}
    for step in steps:
        step_rows = [row for row in rows if int(row["step"]) == step and row["status"] == "success"]
        by_stage = {row["stage"]: row for row in step_rows}
        for stage in stages:
            stages[stage].append(float(by_stage.get(stage, {}).get("duration_sec", 0.0)))
    return steps, stages


def total_duration(rows: list[dict]) -> float:
    return sum(float(row["duration_sec"]) for row in rows if row["status"] == "success")


def style_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def save_metric_plot(
    output_path: Path,
    title: str,
    ylabel: str,
    metric_key: str,
    order_rows: dict[str, list[dict]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = {"normal": "o", "clustered": "s", "interleaved": "^"}
    for key, config in ORDER_CONFIGS.items():
        x, y = metric_series(order_rows[key], metric_key)
        ax.plot(x, y, marker=markers[key], linewidth=2, label=config["label"])
    ax.set_xticks(sorted(set(metric_series(order_rows["normal"], metric_key)[0])))
    style_ax(ax, title, ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_newly_forgotten_plot(output_path: Path, order_rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = {"normal": "o", "clustered": "s", "interleaved": "^"}
    for key, config in ORDER_CONFIGS.items():
        x, y = newly_forgotten_series(order_rows[key], config["order"])
        ax.plot(x, y, marker=markers[key], linewidth=2, label=config["label"])
    ax.axhline(10.0, color="crimson", linestyle="--", linewidth=1, alpha=0.7, label="10% guide")
    ax.set_xticks(list(range(1, len(PILOT_CLASSES) + 1)))
    style_ax(ax, "Newly Forgotten Class Accuracy at Each Step", "Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_selected_class_plot(
    output_path: Path,
    title: str,
    rows: list[dict],
    order: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    forget_step_map = {class_id: step for step, class_id in enumerate(order, start=1)}

    for class_id in order:
        meta = CLASS_META[class_id]
        x, y = class_series(rows, class_id)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=GROUP_COLORS[meta["group"]],
            label=f'{meta["name"]} ({class_id}, forget@{forget_step_map[class_id]})',
        )
        ax.axvline(
            forget_step_map[class_id],
            color=GROUP_COLORS[meta["group"]],
            linestyle="--",
            linewidth=0.8,
            alpha=0.15,
        )

    ax.set_xticks(list(range(1, len(order) + 1)))
    style_ax(ax, title, "Accuracy (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_runtime_plot(output_path: Path, order_progress: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {"mask": "#4C78A8", "unlearn": "#F58518", "eval": "#54A24B"}

    for ax, key in zip(axes, ORDER_CONFIGS):
        config = ORDER_CONFIGS[key]
        steps, stages = runtime_by_step(order_progress[key])
        bottom = [0.0] * len(steps)
        for stage in ["mask", "unlearn", "eval"]:
            values = stages[stage]
            ax.bar(steps, values, bottom=bottom, color=colors[stage], label=stage)
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_title(f'{config["label"]} ({int(total_duration(order_progress[key]))}s)')
        ax.set_xlabel("Step")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(steps[::2] if len(steps) > 10 else steps)

    axes[0].set_ylabel("Duration (sec)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def smoke_status(progress_rows: list[dict]) -> str:
    if not progress_rows:
        return "missing"
    if any(row["status"] == "failed" for row in progress_rows):
        return "failed"
    return "success"


def lag_rows(rows: list[dict], order: list[int]) -> list[dict]:
    items = []
    for step, class_id in enumerate(order, start=1):
        row = next(entry for entry in rows if int(entry["step"]) == step)
        accuracy = float(row[f"class_{class_id}_accuracy"])
        meta = CLASS_META[class_id]
        items.append(
            {
                "step": step,
                "class_id": class_id,
                "class_name": meta["name"],
                "group": meta["group"],
                "accuracy": accuracy,
                "is_failure": accuracy > 10.0,
            }
        )
    return items


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def figure_link(output_dir: Path, filename: str) -> str:
    return f"./{output_dir.name}/{filename}"


def write_report(
    report_path: Path,
    output_dir: Path,
    order_rows: dict[str, list[dict]],
    order_progress: dict[str, list[dict]],
    smoke_progress: dict[str, list[dict]],
) -> None:
    smoke_table = []
    final_metric_table = []
    lag_count_table = []
    worst_lag_sections = []

    for key, config in ORDER_CONFIGS.items():
        smoke_table.append(
            [config["label"], config["smoke_run_id"], smoke_status(smoke_progress[key])]
        )

        final_row = order_rows[key][-1]
        final_metric_table.append(
            [
                config["label"],
                f'{float(final_row["forget_accuracy"]):.2f}',
                f'{float(final_row["UA"]):.2f}',
                f'{float(final_row["retain_accuracy"]):.2f}',
                f'{float(final_row["full_test_accuracy"]):.2f}',
                f"{int(total_duration(order_progress[key]))}",
            ]
        )

        lag_items = lag_rows(order_rows[key], config["order"])
        lag_failures = [item for item in lag_items if item["is_failure"]]
        lag_count_table.append(
            [
                config["label"],
                str(len(lag_failures)),
                f"{max(item['accuracy'] for item in lag_items):.1f}",
            ]
        )

        top_rows = sorted(lag_items, key=lambda item: item["accuracy"], reverse=True)[:5]
        worst_lag_sections.append(
            f"### {config['label']} top lag steps\n\n"
            + markdown_table(
                ["Step", "Class", "Group", "Accuracy", ">10%"],
                [
                    [
                        str(item["step"]),
                        f'{item["class_name"]} ({item["class_id"]})',
                        item["group"],
                        f'{item["accuracy"]:.1f}',
                        "yes" if item["is_failure"] else "no",
                    ]
                    for item in top_rows
                ],
            )
        )

    report = f"""# CIFAR-100 Sequential Incremental Ordered Pilot Analysis

## 1. Experiment Summary

This report analyzes the CIFAR-100 `sequential_incremental_ordered_cifar100_pilot` line under the correct incremental data flow:

- each step only trains on the newly forgotten class
- older forgotten classes are permanently excluded from future train loaders
- cumulative forgotten classes are only used in evaluation
- failure is defined as newly forgotten class accuracy remaining above `10%` at the same step

The pilot uses a fixed 20-class high-confusion animal subset and compares three orders:

- `Normal`
- `Clustered`
- `Interleaved`

## 2. Smoke Summary

{markdown_table(["Order", "Run ID", "Status"], smoke_table)}

## 3. Final Metrics

{markdown_table(["Order", "Forget acc", "UA", "Retain acc", "Full test acc", "Runtime (s)"], final_metric_table)}

## 4. Lag Summary

{markdown_table(["Order", "Lag steps (>10%)", "Worst newly forgotten acc"], lag_count_table)}

![Forget Accuracy vs Step]({figure_link(output_dir, "forget_accuracy_vs_step.png")})

![UA vs Step]({figure_link(output_dir, "ua_vs_step.png")})

![Retain Accuracy vs Step]({figure_link(output_dir, "retain_accuracy_vs_step.png")})

![Full Test Accuracy vs Step]({figure_link(output_dir, "full_test_accuracy_vs_step.png")})

![Newly Forgotten Class Accuracy vs Step]({figure_link(output_dir, "newly_forgotten_class_accuracy_vs_step.png")})

## 5. Selected-Class Trajectories

![Normal Selected Class Accuracy]({figure_link(output_dir, "normal_selected_class_accuracy.png")})

![Clustered Selected Class Accuracy]({figure_link(output_dir, "clustered_selected_class_accuracy.png")})

![Interleaved Selected Class Accuracy]({figure_link(output_dir, "interleaved_selected_class_accuracy.png")})

## 6. Runtime

![Runtime by Order]({figure_link(output_dir, "runtime_by_order.png")})

## 7. Top Lag Steps

{chr(10).join(worst_lag_sections)}
"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    eval_root = root / "results/eval"
    logs_root = root / "results/logs"
    output_dir = args.output_dir
    report_path = args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)

    order_rows = {}
    order_progress = {}
    smoke_progress = {}

    for key, config in ORDER_CONFIGS.items():
        order_rows[key] = read_eval_rows(
            eval_root, args.result_namespace, args.seed, config["order"]
        )
        order_progress[key] = read_progress_rows(
            logs_root / args.result_namespace / config["run_id"] / "progress.csv"
        )
        smoke_progress[key] = read_progress_rows(
            logs_root / args.result_namespace / config["smoke_run_id"] / "progress.csv"
        )

    save_metric_plot(
        output_dir / "forget_accuracy_vs_step.png",
        "Forget Accuracy vs Step",
        "Forget Accuracy (%)",
        "forget_accuracy",
        order_rows,
    )
    save_metric_plot(
        output_dir / "ua_vs_step.png",
        "UA vs Step",
        "UA (%)",
        "UA",
        order_rows,
    )
    save_metric_plot(
        output_dir / "retain_accuracy_vs_step.png",
        "Retain Accuracy vs Step",
        "Retain Accuracy (%)",
        "retain_accuracy",
        order_rows,
    )
    save_metric_plot(
        output_dir / "full_test_accuracy_vs_step.png",
        "Full Test Accuracy vs Step",
        "Full Test Accuracy (%)",
        "full_test_accuracy",
        order_rows,
    )
    save_newly_forgotten_plot(output_dir / "newly_forgotten_class_accuracy_vs_step.png", order_rows)

    for key, config in ORDER_CONFIGS.items():
        save_selected_class_plot(
            output_dir / f"{key}_selected_class_accuracy.png",
            f'{config["label"]} Selected-Class Accuracy',
            order_rows[key],
            config["order"],
        )

    save_runtime_plot(output_dir / "runtime_by_order.png", order_progress)
    write_report(report_path, output_dir, order_rows, order_progress, smoke_progress)


if __name__ == "__main__":
    main()
