#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INTERLEAVED_EVAL_FILES = [
    (1, "sequential_incremental_ordered", "seed1_step1_forgot_0.csv"),
    (2, "sequential_incremental_ordered", "seed1_step2_forgot_0_6.csv"),
    (3, "sequential_incremental_ordered", "seed1_step3_forgot_0_6_1.csv"),
    (4, "sequential_incremental_ordered", "seed1_step4_forgot_0_6_1_4.csv"),
    (5, "sequential_incremental_ordered", "seed1_step5_forgot_0_6_1_4_2.csv"),
    (6, "sequential_incremental_ordered_resume", "seed1_step6_forgot_0_6_1_4_2_7.csv"),
    (7, "sequential_incremental_ordered_resume", "seed1_step7_forgot_0_6_1_4_2_7_3.csv"),
    (8, "sequential_incremental_ordered_resume", "seed1_step8_forgot_0_6_1_4_2_7_3_8.csv"),
    (9, "sequential_incremental_ordered_resume", "seed1_step9_forgot_0_6_1_4_2_7_3_8_5.csv"),
]

CLUSTERED_EVAL_FILES = [
    (1, "sequential_incremental_ordered", "seed1_step1_forgot_2.csv"),
    (2, "sequential_incremental_ordered", "seed1_step2_forgot_2_3.csv"),
    (3, "sequential_incremental_ordered", "seed1_step3_forgot_2_3_4.csv"),
    (4, "sequential_incremental_ordered", "seed1_step4_forgot_2_3_4_5.csv"),
    (5, "sequential_incremental_ordered", "seed1_step5_forgot_2_3_4_5_6.csv"),
    (6, "sequential_incremental_ordered", "seed1_step6_forgot_2_3_4_5_6_7.csv"),
    (7, "sequential_incremental_ordered", "seed1_step7_forgot_2_3_4_5_6_7_0.csv"),
    (8, "sequential_incremental_ordered", "seed1_step8_forgot_2_3_4_5_6_7_0_1.csv"),
    (9, "sequential_incremental_ordered", "seed1_step9_forgot_2_3_4_5_6_7_0_1_8.csv"),
]

INTERLEAVED_FORGET_ORDER = [0, 6, 1, 4, 2, 7, 3, 8, 5]
CLUSTERED_FORGET_ORDER = [2, 3, 4, 5, 6, 7, 0, 1, 8]
ALL_CLASSES = list(range(10))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "sequential_incremental_ordered_analysis_figures",
    )
    return parser.parse_args()


def read_eval_rows(eval_root: Path, mapping: list[tuple[int, str, str]]) -> list[dict]:
    rows = []
    for step, namespace, filename in mapping:
        with open(eval_root / namespace / filename, newline="") as handle:
            row = next(csv.DictReader(handle))
        row["step"] = step
        rows.append(row)
    return rows


def read_progress_rows(progress_csv: Path) -> list[dict]:
    with open(progress_csv, newline="") as handle:
        return list(csv.DictReader(handle))


def metric_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    return [int(row["step"]) for row in rows], [float(row[key]) for row in rows]


def class_series(rows: list[dict], class_id: int) -> tuple[list[int], list[float]]:
    key = f"class_{class_id}_accuracy"
    return metric_series(rows, key)


def newly_forgotten_series(
    rows: list[dict], forget_order: list[int]
) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    for step, class_id in enumerate(forget_order, start=1):
        row = next(row for row in rows if int(row["step"]) == step)
        steps.append(step)
        values.append(float(row[f"class_{class_id}_accuracy"]))
    return steps, values


def runtime_by_step(rows: list[dict]) -> tuple[list[int], dict[str, list[float]]]:
    steps = sorted({int(row["step"]) for row in rows})
    stages = {"mask": [], "unlearn": [], "eval": []}
    for step in steps:
        step_rows = [row for row in rows if int(row["step"]) == step]
        by_stage = {row["stage"]: row for row in step_rows}
        for stage in stages:
            if stage in by_stage:
                stages[stage].append(float(by_stage[stage]["duration_sec"]))
            else:
                stages[stage].append(0.0)
    return steps, stages


def total_duration(rows: list[dict]) -> float:
    return sum(float(row["duration_sec"]) for row in rows)


def stitched_interleaved_progress(
    original_progress: list[dict], resume_progress: list[dict]
) -> list[dict]:
    original_success_prefix = [
        row for row in original_progress if int(row["step"]) <= 5 and row["status"] == "success"
    ]
    resume_success_suffix = [row for row in resume_progress if row["status"] == "success"]
    return original_success_prefix + resume_success_suffix


def style_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def save_metric_plot(
    output_path: Path,
    title: str,
    ylabel: str,
    clustered_rows: list[dict],
    interleaved_rows: list[dict],
    metric_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_c, y_c = metric_series(clustered_rows, metric_key)
    x_i, y_i = metric_series(interleaved_rows, metric_key)
    ax.plot(x_c, y_c, marker="o", linewidth=2, label="clustered_animals")
    ax.plot(x_i, y_i, marker="s", linewidth=2, label="interleaved (stitched complete run)")
    ax.set_xticks(sorted(set(x_c + x_i)))
    style_ax(ax, title, ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_runtime_plot(
    output_path: Path,
    clustered_progress: list[dict],
    interleaved_progress: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"mask": "#4C78A8", "unlearn": "#F58518", "eval": "#54A24B"}

    for ax, rows, title in [
        (
            axes[0],
            clustered_progress,
            f"Clustered animals runtime ({int(total_duration(clustered_progress))}s)",
        ),
        (
            axes[1],
            interleaved_progress,
            f"Interleaved runtime ({int(total_duration(interleaved_progress))}s, success path)",
        ),
    ]:
        steps, stages = runtime_by_step(rows)
        bottom = [0.0] * len(steps)
        for stage in ["mask", "unlearn", "eval"]:
            values = stages[stage]
            ax.bar(steps, values, bottom=bottom, color=colors[stage], label=stage)
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(steps)

    axes[0].set_ylabel("Duration (sec)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_all_class_plot(
    output_path: Path,
    title: str,
    rows: list[dict],
    forget_order: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10").colors
    forget_step_map = {class_id: step for step, class_id in enumerate(forget_order, start=1)}

    for class_id in ALL_CLASSES:
        x, y = class_series(rows, class_id)
        label = (
            f"class_{class_id} (forget@{forget_step_map[class_id]})"
            if class_id in forget_step_map
            else f"class_{class_id} (retain)"
        )
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=colors[class_id % len(colors)],
            label=label,
        )

    for class_id, forget_step in forget_step_map.items():
        ax.axvline(
            forget_step,
            color=colors[class_id % len(colors)],
            linestyle="--",
            linewidth=0.8,
            alpha=0.2,
        )

    ax.set_xticks(sorted(int(row["step"]) for row in rows))
    style_ax(ax, title, "Accuracy (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_newly_forgotten_plot(
    output_path: Path,
    clustered_rows: list[dict],
    interleaved_rows: list[dict],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x_c, y_c = newly_forgotten_series(clustered_rows, CLUSTERED_FORGET_ORDER)
    x_i, y_i = newly_forgotten_series(interleaved_rows, INTERLEAVED_FORGET_ORDER)

    ax.plot(x_c, y_c, marker="o", linewidth=2, label="clustered_animals")
    ax.plot(x_i, y_i, marker="s", linewidth=2, label="interleaved")
    ax.axhline(10.0, color="crimson", linestyle="--", linewidth=1, alpha=0.7, label="10% guide")
    ax.set_xticks(sorted(set(x_c + x_i)))
    style_ax(ax, "Newly Forgotten Class Accuracy at Each Step", "Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root
    eval_root = root / "results/eval"
    logs_root = root / "results/logs"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    interleaved_eval = read_eval_rows(eval_root, INTERLEAVED_EVAL_FILES)
    clustered_eval = read_eval_rows(eval_root, CLUSTERED_EVAL_FILES)
    interleaved_original_progress = read_progress_rows(
        logs_root / "sequential_incremental_ordered/incremental_interleaved_seed1_k9/progress.csv"
    )
    interleaved_resume_progress = read_progress_rows(
        logs_root
        / "sequential_incremental_ordered_resume/incremental_interleaved_seed1_k9_resume_from_step6/progress.csv"
    )
    interleaved_progress = stitched_interleaved_progress(
        interleaved_original_progress, interleaved_resume_progress
    )
    clustered_progress = read_progress_rows(
        logs_root / "sequential_incremental_ordered/incremental_clustered_animals_seed1_k9/progress.csv"
    )

    save_metric_plot(
        output_dir / "forget_accuracy_vs_step.png",
        "Forget Accuracy vs Step",
        "Forget Accuracy (%)",
        clustered_eval,
        interleaved_eval,
        "forget_accuracy",
    )
    save_metric_plot(
        output_dir / "ua_vs_step.png",
        "UA vs Step",
        "UA (%)",
        clustered_eval,
        interleaved_eval,
        "UA",
    )
    save_metric_plot(
        output_dir / "retain_accuracy_vs_step.png",
        "Retain Accuracy vs Step",
        "Retain Accuracy (%)",
        clustered_eval,
        interleaved_eval,
        "retain_accuracy",
    )
    save_metric_plot(
        output_dir / "full_test_accuracy_vs_step.png",
        "Full Test Accuracy vs Step",
        "Full Test Accuracy (%)",
        clustered_eval,
        interleaved_eval,
        "full_test_accuracy",
    )
    save_runtime_plot(
        output_dir / "runtime_by_step.png",
        clustered_progress,
        interleaved_progress,
    )
    save_all_class_plot(
        output_dir / "interleaved_all_class_accuracy.png",
        "Interleaved All-Class Accuracy Trajectories",
        interleaved_eval,
        INTERLEAVED_FORGET_ORDER,
    )
    save_all_class_plot(
        output_dir / "clustered_all_class_accuracy.png",
        "Clustered Animals All-Class Accuracy Trajectories",
        clustered_eval,
        CLUSTERED_FORGET_ORDER,
    )
    save_newly_forgotten_plot(
        output_dir / "newly_forgotten_class_accuracy_vs_step.png",
        clustered_eval,
        interleaved_eval,
    )


if __name__ == "__main__":
    main()
