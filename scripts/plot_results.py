#!/usr/bin/env python3
"""Plot the scaling studies.

    python3 scripts/plot_results.py results/strong_scaling.csv \\
                                    --weak results/weak_scaling.csv \\
                                    --out graphs/

Reads the per-run CSV the benchmark writes - one row per repeat, not per
configuration - so the error bars come from the measurements rather than
from an assumption about them. The original project's graphs were drawn in
a spreadsheet from single samples, which cannot show whether a difference
between two bars is real.

Every chart carries the machine it was measured on and the number of
repeats, because a scaling curve without its core count is not a result.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
from collections import defaultdict

import matplotlib

# Agg: these run in CI and over SSH, where there is no display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  - must follow the backend choice

# Colour-blind safe, and distinguishable in greyscale when printed.
PALETTE = {
    "pthreads": "#0072B2",
    "openmp": "#D55E00",
    "serial": "#009E73",
    "ideal": "#999999",
}


def read_runs(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no measurements")
    for row in rows:
        row["threads"] = int(row["threads"])
        row["points"] = int(row["points"])
        row["wall_seconds"] = float(row["wall_seconds"])
        row["cpu_seconds"] = float(row["cpu_seconds"])
    return rows


def aggregate(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Mean, standard deviation and sample count per (backend, threads)."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    cpu: dict[tuple[str, int], list[float]] = defaultdict(list)
    points: dict[tuple[str, int], int] = {}

    for row in rows:
        key = (row["backend"], row["threads"])
        grouped[key].append(row["wall_seconds"])
        cpu[key].append(row["cpu_seconds"])
        points[key] = row["points"]

    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "mean": statistics.fmean(values),
            # Sample standard deviation; a single repeat has none to report,
            # which is itself worth seeing on the chart.
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "repeats": len(values),
            "cpu_mean": statistics.fmean(cpu[key]),
            "points": points[key],
        }
    return summary


def machine_caption(repeats: int) -> str:
    cores = os.cpu_count() or 0
    return (
        f"{platform.system()} {platform.machine()}, {cores} logical cores, "
        f"{repeats} repeats per point, error bars are one standard deviation"
    )


def plot_strong(summary: dict, out_path: str, caption: str) -> None:
    backends = sorted({backend for backend, _ in summary})
    figure, (speed_axes, time_axes) = plt.subplots(1, 2, figsize=(11, 4.5))

    thread_counts = sorted({threads for _, threads in summary})

    # Ideal linear speedup, for scale. Without it a 1.8x on two cores and a
    # 1.8x on thirty-two look equally good on the page.
    speed_axes.plot(
        thread_counts,
        thread_counts,
        linestyle="--",
        color=PALETTE["ideal"],
        label="ideal (linear)",
        zorder=1,
    )

    for backend in backends:
        points = [(t, summary[(backend, t)]) for t in thread_counts if (backend, t) in summary]
        if not points:
            continue
        threads = [t for t, _ in points]
        baseline = points[0][1]["mean"]
        speedups = [baseline / entry["mean"] for _, entry in points]
        # Propagate the timing spread into the speedup, rather than drawing
        # a bare point that looks more certain than it is.
        errors = [
            baseline / entry["mean"] * (entry["stddev"] / entry["mean"])
            for _, entry in points
        ]
        colour = PALETTE.get(backend, None)

        speed_axes.errorbar(threads, speedups, yerr=errors, marker="o",
                            capsize=3, color=colour, label=backend, zorder=2)
        time_axes.errorbar(threads, [entry["mean"] for _, entry in points],
                           yerr=[entry["stddev"] for _, entry in points],
                           marker="o", capsize=3, color=colour, label=backend)

    cores = os.cpu_count() or 0
    if cores and min(thread_counts) <= cores <= max(thread_counts):
        for axes in (speed_axes, time_axes):
            axes.axvline(cores, color="#444444", linewidth=0.8, linestyle=":")
        # Said plainly on the chart, because a curve that flattens past the
        # core count is not the algorithm failing to scale.
        speed_axes.annotate(
            f"{cores} cores — beyond this\nthreads share cores",
            xy=(cores, 1.05),
            fontsize=8,
            color="#444444",
        )

    speed_axes.set_xlabel("threads")
    speed_axes.set_ylabel("speedup over one thread")
    speed_axes.set_title("Strong scaling")
    speed_axes.set_xticks(thread_counts)
    speed_axes.legend(frameon=False, fontsize=9)
    speed_axes.grid(alpha=0.25)

    time_axes.set_xlabel("threads")
    time_axes.set_ylabel("wall seconds")
    time_axes.set_title("Wall time")
    time_axes.set_xticks(thread_counts)
    time_axes.set_ylim(bottom=0)
    time_axes.legend(frameon=False, fontsize=9)
    time_axes.grid(alpha=0.25)

    figure.suptitle("k-means strong scaling", fontsize=13)
    figure.text(0.5, 0.005, caption, ha="center", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_weak(summary: dict, out_path: str, caption: str) -> None:
    figure, axes = plt.subplots(figsize=(6, 4.5))
    backends = sorted({backend for backend, _ in summary})
    thread_counts = sorted({threads for _, threads in summary})

    for backend in backends:
        points = [(t, summary[(backend, t)]) for t in thread_counts if (backend, t) in summary]
        threads = [t for t, _ in points]
        means = [entry["mean"] for _, entry in points]
        errors = [entry["stddev"] for _, entry in points]
        axes.errorbar(threads, means, yerr=errors, marker="o", capsize=3,
                      color=PALETTE.get(backend), label=backend)

    if thread_counts:
        flat = summary[(backends[0], thread_counts[0])]["mean"]
        axes.axhline(flat, linestyle="--", color=PALETTE["ideal"],
                     label="ideal (flat)")

    axes.set_xlabel("threads (points per thread held constant)")
    axes.set_ylabel("wall seconds")
    axes.set_title("Weak scaling")
    axes.set_xticks(thread_counts)
    axes.set_ylim(bottom=0)
    axes.legend(frameon=False, fontsize=9)
    axes.grid(alpha=0.25)

    figure.text(0.5, 0.005, caption, ha="center", fontsize=7, color="#555555")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def plot_clock_comparison(summary: dict, out_path: str, caption: str) -> None:
    """The original's measurement error, drawn.

    Elapsed time against the CPU time the original measured with clock().
    Wall falls with the thread count; CPU does not, because it is the sum
    over threads. A speedup computed from the second curve is about 1.00 at
    every thread count, which is what the committed program would actually
    have printed.
    """
    backends = sorted({backend for backend, _ in summary})
    backend = backends[0]
    thread_counts = sorted(t for b, t in summary if b == backend)
    if len(thread_counts) < 2:
        return

    wall = [summary[(backend, t)]["mean"] for t in thread_counts]
    cpu = [summary[(backend, t)]["cpu_mean"] for t in thread_counts]

    figure, axes = plt.subplots(figsize=(6.5, 4.5))
    axes.plot(thread_counts, [wall[0] / value for value in wall], marker="o",
              color=PALETTE["pthreads"], label="steady_clock (elapsed) — real speedup")
    axes.plot(thread_counts, [cpu[0] / value for value in cpu], marker="s",
              color=PALETTE["openmp"], label="clock() (CPU time) — what v1.0 measured")
    axes.axhline(1.0, linestyle=":", color="#888888", linewidth=0.8)

    axes.set_xlabel("threads")
    axes.set_ylabel("apparent speedup over one thread")
    axes.set_title("Why clock() cannot measure a speedup")
    axes.set_xticks(thread_counts)
    axes.legend(frameon=False, fontsize=9)
    axes.grid(alpha=0.25)

    figure.text(0.5, 0.005, caption, ha="center", fontsize=7, color="#555555")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strong", help="per-run CSV from --mode strong")
    parser.add_argument("--weak", help="per-run CSV from --mode weak")
    parser.add_argument("--out", default="graphs", help="output directory")
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)

    strong_rows = read_runs(arguments.strong)
    strong = aggregate(strong_rows)
    repeats = max(entry["repeats"] for entry in strong.values())
    caption = machine_caption(repeats)

    plot_strong(strong, os.path.join(arguments.out, "strong_scaling.png"), caption)
    plot_clock_comparison(strong, os.path.join(arguments.out, "clock_vs_wall.png"),
                          caption)

    if arguments.weak:
        weak = aggregate(read_runs(arguments.weak))
        weak_repeats = max(entry["repeats"] for entry in weak.values())
        plot_weak(weak, os.path.join(arguments.out, "weak_scaling.png"),
                  machine_caption(weak_repeats))

    return 0


if __name__ == "__main__":
    sys.exit(main())
