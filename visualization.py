"""
Visualization script for RPG_KDD2025 experiment results.
This script parses log files and creates a comprehensive visualization of results across all datasets.
"""

import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Paper-friendly plotting defaults.
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.figsize": (16, 10),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#D8DEE9",
        "grid.color": "#E6EBF2",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PLOT_COLORS = {
    "loss": "#2A6FBB",
    "loss_raw": "#9AA9BA",
    "ndcg@5": "#0F766E",
    "ndcg@10": "#2563EB",
    "recall@5": "#B7791F",
    "recall@10": "#C2410C",
    "best": "#B91C1C",
    "text": "#172033",
    "muted": "#68758A",
}

ORDERED_METRICS = ("ndcg@5", "ndcg@10", "recall@5", "recall@10")
HYPERPARAM_KEYS = {
    "lr",
    "temperature",
    "diff_temperature",
    "n_codebook",
    "num_beams",
    "n_edges",
    "propagation_steps",
    "num_sampling_steps",
    "rectified_flow_steps",
    "sent_emb_pca",
    "tiger_guidance_weight",
    "use_rectified_flow",
}
METRIC_PATTERN = re.compile(r"'([^']+)':\s*([-+0-9.eE]+)")


def pretty_name(value):
    """Return a label suitable for figures and output directories."""
    if not value:
        return "Unknown"
    return value.replace("_and_", " & ").replace("_", " ")


def safe_path_part(value):
    """Convert a display label into a compact filesystem-safe path component."""
    value = value or "Unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "Unknown"


def extract_metric_values(metrics_str):
    """Extract OrderedDict-style metric values from a log line."""
    metrics = {}
    for metric, value in METRIC_PATTERN.findall(metrics_str):
        try:
            metrics[metric] = float(value)
        except ValueError:
            continue
    return metrics


def format_metric(value):
    if value is None:
        return "-"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def parse_log_file(log_path):
    """
    Parse a log file to extract training information and results.

    Returns:
        dict: Dictionary containing parsed information
    """
    results = {
        "dataset": None,
        "model": None,
        "run_id": None,
        "run_time": None,
        "hyperparams": {},
        "train_losses": [],
        "val_metrics": defaultdict(list),
        "val_epochs": [],
        "best_epoch": None,
        "best_val_score": None,
        "test_results": {},
        "epochs": [],
    }

    # Extract dataset name from filename
    filename = os.path.basename(log_path)
    model_match = re.search(r"--model=([^_]+)", filename)
    if model_match:
        results["model"] = model_match.group(1)

    category_match = re.search(r"--category=([^_]+(?:_and_[^_]+)*)", filename)
    if category_match:
        results["dataset"] = pretty_name(category_match.group(1))

    # Extract hyperparameters from filename
    hyperparam_patterns = {
        "lr": r"--lr=([\d.]+)",
        "temperature": r"--temperature=([\d.]+)",
        "n_codebook": r"--n_codebook=(\d+)",
        "num_beams": r"--num_beams=(\d+)",
        "n_edges": r"--n_edges=(\d+)",
        "propagation_steps": r"--propagation_steps=(\d+)",
    }

    for param, pattern in hyperparam_patterns.items():
        match = re.search(pattern, filename)
        if match:
            results["hyperparams"][param] = match.group(1)

    # Parse log content
    try:
        with open(log_path, "r") as f:
            for line in f:
                # Parse scalar metadata from the hyperparameter block. This
                # supports compact filenames such as logs/.../paper/toy.log.
                scalar_match = re.match(
                    r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$", line
                )
                if scalar_match:
                    key, value = scalar_match.groups()
                    value = value.strip().strip("'\"")
                    if key == "category" and value:
                        results["dataset"] = pretty_name(value)
                    elif key == "model" and value:
                        results["model"] = value
                    elif key == "run_id" and value:
                        results["run_id"] = value
                    elif key == "run_local_time" and value:
                        results["run_time"] = value

                    if key in HYPERPARAM_KEYS and value:
                        results["hyperparams"][key] = value

                # Parse training loss
                train_loss_match = re.search(
                    r"\[Epoch (\d+)\] Train Loss: ([-+0-9.eE]+)", line
                )
                if train_loss_match:
                    epoch = int(train_loss_match.group(1))
                    loss = float(train_loss_match.group(2))
                    results["train_losses"].append(loss)
                    results["epochs"].append(epoch)

                # Parse validation results
                val_match = re.search(
                    r"\[Epoch (\d+)\] Val Results: OrderedDict\(({.*})\)", line
                )
                if val_match:
                    results["val_epochs"].append(int(val_match.group(1)))
                    metrics = extract_metric_values(val_match.group(2))
                    for metric in ORDERED_METRICS:
                        if metric in metrics:
                            results["val_metrics"][metric].append(metrics[metric])

                # Parse best epoch info
                best_epoch_match = re.search(
                    r"Best epoch: (\d+), Best val score: ([-+0-9.eE]+)", line
                )
                if best_epoch_match:
                    results["best_epoch"] = int(best_epoch_match.group(1))
                    results["best_val_score"] = float(best_epoch_match.group(2))

                # Parse test results
                test_match = re.search(r"Test Results: OrderedDict\(({.*})\)", line)
                if test_match:
                    results["test_results"].update(
                        extract_metric_values(test_match.group(1))
                    )

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return results


def metric_epochs(results, metric):
    values = results["val_metrics"].get(metric, [])
    if len(results["val_epochs"]) == len(values):
        return results["val_epochs"]
    return list(range(1, len(values) + 1))


def exponential_moving_average(values, alpha=0.18):
    if not values:
        return []

    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def setup_axis(ax, xlabel=None, ylabel=None, title=None, integer_x=True):
    ax.grid(True, axis="y", alpha=0.95)
    ax.grid(True, axis="x", alpha=0.18)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(colors=PLOT_COLORS["muted"], labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=PLOT_COLORS["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PLOT_COLORS["muted"])
    if title:
        ax.set_title(title, loc="left", fontweight="bold", color=PLOT_COLORS["text"])


def add_line_end_label(ax, xs, ys, label, color, y_offset=0):
    """Put endpoint labels inside the axis to avoid spilling into neighbor subplots."""
    if not xs or not ys:
        return

    ax.scatter(xs[-1], ys[-1], s=22, color=color, zorder=4, clip_on=True)

    ann = ax.annotate(
        label,
        xy=(xs[-1], ys[-1]),
        xytext=(-8, y_offset),  # draw label to the left of the endpoint
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
        clip_on=True,
        annotation_clip=True,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
        ),
    )
    ann.set_clip_on(True)


def add_best_epoch_marker(ax, results, metric="ndcg@10"):
    values = results["val_metrics"].get(metric, [])
    epochs = metric_epochs(results, metric)
    if not values or not epochs:
        return

    if results["best_epoch"] in epochs:
        best_epoch = results["best_epoch"]
        best_value = values[epochs.index(best_epoch)]
    else:
        best_index = int(np.argmax(values))
        best_epoch = epochs[best_index]
        best_value = values[best_index]

    ax.axvline(best_epoch, color=PLOT_COLORS["best"], linewidth=1.2, alpha=0.35)
    ax.scatter(best_epoch, best_value, s=42, color=PLOT_COLORS["best"], zorder=5)

    # If best epoch is near the right side, put annotation on the left.
    x_min, x_max = min(epochs), max(epochs)
    near_right = best_epoch > x_min + 0.68 * (x_max - x_min)
    x_offset = -12 if near_right else 12
    ha = "right" if near_right else "left"

    ann = ax.annotate(
        f"best {format_metric(best_value)}\nepoch {best_epoch}",
        xy=(best_epoch, best_value),
        xytext=(x_offset, 18),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=8,
        color=PLOT_COLORS["best"],
        clip_on=True,
        annotation_clip=True,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#F1C6C6",
            alpha=0.92,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=PLOT_COLORS["best"],
            alpha=0.45,
            linewidth=0.8,
        ),
    )
    ann.set_clip_on(True)

def plot_validation_lines(ax, results, metrics):
    plotted = []
    metrics = tuple(metrics)

    for idx, metric in enumerate(metrics):
        values = results["val_metrics"].get(metric, [])
        epochs = metric_epochs(results, metric)
        if not values or not epochs:
            continue

        color = PLOT_COLORS[metric]
        ax.plot(
            epochs,
            values,
            label=metric.upper(),
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
        )

        # Separate two endpoint labels vertically.
        y_offset = -9 if idx == 0 else 9
        add_line_end_label(ax, epochs, values, metric.upper(), color, y_offset=y_offset)
        plotted.extend(values)

    if plotted:
        ymax = max(plotted)
        ax.set_ylim(0, ymax * 1.24 if ymax > 0 else 1)
        ax.margins(x=0.025)

    ax.legend(
        loc="upper left",
        fontsize=8.5,
        ncol=min(len(metrics), 2),
        borderaxespad=0.35,
        handlelength=1.8,
    )

def save_figure(fig, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = os.path.join(output_dir, "results_visualization.pdf")
    output_png = os.path.join(output_dir, "results_visualization.png")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    print(f"  Saved PDF: {output_pdf}")
    print(f"  Saved PNG: {output_png}")


def plot_single_result(results, output_dir, model=None):
    """
    Create a visualization for a single dataset.
    """
    model_name = model or results.get("model") or "UnknownModel"
    dataset = results.get("dataset") or "Unknown dataset"

    fig = plt.figure(figsize=(14.8, 8.4))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.08, 1.08, 0.88),
        height_ratios=(1.0, 1.0),
        wspace=0.42,
        hspace=0.38,
    )

    ax_loss = fig.add_subplot(grid[0, 0])
    ax_ndcg = fig.add_subplot(grid[0, 1])
    ax_recall = fig.add_subplot(grid[1, 0])
    ax_test = fig.add_subplot(grid[1, 1])
    ax_summary = fig.add_subplot(grid[:, 2])

    if results["train_losses"] and results["epochs"]:
        smoothed_loss = exponential_moving_average(results["train_losses"])
        ax_loss.plot(
            results["epochs"],
            results["train_losses"],
            color=PLOT_COLORS["loss_raw"],
            linewidth=1.0,
            alpha=0.35,
            label="raw",
        )
        ax_loss.plot(
            results["epochs"],
            smoothed_loss,
            color=PLOT_COLORS["loss"],
            linewidth=2.6,
            label="EMA",
        )
        add_line_end_label(
            ax_loss,
            results["epochs"],
            smoothed_loss,
            format_metric(smoothed_loss[-1]),
            PLOT_COLORS["loss"],
        )
        ax_loss.legend(loc="upper right", fontsize=8.5)
    setup_axis(ax_loss, xlabel="Epoch", ylabel="Loss", title="Training Loss")

    plot_validation_lines(ax_ndcg, results, ("ndcg@5", "ndcg@10"))
    add_best_epoch_marker(ax_ndcg, results, metric="ndcg@10")
    setup_axis(ax_ndcg, xlabel="Epoch", ylabel="NDCG", title="Validation NDCG")

    plot_validation_lines(ax_recall, results, ("recall@5", "recall@10"))
    setup_axis(ax_recall, xlabel="Epoch", ylabel="Recall", title="Validation Recall")

    test_metrics = [m for m in ORDERED_METRICS if m in results["test_results"]]
    if test_metrics:
        values = [results["test_results"][metric] for metric in test_metrics]
        colors = [PLOT_COLORS[metric] for metric in test_metrics]
        y_pos = np.arange(len(test_metrics))
        bars = ax_test.barh(y_pos, values, color=colors, height=0.58, alpha=0.88)
        ax_test.set_yticks(y_pos)
        ax_test.set_yticklabels([m.upper() for m in test_metrics])
        ax_test.invert_yaxis()
        # ax_test.set_xlim(0, max(values) * 1.22)
        xmax = max(values) if max(values) > 0 else 1.0
        ax_test.set_xlim(0, xmax * 1.34)
        ax_test.tick_params(axis="y", pad=8)
        for bar, value in zip(bars, values):
            ax_test.text(
                # bar.get_width() + max(values) * 0.025,
                bar.get_width() + xmax * 0.025,
                bar.get_y() + bar.get_height() / 2,
                format_metric(value),
                va="center",
                fontsize=8.5,
                color=PLOT_COLORS["text"],
                fontweight="bold",
            )
    setup_axis(
        ax_test,
        xlabel="Score",
        title="Held-out Test Metrics",
        integer_x=False,
    )

    ax_summary.set_facecolor("#F8FAFC")
    for spine in ax_summary.spines.values():
        spine.set_visible(True)
        spine.set_color("#DFE7F1")
    ax_summary.set_xticks([])
    ax_summary.set_yticks([])

    subtitle_parts = ["AmazonReviews2014"]
    if results.get("run_id"):
        subtitle_parts.append(f"run_id={results['run_id']}")
    if results.get("run_time"):
        subtitle_parts.append(results["run_time"])

    fig.text(
        0.055,
        0.982,
        f"{model_name} on {dataset}",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=PLOT_COLORS["text"],
    )
    fig.text(
        0.055,
        0.948,
        " | ".join(subtitle_parts),
        ha="left",
        va="top",
        fontsize=9.5,
        color=PLOT_COLORS["muted"],
    )

    summary_items = [
        ("Best val NDCG@10", format_metric(results.get("best_val_score"))),
        ("Best epoch", str(results["best_epoch"]) if results["best_epoch"] else "-"),
        (
            "Test NDCG@10",
            format_metric(results["test_results"].get("ndcg@10")),
        ),
        (
            "Test Recall@10",
            format_metric(results["test_results"].get("recall@10")),
        ),
    ]

    ax_summary.text(
        0.08,
        0.94,
        "Run Summary",
        transform=ax_summary.transAxes,
        fontsize=13,
        fontweight="bold",
        color=PLOT_COLORS["text"],
        va="top",
    )
    y = 0.83
    for label, value in summary_items:
        ax_summary.text(
            0.08,
            y,
            label,
            transform=ax_summary.transAxes,
            fontsize=8.5,
            color=PLOT_COLORS["muted"],
        )
        ax_summary.text(
            0.08,
            y - 0.043,
            value,
            transform=ax_summary.transAxes,
            fontsize=15,
            fontweight="bold",
            color=PLOT_COLORS["text"],
        )
        y -= 0.135

    ax_summary.text(
        0.08,
        y - 0.005,
        "Key Config",
        transform=ax_summary.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=PLOT_COLORS["text"],
    )

    config_lines = []
    for key in (
        "lr",
        "temperature",
        "num_sampling_steps",
        "rectified_flow_steps",
        "sent_emb_pca",
        "tiger_guidance_weight",
        "n_codebook",
        "num_beams",
        "n_edges",
        "propagation_steps",
    ):
        if key in results["hyperparams"]:
            config_lines.append(f"{key}: {results['hyperparams'][key]}")

    ax_summary.text(
        0.08,
        y - 0.065,
        "\n".join(config_lines) if config_lines else "No config metadata found.",
        transform=ax_summary.transAxes,
        fontsize=8.0,
        color=PLOT_COLORS["text"],
        family="monospace",
        va="top",
        linespacing=1.3,
    )

    fig.subplots_adjust(top=0.88, left=0.06, right=0.985, bottom=0.08)
    save_figure(fig, output_dir)
    plt.close(fig)


def plot_all_results(log_paths, output_dir, model=None):
    """
    Create a comprehensive visualization of all experiment results.
    """
    log_files = []
    for path in log_paths:
        if os.path.isfile(path):
            log_files.append(path)
        elif os.path.isdir(path):
            log_files.extend(glob.glob(os.path.join(path, "*.log")))

    if not log_files:
        print("No log files found in the provided paths")
        return

    print(f"Found {len(log_files)} log files")

    # Parse all log files
    all_results = []
    for log_file in sorted(log_files):
        print(f"Parsing {os.path.basename(log_file)}...")
        results = parse_log_file(log_file)
        if results["dataset"]:
            all_results.append(results)

    if not all_results:
        print("No valid results found")
        return

    model_name = model or all_results[0].get("model") or "UnknownModel"
    fig = plt.figure(figsize=(13.8, 8.6))
    grid = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.38)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_ndcg = fig.add_subplot(grid[0, 1])
    ax_heatmap = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    dataset_palette = sns.color_palette("colorblind", n_colors=len(all_results))

    for color, results in zip(dataset_palette, all_results):
        label = results["dataset"]
        if results["train_losses"] and results["epochs"]:
            ax_loss.plot(
                results["epochs"],
                exponential_moving_average(results["train_losses"]),
                label=label,
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
            )
        if results["val_metrics"].get("ndcg@10"):
            ax_ndcg.plot(
                metric_epochs(results, "ndcg@10"),
                results["val_metrics"]["ndcg@10"],
                label=label,
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
            )

    setup_axis(ax_loss, xlabel="Epoch", ylabel="EMA Loss", title="Training Loss")
    setup_axis(ax_ndcg, xlabel="Epoch", ylabel="NDCG@10", title="Validation NDCG@10")
    ax_loss.legend(loc="upper right", fontsize=8.5)
    ax_ndcg.legend(loc="lower right", fontsize=8.5)

    datasets_with_test = [r for r in all_results if r["test_results"]]
    if datasets_with_test:
        heatmap_data = np.array(
            [
                [r["test_results"].get(metric, np.nan) for metric in ORDERED_METRICS]
                for r in datasets_with_test
            ]
        )
        sns.heatmap(
            heatmap_data,
            ax=ax_heatmap,
            annot=True,
            fmt=".4f",
            cmap=sns.light_palette("#2563EB", as_cmap=True),
            cbar=False,
            linewidths=1,
            linecolor="white",
            xticklabels=[m.upper() for m in ORDERED_METRICS],
            yticklabels=[r["dataset"] for r in datasets_with_test],
            annot_kws={"fontsize": 8.5, "fontweight": "bold"},
        )
        ax_heatmap.set_title(
            "Held-out Test Metrics",
            loc="left",
            fontweight="bold",
            color=PLOT_COLORS["text"],
        )
        ax_heatmap.tick_params(axis="x", rotation=0, colors=PLOT_COLORS["muted"])
        ax_heatmap.tick_params(axis="y", rotation=0, colors=PLOT_COLORS["muted"])
    else:
        ax_heatmap.axis("off")
        ax_heatmap.text(
            0.5,
            0.5,
            "No test metrics found",
            ha="center",
            va="center",
            color=PLOT_COLORS["muted"],
        )

    ax_summary.axis("off")
    ax_summary.set_facecolor("#F8FAFC")
    for spine in ax_summary.spines.values():
        spine.set_visible(True)
        spine.set_color("#DFE7F1")

    ax_summary.text(
        0.02,
        0.98,
        "Best Validation Runs",
        transform=ax_summary.transAxes,
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PLOT_COLORS["text"],
    )
    rows = []
    for results in all_results:
        rows.append(
            (
                results["dataset"],
                str(results["best_epoch"]) if results["best_epoch"] else "-",
                format_metric(results.get("best_val_score")),
                format_metric(results["test_results"].get("ndcg@10")),
            )
        )

    header = f"{'Dataset':<18} {'Epoch':>6} {'Best':>9} {'Test@10':>9}"
    body = [header, "-" * len(header)]
    body.extend(
        f"{dataset[:18]:<18} {epoch:>6} {best:>9} {test:>9}"
        for dataset, epoch, best, test in rows
    )
    ax_summary.text(
        0.02,
        0.84,
        "\n".join(body),
        transform=ax_summary.transAxes,
        va="top",
        fontsize=9,
        color=PLOT_COLORS["text"],
        family="monospace",
        linespacing=1.55,
    )

    fig.text(
        0.055,
        0.982,
        f"{model_name} Performance Overview",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=PLOT_COLORS["text"],
    )
    fig.text(
        0.055,
        0.948,
        f"{len(all_results)} run(s) from AmazonReviews2014 logs",
        ha="left",
        va="top",
        fontsize=9.5,
        color=PLOT_COLORS["muted"],
    )

    fig.subplots_adjust(top=0.88, left=0.06, right=0.985, bottom=0.08)
    save_figure(fig, output_dir)
    plt.close(fig)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for results in all_results:
        print(f"\n{results['dataset']}:")
        print(f"  Total epochs: {len(results['train_losses'])}")
        if results["best_epoch"]:
            print(f"  Best epoch: {results['best_epoch']}")
            print(f"  Best validation score: {results['best_val_score']:.6f}")
        if results["test_results"]:
            print("  Test Results:")
            for metric, value in results["test_results"].items():
                print(f"    {metric}: {value:.6f}")


def extract_metadata_from_filename(filepath):
    """Extract model, category, and timestamp from a log filename."""
    filename = os.path.basename(filepath)
    model_m = re.search(r"--model=([^_]+)", filename)
    cat_m = re.search(r"--category=([^_]+(?:_and_[^_]+)*)", filename)
    time_m = re.search(
        r"-([A-Z][a-z]{2}-\d{2}-\d{4}_\d{2}-\d{2}(?:-[a-z0-9]+)?)\.log", filename
    )
    model = model_m.group(1) if model_m else "UnknownModel"
    category = pretty_name(cat_m.group(1)) if cat_m else "UnknownCategory"
    timestamp = time_m.group(1) if time_m else "UnknownTime"
    return model, category, timestamp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize RPG experiment results")
    parser.add_argument(
        "--log_paths",
        nargs="+",
        default=[
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=Toys_and_Games_--lr=0.003_--temperature=0.03-May-05-2026_18-52-833a9d.log",
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=Sports_and_Outdoors_--lr=0.003_--temperature=0.03-May-05-2026_18-53-91572e.log",
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=CDs_and_Vinyl_--lr=0.001_--temperature=0.03-May-05-2026_18-53-2eeead.log",
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=Beauty_--lr=0.01_--temperature=0.03_--n_codebook=32_--num_beams=20_--n_edges=200_--propagation_steps=3-May-05-2026_18-31-d5929c.log",
            # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=Beauty_--lr=0.01_--temperature=0.03_--n_codebook=32_--num_beams=20_--n_edges=200_--propagation_steps=3-Apr-27-2026_20-30-ba460a.log"
            # # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=CDs_and_Vinyl_--lr=0.001_--temperature=0.03_--n_codebook=64_--num_beams=20_--n_edges=500_--propagation_steps=5-May-06-2026_01-21-70824e.log",
            # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=Sports_and_Outdoors_--lr=0.003_--temperature=0.03_--n_codebook=16_--num_beams=100_--n_edges=30_--propagation_steps=5-May-06-2026_01-21-52057a.log",
            # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=Toys_and_Games_--lr=0.003_--temperature=0.03_--n_codebook=16_--num_beams=200_--n_edges=20_--propagation_steps=3-May-06-2026_01-22-2b595c.log",
            # "logs/AmazonReviews2014/RPGDiff"
            "logs/AmazonReviews2014/paper"
        ],
        help="List of full paths to the log files (or directories)",
    )

    args = parser.parse_args()

    # Resolve log files (files or directories)
    log_files = []
    for path in args.log_paths:
        if os.path.isfile(path):
            log_files.append(path)
        elif os.path.isdir(path):
            log_files.extend(glob.glob(os.path.join(path, "*.log")))

    if not log_files:
        print("No log files found in the provided paths")
        exit(1)

    print(f"Found {len(log_files)} log files")

    # Step 1: Generate individual per-dataset PDFs/PNGs
    all_results = []
    model = "UnknownModel"
    first_ts = "UnknownTime"
    for log_file in sorted(log_files):
        print(f"Processing {os.path.basename(log_file)}...")
        results = parse_log_file(log_file)
        if results["dataset"]:
            all_results.append(results)
            file_model, file_category, file_ts = extract_metadata_from_filename(
                log_file
            )
            mdl = results.get("model") or file_model
            cat = results.get("dataset") or file_category
            ts = results.get("run_time") or file_ts

            if model == "UnknownModel":
                model = mdl
            if first_ts == "UnknownTime":
                first_ts = ts

            individual_output_dir = os.path.join(
                "vis_results",
                safe_path_part(mdl),
                safe_path_part(cat),
                safe_path_part(ts),
            )
            plot_single_result(results, individual_output_dir, model=mdl)

    if not all_results:
        print("No valid results found")
        exit(1)

    # Step 2: Generate combined comparison PDF/PNG
    combined_output_dir = os.path.join(
        "vis_results", safe_path_part(model), f"combined_{safe_path_part(first_ts)}"
    )
    plot_all_results(log_files, combined_output_dir, model)
