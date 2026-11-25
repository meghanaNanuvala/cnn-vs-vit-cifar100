"""Compares the two trained models.

Reads results/cnn.json and results/vit.json, prints the comparison table and
writes curves.png and tsne.png.

    python compare.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"cnn": "#2a78d6", "vit": "#eb6834"}
MARKERS = {"cnn": "o", "vit": "s"}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

ROWS = [
    ("Test top-1 accuracy (%)", "top1", ".2f"),
    ("Test top-5 accuracy (%)", "top5", ".2f"),
    ("Training top-1 accuracy (%)", "train_top1", ".2f"),
    ("k-NN accuracy on features (%)", "knn_top1", ".2f"),
    ("Parameters (M)", "params_M", ".2f"),
    ("FLOPs per image (G)", "flops_G", ".2f"),
    ("Inference speed (images/s)", "inference_img_per_s", ",.0f"),
    ("Training time per epoch (s)", "train_s_per_epoch", ".1f"),
    ("Peak GPU memory in training (GB)", "peak_train_mem_GB", ".2f"),
]


def print_table(results):
    cnn, vit = results["cnn"], results["vit"]
    print(f"{'':<34}{cnn['name']:>12}{vit['name']:>12}")
    for label, key, fmt in ROWS:
        print(f"{label:<34}{format(cnn[key], fmt):>12}{format(vit[key], fmt):>12}")
    for r in results.values():
        print(f"\nHardest classes for {r['name']}:")
        for h in r["hardest_classes"]:
            print(f"  {h['class']:<14}{h['accuracy']:5.0f}%   mostly predicted as {h['most_confused_with']}")


def plot_curves(results, path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for key, r in results.items():
        epochs = [h["epoch"] for h in r["history"]]
        every = max(1, len(epochs) // 10)
        style = dict(color=COLORS[key], marker=MARKERS[key], markevery=every, linewidth=2)
        ax_loss.plot(epochs, [h["train_loss"] for h in r["history"]], label=r["name"], **style)
        ax_acc.plot(epochs, [h["test_top1"] for h in r["history"]], label=f"{r['name']} test", **style)
        ax_acc.plot(epochs, [h["train_top1"] for h in r["history"]], label=f"{r['name']} train",
                    color=COLORS[key], linewidth=1.2, linestyle="--")
    ax_loss.set(xlabel="epoch", ylabel="training loss")
    ax_acc.set(xlabel="epoch", ylabel="top-1 accuracy (%)")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tsne(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, r in zip(axes, results.values()):
        points, labels = np.array(r["tsne"]["points"]), np.array(r["tsne"]["labels"])
        for c, class_name in enumerate(r["tsne"]["classes"]):
            xy = points[labels == c]
            ax.scatter(xy[:, 0], xy[:, 1], s=14, color=CLASS_COLORS[c], label=class_name)
        ax.set_title(r["name"])
        ax.set_xticks([])
        ax.set_yticks([])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, markerscale=1.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    out = Path(parser.parse_args().out)

    for key in ("cnn", "vit"):
        if not (out / f"{key}.json").exists():
            raise SystemExit(f"{out}/{key}.json not found. Run {key}.py first.")
    results = {key: json.load(open(out / f"{key}.json")) for key in ("cnn", "vit")}
    print_table(results)
    plot_curves(results, out / "curves.png")
    plot_tsne(results, out / "tsne.png")
