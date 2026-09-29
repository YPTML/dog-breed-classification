"""Plotting utility for Outlier Exposure Stage B Training History.

Generates:
1. outputs/plots/oe_v2/stage_b_losses.png
2. outputs/plots/oe_v2/stage_b_id_accuracy.png
3. outputs/plots/oe_v2/stage_b_ood_diagnostics.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_stage_b_plots(
    history_json_path: Path,
    output_dir: Path,
) -> None:
    """Generate high-resolution PNG plots from Stage B training history JSON."""
    if not history_json_path.is_file():
        raise FileNotFoundError(f"History file not found: {history_json_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(history_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage_a_ref = data.get("stage_a_reference", {})
    epochs_data = data.get("epochs", [])
    if not epochs_data:
        print("[WARN] No epoch data found to plot.")
        return

    epochs = [e["epoch"] for e in epochs_data]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # -----------------------------------------------------------------------
    # Plot 1: Losses
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    train_total = [e["train_total_loss"] for e in epochs_data]
    train_id = [e["train_id_loss"] for e in epochs_data]
    train_oe = [e["train_oe_loss"] for e in epochs_data]
    val_id = [e["val_id_loss"] for e in epochs_data]

    ax.plot(epochs, train_total, marker="o", color="#2b5c8f", label="Train Total Loss ($L_{ID} + 0.5 L_{OE}$)", linewidth=2)
    ax.plot(epochs, train_id, marker="s", color="#3b9ab2", linestyle="--", label="Train ID Loss", linewidth=1.5)
    ax.plot(epochs, train_oe, marker="^", color="#e87d3e", linestyle=":", label="Train OE Loss", linewidth=1.5)
    ax.plot(epochs, val_id, marker="D", color="#b22222", label="Val ID Loss", linewidth=2)

    if "val_id_loss" in stage_a_ref:
        ax.axhline(y=stage_a_ref["val_id_loss"], color="#b22222", linestyle="--", alpha=0.7, label=f"Stage A Best Val ID Loss ({stage_a_ref['val_id_loss']:.4f})")

    ax.set_title("Stage B Training & Validation Losses", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_xticks(epochs)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    loss_plot_path = output_dir / "stage_b_losses.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {loss_plot_path}")

    # -----------------------------------------------------------------------
    # Plot 2: ID Accuracy (Top-1 & Top-5)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    train_acc = [e["train_id_accuracy"] * 100 for e in epochs_data]
    val_acc = [e["val_id_accuracy"] * 100 for e in epochs_data]
    val_top5 = [e["val_id_top5"] * 100 for e in epochs_data]

    ax.plot(epochs, train_acc, marker="o", color="#1f77b4", label="Train ID Top-1 (%)", linewidth=2)
    ax.plot(epochs, val_acc, marker="s", color="#2ca02c", label="Val ID Top-1 (%)", linewidth=2)
    ax.plot(epochs, val_top5, marker="^", color="#9467bd", label="Val ID Top-5 (%)", linewidth=2)

    if "val_id_accuracy" in stage_a_ref:
        sa_acc = stage_a_ref["val_id_accuracy"] * 100
        ax.axhline(y=sa_acc, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Stage A Best Val Top-1 ({sa_acc:.2f}%)")
        ax.axhline(y=sa_acc - 1.0, color="#d62728", linestyle=":", alpha=0.7, label=f"Safety Guard Threshold ({sa_acc - 1.0:.2f}%)")
    if "val_id_top5" in stage_a_ref:
        sa_top5 = stage_a_ref["val_id_top5"] * 100
        ax.axhline(y=sa_top5, color="#9467bd", linestyle="--", alpha=0.5, label=f"Stage A Best Val Top-5 ({sa_top5:.2f}%)")

    ax.set_title("Stage B Dog Breed Classification Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(epochs)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)

    acc_plot_path = output_dir / "stage_b_id_accuracy.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {acc_plot_path}")

    # -----------------------------------------------------------------------
    # Plot 3: OOD Calibration Diagnostics (MSP, Entropy, Energy)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Subplot 3a: Mean MSP
    ood_msp = [e["ood_cal_max_probability_mean"] for e in epochs_data]
    ax1.plot(epochs, ood_msp, marker="o", color="#d95f02", linewidth=2, label="Stage B OOD MSP")
    if "ood_cal_max_probability_mean" in stage_a_ref:
        ax1.axhline(y=stage_a_ref["ood_cal_max_probability_mean"], color="#7570b3", linestyle="--", label=f"Stage A Best ({stage_a_ref['ood_cal_max_probability_mean']:.4f})")
    ax1.set_title("OOD Mean Max Softmax Prob (MSP)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSP (lower is better)")
    ax1.set_xticks(epochs)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Subplot 3b: Mean Entropy
    ood_ent = [e["ood_cal_entropy_mean"] for e in epochs_data]
    ax2.plot(epochs, ood_ent, marker="s", color="#1b9e77", linewidth=2, label="Stage B OOD Entropy")
    if "ood_cal_entropy_mean" in stage_a_ref:
        ax2.axhline(y=stage_a_ref["ood_cal_entropy_mean"], color="#7570b3", linestyle="--", label=f"Stage A Best ({stage_a_ref['ood_cal_entropy_mean']:.4f})")
    ax2.set_title("OOD Mean Entropy (nats)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Entropy (higher is better)")
    ax2.set_xticks(epochs)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Subplot 3c: Mean Energy
    ood_nrg = [e["ood_cal_energy_mean"] for e in epochs_data]
    ax3.plot(epochs, ood_nrg, marker="^", color="#7570b3", linewidth=2, label="Stage B OOD Energy")
    if "ood_cal_energy_mean" in stage_a_ref:
        ax3.axhline(y=stage_a_ref["ood_cal_energy_mean"], color="#d95f02", linestyle="--", label=f"Stage A Best ({stage_a_ref['ood_cal_energy_mean']:.4f})")
    ax3.set_title("OOD Mean Free Energy", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Energy (lower is better for OOD)")
    ax3.set_xticks(epochs)
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Stage B OOD Calibration Diagnostics on OOD OE V2 Calibration Split", fontsize=14, fontweight="bold", y=1.02)
    diag_plot_path = output_dir / "stage_b_ood_diagnostics.png"
    plt.savefig(diag_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {diag_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for Stage B OE Training")
    parser.add_argument("--history-json", type=Path, required=True, help="Path to stage_b_history.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save plot PNGs")
    args = parser.parse_args()

    generate_stage_b_plots(args.history_json, args.output_dir)


if __name__ == "__main__":
    main()
