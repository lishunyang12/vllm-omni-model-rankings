"""Standalone figures for the Qwen3-Omni DSpark **audio** draft head.

One figure per file (DSpark-style: accepted length tau is the headline metric),
single-column publication size. Emits, each as PDF (vector) + PNG:
  dspark_audio_convergence.*   accepted length tau over training
  dspark_audio_perposition.*   per-position acceptance across the block
  dspark_audio_pipeline.*      the training pipeline

Data: 28,539 LibriSpeech train-clean-100 utts, 3 epochs, block_size=7, aux
[2,13,24,35,46], vanilla Markov (rank 256). Per-step tau from
dspark_audio_train_curve.json; final metrics from checkpoint_best/val_metrics.json.

Run: python scripts/plot_dspark_audio_profile.py [--outdir .]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

POS_ACC = [0.9527, 0.9432, 0.9359, 0.9252, 0.9121, 0.8983, 0.8790]
BLOCK = 7
ACCEPT_LEN, ACCEPT_RATE = 6.19, 0.907

C = "#3b6fb6"; C2 = "#9db8dc"; INK = "#1a1a1a"; MUTED = "#555555"
GRID = "#dddddd"; BOX = "#eaf0f8"


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.6, "font.size": 9,
        "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"saved -> {p}")
    plt.close(fig)


def _smooth(y, w=31):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")


def fig_convergence(curve, outdir):
    al = np.array(curve["accept_len"]); val = curve["val_epoch_accept_len"]
    spe = curve["steps_per_epoch"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6)); _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    x = np.arange(len(al))
    ax.plot(x, al, color=C2, lw=0.5, alpha=0.7)
    ax.plot(x, _smooth(al), color=C, lw=1.6, label="train")
    for e, v in enumerate(val, start=1):
        ax.plot(e * spe - 1, v, marker="o", ms=5, mfc="white", mec=C, mew=1.3,
                zorder=5, label="val" if e == 1 else None)
    ax.annotate(rf"val $\tau={val[-1]:.2f}$", (len(al) - 1, val[-1]),
                textcoords="offset points", xytext=(-14, -20), ha="right",
                color=C, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C, lw=0.6))
    ax.set_ylim(0.8, 8.1); ax.set_xlabel("Training step")
    ax.set_ylabel(r"Accepted length $\tau$ ($\leq 8$)")
    ax.set_title("Validation (teacher-forced)", fontsize=9, color=MUTED, pad=3)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    _save(fig, outdir, "dspark_audio_convergence")


def fig_perposition(outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6)); _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.plot(range(BLOCK), POS_ACC, color=C, lw=1.8, marker="o", ms=5.5,
            mfc=C, mec="white", mew=0.9)
    ax.annotate(f"{POS_ACC[0]:.1%}", (0, POS_ACC[0]), textcoords="offset points",
                xytext=(3, 7), ha="left", color=C, fontsize=8.5)
    ax.annotate(f"{POS_ACC[-1]:.1%}", (6, POS_ACC[-1]), textcoords="offset points",
                xytext=(0, -14), ha="center", color=C, fontsize=8.5)
    ax.set_ylim(0.82, 1.0); ax.set_xticks(range(BLOCK))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Draft position in block")
    ax.set_ylabel("Top-1 agreement")
    ax.set_title("Validation (teacher-forced)", fontsize=9, color=MUTED, pad=3)
    fig.tight_layout()
    _save(fig, outdir, "dspark_audio_perposition")


def fig_dataset(outdir, dist_path):
    d = json.load(open(dist_path))
    al = np.array(d["audio_len"]); an = np.array(d["answer_len"]); n = d["n"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    fig.suptitle(f"Dataset: {n:,} LibriSpeech train-clean-100 utterances (ASR)",
                 fontsize=10, fontweight="bold", color=INK, y=1.02)
    for ax, arr, xlab, title, med in [
        (axa, al, "Audio length (tokens)", "Input audio", int(np.median(al))),
        (axb, an, "Answer length (tokens)", "Output transcript", int(np.median(an))),
    ]:
        _style(ax)
        ax.hist(arr, bins=40, color=C2, edgecolor=C, linewidth=0.4)
        ax.axvline(med, color=C, lw=1.2, ls="--")
        ax.annotate(f"median {med}", (med, ax.get_ylim()[1] * 0.9),
                    xytext=(5, 0), textcoords="offset points", color=C, fontsize=8)
        ax.set_xlabel(xlab); ax.set_ylabel("Utterances")
        ax.set_title(title, fontsize=9.5, color=INK, pad=3)
        ax.yaxis.set_major_formatter(
            lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    fig.tight_layout()
    _save(fig, outdir, "dspark_audio_dataset")


def fig_pipeline(outdir):
    fig, ax = plt.subplots(figsize=(7.2, 1.5)); ax.axis("off")
    steps = ["LibriSpeech\n28.5k utts", "Prompts\n(audio+instr)",
             "On-policy gen\n(target answers)", "Arrow\n(tokenize)",
             "Online train\n(extract-hidden\n+ DSpark)", "Audio draft\nhead"]
    n = len(steps); xw, gap = 1.0 / n, 0.012; y, h = 0.12, 0.76
    for i, s in enumerate(steps):
        x = i * xw
        fc = C if i == n - 1 else BOX; tc = "white" if i == n - 1 else INK
        ax.add_patch(FancyBboxPatch((x + gap, y), xw - 2 * gap, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=0.8, edgecolor=C, facecolor=fc,
                     transform=ax.transAxes, clip_on=False))
        ax.text(x + xw / 2, y + h / 2, s, transform=ax.transAxes, ha="center",
                va="center", fontsize=7.6, color=tc, linespacing=1.25)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + xw - gap, y + h / 2), (x + xw + gap, y + h / 2),
                         transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
                         lw=1.0, color=MUTED, clip_on=False))
    _save(fig, outdir, "dspark_audio_pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(__file__) or ".")
    ap.add_argument("--curve", default=os.path.join(os.path.dirname(__file__),
                    "dspark_audio_train_curve.json"))
    a = ap.parse_args()
    _rc()
    curve = json.load(open(a.curve))
    fig_convergence(curve, a.outdir)
    fig_perposition(a.outdir)
    fig_pipeline(a.outdir)
    dist = os.path.join(os.path.dirname(__file__), "dspark_audio_dataset_dist.json")
    if os.path.exists(dist):
        fig_dataset(a.outdir, dist)


if __name__ == "__main__":
    main()
