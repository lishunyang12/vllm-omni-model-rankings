"""Self-contained profile figure for the Qwen3-Omni DSpark **audio** draft head.

DSpark-style presentation of THIS training run — no cross-model comparison.
Shows: (top) the pipeline we built; (bottom) dataset distribution, training
convergence, and the final per-position acceptance profile.

Outputs vector PDF (LaTeX) + PNG. Serif (STIX), editable text embedded.
Run: python scripts/plot_dspark_audio_profile.py [--out-prefix PATH]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

# ── measured results (Qwen3-Omni-30B-A3B, LibriSpeech train-clean-100, 3 ep) ──
N_SAMPLES = 28539
POS_ACC = [0.9527, 0.9432, 0.9359, 0.9252, 0.9121, 0.8983, 0.8790]
EPOCHS = [1, 2, 3]
ACCEPT_LEN = [5.448, 5.927, 6.193]
VAL_LOSS = [0.254, 0.194, 0.178]
BLOCK = 7
ACCEPT_RATE, FULL_ACC = 0.907, 0.922

# dataset histogram fallback (used only if the raw array is unavailable)
PAD_BINS = [(0, 50, 1129), (50, 100, 2498), (100, 150, 3321),
            (150, 200, 15592), (200, 250, 5996), (250, 320, 3)]

C = "#3b6fb6"
C2 = "#7aa6d6"
INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#dddddd"
BOX = "#eaf0f8"


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.6, "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)


def _plabel(ax, t):
    ax.text(-0.02, 1.12, t, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def _pipeline(ax):
    ax.axis("off")
    steps = [
        "LibriSpeech\n28.5k utts",
        "Prompts\n(audio+instr)",
        "On-policy gen\n(target answers)",
        "Arrow\n(tokenize)",
        "Online train\n(extract-hidden\n+ DSpark)",
        "Audio draft\nhead",
    ]
    n = len(steps)
    xw, gap = 1.0 / n, 0.012
    y, h = 0.15, 0.7
    for i, s in enumerate(steps):
        x = i * xw
        fc = C if i == n - 1 else BOX
        tc = "white" if i == n - 1 else INK
        ax.add_patch(FancyBboxPatch((x + gap, y), xw - 2 * gap, h,
                     boxstyle="round,pad=0.006,rounding_size=0.02",
                     linewidth=0.8, edgecolor=C, facecolor=fc,
                     transform=ax.transAxes, clip_on=False))
        ax.text(x + xw / 2, y + h / 2, s, transform=ax.transAxes, ha="center",
                va="center", fontsize=7.4, color=tc, linespacing=1.25)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + xw - gap, y + h / 2), (x + xw + gap, y + h / 2),
                transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
                lw=1.0, color=MUTED, clip_on=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="dspark_audio_profile")
    ap.add_argument("--pad-npy", default="/home/zjy/wan_probe/ds_pad.npy")
    a = ap.parse_args()
    _rc()

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1], hspace=0.55, wspace=0.34,
                          left=0.08, right=0.97, top=0.9, bottom=0.11)

    axp = fig.add_subplot(gs[0, :])
    _pipeline(axp)
    axp.set_title("Pipeline: Qwen3-Omni DSpark audio draft head (this run)",
                  fontsize=10.5, fontweight="bold", color=INK, pad=6)

    # (a) dataset: audio token-length distribution
    axa = fig.add_subplot(gs[1, 0]); _style(axa)
    if os.path.exists(a.pad_npy):
        pad = np.load(a.pad_npy)
        axa.hist(pad, bins=40, color=C2, edgecolor=C, linewidth=0.4)
    else:
        centers = [(lo + hi) / 2 for lo, hi, _ in PAD_BINS]
        axa.bar(centers, [c for *_, c in PAD_BINS],
                width=45, color=C2, edgecolor=C, linewidth=0.4)
    axa.set_xlabel("Audio length (tokens)")
    axa.set_ylabel("Utterances")
    axa.set_title(f"Dataset: {N_SAMPLES:,} ASR utts", fontsize=9.5, color=INK, pad=4)
    axa.set_yticks([0, 1000, 2000, 3000, 4000])
    axa.set_yticklabels(["0", "1k", "2k", "3k", "4k"])
    _plabel(axa, "(a)")

    # (b) training convergence: accept_len per epoch
    axb = fig.add_subplot(gs[1, 1]); _style(axb)
    axb.yaxis.grid(True, color=GRID, linewidth=0.6); axb.set_axisbelow(True)
    axb.plot(EPOCHS, ACCEPT_LEN, color=C, lw=1.8, marker="o", ms=6,
             mfc=C, mec="white", mew=1.0)
    for x, y in zip(EPOCHS, ACCEPT_LEN):
        axb.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", color=C, fontsize=8)
    axb.set_ylim(5.0, 6.6)
    axb.set_xticks(EPOCHS)
    axb.set_xlabel("Epoch")
    axb.set_ylabel(r"Accepted length ($\leq 7$)")
    axb.set_title("Training convergence", fontsize=9.5, color=INK, pad=4)
    _plabel(axb, "(b)")

    # (c) final per-position acceptance
    axc = fig.add_subplot(gs[1, 2]); _style(axc)
    axc.yaxis.grid(True, color=GRID, linewidth=0.6); axc.set_axisbelow(True)
    axc.plot(range(BLOCK), POS_ACC, color=C, lw=1.8, marker="o", ms=5.5,
             mfc=C, mec="white", mew=0.9)
    axc.annotate(f"{POS_ACC[0]:.0%}", (0, POS_ACC[0]), textcoords="offset points",
                 xytext=(2, 7), ha="left", color=C, fontsize=8)
    axc.annotate(f"{POS_ACC[-1]:.0%}", (6, POS_ACC[-1]), textcoords="offset points",
                 xytext=(0, -14), ha="center", color=C, fontsize=8)
    axc.set_ylim(0.82, 1.0)
    axc.set_xticks(range(BLOCK))
    axc.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axc.set_xlabel("Draft position")
    axc.set_ylabel("Acceptance")
    axc.set_title(f"Per-position (accept_len {ACCEPT_LEN[-1]:.2f})",
                  fontsize=9.5, color=INK, pad=4)
    _plabel(axc, "(c)")

    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out_prefix}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved -> {a.out_prefix}.{ext}")


if __name__ == "__main__":
    main()
