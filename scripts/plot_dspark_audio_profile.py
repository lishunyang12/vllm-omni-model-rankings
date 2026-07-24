"""Self-contained profile figure for the Qwen3-Omni DSpark **audio** draft head.

DSpark-style presentation of THIS training run (no cross-model comparison):
(top) the pipeline; (bottom) fine-grained training convergence, final per-position
acceptance, and the headline speculative-decoding metrics.

Data: 28,539 LibriSpeech train-clean-100 utterances, 3 epochs, block_size=7, aux
[2,13,24,35,46], vanilla Markov head (rank 256). Per-step accept_len from
dspark_audio_train_curve.json; final metrics from checkpoint_best/val_metrics.json.

Outputs vector PDF (LaTeX) + PNG. Serif (STIX), editable text embedded.
Run: python scripts/plot_dspark_audio_profile.py [--out-prefix PATH]
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

# ── final measured results ──────────────────────────────────────────────────
POS_ACC = [0.9527, 0.9432, 0.9359, 0.9252, 0.9121, 0.8983, 0.8790]
BLOCK = 7
ACCEPT_LEN, ACCEPT_RATE, FULL_ACC = 6.19, 0.907, 0.922

C = "#3b6fb6"
C2 = "#9db8dc"
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
    ax.text(-0.02, 1.13, t, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def _pipeline(ax):
    ax.axis("off")
    steps = ["LibriSpeech\n28.5k utts", "Prompts\n(audio+instr)",
             "On-policy gen\n(target answers)", "Arrow\n(tokenize)",
             "Online train\n(extract-hidden\n+ DSpark)", "Audio draft\nhead"]
    n = len(steps); xw, gap = 1.0 / n, 0.012; y, h = 0.15, 0.7
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
            ax.add_patch(FancyArrowPatch((x + xw - gap, y + h / 2), (x + xw + gap, y + h / 2),
                         transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
                         lw=1.0, color=MUTED, clip_on=False))


def _smooth(y, w=31):
    # Edge-aware moving average: divide by the actual window count at the
    # boundaries so the smoothed line doesn't droop toward zero at the ends.
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    kernel = np.ones(w)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="dspark_audio_profile")
    ap.add_argument("--curve", default=os.path.join(os.path.dirname(__file__),
                    "dspark_audio_train_curve.json"))
    a = ap.parse_args()
    _rc()

    curve = json.load(open(a.curve))
    al = np.array(curve["accept_len"])
    val = curve["val_epoch_accept_len"]
    spe = curve["steps_per_epoch"]

    fig = plt.figure(figsize=(7.6, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.48, 1], hspace=0.6, wspace=0.36,
                          left=0.075, right=0.975, top=0.9, bottom=0.12)

    axp = fig.add_subplot(gs[0, :]); _pipeline(axp)
    axp.set_title("Pipeline: Qwen3-Omni DSpark audio draft head (this run)",
                  fontsize=10.5, fontweight="bold", color=INK, pad=6)

    # (a) fine-grained training convergence — accept_len over every step
    axa = fig.add_subplot(gs[1, 0]); _style(axa)
    axa.yaxis.grid(True, color=GRID, linewidth=0.6); axa.set_axisbelow(True)
    x = np.arange(len(al))
    axa.plot(x, al, color=C2, lw=0.5, alpha=0.7)
    axa.plot(x, _smooth(al), color=C, lw=1.6)
    for e, v in enumerate(val, start=1):
        axa.plot(e * spe - 1, v, marker="o", ms=5, mfc="white", mec=C, mew=1.3, zorder=5)
    axa.annotate(f"{val[-1]:.2f}", (len(al) - 1, val[-1]), textcoords="offset points",
                 xytext=(-2, 6), ha="right", color=C, fontsize=8, fontweight="bold")
    axa.set_ylim(0.8, 7.1)
    axa.set_xlabel("Training step")
    axa.set_ylabel(r"Accepted length ($\leq 7$)")
    axa.set_title("Convergence (2196 steps, 3 ep)", fontsize=9.5, color=INK, pad=4)
    axa.text(0.97, 0.06, "○ = val / epoch", transform=axa.transAxes, ha="right",
             va="bottom", fontsize=7, color=MUTED)
    _plabel(axa, "(a)")

    # (b) final per-position acceptance
    axb = fig.add_subplot(gs[1, 1]); _style(axb)
    axb.yaxis.grid(True, color=GRID, linewidth=0.6); axb.set_axisbelow(True)
    axb.plot(range(BLOCK), POS_ACC, color=C, lw=1.8, marker="o", ms=5.5,
             mfc=C, mec="white", mew=0.9)
    axb.annotate(f"{POS_ACC[0]:.0%}", (0, POS_ACC[0]), textcoords="offset points",
                 xytext=(2, 7), ha="left", color=C, fontsize=8)
    axb.annotate(f"{POS_ACC[-1]:.0%}", (6, POS_ACC[-1]), textcoords="offset points",
                 xytext=(0, -14), ha="center", color=C, fontsize=8)
    axb.set_ylim(0.82, 1.0)
    axb.set_xticks(range(BLOCK))
    axb.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axb.set_xlabel("Draft token position in block")
    axb.set_ylabel("Acceptance")
    axb.set_title("Per-position acceptance", fontsize=9.5, color=INK, pad=4)
    _plabel(axb, "(b)")

    # (c) headline speculative-decoding metrics (stat tile)
    axc = fig.add_subplot(gs[1, 2]); axc.axis("off")
    _plabel(axc, "(c)")
    axc.set_title("Key spec-decode metrics", fontsize=9.5, color=INK, pad=4)
    tiles = [
        ("Block drafted", f"{BLOCK}", "tokens / verify"),
        ("Accept length", f"{ACCEPT_LEN:.2f}", f"of {BLOCK}"),
        ("Accept rate", f"{ACCEPT_RATE:.1%}", "of drafted"),
        ("Full-block acc", f"{FULL_ACC:.1%}", "all 7 correct"),
    ]
    for i, (label, val_s, sub) in enumerate(tiles):
        yy = 0.90 - i * 0.245
        axc.add_patch(FancyBboxPatch((0.02, yy - 0.16), 0.96, 0.19,
                      boxstyle="round,pad=0.01,rounding_size=0.03",
                      linewidth=0.7, edgecolor=C, facecolor=BOX,
                      transform=axc.transAxes, clip_on=False))
        axc.text(0.07, yy - 0.065, label, transform=axc.transAxes, fontsize=7.6,
                 color=MUTED, va="center")
        axc.text(0.93, yy - 0.065, val_s, transform=axc.transAxes, fontsize=13,
                 fontweight="bold", color=C, va="center", ha="right")
        axc.text(0.07, yy - 0.125, sub, transform=axc.transAxes, fontsize=6.6,
                 color=MUTED, va="center", style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out_prefix}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved -> {a.out_prefix}.{ext}")


if __name__ == "__main__":
    main()
