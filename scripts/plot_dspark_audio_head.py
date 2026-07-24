"""Publication-quality figure: Qwen3-Omni DSpark **audio** draft-head acceptance.

Outputs vector PDF (primary, for LaTeX \\includegraphics) + high-res PNG.
Double-column width (~7in). Serif (STIX, Times-like) to match LaTeX body text;
editable text embedded (pdf.fonttype=42). Editorial framing belongs in the
caption, not the figure — a suggested caption is printed on run.

Data: on-policy training on 28,539 LibriSpeech (train-clean-100) samples, 3 epochs,
block_size=7, aux [2,13,24,35,46], vanilla Markov head (rank 256). Metrics from
checkpoint_best/val_metrics.json. Image baseline
(feizhai123/Qwen3-Omni-30B-A3B-Instruct-DSpark-Thinker-Image) reported only its two
endpoint acceptances (pos0=0.71, pos6=0.42) + accept_len=3.16, so its intermediate
curve is drawn dashed and marked "endpoints only".

Run: python scripts/plot_dspark_audio_head.py [--out-prefix PATH]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ── measured results ────────────────────────────────────────────────────────
POS = list(range(7))
AUDIO_ACC = [0.9527, 0.9432, 0.9359, 0.9252, 0.9121, 0.8983, 0.8790]
IMAGE_ENDPOINTS = {0: 0.71, 6: 0.42}          # only endpoints reported
ACCEPT_LEN = {"Audio (ASR)": 6.19, "Image (VQA)": 3.16}
BLOCK = 7

# ── CVD-safe pair (validated); direct labels supply the contrast relief ──────
C_AUDIO = "#3b6fb6"
C_IMAGE = "#e8863c"
INK = "#1a1a1a"
MUTED = "#555555"
GRID = "#dddddd"


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed editable TrueType, not paths
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    })


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)


def _panel_label(ax, txt):
    ax.text(-0.14, 1.04, txt, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="dspark_audio_head_acceptance")
    args = ap.parse_args()
    _rc()

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # ── (a) per-position acceptance ─────────────────────────────────────────
    _style(axL)
    axL.yaxis.grid(True, color=GRID, linewidth=0.6)
    axL.set_axisbelow(True)
    axL.plot(POS, AUDIO_ACC, color=C_AUDIO, lw=1.6, marker="o", ms=4.5,
             mfc=C_AUDIO, mec="white", mew=0.8, zorder=3, label="Audio (ASR)")
    ix = sorted(IMAGE_ENDPOINTS)
    axL.plot(ix, [IMAGE_ENDPOINTS[i] for i in ix], color=C_IMAGE, lw=1.6,
             ls=(0, (4, 2.5)), marker="s", ms=4.5, mfc=C_IMAGE, mec="white",
             mew=0.8, zorder=3, label="Image (VQA), endpoints only")
    for x, y, dy in [(0, AUDIO_ACC[0], 7), (6, AUDIO_ACC[-1], 7)]:
        axL.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                     xytext=(0, dy), ha="center", color=C_AUDIO, fontsize=8)
    for x, dy in [(0, -13), (6, -13)]:
        axL.annotate(f"{IMAGE_ENDPOINTS[x]:.0%}", (x, IMAGE_ENDPOINTS[x]),
                     textcoords="offset points", xytext=(0, dy), ha="center",
                     color=C_IMAGE, fontsize=8)
    axL.set_ylim(0.35, 1.0)
    axL.set_xlim(-0.35, 6.35)
    axL.set_xticks(POS)
    axL.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axL.set_xlabel("Draft position within block")
    axL.set_ylabel("Per-position acceptance")
    axL.legend(frameon=False, loc="lower left", handlelength=1.8,
               labelcolor=INK, borderpad=0.1)
    _panel_label(axL, "(a)")

    # ── (b) mean accepted length ────────────────────────────────────────────
    _style(axR)
    axR.xaxis.grid(True, color=GRID, linewidth=0.6)
    axR.set_axisbelow(True)
    for name, y, c in zip(ACCEPT_LEN, (1, 0), (C_AUDIO, C_IMAGE)):
        v = ACCEPT_LEN[name]
        axR.barh(y, BLOCK, height=0.46, color="#eeeeee", zorder=1)
        axR.barh(y, v, height=0.46, color=c, zorder=3)
        axR.annotate(f"{v:.2f}", (v, y), textcoords="offset points",
                     xytext=(4, 0), va="center", color=c, fontsize=10,
                     fontweight="bold")
    axR.set_xlim(0, BLOCK + 0.3)
    axR.set_ylim(-0.6, 1.6)
    axR.set_yticks((1, 0))
    axR.set_yticklabels(list(ACCEPT_LEN), color=INK)
    axR.set_xticks([0, BLOCK])
    axR.set_xticklabels(["0", f"{BLOCK}"])
    axR.set_xlabel(r"Mean accepted tokens / block ($\leq 7$)")
    _panel_label(axR, "(b)")

    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        path = f"{args.out_prefix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved → {path}")

    print(
        "\nSuggested LaTeX caption:\n"
        "\\caption{Acceptance of the Qwen3-Omni DSpark audio draft head "
        "(28.5k LibriSpeech utterances, 3 epochs, block size 7). "
        "(a) Per-position acceptance decays only from 95.3\\% to 87.9\\% across "
        "the block, versus 71\\%$\\to$42\\% for the image baseline. "
        "(b) Mean accepted length reaches 6.19/7 tokens versus 3.16 for image. "
        "The high acceptance reflects that ASR transcription is strongly "
        "constrained by the input audio; it is not a general claim that audio "
        "outperforms image.}"
    )


if __name__ == "__main__":
    main()
