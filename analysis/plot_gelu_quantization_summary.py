#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# Paths
# ================================================================

ROOT = Path(
    "ACTIVATION_DUMP/"
    "smoothing_base_regression10"
)

BEFORE_PATH = (
    ROOT
    / "before"
    / "encoder.blocks.5.mlp.1.npy"
)

AFTER_PATH = (
    ROOT
    / "after"
    / "encoder.blocks.5.mlp.1.npy"
)

OUTPUT = Path(
    "analysis/figures/"
    "gelu_quantization_summary.png"
)


# ================================================================
# Load data
# ================================================================

before = np.load(
    BEFORE_PATH,
    mmap_mode="r",
)

after = np.load(
    AFTER_PATH,
    mmap_mode="r",
)

before_flat = np.asarray(
    before,
    dtype=np.float32,
).reshape(-1)

after_flat = np.asarray(
    after,
    dtype=np.float32,
).reshape(-1)


# ================================================================
# Statistics
# ================================================================

before_neg = before_flat[
    before_flat < 0
]

after_neg = after_flat[
    after_flat < 0
]

before_neg_fraction = (
    np.mean(before_flat < 0) * 100
)

before_neg_p99 = np.percentile(
    np.abs(before_neg),
    99,
)

after_neg_p99 = np.percentile(
    np.abs(after_neg),
    99,
)

before_max = float(
    before_flat.max()
)

after_max = float(
    after_flat.max()
)


# Block 5 MLP2 output NRMSE
quantizer_names = [
    "Signed\nsymmetric",
    "Standard\nasymmetric",
    "True-zero",
]

quantizer_nrmse = [
    0.04775096,
    0.02981970,
    0.02976998,
]


# ================================================================
# Plot
# ================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.0),
)


# ----------------------------------------------------------------
# (a) GeLU asymmetry
# ----------------------------------------------------------------

ax = axes[0]

ax.set_xlim(
    0.0,
    1.0,
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.axis("off")

ax.set_title(
    "(a) Highly asymmetric GeLU output"
)

# --------------------------------------------------------------
# Main takeaway
# --------------------------------------------------------------

ax.text(
    0.50,
    0.86,
    "98.1% of activations are slightly negative",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
)

# --------------------------------------------------------------
# Range schematic
# --------------------------------------------------------------

zero_x = 0.30
neg_left = 0.08
pos_right = 0.90
range_y = 0.58

# Negative side
ax.annotate(
    "",
    xy=(zero_x - 0.01, range_y),
    xytext=(neg_left, range_y),
    arrowprops=dict(
        arrowstyle="<->",
        linewidth=2.0,
    ),
)

# Positive side
ax.annotate(
    "",
    xy=(pos_right, range_y),
    xytext=(zero_x + 0.01, range_y),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=2.0,
    ),
)

# Zero marker
ax.plot(
    [zero_x, zero_x],
    [0.49, 0.63],
    linestyle="--",
    linewidth=1.2,
)

# Labels above arrows
ax.text(
    0.14,
    0.67,
    "Tiny negative\nrange",
    ha="center",
    va="center",
    fontsize=9.5,
    linespacing=1.05,
)

ax.text(
    0.60,
    0.67,
    "Long positive tail",
    ha="center",
    va="center",
    fontsize=9.5,
)

# Endpoint labels
ax.text(
    neg_left,
    0.50,
    "-0.164",
    ha="center",
    va="top",
    fontsize=10,
)

ax.text(
    zero_x,
    0.47,
    "0",
    ha="center",
    va="top",
    fontsize=10,
)

ax.text(
    pos_right,
    0.50,
    "+204.9",
    ha="center",
    va="top",
    fontsize=10,
)

# --------------------------------------------------------------
# Supporting statistics
# --------------------------------------------------------------

ax.text(
    0.50,
    0.27,
    "Negative P99 |x| = 0.124\n"
    "Positive P99 = 15.64\n"
    "Positive Max = 204.9",
    ha="center",
    va="center",
    fontsize=10,
    linespacing=1.35,
)

ax.text(
    0.50,
    0.08,
    "Range illustration not to scale",
    ha="center",
    va="center",
    fontsize=8,
)



# ----------------------------------------------------------------
# (b) Smoothing effect
# ----------------------------------------------------------------

ax = axes[1]

labels = [
    "Negative P99\nmagnitude",
    "Positive\nmaximum",
]

before_values = [
    before_neg_p99,
    before_max,
]

after_values = [
    after_neg_p99,
    after_max,
]

x = np.arange(
    len(labels)
)

width = 0.35

ax.bar(
    x - width / 2,
    before_values,
    width,
    label="Before",
)

ax.bar(
    x + width / 2,
    after_values,
    width,
    label="After",
)

ax.set_yscale(
    "log"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_ylabel(
    "Activation magnitude"
)

ax.set_title(
    "(b) Smooth@GeLU compresses activation range"
)

ax.legend(
    frameon=False
)

ax.grid(
    True,
    axis="y",
    which="both",
    alpha=0.2,
)

for xpos, value in zip(
    x - width / 2,
    before_values,
):
    ax.text(
        xpos,
        value * 1.12,
        f"{value:.3g}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

for xpos, value in zip(
    x + width / 2,
    after_values,
):
    ax.annotate(
    f"{value:.3g}",
    xy=(xpos, value),
    xytext=(0, 4),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=9,
)

ymin, ymax = ax.get_ylim()
ax.set_ylim(
    ymin,
    ymax * 1.35,
)


# ----------------------------------------------------------------
# (c) Smaller step narrows the zero bin
# ----------------------------------------------------------------

ax = axes[2]

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.axis("off")

ax.set_title(
    "(c) Smaller step narrows the zero bin",
    fontsize=13,
    pad=14,
)

before_color = "tab:blue"
after_color = "tab:orange"
zero_bin_color = "tab:red"


# ================================================================
# BEFORE
# ================================================================

y_before = 0.66

ax.text(
    0.06,
    0.79,
    "Before",
    ha="left",
    va="center",
    fontsize=11.5,
    fontweight="bold",
)

ax.text(
    0.58,
    0.79,
    "Wide zero bin",
    ha="center",
    va="center",
    fontsize=10.5,
)

ax.plot(
    [0.14, 0.92],
    [y_before, y_before],
    linewidth=2.6,
    color=before_color,
)

ax.plot(
    [0.35, 0.69],
    [y_before, y_before],
    linewidth=10,
    color=zero_bin_color,
    solid_capstyle="butt",
)

ax.scatter(
    [0.52],
    [y_before],
    s=36,
    color=before_color,
    zorder=3,
)

ax.text(
    0.52,
    0.57,
    "0",
    ha="center",
    va="center",
    fontsize=9.5,
)

ax.text(
    0.52,
    0.48,
    "many small nonzero values collapse to 0",
    ha="center",
    va="center",
    fontsize=9.2,
)


# ================================================================
# AFTER
# ================================================================

y_after = 0.26

ax.text(
    0.06,
    0.39,
    "After Smooth@GeLU",
    ha="left",
    va="center",
    fontsize=11.5,
    fontweight="bold",
)

ax.text(
    0.72,
    0.33,
    "Narrow zero bin",
    ha="center",
    va="center",
    fontsize=10.5,
)

ax.plot(
    [0.14, 0.92],
    [y_after, y_after],
    linewidth=2.6,
    color=after_color,
)

ax.plot(
    [0.47, 0.57],
    [y_after, y_after],
    linewidth=10,
    color=zero_bin_color,
    solid_capstyle="butt",
)

ax.scatter(
    [0.52],
    [y_after],
    s=36,
    color=after_color,
    zorder=3,
)

ax.text(
    0.52,
    0.17,
    "0",
    ha="center",
    va="center",
    fontsize=9.5,
)

ax.text(
    0.52,
    0.06,
    "finer quantization → less collapse to zero",
    ha="center",
    va="center",
    fontsize=9.2,
    fontweight="bold",
)

# ================================================================
# Finish
# ================================================================

fig.suptitle(
    "Smooth@GeLU Reduces Activation Range for Signed Quantization",
    fontsize=14,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.93,
    ]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {OUTPUT}"
)

"""
GeLU has many tiny negative values but a long positive tail. The outliers force a large quantization step, creating a wide zero bin where many nonzero values collapse to zero. Smooth@GeLU reduces the range first, so we can use finer signed quantization without clipping the important positive activations.
"""