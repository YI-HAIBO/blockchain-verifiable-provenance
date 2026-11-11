#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the two camera-ready figures referenced by the ICDE 2027 R2 paper.

Outputs (into the same directory as this script):
  - fig_architecture.png : system architecture (on-chain MST + off-chain index
                          + lazy client verification).
  - fig_cover.png        : canonical-cover decomposition for an authenticated
                          range proof (O(log n) subtrees + sibling-path folding).

Pure matplotlib, deterministic, no external assets. Run:

    python3 figures/generate_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))


def _box(ax, x, y, w, h, text, fc, ec="#333333", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#111111")


def _arrow(ax, x1, y1, x2, y2, color="#444444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=12,
                                 lw=1.3, color=color))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # On-chain layer
    ax.text(5, 4.6, "Blockchain (consensus-committed)", ha="center",
            fontsize=10, fontweight="bold")
    _box(ax, 0.6, 3.4, 3.2, 0.9, "Merkle State Tree (MST)\ncommitted root_n", "#cfe8ff")
    _box(ax, 4.2, 3.4, 2.0, 0.9, "Block / Tx log", "#d6f5d6")
    _box(ax, 6.6, 3.4, 2.8, 0.9, "Consensus finality\n(finalized roots only)", "#ffe2b3")
    _arrow(ax, 3.8, 3.85, 4.2, 3.85)
    _arrow(ax, 6.2, 3.85, 6.6, 3.85)

    # Off-chain indexer (untrusted) - label on the left, away from arrows
    ax.text(0.6, 2.95, "Off-chain index server (untrusted)", ha="left",
            fontsize=9.5, fontweight="bold", color="#8a1f1f")
    _box(ax, 0.6, 1.55, 3.6, 1.0, "Queryable index\n(point / range / agg / lineage)", "#ffd6d6")
    _box(ax, 4.5, 1.55, 2.0, 1.0, "Answer R\n+ certificate C", "#ffd6d6")
    _box(ax, 6.7, 1.55, 2.7, 1.0, "Derives roots\nfrom chain\n(no self-commit)", "#ffe2b3", fs=8)
    _arrow(ax, 4.2, 2.05, 4.5, 2.05)
    _arrow(ax, 6.5, 2.05, 6.7, 2.05)
    # red emphasis: index derives roots FROM consensus (up to the chain layer)
    _arrow(ax, 8.05, 2.55, 8.05, 3.4, "#cc4444")

    # Client
    ax.text(2.4, 1.18, "Verifying client (lazy)", ha="center",
            fontsize=9.5, fontweight="bold")
    _box(ax, 3.0, 0.2, 4.0, 0.8,
         "Holds on-chain root + C;\nverifies R vs root (sound/complete)", "#eeeeee", fs=8)
    _arrow(ax, 5.8, 1.55, 5.8, 1.0)   # server -> client answer
    _arrow(ax, 5.0, 1.0, 5.0, 1.55)   # client challenge / verify

    ax.text(0.2, 0.0,
            "Indexer returns (R, C); client re-derives root from finalized chain and "
            "verifies C. No data owner signs; trust comes from consensus.",
            fontsize=7.5, color="#555555")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_architecture.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_cover():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(5, 4.6, "Authenticated range proof: canonical-cover decomposition",
            ha="center", fontsize=10, fontweight="bold")

    # sorted key space
    n = 16
    y = 3.4
    for i in range(n):
        _box(ax, 0.2 + i * 0.6, y, 0.55, 0.5, str(i), "#e6e6e6")
    # range [k3, k11]
    ax.add_patch(FancyBboxPatch((0.2 + 3 * 0.6 - 0.02, y - 0.18),
                                9 * 0.6 + 0.04, 0.86,
                                boxstyle="round,pad=0.0",
                                fc="none", ec="#cc4444", lw=1.6))
    ax.text(0.2 + 3 * 0.6 + 4.5 * 0.6, y + 0.75,
            "query range [k3, k11] (contiguous)", ha="center", fontsize=8,
            color="#cc4444")

    # O(log n) canonical-cover subtrees
    labels = ["T3", "T4-7", "T8", "T9-10", "T11"]
    xs = [0.2 + 3 * 0.6, 0.2 + 4 * 0.6, 0.2 + 8 * 0.6,
          0.2 + 9 * 0.6, 0.2 + 11 * 0.6]
    ys = 1.7
    for lb, x in zip(labels, xs):
        _box(ax, x, ys, 0.9, 0.6, lb, "#cfe8ff")
    ax.text(5, ys + 0.95, "O(log n) canonical-cover subtrees (proven against root_n)",
            ha="center", fontsize=8.5)

    # sibling-path folding
    _box(ax, 1.6, 0.3, 5.6, 0.6, "sibling-path folding ->\ncompact cert C",
         "#eeeeee", fs=8)
    for x in xs:
        _arrow(ax, x + 0.45, ys, x + 0.45, 0.95)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_cover.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    a = fig_architecture()
    b = fig_cover()
    print("wrote", a)
    print("wrote", b)
