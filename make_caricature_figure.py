"""Compact one-column schematic for Fig. 1.

The LaTeX file includes this figure at ``width=\\columnwidth``.  The bound is
described in the caption, so the artwork focuses only on the motor-to-network
caricature and fills a short single-column panel.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent

OBS = "#1f77b4"
HID = "#9aa0a6"
HID_FILL = "#eef1f4"
TEXT = "#272a2d"
MUTED = "#707780"
BODY = "#506675"
BODY_EDGE = "#33424d"
FOOT = "#e8973a"
TRACK = "#c9ccd1"
GREEN = "#5fa84e"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.6,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def eye(ax, x, y, s=0.11, color=OBS, z=8):
    th = np.linspace(0, np.pi, 48)
    ax.plot(
        x + s * 1.55 * np.cos(th),
        y + s * 0.85 * np.sin(th),
        color=color,
        lw=1.0,
        solid_capstyle="round",
        zorder=z,
    )
    ax.plot(
        x + s * 1.55 * np.cos(th),
        y - s * 0.85 * np.sin(th),
        color=color,
        lw=1.0,
        solid_capstyle="round",
        zorder=z,
    )
    ax.add_patch(Circle((x, y), 0.043, facecolor=color, edgecolor="white", lw=0.18, zorder=z + 1))


def motor(ax, xc, y0, alpha=1.0, label=False):
    for fx in (xc - 0.43, xc + 0.43):
        ax.plot([fx, fx], [y0 + 0.17, y0 + 0.76], color=BODY_EDGE, lw=1.78, alpha=alpha, zorder=4)
        ax.add_patch(
            Circle(
                (fx, y0 + 0.17),
                0.135,
                facecolor=FOOT,
                edgecolor="#9a6011",
                lw=0.72,
                alpha=alpha,
                zorder=5,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (xc - 0.58, y0 + 0.75),
            1.16,
            0.68,
            boxstyle="round,pad=0.014,rounding_size=0.145",
            facecolor=BODY,
            edgecolor=BODY_EDGE,
            lw=0.96,
            alpha=alpha,
            zorder=5,
        )
    )
    if label:
        ax.text(xc, y0 + 1.09, "motor", ha="center", va="center", fontsize=7.8, color="white", zorder=6)


def edge(ax, p, q, color, lw, linestyle="solid", z=4):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=6.8,
            lw=lw,
            color=color,
            linestyle=linestyle,
            shrinkA=9.8,
            shrinkB=9.8,
            zorder=z,
        )
    )


fig = plt.figure(figsize=(3.35, 1.45))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 8.6)
ax.set_ylim(0, 3.72)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------------------
# Motor caricature
# ---------------------------------------------------------------------------
ax.text(1.92, 3.48, "motor", ha="center", va="center", fontsize=7.6, fontstyle="italic", color=TEXT)

track_y = 0.83
ax.add_patch(Rectangle((0.20, track_y), 3.64, 0.112, facecolor=TRACK, edgecolor="none", zorder=1))
for xt in np.arange(0.48, 3.82, 0.43):
    ax.add_patch(Circle((xt, track_y + 0.056), 0.053, facecolor="#aeb4ba", edgecolor="none", zorder=2))

motor(ax, 2.36, track_y, alpha=0.24)
motor(ax, 1.40, track_y, alpha=1.0, label=True)

ax.add_patch(
    FancyArrowPatch(
        (1.80, 2.72),
        (2.55, 2.72),
        connectionstyle="arc3,rad=-0.34",
        arrowstyle="-|>",
        mutation_scale=9.8,
        lw=1.28,
        color=OBS,
        zorder=7,
    )
)
eye(ax, 1.50, 2.90, s=0.112)
ax.text(2.06, 3.14, "observed step", ha="center", va="center", fontsize=6.45, color=OBS)

ax.add_patch(Circle((0.66, 2.22), 0.118, facecolor=GREEN, edgecolor="#3c7a30", lw=0.66, zorder=5))
ax.text(0.66, 2.22, "ATP", ha="center", va="center", fontsize=4.4, color="white", zorder=6)
ax.add_patch(
    FancyArrowPatch(
        (0.83, 2.08),
        (1.03, 1.85),
        arrowstyle="-|>",
        mutation_scale=6.4,
        lw=0.82,
        color=HID,
        linestyle=(0, (1.7, 1.3)),
        zorder=4,
    )
)
ax.text(1.66, 0.45, "hidden chemistry", ha="center", va="center", fontsize=5.8, fontstyle="italic", color=HID)

# Bridge
ax.add_patch(
    FancyArrowPatch((3.78, 2.12), (4.52, 2.12), arrowstyle="-|>", mutation_scale=11.5, lw=1.35, color="#55585c")
)
ax.text(4.15, 2.46, "states", ha="center", va="center", fontsize=5.7, color="#55585c")

# ---------------------------------------------------------------------------
# Markov-state network
# ---------------------------------------------------------------------------
ax.text(6.55, 3.48, "state network", ha="center", va="center", fontsize=7.6, fontstyle="italic", color=TEXT)

cx, cy, rr = 6.55, 1.90, 0.90
angles = np.deg2rad([90, 30, -30, -90, -150, 150])
nodes = np.array([[cx + rr * np.cos(a), cy + rr * np.sin(a)] for a in angles])
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
observed = {(2, 3), (3, 4)}
observed_nodes = {k for ij in observed for k in ij}

ax.add_patch(
    FancyBboxPatch(
        (cx - 1.30, cy + 0.16),
        2.60,
        1.26,
        boxstyle="round,pad=0.025,rounding_size=0.115",
        facecolor=HID_FILL,
        edgecolor=HID,
        lw=0.82,
        linestyle=(0, (3.2, 1.9)),
        zorder=1,
    )
)
ax.text(cx, cy + 1.22, "hidden", ha="center", va="center", fontsize=6.1, fontstyle="italic", color=MUTED, zorder=2)

for i, j in edges:
    if (i, j) in observed:
        edge(ax, nodes[i], nodes[j], OBS, 1.95, z=5)
    else:
        edge(ax, nodes[i], nodes[j], HID, 1.12, linestyle=(0, (2.3, 1.55)), z=4)

for k, (x, y) in enumerate(nodes):
    watched = k in observed_nodes
    ax.add_patch(
        Circle(
            (x, y),
            0.168,
            facecolor="white",
            edgecolor=OBS if watched else HID,
            lw=1.36 if watched else 1.15,
            zorder=7,
        )
    )

ax.add_patch(
    FancyArrowPatch(
        (cx + 0.49, cy + 0.04),
        (cx - 0.49, cy + 0.04),
        connectionstyle="arc3,rad=0.52",
        arrowstyle="-|>",
        mutation_scale=8.0,
        lw=1.15,
        color="#55585c",
        zorder=6,
    )
)
ax.text(cx, cy + 0.40, r"$K$", ha="center", va="center", fontsize=8.0, color=TEXT, zorder=7)
eye(ax, nodes[3][0], nodes[3][1] - 0.34, s=0.092)
ax.text(nodes[3][0], nodes[3][1] - 0.56, "observed", ha="center", va="center", fontsize=5.9, color=OBS)

fig.savefig(OUT_DIR / "fig_caricature.pdf")
fig.savefig(OUT_DIR / "fig_caricature.png", dpi=300)
plt.close(fig)
print(f"saved {OUT_DIR / 'fig_caricature.pdf'}")
print(f"saved {OUT_DIR / 'fig_caricature.png'}")
