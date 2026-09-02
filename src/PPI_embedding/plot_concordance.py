"""
ICLR appendix: macro-averaged pairwise concordance A(r) against embedding rank r
for the selected candidate (tau=700, K_diff=4, b=1).

Reads PPI_rank_concordance.csv and plots A(r) on a linear rank axis. The
selection baseline 1/2 + 0.95 (A_max - 1/2) is drawn as a recessive dashed rule
and the smallest r reaching it is filled in and labelled. Single series, so no
legend box: the ordinate names the quantity.

Figure size matches plot_spectrum.py so the two PDFs sit side by side inside the
5.5 in ICLR text block (2 x 2.65 in + gutter); insert at this size, do not rescale.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------- CONFIG ---------------------------------
CSV = (
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/"
    "PPI_rank_concordance.csv"
)
OUT = (
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/"
    "PPI_rank_concordance.pdf"
)

XCOL, YCOL = "r", "A"
CHANCE_LEVEL = 0.5            # A = 1/2 是随机排序的期望
BASELINE_FRACTION = 0.95      # 选择条件：达到 A_max 相对随机水平增益的 95%
FIGSIZE = (2.65, 2.0)         # inches; paired with plot_spectrum.py
# ----------------------------------------------------------------------

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.5, "legend.frameon": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
})

df = pd.read_csv(CSV).sort_values(XCOL)
ranks = df[XCOL].to_numpy()
areas = df[YCOL].to_numpy()

baseline = CHANCE_LEVEL + BASELINE_FRACTION * (areas.max() - CHANCE_LEVEL)
reaching = ranks[areas >= baseline]
selected = int(reaching[0]) if reaching.size else None

fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

ax.axhline(baseline, color="0.55", linestyle=(0, (3.0, 1.6)), linewidth=0.6,
           zorder=1)
ax.plot(ranks, areas, color="black", linestyle="-", linewidth=1.0,
        marker="o", markersize=2.6, markerfacecolor="white",
        markeredgewidth=0.7, zorder=3)

if selected is not None:
    selected_area = float(areas[ranks == selected][0])
    ax.plot([selected], [selected_area], marker="o", markersize=2.6,
            color="black", zorder=4)
    ax.annotate(rf"$r={selected}$", xy=(selected, selected_area),
                xytext=(3.5, -10), textcoords="offset points",
                va="top", fontsize=6.5)

span = areas.max() - areas.min()
ax.set_ylim(areas.min() - 0.12 * span, areas.max() + 0.14 * span)
ax.set_yticks([0.92, 0.94, 0.96])
ax.set_ylabel(r"$\mathcal{A}(r)$")

ax.set_xlabel(r"embedding rank $r$")
ax.set_xlim(0, ranks.max() + ranks.min())
ax.set_xticks([8, 32, 64, 96, 128])

# 基线的解析式标在右端、基线之下——那一带被曲线让空，不必再开图例框。
ax.text(0.97, baseline, r"$1/2 + 0.95\,(\mathcal{A}_{\mathrm{max}} - 1/2)$",
        transform=ax.get_yaxis_transform(), va="top", ha="right",
        fontsize=5.8, color="0.35")

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=2.2)
ax.grid(axis="y", which="major", color="0.9", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=400)
print(f"baseline = {baseline:.9f}; smallest r reaching it = {selected}")
print("wrote", OUT)
