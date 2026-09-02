"""
ICLR appendix: leading normalized positive spectrum for the top-4 candidates
by validation J (tau=700, b=1, K_diff in 4..7). Among them K_diff=4 is the
selected one, not the marginally higher-scoring K_diff=5.

Reads PPI_rank_scan_spectrum.csv (4 candidates x 256 ranks) and plots
sigma_j / sigma_1 against rank j. Linear rank axis, logarithmic ordinate.
The candidate with selection_rank == 1 is the selected one: solid black and
labelled "(selected)" in the legend. Runners-up: light dashed.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter

# ----------------------------- CONFIG ---------------------------------
CSV = (
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/"
    "PPI_rank_scan_spectrum.csv"
)
SUMMARY = (
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/"
    "PPI_rank_scan_summary.csv"
)
OUT = (
    "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/"
    "PPI_rank_scan_spectrum.pdf"
)

XCOL, YCOL, IDCOL, RANKCOL = "j", "sigma_normalized", "candidate_id", "selection_rank"
FIGSIZE = (2.65, 2.0)         # inches; paired with plot_concordance.py so the two
                              # PDFs fit side by side in the 5.5 in ICLR text
                              # block. Insert at this size, do not rescale.
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

df = pd.read_csv(CSV)
summary = pd.read_csv(SUMMARY).sort_values(RANKCOL)


# 四个候选共享的超参数进图例标题，只让变化的那一维出现在每条曲线的标签上。
PARAMETER_SYMBOLS = (("tau", r"\tau"), ("K_diff", r"K_{\mathrm{diff}}"), ("b", "b"))
shared = [(c, s) for c, s in PARAMETER_SYMBOLS if df[c].nunique() == 1]
varying = [(c, s) for c, s in PARAMETER_SYMBOLS if df[c].nunique() > 1]
legend_title = (
    "$" + r",\ ".join(rf"{s}={int(df[c].iloc[0])}" for c, s in shared) + "$"
    if shared
    else None
)


def label(g, is_best):
    r = g.iloc[0]
    text = "$" + r",\ ".join(rf"{s}={int(getattr(r, c))}" for c, s in varying) + "$"
    return (text + " (selected)") if is_best else text


order = df.groupby(IDCOL)[RANKCOL].first().sort_values().index.tolist()
best_id = order[0]
r_scan = int(summary["R_scan"].iloc[0])
right_censored = summary["r_plus_right_censored"].all()

runner_style = [
    dict(color="#7B9FD4", linestyle=(0, (4.0, 1.8))),
    dict(color="#E0A276", linestyle=(0, (1.8, 1.4))),
    dict(color="#83C295", linestyle=(0, (0.9, 1.3))),
]

fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

k = 0
for cid in order:
    g = df[df[IDCOL] == cid].sort_values(XCOL)
    if cid == best_id:
        st = dict(color="black", linestyle="-", linewidth=1.0, zorder=3)
    else:
        st = dict(**runner_style[k], linewidth=0.8, zorder=2)
        k += 1
    if right_censored:
        st.update(marker=">", markevery=[-1], markersize=3.2, markeredgewidth=0.6,
                  clip_on=False)
    ax.plot(g[XCOL], g[YCOL], label=label(g, cid == best_id), **st)

ax.set_yscale("log")
ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1, numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel(r"$\sigma_j/\sigma_1$")

ax.set_xlabel(r"eigenvalue index $j$")
ax.set_xlim(1, r_scan)
ax.set_xticks([1, 50, 100, 150, 200, r_scan])

if right_censored:
    ax.text(0.03, 0.04, rf"right-censored at $R_{{\mathrm{{scan}}}}={r_scan}$",
            transform=ax.transAxes, fontsize=6, color="0.25")

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=2.2)
ax.grid(axis="y", which="major", color="0.9", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)

ax.legend(title=legend_title, loc="upper right", fontsize=6, title_fontsize=6,
          handlelength=2.6, borderpad=0.2, labelspacing=0.35)

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=400)
print("wrote", OUT)
