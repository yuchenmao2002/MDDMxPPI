"""
ICLR appendix: 1x3 faceted heatmap for a 3-hyperparameter grid search.

CSV is expected in long format, one row per configuration:
    tau,K_diff,b,J
    400,1,1,-0.496
    ...

Edit the CONFIG block below, then:  python plot_heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, Normalize, ListedColormap

# ----------------------------- CONFIG ---------------------------------
CSV = "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/PPI_hyperparameter_J.csv"
OUT = "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/PPI_hyperparameter_J_heatmap.pdf"

FACET = "b"       # 3 values -> one panel each
XCOL  = "tau"     # 6 values -> columns (keep the SMALLER one on x)
YCOL  = "K_diff"  # 8 values -> rows
VCOL  = "J"

FACET_LABEL = r"b"                  # bare symbol, no $...$: wrapped in math mode below
XLABEL      = r"$\tau$"
YLABEL      = r"$K_{\mathrm{diff}}$"
CBAR_LABEL  = r"$\mathcal{J}$"      # keep identical to the notation used in the table

TEXTWIDTH_IN = 5.5     # ICLR \textwidth; verify with \the\textwidth (pt/72.27)
FIGHEIGHT_IN = 2.75
CELL_FS      = 5.5     # in-cell font size (pt); do not go below ~5.2

# --- marking ---------------------------------------------------------
# Cells within TOL of the global maximum form the "plateau" and get a dashed
# outline. Among them, PREFER picks the configuration actually selected (solid
# outline): the columns are minimised in order, so the cheapest point on the
# plateau wins rather than a noise-level argmax. Set PREFER = None for argmax.
TOL    = 0.01
PREFER = ["K_diff", "tau"]
PAD    = 0.06          # box inset, so adjacent plateau boxes do not merge

# --- degenerate cells ------------------------------------------------
# Rows matching this pandas query are blanked out instead of being plotted as an
# exact 0. They are drawn as a hatched cell rather than a grey fill: a grey light
# enough not to look like data is too close in lightness to the bottom of the
# colour ramp, so "no value" has to be signalled by texture, not by brightness.
# Example: "K_diff == 1 and tau >= 700"
MASK_QUERY = None
HATCH      = "///"
HATCH_COL  = "0.75"

# --- colour ----------------------------------------------------------
# BuPu truncated to (0.12, 0.92) measured best of the candidates tested:
# strictly monotone lightness, zero monotonicity violation under simulated
# deuteran/protan/tritanopia, and the highest in-cell text contrast (4.2:1).
CENTER     = None            # diverging midpoint; None -> plain linear Normalize
CMAP       = "BuPu"          # sequential, light -> dark: high values read as heavy
CMAP_RANGE = (0.12, 0.92)    # avoid pure white at the bottom / near-black at the top
FMT        = "{:.2f}"

# Clip the colour scale so the ramp is spent on the range the data actually
# occupies. Values outside are drawn in the extreme colour and the colourbar
# grows an arrow to say so. None -> take the limit from the data.
VLIM       = (0.8, None)
# ----------------------------------------------------------------------

mpl.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType, never Type 3
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Times-compatible maths, matches ICLR body text
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "hatch.linewidth": 0.4,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})

MINUS = "\u2212"          # true minus sign, not a hyphen


def fmt(v):
    return FMT.format(v).replace("-", MINUS)


def tick(v):
    return f"{v:g}".replace("-", MINUS)


df = pd.read_csv(CSV)

if MASK_QUERY:
    df.loc[df.eval(MASK_QUERY), VCOL] = np.nan

facets = sorted(df[FACET].unique())
xs     = sorted(df[XCOL].unique())
ys     = sorted(df[YCOL].unique())

vmin, vmax = float(df[VCOL].min()), float(df[VCOL].max())
lo = vmin if VLIM[0] is None else float(VLIM[0])
hi = vmax if VLIM[1] is None else float(VLIM[1])

if CENTER is not None and lo < CENTER < hi:
    norm = TwoSlopeNorm(vmin=lo, vcenter=CENTER, vmax=hi)
else:
    norm = Normalize(vmin=lo, vmax=hi)

under, over = vmin < lo, vmax > hi
extend = {(0, 0): "neither", (1, 0): "min",
          (0, 1): "max", (1, 1): "both"}[(int(under), int(over))]

cmap = ListedColormap(plt.get_cmap(CMAP)(np.linspace(*CMAP_RANGE, 256)), name="bp")
cmap.set_bad(alpha=0.0)          # masked cells fall through to the hatched axes patch

# ---- plateau: every cell within TOL of the global maximum -------------
plateau = df[df[VCOL] >= vmax - TOL].copy()
if PREFER:
    plateau = plateau.sort_values(PREFER, ascending=True)
    selected = plateau.iloc[0]
else:
    selected = plateau.sort_values(VCOL, ascending=False).iloc[0]
others = plateau.drop(index=selected.name)

fig, axes = plt.subplots(
    1, len(facets),
    figsize=(TEXTWIDTH_IN, FIGHEIGHT_IN),
    sharey=True, constrained_layout=True,
)
axes = np.atleast_1d(axes)


def ink(v):
    """Black or white, whichever reads on top of the colour for v."""
    r, g, b, _ = cmap(norm(v))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def outline(ax, row, **kw):
    j, i = xs.index(row[XCOL]), ys.index(row[YCOL])
    ax.add_patch(Rectangle((j - .5 + PAD, i - .5 + PAD), 1 - 2 * PAD, 1 - 2 * PAD,
                           fill=False, clip_on=False,
                           edgecolor=ink(row[VCOL]), **kw))


for ax, f in zip(axes, facets):
    sub = df[df[FACET] == f]
    M = (sub.pivot(index=YCOL, columns=XCOL, values=VCOL)
            .reindex(index=ys, columns=xs)
            .to_numpy(dtype=float))

    ax.set_facecolor("white")
    ax.patch.set_hatch(HATCH)
    ax.patch.set_edgecolor(HATCH_COL)

    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm,
                   aspect="auto", origin="upper", zorder=2)

    ax.set_xticks(range(len(xs)), [tick(v) for v in xs])
    ax.set_yticks(range(len(ys)), [tick(v) for v in ys])
    ax.set_title(rf"${FACET_LABEL} = {f:g}$", pad=3)
    for s in ax.spines.values():
        s.set_visible(False)

    # in-cell values, with contrast-aware text colour
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", zorder=3,
                    fontsize=CELL_FS, color=ink(v))

    if selected[FACET] == f:
        outline(ax, selected, linewidth=1.5, zorder=4)
    for _, r_ in others.iterrows():
        if r_[FACET] == f:
            outline(ax, r_, linewidth=0.8, linestyle=(0, (2, 1.5)), zorder=4)

cb = fig.colorbar(im, ax=axes.tolist(), pad=0.015, aspect=28, extend=extend)
cb.set_label(CBAR_LABEL, fontsize=8, rotation=90)
cb.outline.set_linewidth(0.6)
cb.ax.tick_params(labelsize=6, size=0)

axes[0].set_ylabel(YLABEL)
fig.supxlabel(XLABEL, fontsize=8)

fig.savefig(OUT)                      # exact 5.5in width -> insert with no rescaling
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=400)

print(f"wrote {OUT}")
print(f"colour scale [{lo:.3f}, {hi:.3f}] (data [{vmin:.3f}, {vmax:.3f}], extend={extend})")
print(f"max {VCOL} = {vmax:.4f}; plateau (within {TOL}) has {len(plateau)} cells")
print("selected:", {k: selected[k] for k in (FACET, XCOL, YCOL, VCOL)})