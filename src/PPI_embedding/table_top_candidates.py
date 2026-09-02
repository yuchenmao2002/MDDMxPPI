"""
ICLR appendix: top-k table for a 3-hyperparameter grid search.

Companion to plot_heatmap.py — the heatmap shows the whole grid, this table
lists the best candidate and its runners-up with exact numbers.

CSV is expected in long format, one row per configuration:
    tau,K_diff,b,J
    400,1,1,-0.484
    ...

Edit the CONFIG block below, then:  python table_top_candidates.py
"""

import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------
CSV      = "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/PPI_hyperparameter_J.csv"
OUT      = "/home/svu/e1538713/CodeNo0/outputs/evaluation/PPI_embedding/PPI_hyperparameter_J_top.csv"

KEY_COLS = ["tau", "K_diff", "b"]  # 前三列：超参数
VCOL     = "J"                     # 第四列：指标
N_RUNNERUP = 6                     # 最优候选之外再列出几个次优候选
DECIMALS   = 3
# ----------------------------------------------------------------------

frame = pd.read_csv(CSV)

# 按指标从大到小排序，取最优候选和 N_RUNNERUP 个次优候选。
top = frame.sort_values(VCOL, ascending=False).head(1 + N_RUNNERUP).reset_index(drop=True)
best = top[VCOL].iloc[0]

table = top[KEY_COLS].copy()
table[VCOL] = top[VCOL].round(DECIMALS)
table[f"exp({VCOL})"] = np.exp(top[VCOL]).round(DECIMALS)
table[f"{VCOL} - {VCOL}_best"] = (top[VCOL] - best).round(DECIMALS)

table.to_csv(OUT, index=False)
print(table.to_string(index=False, float_format=lambda value: f"{value:.{DECIMALS}f}"))
print(f"\nSaved: {OUT}")
