# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ali Rıza Saral
# Licensed under the MIT License.
"""
Created on Mon Aug  3 21:55:45 2026

@author: Ali Rıza SARAL
"""

# -*- coding: utf-8 -*-
"""
Program 2 - Orchestral Balance

Reads an Excel matrix containing entries such as

    S5
    D10
    T16
    Q20

and computes row statistics.

Outputs:

orchestral_balance/
    tables/
        row_statistics.csv

    plots/
        occupied_cells.png
        total_density.png
        mean_density.png

"""

import os
import re

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

filename = "Matrix_M.xlsx"
sheet = 0

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

base_dir = "orchestral_balance"

plot_dir = os.path.join(base_dir, "plots")
table_dir = os.path.join(base_dir, "tables")

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

# ============================================================
# READ EXCEL
# ============================================================

df = pd.read_excel(
    filename,
    sheet_name=sheet,
    index_col=0,
    header=0
)

pattern = r"([SDTQ])\s*([0-9]+(?:\.[0-9]+)?)"

# ============================================================
# COMPUTE ROW STATISTICS
# ============================================================

results = []

for row in df.index:

    occupied = 0
    total_density = 0.0
    
    S = 0
    D = 0
    T = 0
    Q = 0

    for col in df.columns:

        value = df.loc[row, col]

        if pd.isna(value):
            continue

        text = str(value).strip()

        m = re.match(pattern, text)

        if m is None:
            continue

        density = float(m.group(2))
        
        event = m.group(1)

        if event == "S":
            S += 1
        elif event == "D":
            D += 1
        elif event == "T":
            T += 1
        elif event == "Q":
            Q += 1

        occupied += 1
        total_density += density

    if occupied > 0:
        mean_density = total_density / occupied
    else:
        mean_density = 0

    results.append({
    "Row": row,
    "S": S,
    "D": D,
    "T": T,
    "Q": Q,
    "Occupied Cells": occupied,
    "Total Density": total_density,
    "Mean Density": round(mean_density, 3)
})

# ============================================================
# SAVE TABLE
# ============================================================

stats = pd.DataFrame(results)

outfile = os.path.join(
    table_dir,
    "row_statistics.csv"
)

stats.to_csv(outfile, index=False)

print()
print(stats)

# ============================================================
# GRAPH 1
# OCCUPIED CELLS
# ============================================================

plt.figure(figsize=(8,4))

plt.bar(stats["Row"], stats["Occupied Cells"])

plt.title("Occupied Cells")

plt.ylabel("Cells")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir,
    "occupied_cells.png"),
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 2
# TOTAL DENSITY
# ============================================================

plt.figure(figsize=(8,4))

plt.bar(stats["Row"], stats["Total Density"])

plt.title("Total Density")

plt.ylabel("Density")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir,
    "total_density.png"),
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 3
# MEAN DENSITY
# ============================================================

plt.figure(figsize=(8,4))

plt.bar(stats["Row"], stats["Mean Density"])

plt.title("Mean Density")

plt.ylabel("Density")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(plot_dir,
    "mean_density.png"),
    dpi=300
)

plt.close()

print()
print("Finished.")

print("Tables saved in:")
print(table_dir)

print("Plots saved in:")
print(plot_dir)