# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ali Rıza Saral
# Licensed under the MIT License.
"""
Program 1 - Event Statistics

Reads an Excel matrix whose cells contain entries such as

    S5
    S4.5
    D10
    T16
    Q20

Empty cells are ignored.

Outputs:

event_statistics/
    plots/
        S_histogram.png
        D_histogram.png
        T_histogram.png
        Q_histogram.png

    tables/
        density_frequencies.csv
        summary_statistics.csv

"""

import os
import re

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

filename = "Matrix_M.xlsx"      # <-- change if necessary
sheet = 0

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

base_dir = "event_statistics"

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

rows = len(df.index)
cols = len(df.columns)

# ============================================================
# STORAGE
# ============================================================

event_counts = {
    "S": 0,
    "D": 0,
    "T": 0,
    "Q": 0
}

densities = {
    "S": [],
    "D": [],
    "T": [],
    "Q": []
}

occupied = 0

pattern = r"([SDTQ])\s*([0-9]+(?:\.[0-9]+)?)"

# ============================================================
# PARSE MATRIX
# ============================================================

for r in df.index:

    for c in df.columns:

        value = df.loc[r, c]

        if pd.isna(value):
            continue

        text = str(value).strip()

        m = re.match(pattern, text)

        if m is None:
            print("Ignored:", text)
            continue

        event = m.group(1)

        density = float(m.group(2))

        occupied += 1

        event_counts[event] += 1

        densities[event].append(density)

# ============================================================
# MATRIX SUMMARY
# ============================================================

print()
print("=" * 60)
print("EVENT STATISTICS")
print("=" * 60)

print("Rows            :", rows)
print("Columns         :", cols)
print("Cells           :", rows * cols)
print("Occupied Cells  :", occupied)
print("Empty Cells     :", rows * cols - occupied)

print()

# ============================================================
# EVENT COUNTS
# ============================================================

print("EVENT COUNTS")
print("-" * 60)

total_events = sum(event_counts.values())

for e in ["S", "D", "T", "Q"]:

    count = event_counts[e]

    pct = 100 * count / total_events if total_events else 0

    print(f"{e} : {count:4d} ({pct:6.2f} %)")

print()

# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = []

print("DENSITY STATISTICS")
print("-" * 60)

for e in ["S", "D", "T", "Q"]:

    if len(densities[e]) == 0:
        continue

    vals = pd.Series(densities[e])

    mean = vals.mean()
    minimum = vals.min()
    maximum = vals.max()

    print(f"{e}")
    print(f"   Count : {len(vals)}")
    print(f"   Mean  : {mean:.3f}")
    print(f"   Min   : {minimum}")
    print(f"   Max   : {maximum}")
    print()

    summary.append({
        "Event": e,
        "Count": len(vals),
        "Mean": mean,
        "Minimum": minimum,
        "Maximum": maximum
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(table_dir, "summary_statistics.csv"),
    index=False
)

# ============================================================
# DENSITY FREQUENCIES
# ============================================================

all_densities = []

for e in ["S", "D", "T", "Q"]:
    all_densities.extend(densities[e])

freq = (
    pd.Series(all_densities)
    .value_counts()
    .sort_index()
)

print("DENSITY FREQUENCIES")
print("-" * 60)

print(freq)

freq.to_csv(
    os.path.join(table_dir, "density_frequencies.csv"),
    header=["Frequency"]
)

# ============================================================
# HISTOGRAMS
# ============================================================

for e in ["S", "D", "T", "Q"]:

    if len(densities[e]) == 0:
        continue

    plt.figure(figsize=(6,4))

    plt.hist(densities[e], bins="auto")

    plt.title(f"{e} Event Density Histogram")

    plt.xlabel("Density")

    plt.ylabel("Frequency")

    plt.grid(True)

    filename_out = os.path.join(
        plot_dir,
        f"{e}_histogram.png"
    )

    plt.savefig(filename_out,
                dpi=300,
                bbox_inches="tight")

    plt.close()

# ============================================================
# COMBINED HISTOGRAMS (2 x 2)
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

events = ["S", "D", "T", "Q"]
titles = ["Singles", "Doubles", "Triples", "Quadruples"]

for ax, e, title in zip(axs.flatten(), events, titles):

    if len(densities[e]) > 0:

        ax.hist(densities[e], bins='auto')

        ax.set_title(title)

        ax.set_xlabel("Density")

        ax.set_ylabel("Frequency")

        ax.grid(True)

    else:
        ax.set_title(title + "\n(No Data)")

plt.suptitle("Event Density Histograms", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])

combined_file = os.path.join(
    plot_dir,
    "All_Event_Histograms.png"
)

plt.savefig(
    combined_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Combined histogram saved:")
print(combined_file)

print()
print("Histogram images saved in:")
print(plot_dir)

print()
print("CSV tables saved in:")
print(table_dir)

print()
print("Finished.")