# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ali Rıza Saral
# Licensed under the MIT License.
"""
Created on Thu Aug  6 14:00:04 2026

@author: Ali Rıza SARAL
"""

# -*- coding: utf-8 -*-
"""
=========================================================
Program 3 : Temporal Structure
=========================================================

Input:
    Any Excel matrix

Each occupied cell should contain

    S5
    D10
    T15
    Q20

etc.

Empty cells are left blank.

Outputs

temporal_structure/

    tables/
        column_statistics.csv
        summary.txt

    plots/
        density_curve.png
        occupied_cells_curve.png

Author:
    Ali Saral
=========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# INPUT FILE
# ----------------------------------------------------------

INPUT_FILE = "Matrix_M.xlsx"

# ----------------------------------------------------------
# OUTPUT DIRECTORIES
# ----------------------------------------------------------

OUTDIR = "temporal_structure"

TABLEDIR = os.path.join(OUTDIR, "tables")
PLOTDIR  = os.path.join(OUTDIR, "plots")

os.makedirs(TABLEDIR, exist_ok=True)
os.makedirs(PLOTDIR, exist_ok=True)

# ----------------------------------------------------------
# READ EXCEL
# ----------------------------------------------------------

print("Reading:", INPUT_FILE)

df = pd.read_excel(INPUT_FILE, index_col=0)

ROWS, COLS = df.shape

print("Rows   :", ROWS)
print("Columns:", COLS)

# ----------------------------------------------------------
# EXTRACT DENSITY
# ----------------------------------------------------------

def get_density(cell):

    if pd.isna(cell):
        return None

    text = str(cell).strip()

    if text == "":
        return None

    if text[0] not in "SDTQ":
        return None

    try:
        return float(text[1:])
    except:
        return None


# ----------------------------------------------------------
# COLUMN ANALYSIS
# ----------------------------------------------------------

results = []

for col in range(COLS):

    occupied = 0
    densities = []

    for row in range(ROWS):

        density = get_density(df.iat[row, col])

        if density is not None:

            occupied += 1
            densities.append(density)

    if occupied > 0:

        total_density = sum(densities)

        mean_density = total_density / occupied

        minimum_density = min(densities)

        maximum_density = max(densities)

    else:

        total_density = 0

        mean_density = 0

        minimum_density = 0

        maximum_density = 0

    occupancy = occupied / ROWS * 100

    results.append([

        col + 1,

        occupied,

        occupancy,

        total_density,

        mean_density,

        minimum_density,

        maximum_density

    ])

# ----------------------------------------------------------
# DATAFRAME
# ----------------------------------------------------------

out = pd.DataFrame(

    results,

    columns=[

        "Column",

        "Occupied Cells",

        "Occupancy %",

        "Total Density",

        "Mean Density",

        "Minimum Density",

        "Maximum Density"

    ]

)

print()
print(out)

# ----------------------------------------------------------
# SAVE CSV
# ----------------------------------------------------------

csv_file = os.path.join(

    TABLEDIR,

    "column_statistics.csv"

)

out.to_csv(csv_file, index=False)

# ----------------------------------------------------------
# SUMMARY REPORT
# ----------------------------------------------------------

txt_file = os.path.join(

    TABLEDIR,

    "summary.txt"

)

with open(txt_file, "w") as f:

    f.write("TEMPORAL STRUCTURE\n")
    f.write("==================\n\n")

    f.write(f"Rows               : {ROWS}\n")
    f.write(f"Columns            : {COLS}\n\n")

    f.write(out.to_string(index=False))

# ----------------------------------------------------------
# GRAPH 1
# TOTAL DENSITY
# ----------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(

    out["Column"],

    out["Total Density"],

    marker="o",

    linewidth=2

)

plt.grid(True)

plt.xlabel("Column")

plt.ylabel("Total Density")

plt.title("Temporal Density")

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOTDIR,

        "density_curve.png"

    ),

    dpi=300

)

plt.close()

# ----------------------------------------------------------
# GRAPH 2
# OCCUPIED CELLS
# ----------------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(

    out["Column"],

    out["Occupied Cells"],

    marker="o",

    linewidth=2

)

plt.grid(True)

plt.xlabel("Column")

plt.ylabel("Occupied Cells")

plt.title("Occupied Cells per Column")

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOTDIR,

        "occupied_cells_curve.png"

    ),

    dpi=300

)

plt.close()

# ----------------------------------------------------------
# FINISHED
# ----------------------------------------------------------

print()
print("Output written to:")
print(TABLEDIR)
print(PLOTDIR)