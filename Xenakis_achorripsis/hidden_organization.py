# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ali Rıza Saral
# Licensed under the MIT License.
"""
Created on Thu Aug  6 14:19:13 2026

@author: Ali Rıza SARAL
"""

# -*- coding: utf-8 -*-
"""
=========================================================
Program 4 : Hidden Organization
=========================================================

Input:
    Any Excel matrix

Each occupied cell should contain

    S5
    D10
    T15
    Q20

Empty cells are left blank.

Outputs

hidden_organization/

    tables/
        event_distribution.csv
        row_column_summary.csv
        summary.txt

    plots/
        occupancy_heatmap.png
        density_heatmap.png
        eventtype_heatmap.png

=========================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# INPUT FILE
# ----------------------------------------------------------

INPUT_FILE = "Matrix_M.xlsx"

# ----------------------------------------------------------
# OUTPUT DIRECTORIES
# ----------------------------------------------------------

OUTDIR = "hidden_organization"

TABLEDIR = os.path.join(OUTDIR, "tables")
PLOTDIR = os.path.join(OUTDIR, "plots")

os.makedirs(TABLEDIR, exist_ok=True)
os.makedirs(PLOTDIR, exist_ok=True)

# ----------------------------------------------------------
# READ EXCEL
# ----------------------------------------------------------

df = pd.read_excel(INPUT_FILE, index_col=0)

ROWS, COLS = df.shape

print("Rows :", ROWS)
print("Cols :", COLS)

# ----------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------

def get_event(cell):

    if pd.isna(cell):
        return None

    text = str(cell).strip().upper()

    if text == "":
        return None

    if text[0] in "SDTQ":
        return text[0]

    return None


def get_density(cell):

    if pd.isna(cell):
        return None

    text = str(cell)

    m = re.search(r'(\d+(\.\d+)?)', text)

    if m:
        return float(m.group(1))

    return None

# ----------------------------------------------------------
# MATRICES
# ----------------------------------------------------------

occupancy = np.zeros((ROWS, COLS))

density = np.zeros((ROWS, COLS))

event_type = np.zeros((ROWS, COLS))

event_code = {

    "S":1,
    "D":2,
    "T":3,
    "Q":4

}

# ----------------------------------------------------------
# FILL MATRICES
# ----------------------------------------------------------

for r in range(ROWS):

    for c in range(COLS):

        cell = df.iat[r,c]

        e = get_event(cell)

        d = get_density(cell)

        if e is not None:

            occupancy[r,c] = 1

            density[r,c] = d

            event_type[r,c] = event_code[e]

# ----------------------------------------------------------
# EVENT DISTRIBUTION
# ----------------------------------------------------------

rows=[]

for r in range(ROWS):

    S=D=T=Q=0

    for c in range(COLS):

        e=get_event(df.iat[r,c])

        if e=="S":
            S+=1

        elif e=="D":
            D+=1

        elif e=="T":
            T+=1

        elif e=="Q":
            Q+=1

    rows.append([

        df.index[r],

        S,D,T,Q,

        S+D+T+Q

    ])

event_df = pd.DataFrame(

    rows,

    columns=[

        "Instrument",

        "S",

        "D",

        "T",

        "Q",

        "Total"

    ]

)

event_df.to_csv(

    os.path.join(

        TABLEDIR,

        "event_distribution.csv"

    ),

    index=False

)

# ----------------------------------------------------------
# ROW/COLUMN SUMMARY
# ----------------------------------------------------------

summary=[]

for r in range(ROWS):

    occ=int(np.sum(occupancy[r]))

    total=np.sum(density[r])

    mean=total/occ if occ>0 else 0

    summary.append([

        df.index[r],

        occ,

        total,

        mean

    ])

summary_df=pd.DataFrame(

    summary,

    columns=[

        "Instrument",

        "Occupied",

        "Total Density",

        "Mean Density"

    ]

)

summary_df.to_csv(

    os.path.join(

        TABLEDIR,

        "row_summary.csv"

    ),

    index=False

)

print(summary_df)

# ----------------------------------------------------------
# COLUMN SUMMARY
# ----------------------------------------------------------

column_summary = []

for c in range(COLS):

    occ = int(np.sum(occupancy[:, c]))

    total = np.sum(density[:, c])

    mean = total / occ if occ > 0 else 0

    column_summary.append([

        df.columns[c],

        occ,

        total,

        mean

    ])

column_df = pd.DataFrame(

    column_summary,

    columns=[

        "Column",

        "Occupied",

        "Total Density",

        "Mean Density"

    ]

)

column_df.to_csv(

    os.path.join(

        TABLEDIR,

        "column_summary.csv"

    ),

    index=False

)

# ----------------------------------------------------------
# HEAT MAP
# OCCUPANCY
# ----------------------------------------------------------

plt.figure(figsize=(12,4))

plt.imshow(
    occupancy,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Occupied")

plt.xticks(range(COLS), df.columns, rotation=90)

plt.yticks(range(ROWS), df.index)

plt.title("Occupancy Heat Map")

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOTDIR,

        "occupancy_heatmap.png"

    ),

    dpi=300

)

plt.close()

# ----------------------------------------------------------
# HEAT MAP
# DENSITY
# ----------------------------------------------------------

plt.figure(figsize=(12,4))

plt.imshow(
    density,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Density")

plt.xticks(range(COLS), df.columns, rotation=90)

plt.yticks(range(ROWS), df.index)

plt.title("Density Heat Map")

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOTDIR,

        "density_heatmap.png"

    ),

    dpi=300

)

plt.close()

# ----------------------------------------------------------
# HEAT MAP
# EVENT TYPE
# ----------------------------------------------------------

plt.figure(figsize=(12,4))

plt.imshow(
    event_type,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="0=Empty 1=S 2=D 3=T 4=Q")

plt.xticks(range(COLS), df.columns, rotation=90)

plt.yticks(range(ROWS), df.index)

plt.title("Event Type Heat Map")

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOTDIR,

        "eventtype_heatmap.png"

    ),

    dpi=300

)

plt.close()

# ----------------------------------------------------------
# GLOBAL STATISTICS
# ----------------------------------------------------------

occupied_cells = int(np.sum(occupancy))

empty_cells = ROWS * COLS - occupied_cells

occupancy_percent = 100 * occupied_cells / (ROWS * COLS)

max_row = summary_df.iloc[summary_df["Total Density"].idxmax()]

min_row = summary_df.iloc[summary_df["Total Density"].idxmin()]

max_col = column_df.iloc[column_df["Total Density"].idxmax()]

min_col = column_df.iloc[column_df["Total Density"].idxmin()]

# ----------------------------------------------------------
# LONGEST EMPTY COLUMN SEQUENCE
# ----------------------------------------------------------

longest_empty = 0
current = 0

for c in range(COLS):

    if np.sum(occupancy[:, c]) == 0:

        current += 1

        longest_empty = max(longest_empty, current)

    else:

        current = 0

# ----------------------------------------------------------
# LONGEST OCCUPIED COLUMN SEQUENCE
# ----------------------------------------------------------

longest_occupied = 0
current = 0

for c in range(COLS):

    if np.sum(occupancy[:, c]) > 0:

        current += 1

        longest_occupied = max(longest_occupied, current)

    else:

        current = 0

# ----------------------------------------------------------
# SUMMARY REPORT
# ----------------------------------------------------------

txtfile = os.path.join(

    TABLEDIR,

    "summary.txt"

)

with open(txtfile, "w", encoding="utf-8") as f:

    f.write("HIDDEN ORGANIZATION\n")
    f.write("===================\n\n")

    f.write(f"Rows                 : {ROWS}\n")
    f.write(f"Columns              : {COLS}\n")
    f.write(f"Total cells          : {ROWS*COLS}\n")
    f.write(f"Occupied cells       : {occupied_cells}\n")
    f.write(f"Empty cells          : {empty_cells}\n")
    f.write(f"Occupancy (%)        : {occupancy_percent:.2f}\n\n")

    f.write(f"Longest empty column sequence     : {longest_empty}\n")
    f.write(f"Longest occupied column sequence  : {longest_occupied}\n\n")

    f.write("ROW WITH MAXIMUM DENSITY\n")
    f.write("------------------------\n")
    f.write(max_row.to_string())
    f.write("\n\n")

    f.write("ROW WITH MINIMUM DENSITY\n")
    f.write("------------------------\n")
    f.write(min_row.to_string())
    f.write("\n\n")

    f.write("COLUMN WITH MAXIMUM DENSITY\n")
    f.write("---------------------------\n")
    f.write(max_col.to_string())
    f.write("\n\n")

    f.write("COLUMN WITH MINIMUM DENSITY\n")
    f.write("---------------------------\n")
    f.write(min_col.to_string())
    f.write("\n")

print()
print("===================================")
print("Hidden Organization completed.")
print("===================================")
print()
print("Tables :", TABLEDIR)
print("Plots  :", PLOTDIR)