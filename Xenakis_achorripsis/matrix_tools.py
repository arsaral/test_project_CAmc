# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:25:16 2026

@author: USER
"""

# -*- coding: utf-8 -*-
"""
=========================================================
matrix_tools.py

Common routines for Matrix M analysis

Author:
    Ali Saral

=========================================================
"""

import os
import re
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# READ MATRIX
# ----------------------------------------------------------

def read_matrix(filename):

    """
    Reads an Excel matrix.

    First row     : column headers
    First column  : row names

    Returns

        df
        ROWS
        COLS
    """

    df = pd.read_excel(filename, index_col=0)

    rows, cols = df.shape

    return df, rows, cols


# ----------------------------------------------------------
# GET EVENT TYPE
# ----------------------------------------------------------

def get_event(cell):

    if pd.isna(cell):
        return None

    text = str(cell).strip().upper()

    if len(text) == 0:
        return None

    if text[0] in "SDTQ":

        return text[0]

    return None


# ----------------------------------------------------------
# GET DENSITY
# ----------------------------------------------------------

def get_density(cell):

    if pd.isna(cell):
        return None

    text = str(cell)

    m = re.search(r'(\d+(\.\d+)?)', text)

    if m:

        return float(m.group(1))

    return None


# ----------------------------------------------------------
# EVENT CODE
# ----------------------------------------------------------

EVENT_CODE = {

    "S":1,
    "D":2,
    "T":3,
    "Q":4

}


# ----------------------------------------------------------
# CREATE MATRICES
# ----------------------------------------------------------

def build_matrices(df):

    """
    Returns

    occupancy
    density
    event_type
    """

    rows, cols = df.shape

    occupancy = np.zeros((rows, cols))

    density = np.zeros((rows, cols))

    event_type = np.zeros((rows, cols))

    for r in range(rows):

        for c in range(cols):

            cell = df.iat[r, c]

            e = get_event(cell)

            d = get_density(cell)

            if e is not None:

                occupancy[r, c] = 1

                density[r, c] = d

                event_type[r, c] = EVENT_CODE[e]

    return occupancy, density, event_type


# ----------------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# ----------------------------------------------------------

def create_output(outdir):

    tabledir = os.path.join(outdir, "tables")

    plotdir = os.path.join(outdir, "plots")

    os.makedirs(tabledir, exist_ok=True)

    os.makedirs(plotdir, exist_ok=True)

    return tabledir, plotdir


# ----------------------------------------------------------
# ROW STATISTICS
# ----------------------------------------------------------

def row_statistics(df):

    results = []

    for r in range(df.shape[0]):

        densities = []

        occupied = 0

        for c in range(df.shape[1]):

            d = get_density(df.iat[r, c])

            if d is not None:

                occupied += 1

                densities.append(d)

        total = sum(densities)

        mean = total / occupied if occupied else 0

        minimum = min(densities) if occupied else 0

        maximum = max(densities) if occupied else 0

        results.append([

            df.index[r],

            occupied,

            total,

            mean,

            minimum,

            maximum

        ])

    return pd.DataFrame(

        results,

        columns=[

            "Row",

            "Occupied",

            "Total Density",

            "Mean Density",

            "Minimum Density",

            "Maximum Density"

        ]

    )


# ----------------------------------------------------------
# COLUMN STATISTICS
# ----------------------------------------------------------

def column_statistics(df):

    results = []

    for c in range(df.shape[1]):

        densities = []

        occupied = 0

        for r in range(df.shape[0]):

            d = get_density(df.iat[r, c])

            if d is not None:

                occupied += 1

                densities.append(d)

        total = sum(densities)

        mean = total / occupied if occupied else 0

        minimum = min(densities) if occupied else 0

        maximum = max(densities) if occupied else 0

        results.append([

            df.columns[c],

            occupied,

            total,

            mean,

            minimum,

            maximum

        ])

    return pd.DataFrame(

        results,

        columns=[

            "Column",

            "Occupied",

            "Total Density",

            "Mean Density",

            "Minimum Density",

            "Maximum Density"

        ]

    )