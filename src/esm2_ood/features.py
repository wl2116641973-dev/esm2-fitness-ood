"""Position-agnostic 44-dim amino acid substitution features (WT + Mut + BLOSUM62 + 3 physicochemical)."""

from __future__ import annotations

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Kyte-Doolittle hydropathy index
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Residue volume in A^3 (Zamyatnin, 1972)
VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

# Approximate side-chain formal charge at pH 7
CHARGE = {
    "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, "F": 0.0,
    "G": 0.0, "H": 0.1, "I": 0.0, "K": 1.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}

# Standard BLOSUM62 lower-triangular rows in AA_ALPHABET order: A C D E F G H I K L M N P Q R S T V W Y
_BLOSUM62_ROWS = [
    [4],
    [0, 9],
    [-2, -3, 6],
    [-1, -4, 2, 5],
    [-2, -2, -3, -3, 6],
    [0, -3, -1, -2, -3, 6],
    [-2, -3, -1, 0, -1, -2, 8],
    [-1, -1, -3, -3, 0, -4, -3, 4],
    [-1, -3, -1, 1, -3, -2, -1, -3, 5],
    [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4],
    [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5],
    [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6],
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7],
    [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5],
    [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5],
    [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4],
    [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5],
    [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4],
    [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11],
    [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
]

BLOSUM62 = np.zeros((20, 20), dtype=np.float32)
for i, row in enumerate(_BLOSUM62_ROWS):
    for j, val in enumerate(row):
        BLOSUM62[i, j] = val
        BLOSUM62[j, i] = val


def featurize_aa_substitution(df: pd.DataFrame) -> np.ndarray:
    """Build 44-dim position-agnostic substitution features:
    [20 WT one-hot | 20 Mut one-hot | BLOSUM62 | |dHydro| | |dVol| | |dCharge|].
    """
    n = len(df)
    X = np.zeros((n, 44), dtype=np.float32)
    for row_idx, (wt, mut) in enumerate(zip(df["wt_aa"], df["mut_aa"])):
        i = AA_TO_IDX[wt]
        j = AA_TO_IDX[mut]
        X[row_idx, i] = 1.0
        X[row_idx, 20 + j] = 1.0
        X[row_idx, 40] = BLOSUM62[i, j]
        X[row_idx, 41] = abs(HYDROPATHY[mut] - HYDROPATHY[wt])
        X[row_idx, 42] = abs(VOLUME[mut] - VOLUME[wt])
        X[row_idx, 43] = abs(CHARGE[mut] - CHARGE[wt])
    return X
