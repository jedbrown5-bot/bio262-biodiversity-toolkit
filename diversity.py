"""
diversity.py
============
Species-diversity indices for teaching (BIO262).

Works identically on **abundance counts** (individuals per species) or
**percent-cover** data (e.g. Braun-Blanquet / point-intercept cover), because
every index is computed from the relative proportions p_i = x_i / sum(x).

Indices
-------
- S            observed species richness
- H'           Shannon diversity (natural log)          -Σ p_i ln p_i
- exp(H')      Shannon effective number of species      e^{H'}
- D            Simpson's index (dominance)              Σ p_i^2
- 1 - D        Gini-Simpson diversity                   1 - Σ p_i^2
- 1 / D        Inverse Simpson (effective species)      1 / Σ p_i^2
- J'           Pielou's evenness                        H' / ln(S)

All are standard (Magurran 2004; Jost 2006).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Diversity:
    S: int
    total: float          # total individuals or total cover
    shannon_H: float      # H'
    shannon_exp: float    # e^{H'}
    simpson_D: float      # Σ p^2  (dominance)
    gini_simpson: float   # 1 - D
    inv_simpson: float    # 1 / D
    pielou_J: float       # evenness (nan if S < 2)

    def as_row(self) -> dict:
        return asdict(self)


def _proportions(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    total = x.sum()
    if total <= 0:
        raise ValueError("All values are zero or missing; nothing to compute.")
    return x / total


def diversity_from_vector(x) -> Diversity:
    """Compute all indices for one sample (a vector of counts or cover values)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    total = float(x[x > 0].sum())
    p = _proportions(x)
    S = int(p.size)

    H = float(-np.sum(p * np.log(p)))          # natural-log Shannon
    H = max(H, 0.0)
    shannon_exp = math.exp(H)
    D = float(np.sum(p ** 2))
    gini = 1.0 - D
    inv = 1.0 / D if D > 0 else math.inf
    J = H / math.log(S) if S > 1 else float("nan")

    return Diversity(S=S, total=total, shannon_H=H, shannon_exp=shannon_exp,
                     simpson_D=D, gini_simpson=gini, inv_simpson=inv, pielou_J=J)


def diversity_table(matrix: np.ndarray, sample_names=None,
                    include_pooled: bool = True):
    """Per-sample indices for a species x samples matrix.

    Returns a list of (name, Diversity) tuples. Columns are samples/sites.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    n_samples = mat.shape[1]
    if sample_names is None:
        sample_names = [f"Sample {i+1}" for i in range(n_samples)]

    rows = []
    for j, name in enumerate(sample_names):
        col = mat[:, j]
        if col[col > 0].size == 0:
            continue                       # skip empty columns
        rows.append((name, diversity_from_vector(col)))

    if include_pooled and n_samples > 1:
        pooled = mat.sum(axis=1)
        rows.append(("Pooled (all samples)", diversity_from_vector(pooled)))

    return rows
