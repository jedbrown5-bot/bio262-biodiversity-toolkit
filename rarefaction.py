"""
rarefaction.py
==============
Individual-based and sample-based rarefaction, extrapolation and asymptotic
richness estimators, following:

    Colwell, R. K., A. Chao, N. J. Gotelli, S.-Y. Lin, C. X. Mao, R. L. Chazdon,
    & J. T. Longino. 2012. Models and estimators linking individual-based and
    sample-based rarefaction, extrapolation and comparison of assemblages.
    Journal of Plant Ecology 5:3-21.

    Colwell, R. K., C. X. Mao, & J. Chang. 2004. Interpolating, extrapolating,
    and comparing incidence-based species accumulation curves. Ecology 85:2717-2727.

This is a clean-room re-implementation of the core mathematics behind EstimateS
(R. K. Colwell). Equation numbers below refer to Colwell et al. (2012).

Point estimates (Eqs. 4, 9, 17, 18) are exact. Rarefaction unconditional
variances follow Eq. 5 (individual) and its incidence analogue (Colwell et al.
2004, Eq. 6). Extrapolation variances follow the delta method of Eqs. 10 & 19.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

Z95 = 1.959963984540054  # 1.96, standard-normal 0.975 quantile


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _log_binom(a: float, b: float) -> float:
    """log C(a, b) via log-gamma; returns -inf when undefined (b<0 or b>a)."""
    if b < 0 or b > a:
        return -math.inf
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def _alpha(N: int, k: int, m: int) -> float:
    """
    a_{km} = C(N-k, m) / C(N, m)  -- probability that a species detected
    exactly k times (out of N total draws) is *absent* from a random subset of
    size m. Returns 0 when m > N-k. (Colwell et al. 2012, def. under Eq. 4.)
    """
    if m > N - k:
        return 0.0
    if m <= 0:
        return 1.0
    return math.exp(_log_binom(N - k, m) - _log_binom(N, m))


# --------------------------------------------------------------------------- #
#  Frequency-count structures
# --------------------------------------------------------------------------- #
@dataclass
class Counts:
    """Frequency-count summary shared by both data types.

    f[k] = number of species detected exactly k times.
    For individual-based data k counts individuals (total N individuals).
    For sample-based data k counts sampling-unit occurrences (T sampling units).
    """

    N: int                 # n (individuals) or T (sampling units)
    S_obs: int             # observed species richness
    f: dict                # {k: number of species with frequency k}, k >= 1

    @property
    def f1(self) -> int:
        return self.f.get(1, 0)

    @property
    def f2(self) -> int:
        return self.f.get(2, 0)


def counts_from_abundances(abund: Sequence[float]) -> Counts:
    """Individual-based: abundance vector -> Counts (over individuals)."""
    x = np.asarray([a for a in abund], dtype=float)
    x = np.rint(x).astype(int)              # EstimateS treats data as integer counts
    x = x[x > 0]
    if x.size == 0:
        raise ValueError("No positive abundances supplied.")
    N = int(x.sum())
    S_obs = int(x.size)
    f: dict[int, int] = {}
    for k in x:
        f[int(k)] = f.get(int(k), 0) + 1
    return Counts(N=N, S_obs=S_obs, f=f)


def counts_from_incidence(matrix: np.ndarray) -> Counts:
    """Sample-based: species x samples matrix -> Counts (over sampling units).

    Entries may be abundances or 0/1; anything > 0 counts as a presence.
    Y_i = number of sampling units in which species i occurs.
    Q_k = number of species with Y_i == k.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("Incidence data must be a 2-D species x samples matrix.")
    T = mat.shape[1]
    presence = (mat > 0).astype(int)
    Y = presence.sum(axis=1)               # incidence frequency per species
    Y = Y[Y > 0]
    if Y.size == 0:
        raise ValueError("No species present in the matrix.")
    S_obs = int(Y.size)
    Q: dict[int, int] = {}
    for k in Y:
        Q[int(k)] = Q.get(int(k), 0) + 1
    return Counts(N=int(T), S_obs=S_obs, f=Q)


# --------------------------------------------------------------------------- #
#  Asymptotic richness estimators (extrapolation targets)
# --------------------------------------------------------------------------- #
def chao1(c: Counts) -> float:
    """Chao1 asymptotic richness for abundance data (Eq. 15a/15b)."""
    f1, f2 = c.f1, c.f2
    if f2 > 0:
        f0 = f1 * f1 / (2.0 * f2)
    else:
        f0 = f1 * (f1 - 1) / 2.0            # bias-corrected form, f2 == 0
    return c.S_obs + f0


def chao2(c: Counts) -> float:
    """Chao2 asymptotic richness for incidence data (Eq. 21/22)."""
    Q1, Q2, T = c.f1, c.f2, c.N
    corr = (T - 1) / T if T > 0 else 1.0
    if Q2 > 0:
        Q0 = corr * Q1 * Q1 / (2.0 * Q2)
    else:
        Q0 = corr * Q1 * (Q1 - 1) / 2.0
    return c.S_obs + Q0


def _f0_estimate(c: Counts, incidence: bool) -> float:
    """Undetected-species estimate used as the extrapolation asymptote."""
    return (chao2(c) if incidence else chao1(c)) - c.S_obs


# --------------------------------------------------------------------------- #
#  Rarefaction (interpolation)  --  Eqs. 4, 5, 17
# --------------------------------------------------------------------------- #
def _rarefy_point(c: Counts, m: int) -> float:
    """Expected species in a random subsample of size m (Eq. 4 / Eq. 17)."""
    N, S = c.N, c.S_obs
    if m <= 0:
        return 0.0
    if m >= N:
        return float(S)
    total = 0.0
    for k, fk in c.f.items():
        total += _alpha(N, k, m) * fk
    return S - total


def _rarefy_var(c: Counts, m: int, S_est: float) -> float:
    """Unconditional variance of rarefied richness (Eq. 5 / Colwell 2004 Eq. 6).

        var = sum_k (1 - a_{km})^2 f_k  -  Shat(m)^2 / S_est
    """
    N = c.N
    if m <= 0:
        return 0.0
    m = min(m, N)
    s_hat = _rarefy_point(c, m)
    acc = 0.0
    for k, fk in c.f.items():
        a = _alpha(N, k, m)
        acc += (1.0 - a) ** 2 * fk
    var = acc - (s_hat ** 2) / S_est if S_est > 0 else acc
    return max(var, 0.0)


# --------------------------------------------------------------------------- #
#  Extrapolation  --  Eqs. 9, 10, 18, 19
# --------------------------------------------------------------------------- #
def _extrap_point_from_f(f_arr: np.ndarray, ks: np.ndarray, N: int,
                         extra: int, incidence: bool) -> float:
    """Extrapolated richness for N+extra draws, as a function of the frequency
    vector f_arr (aligned with frequency classes ks). Eqs. 9 (abundance) & 18
    (incidence). Written to accept a perturbed f_arr so the delta-method
    gradient can be obtained by finite differences.
    """
    S_obs = float(f_arr.sum())
    f1 = float(f_arr[ks == 1].sum()) if (ks == 1).any() else 0.0
    f2 = float(f_arr[ks == 2].sum()) if (ks == 2).any() else 0.0

    if incidence:
        corr = (N - 1) / N if N > 0 else 1.0
        f0 = corr * f1 * f1 / (2.0 * f2) if f2 > 0 else corr * f1 * (f1 - 1) / 2.0
    else:
        f0 = f1 * f1 / (2.0 * f2) if f2 > 0 else f1 * (f1 - 1) / 2.0

    if extra <= 0:
        return S_obs
    if f0 <= 0 or f1 <= 0:
        return S_obs  # nothing left to discover

    if incidence:
        # Eq. 18: rate = Q1 / (Q1 + T*Q0)
        rate = f1 / (f1 + N * f0)
    else:
        # Eq. 9: rate = f1 / (n * f0)
        rate = f1 / (N * f0)
    return S_obs + f0 * (1.0 - (1.0 - rate) ** extra)


def _extrap_var(c: Counts, extra: int, incidence: bool) -> float:
    """Delta-method variance of extrapolated richness (Eqs. 10 & 19).

    With cov(f_i,f_i) = f_i(1 - f_i/(S_obs+f0)) and
         cov(f_i,f_j) = -f_i f_j/(S_obs+f0),
    the quadratic form collapses to
        var = sum_i d_i^2 f_i  -  (sum_i d_i f_i)^2 / (S_obs + f0)
    where d_i = dShat/df_i is obtained by finite differences.
    """
    ks = np.array(sorted(c.f.keys()), dtype=float)
    f_arr = np.array([c.f[int(k)] for k in ks], dtype=float)
    f0 = _f0_estimate(c, incidence)
    denom = c.S_obs + f0
    if extra <= 0 or denom <= 0:
        return 0.0

    base = _extrap_point_from_f(f_arr, ks, c.N, extra, incidence)
    d = np.zeros_like(f_arr)
    for i in range(len(f_arr)):
        h = 1e-4
        fp = f_arr.copy(); fp[i] += h
        d[i] = (_extrap_point_from_f(fp, ks, c.N, extra, incidence) - base) / h

    var = float((d ** 2 * f_arr).sum() - (d * f_arr).sum() ** 2 / denom)
    return max(var, 0.0)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
@dataclass
class Curve:
    x: np.ndarray          # sample size (individuals or samples)
    y: np.ndarray          # expected richness
    lo: np.ndarray         # lower 95% CI
    hi: np.ndarray         # upper 95% CI
    sd: np.ndarray         # standard deviation
    is_extrap: np.ndarray  # boolean: point is extrapolated
    reference_x: int       # x of the reference sample (rarefaction/extrapolation boundary)
    reference_y: float     # observed richness at the reference sample
    kind: str              # 'individual' or 'sample'
    estimator: float       # Chao1 (individual) or Chao2 (sample)


def compute_curve(c: Counts,
                  incidence: bool,
                  extrapolate_to: int | None = None,
                  n_knots: int = 60,
                  ci: float = 0.95) -> Curve:
    """Full rarefaction (+ optional extrapolation) curve with CIs.

    Parameters
    ----------
    c : Counts
    incidence : bool          sample-based (True) or individual-based (False)
    extrapolate_to : int      final x value (>= reference size) to extrapolate to;
                              None or <= N disables extrapolation
    n_knots : int             approximate number of evaluation points
    ci : float                confidence level (default 0.95)
    """
    N = c.N
    S_est = chao2(c) if incidence else chao1(c)
    z = Z95 if abs(ci - 0.95) < 1e-9 else float(_norm_ppf(0.5 + ci / 2.0))

    end = N if (extrapolate_to is None or extrapolate_to <= N) else int(extrapolate_to)

    # Choose evenly spaced integer knots, always including 1, N and end.
    knots = _make_knots(N, end, n_knots)

    ks_sorted = np.array(sorted(c.f), dtype=float)
    f_sorted = np.array([c.f[int(k)] for k in ks_sorted], dtype=float)

    xs, ys, sds, extern = [], [], [], []
    for x in knots:
        if x <= N:
            y = _rarefy_point(c, x)
            var = _rarefy_var(c, x, S_est)
            ext = False
        else:
            extra = x - N
            y = _extrap_point_from_f(f_sorted, ks_sorted, N, extra, incidence)
            var = _extrap_var(c, extra, incidence)
            ext = True
        xs.append(x); ys.append(y); sds.append(math.sqrt(max(var, 0.0)))
        extern.append(ext)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    sds = np.array(sds, dtype=float)
    extern = np.array(extern, dtype=bool)
    lo = np.maximum(ys - z * sds, 0.0)
    hi = ys + z * sds

    return Curve(x=xs, y=ys, lo=lo, hi=hi, sd=sds, is_extrap=extern,
                 reference_x=N, reference_y=float(c.S_obs),
                 kind="sample" if incidence else "individual",
                 estimator=S_est)


def _make_knots(N: int, end: int, n_knots: int) -> list[int]:
    pts = {1, N, end}
    if n_knots >= end:
        pts.update(range(1, end + 1))
    else:
        step = max(1, round(end / n_knots))
        pts.update(range(1, end + 1, step))
        pts.add(N)  # guarantee the reference boundary is present
    return sorted(p for p in pts if 1 <= p <= end)


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
