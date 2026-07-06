"""
anova.py
========
One-way ANOVA and companions for teaching (BIO262):

- one_way_anova   : SS / df / MS / F / p summary table (+ eta^2)
- tukey_hsd       : all pairwise Tukey HSD comparisons
- levene_test     : homogeneity-of-variance assumption
- shapiro_test    : normality of residuals assumption
- kruskal_wallis  : non-parametric alternative to ANOVA
- dunn_test       : non-parametric pairwise post-hoc (Bonferroni-adjusted)

Groups are supplied as a list of 1-D arrays (one per group) plus their labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
#  One-way ANOVA
# --------------------------------------------------------------------------- #
@dataclass
class AnovaResult:
    F: float
    p: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ss_total: float
    ms_between: float
    ms_within: float
    eta_sq: float          # effect size (proportion of variance explained)
    grand_mean: float
    group_means: dict
    group_ns: dict


def one_way_anova(groups, labels) -> AnovaResult:
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    all_vals = np.concatenate(groups)
    N = all_vals.size
    k = len(groups)
    grand = all_vals.mean()

    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_total = ((all_vals - grand) ** 2).sum()

    df_b = k - 1
    df_w = N - k
    ms_b = ss_between / df_b if df_b > 0 else float("nan")
    ms_w = ss_within / df_w if df_w > 0 else float("nan")
    F = ms_b / ms_w if ms_w > 0 else float("nan")
    p = float(stats.f.sf(F, df_b, df_w)) if np.isfinite(F) else float("nan")
    eta = ss_between / ss_total if ss_total > 0 else float("nan")

    return AnovaResult(
        F=float(F), p=p, df_between=df_b, df_within=df_w,
        ss_between=float(ss_between), ss_within=float(ss_within),
        ss_total=float(ss_total), ms_between=float(ms_b), ms_within=float(ms_w),
        eta_sq=float(eta), grand_mean=float(grand),
        group_means={lab: float(g.mean()) for lab, g in zip(labels, groups)},
        group_ns={lab: int(g.size) for lab, g in zip(labels, groups)})


# --------------------------------------------------------------------------- #
#  Post-hoc: Tukey HSD (parametric)
# --------------------------------------------------------------------------- #
def tukey_hsd(groups, labels):
    """All pairwise Tukey HSD comparisons.

    Returns list of dicts: group1, group2, meandiff, p_adj, lower, upper, reject.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    res = stats.tukey_hsd(*groups)
    ci = res.confidence_interval(confidence_level=0.95)
    out = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            out.append({
                "group1": labels[i],
                "group2": labels[j],
                "meandiff": float(groups[j].mean() - groups[i].mean()),
                "p_adj": float(res.pvalue[i, j]),
                "ci_low": float(ci.low[i, j]),
                "ci_high": float(ci.high[i, j]),
                "significant": bool(res.pvalue[i, j] < 0.05),
            })
    return out


# --------------------------------------------------------------------------- #
#  Assumption checks
# --------------------------------------------------------------------------- #
def levene_test(groups):
    """Levene's test for equal variances (uses median centre = Brown-Forsythe)."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    stat, p = stats.levene(*groups, center="median")
    return {"statistic": float(stat), "p": float(p),
            "equal_variances": bool(p >= 0.05)}


def shapiro_test(groups):
    """Shapiro-Wilk normality test on the pooled within-group residuals."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    resid = np.concatenate([g - g.mean() for g in groups])
    if resid.size < 3:
        return {"statistic": float("nan"), "p": float("nan"),
                "normal": None, "note": "Too few observations to test."}
    stat, p = stats.shapiro(resid)
    return {"statistic": float(stat), "p": float(p), "normal": bool(p >= 0.05)}


# --------------------------------------------------------------------------- #
#  Non-parametric: Kruskal-Wallis + Dunn's post-hoc
# --------------------------------------------------------------------------- #
def kruskal_wallis(groups, labels):
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    H, p = stats.kruskal(*groups)
    # epsilon^2 effect size
    N = sum(g.size for g in groups)
    eps2 = (H - len(groups) + 1) / (N - len(groups)) if N - len(groups) > 0 else float("nan")
    return {"H": float(H), "p": float(p), "df": len(groups) - 1,
            "epsilon_sq": float(eps2)}


def dunn_test(groups, labels):
    """Dunn's (1964) pairwise test with tie correction and Bonferroni adjustment.

    Returns list of dicts: group1, group2, z, p_adj, significant.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    ns = [g.size for g in groups]
    all_vals = np.concatenate(groups)
    N = all_vals.size

    ranks = stats.rankdata(all_vals)
    idx = np.cumsum([0] + ns)
    grp_ranks = [ranks[idx[i]:idx[i + 1]] for i in range(len(groups))]
    mean_rank = [r.mean() for r in grp_ranks]

    _, counts = np.unique(all_vals, return_counts=True)
    tie = np.sum(counts ** 3 - counts)
    sigma2 = (N * (N + 1) / 12.0) - tie / (12.0 * (N - 1))

    k = len(groups)
    n_comp = k * (k - 1) // 2
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(sigma2 * (1.0 / ns[i] + 1.0 / ns[j]))
            z = (mean_rank[i] - mean_rank[j]) / se if se > 0 else 0.0
            p_raw = 2.0 * stats.norm.sf(abs(z))
            p_adj = min(1.0, p_raw * n_comp)     # Bonferroni
            out.append({
                "group1": labels[i],
                "group2": labels[j],
                "z": float(z),
                "p_adj": float(p_adj),
                "significant": bool(p_adj < 0.05),
            })
    return out
