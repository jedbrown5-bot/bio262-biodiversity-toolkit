"""Verify rarefaction math against Monte Carlo resampling and hand checks."""
import numpy as np
import rarefaction as R

rng = np.random.default_rng(42)


def mc_individual(abund, m, reps=20000):
    """Monte Carlo expected species in m individuals (without replacement)."""
    pool = np.repeat(np.arange(len(abund)), abund)
    out = []
    for _ in range(reps):
        pick = rng.choice(pool, size=m, replace=False)
        out.append(len(np.unique(pick)))
    return np.mean(out), np.std(out)


def mc_sample(mat, t, reps=20000):
    """Monte Carlo expected species in t sampling units (without replacement)."""
    T = mat.shape[1]
    pres = (mat > 0)
    out = []
    for _ in range(reps):
        cols = rng.choice(T, size=t, replace=False)
        out.append(int((pres[:, cols].any(axis=1)).sum()))
    return np.mean(out), np.std(out)


print("=" * 64)
print("TEST 1  Individual-based point estimates vs Monte Carlo")
print("=" * 64)
abund = [40, 20, 15, 10, 8, 5, 3, 3, 2, 2, 1, 1, 1, 1]
c = R.counts_from_abundances(abund)
print(f"n={c.N}, S_obs={c.S_obs}, f1={c.f1}, f2={c.f2}, Chao1={R.chao1(c):.2f}")
for m in [1, 5, 20, 50, c.N]:
    y = R._rarefy_point(c, m)
    mc, mcsd = mc_individual(abund, m)
    print(f"  m={m:3d}  formula={y:7.4f}   MC={mc:7.4f}   diff={abs(y-mc):.4f}")

print()
print("=" * 64)
print("TEST 2  Sample-based (Mao Tau) point estimates vs Monte Carlo")
print("=" * 64)
# 12 species x 8 samples incidence matrix
mat = np.array([
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0],
    [1,1,1,1,1,0,0,0],
    [1,1,1,1,0,0,0,0],
    [1,1,1,0,0,0,0,0],
    [1,1,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,1],
])
c2 = R.counts_from_incidence(mat)
print(f"T={c2.N}, S_obs={c2.S_obs}, Q1={c2.f1}, Q2={c2.f2}, Chao2={R.chao2(c2):.2f}")
for t in [1, 2, 4, 6, c2.N]:
    y = R._rarefy_point(c2, t)
    mc, mcsd = mc_sample(mat, t)
    print(f"  t={t}  formula={y:7.4f}   MC={mc:7.4f}   diff={abs(y-mc):.4f}")

print()
print("=" * 64)
print("TEST 3  Hand check: 2 species, [1,1]; m=1 -> expect 1.0 species")
print("=" * 64)
c3 = R.counts_from_abundances([1, 1])
print(f"  _rarefy_point(m=1) = {R._rarefy_point(c3,1)} (expect 1.0)")
print(f"  _rarefy_point(m=2) = {R._rarefy_point(c3,2)} (expect 2.0)")

print()
print("=" * 64)
print("TEST 4  Boundary continuity: rarefaction end == extrapolation start")
print("=" * 64)
cur = R.compute_curve(c, incidence=False, extrapolate_to=2*c.N, n_knots=40)
# value at reference should equal S_obs
at_ref = cur.y[cur.x == c.N][0]
print(f"  curve at reference n={c.N}: y={at_ref:.4f} (expect S_obs={c.S_obs})")
print(f"  extrapolation approaches Chao1={R.chao1(c):.2f}; "
      f"y at 2n={cur.y[-1]:.3f}")
print(f"  SD at reference (unconditional, should be > 0): {cur.sd[cur.x==c.N][0]:.4f}")

print()
print("=" * 64)
print("TEST 5  Extrapolation delta-method var vs analytic collapse form")
print("=" * 64)
# var = sum d_i^2 f_i - (sum d_i f_i)^2/(S_obs+f0). Compare to a bootstrap.
def boot_extrap(abund, extra, reps=4000):
    pool = np.repeat(np.arange(len(abund)), abund)
    n = len(pool)
    est = []
    for _ in range(reps):
        bs = rng.choice(pool, size=n, replace=True)
        vals, cnts = np.unique(bs, return_counts=True)
        cc = R.counts_from_abundances(list(cnts))
        ks = np.array(sorted(cc.f), float); fs = np.array([cc.f[int(k)] for k in ks], float)
        est.append(R._extrap_point_from_f(fs, ks, cc.N, extra, False))
    return np.std(est)

extra = c.N
var_delta = R._extrap_var(c, extra, incidence=False)
sd_delta = var_delta ** 0.5
sd_boot = boot_extrap(abund, extra)
print(f"  extra={extra}: delta-method SD={sd_delta:.3f}   bootstrap SD={sd_boot:.3f}")
print("  (same order of magnitude expected; delta method is an approximation)")

print("\nAll tests executed.")
