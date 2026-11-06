"""Statistics across seeds.

Each seed is one independently initialised and trained network, so seeds are
the unit of replication: confidence intervals and tests are computed over
seeds, never over trials (trials from one network are not independent
evidence about "the model").

With only 5 seeds, tests are weak. That is deliberate honesty, not a flaw to
fix by counting trials as replicates.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-distribution confidence interval. With 1 value the CI is undefined (NaN)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    half = _st.t.ppf(0.5 + level / 2, df=len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - half, m + half


def t_test(diffs, alternative: str = "two-sided") -> float:
    """One-sample t-test of paired differences against 0, over seeds.

    If every seed shows exactly the same difference (zero spread), the
    t-statistic is undefined; we return p = 0 for a non-zero difference in the
    tested direction and p = 1 otherwise.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    if np.allclose(d, d[0]):
        m = d[0]
        if m == 0:
            return 1.0
        if alternative == "greater":
            return 0.0 if m > 0 else 1.0
        if alternative == "less":
            return 0.0 if m < 0 else 1.0
        return 0.0
    return float(_st.ttest_1samp(d, 0.0, alternative=alternative).pvalue)


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the chance of ANY false positive).

    Sort p-values; multiply the smallest by m, the next by m-1, ...; keep the
    sequence non-decreasing; cap at 1. NaNs stay NaN and are not counted.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out
