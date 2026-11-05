"""Statistics across seeds.

Each seed is one independently initialised and trained network, so seeds are
the unit of replication: confidence intervals are computed over seeds, never
over trials (trials from one network are not independent evidence about
"the model").
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-distribution confidence interval. With 1 value the CI is undefined (NaN)."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    half = _st.t.ppf(0.5 + level / 2, df=len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - half, m + half
