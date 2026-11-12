"""Unit-root and cointegration tests done on the right objects.

Cointegration is a property of I(1) *levels* (log prices). Returns are already stationary, so
running Engle-Granger or Johansen on returns "rejects" almost surely and says nothing about a
long-run relationship. `spurious_returns_demo` shows exactly that failure mode."""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex.*")
warnings.filterwarnings("ignore", message=".*date index has been provided.*")
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order


def unit_root_table(df: pd.DataFrame) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) on levels and first differences."""
    rows = {}
    for c in df:
        for label, s in (("level", df[c]), ("diff", df[c].diff().dropna())):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = adfuller(s, regression="c", autolag="AIC")[1]
                kpss_p = kpss(s, regression="c", nlags="auto")[1]
            verdict = "I(0)" if adf_p < 0.05 and kpss_p > 0.05 else ("unit root" if adf_p > 0.05 and kpss_p < 0.05 else "ambiguous")
            rows[(c, label)] = {"ADF_p": adf_p, "KPSS_p": kpss_p, "verdict": verdict}
    return pd.DataFrame(rows).T


def engle_granger_matrix(logp: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Engle-Granger p-values (row regressed on column)."""
    cols = list(logp)
    m = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            if a != b:
                m.loc[a, b] = coint(logp[a], logp[b], trend="c")[1]
    return m


def johansen_table(logp: pd.DataFrame, k_ar_diff: int = 1, det_order: int = 0) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = coint_johansen(np.asarray(logp, float), det_order, k_ar_diff)
    rows = []
    for r in range(logp.shape[1]):
        rows.append({"H0: rank <=": r, "trace": res.lr1[r], "trace_crit_5%": res.cvt[r, 1],
                     "max_eig": res.lr2[r], "max_eig_crit_5%": res.cvm[r, 1],
                     "reject_trace_5%": res.lr1[r] > res.cvt[r, 1]})
    return pd.DataFrame(rows)


def johansen_rank(logp: pd.DataFrame, k_ar_diff: int = 1) -> int:
    t = johansen_table(logp, k_ar_diff)
    rank = 0
    for _, row in t.iterrows():
        if row["reject_trace_5%"]:
            rank += 1
        else:
            break
    return rank


def fit_vecm(logp: pd.DataFrame, maxlags: int = 10):
    """Choose lag by BIC, rank by Johansen trace test; fit VECM (rank>=1 required)."""
    X = np.asarray(logp, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag = max(1, select_order(X, maxlags=maxlags, deterministic="ci").bic)
        rank = johansen_rank(logp, lag)
        if rank == 0:
            return None, lag, rank
        return VECM(X, k_ar_diff=lag, coint_rank=rank, deterministic="ci").fit(), lag, rank


def rolling_eg_pvalue(a: pd.Series, b: pd.Series, window: int = 500, step: int = 20) -> pd.Series:
    """Engle-Granger p-value on a rolling window: is any relationship stable over time?"""
    idx, vals = [], []
    for end in range(window, len(a) + 1, step):
        idx.append(a.index[end - 1])
        vals.append(coint(a.iloc[end - window:end], b.iloc[end - window:end])[1])
    return pd.Series(vals, index=idx)


def spurious_returns_demo(n: int = 1000, n_sims: int = 200, seed: int = 0) -> dict:
    """Simulate *independent* random walks. Testing their levels should reject ~5% of the time;
    testing their returns (the common mistake) rejects ~100% of the time."""
    rng = np.random.default_rng(seed)
    lev = ret = 0
    for _ in range(n_sims):
        e = rng.standard_normal((n, 2))
        x = e.cumsum(axis=0)
        lev += coint(x[:, 0], x[:, 1])[1] < 0.05
        ret += coint(e[:, 0], e[:, 1])[1] < 0.05
    return {"reject_rate_levels": lev / n_sims, "reject_rate_returns": ret / n_sims}
