"""GARCH-family estimation and honest out-of-sample variance-forecast comparison."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

SPECS = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, q=1),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
    "FIGARCH(1,d,1)": dict(vol="FIGARCH", p=1, q=1),
}


def _model(r_pct, spec, dist="t"):
    return arch_model(r_pct, mean="Constant", dist=dist, rescale=False, **spec)


def fit_family(returns: pd.Series, dist: str = "t") -> pd.DataFrame:
    """In-sample fit of each spec on returns in percent. Returns log-likelihood, AIC, BIC
    and the key shape parameters."""
    r = 100 * returns.dropna()
    rows = {}
    for name, spec in SPECS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _model(r, spec, dist).fit(disp="off")
        p = res.params
        persistence = p.get("alpha[1]", 0) + p.get("beta[1]", 0) + 0.5 * p.get("gamma[1]", 0)
        if name.startswith("EGARCH"):
            persistence = p.get("beta[1]", np.nan)
        if name.startswith("FIGARCH"):
            persistence = np.nan
        rows[name] = {"loglik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
                      "persistence": persistence, "leverage(gamma)": p.get("gamma[1]", np.nan),
                      "d": p.get("d", np.nan), "t_dof": p.get("nu", np.nan)}
    return pd.DataFrame(rows).T


def qlike(realized, forecast):
    """QLIKE loss (Patton 2011): robust to noise in the realised-variance proxy."""
    realized, forecast = np.asarray(realized), np.asarray(forecast)
    ratio = realized / forecast
    return float(np.mean(ratio - np.log(ratio) - 1))


def rolling_forecasts(returns: pd.Series, n_test: int = 500, refit_every: int = 20,
                      window: int | None = None, dist: str = "t", specs=None) -> pd.DataFrame:
    """One-day-ahead variance forecasts over the last `n_test` observations.

    Parameters are re-estimated every `refit_every` days using only data up to that day;
    between refits the filter is rolled forward with fixed parameters. No look-ahead."""
    specs = specs or SPECS
    r = 100 * returns.dropna()
    n = len(r)
    start = n - n_test
    out = pd.DataFrame(index=r.index[start:])
    out["realized"] = (r.iloc[start:] - r.iloc[:start].mean()) ** 2
    # benchmark: 60-day rolling variance known at t-1, and RiskMetrics EWMA
    out["Rolling-60d"] = r.rolling(60).var().shift(1).iloc[start:]
    ewma = np.empty(n)
    ewma[0] = r.iloc[:60].var()
    for t in range(1, n):
        ewma[t] = 0.94 * ewma[t - 1] + 0.06 * (r.iloc[t - 1] - 0) ** 2
    out["EWMA(0.94)"] = ewma[start:]
    for name, spec in specs.items():
        fc = np.empty(n_test)
        for k in range(0, n_test, refit_every):
            t0 = start + k
            lo = 0 if window is None else max(0, t0 - window)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = _model(r.iloc[lo:t0], spec, dist).fit(disp="off")
                # filter through the next block with fixed params; forecast for t uses data < t
                seg = r.iloc[lo:min(t0 + refit_every, n)]
                f = _model(seg, spec, dist).fix(res.params).forecast(horizon=1, start=t0 - lo - 1,
                                                                     reindex=False)
            v = f.variance.values[:, 0]
            m = min(refit_every, n_test - k)
            fc[k:k + m] = v[:m]
        out[name] = fc
    return out


def loss_table(fc: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in fc if c != "realized"]
    rv = fc["realized"].clip(lower=1e-8)
    return pd.DataFrame({
        "QLIKE": {c: qlike(rv, fc[c]) for c in cols},
        "MSE": {c: float(np.mean((fc["realized"] - fc[c]) ** 2)) for c in cols},
    }).sort_values("QLIKE")
