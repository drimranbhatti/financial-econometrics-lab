"""Walk-forward return forecasting (ARMA, VAR) with honest evaluation:
out-of-sample R^2 vs the historical mean, Diebold-Mariano test, hit rate and a cost-aware
sign strategy. Model order is chosen on the training window only."""
from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")


def select_arma(y, max_p: int = 3, max_q: int = 3):
    best, order = np.inf, (0, 0)
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            aic = ARIMA(y, order=(p, 0, q)).fit().aic
        except Exception:
            continue
        if aic < best:
            best, order = aic, (p, q)
    return order


def walk_forward_arma(r: pd.Series, n_test: int, refit_every: int = 20, order=None,
                      train_window: int = 750) -> pd.DataFrame:
    """One-step forecasts for the last n_test points. The order is selected once on the
    first training window; parameters are re-estimated every `refit_every` steps."""
    y = r.values
    n = len(y)
    start = n - n_test
    if order is None:
        order = select_arma(y[start - train_window:start])
    fc = np.empty(n_test)
    for k in range(0, n_test, refit_every):
        t0 = start + k
        res = ARIMA(y[t0 - train_window:t0], order=(order[0], 0, order[1])).fit()
        m = min(refit_every, n_test - k)
        seg = res.apply(y[t0 - train_window:t0 + m])  # same params, filter through new data
        pred = seg.predict(start=train_window, end=train_window + m - 1)
        fc[k:k + m] = pred
    return pd.DataFrame({"actual": y[start:], "forecast": fc,
                         "hist_mean": pd.Series(y).expanding().mean().shift(1).values[start:]},
                        index=r.index[start:]).assign(order=str(order))


def walk_forward_var(R: pd.DataFrame, target: str, n_test: int, refit_every: int = 20,
                     train_window: int = 750, maxlags: int = 5) -> pd.DataFrame:
    n = len(R)
    start = n - n_test
    fc = np.empty(n_test)
    lag = VAR(R.iloc[start - train_window:start].values).select_order(maxlags).bic or 1
    j = list(R).index(target)
    for k in range(0, n_test, refit_every):
        t0 = start + k
        res = VAR(R.iloc[t0 - train_window:t0].values).fit(lag)
        for i in range(min(refit_every, n_test - k)):
            hist = R.iloc[t0 + i - lag:t0 + i].values
            fc[k + i] = res.forecast(hist, 1)[0, j]
    y = R[target].values
    return pd.DataFrame({"actual": y[start:], "forecast": fc,
                         "hist_mean": pd.Series(y).expanding().mean().shift(1).values[start:]},
                        index=R.index[start:]).assign(order=f"VAR({lag})")


def diebold_mariano(e1, e2, h: int = 1):
    """DM test on squared-error loss; positive stat => model 2 more accurate than model 1."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0 + 2 * sum((1 - k / h) * np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)) if h > 1 else gamma0
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def evaluate(df: pd.DataFrame, cost_bps: float = 10, periods: int = 252) -> dict:
    a, f, m = df["actual"].values, df["forecast"].values, df["hist_mean"].values
    oos_r2 = 1 - np.sum((a - f) ** 2) / np.sum((a - m) ** 2)
    dm, dm_p = diebold_mariano(a - m, a - f)
    hits = np.sum(np.sign(f) == np.sign(a))
    hit_p = stats.binomtest(int(hits), len(a), 0.5).pvalue
    pos = np.sign(f)                          # long/short next day
    turnover = np.abs(np.diff(np.r_[0, pos]))
    strat = pos * a - turnover * cost_bps / 1e4
    def sharpe(x):
        return float(np.mean(x) / np.std(x) * np.sqrt(periods)) if np.std(x) > 0 else np.nan
    return {"model": df["order"].iloc[0], "OOS_R2_%": 100 * oos_r2, "DM_stat": dm, "DM_p": dm_p,
            "hit_rate_%": 100 * hits / len(a), "always_long_hit_%": 100 * np.mean(a > 0), "hit_p": hit_p,
            "strategy_ann_ret_%": 100 * np.mean(strat) * periods, "strategy_sharpe": sharpe(strat),
            "buyhold_ann_ret_%": 100 * np.mean(a) * periods, "buyhold_sharpe": sharpe(a),
            "trades": int(turnover.sum() / 2 if turnover.sum() else 0)}
