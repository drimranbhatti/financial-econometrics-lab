"""Loading and aligning the snapshot data."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
MATURITIES = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS3": 3,
              "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}


def load_prices(columns=None, common_days: bool = True) -> pd.DataFrame:
    """Daily adjusted closes. Bitcoin trades 7 days a week and ETFs do not, so by default
    only days on which *every* series has a price are kept (no forward-filling, which would
    create artificial zero returns)."""
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    if columns is not None:
        px = px[list(columns)]
    return px.dropna() if common_days else px


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def load_yields() -> pd.DataFrame:
    """US Treasury constant-maturity yields in percent; columns renamed to maturity in years."""
    y = pd.read_csv(DATA / "treasury_yields.csv", index_col=0, parse_dates=True)
    return y.rename(columns=MATURITIES).dropna()


def summary_table(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    from scipy import stats
    out = {}
    for c in returns:
        r = returns[c].dropna()
        jb = stats.jarque_bera(r)
        out[c] = {"ann_mean_%": 100 * r.mean() * periods, "ann_vol_%": 100 * r.std() * np.sqrt(periods),
                  "skew": stats.skew(r), "excess_kurtosis": stats.kurtosis(r),
                  "min_%": 100 * r.min(), "max_%": 100 * r.max(), "JB_pvalue": jb.pvalue, "n": len(r)}
    return pd.DataFrame(out).T
