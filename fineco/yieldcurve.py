"""Nelson-Siegel (1987) and Svensson (1994) yield-curve models.

For fixed decay parameters the models are linear in the betas, so we profile:
grid/optimise over the decay parameter(s) and solve the betas by least squares.
This avoids the local minima that plague a joint non-linear fit."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar, minimize


def _ns_loadings(tau, lam):
    tau = np.asarray(tau, float)
    x = tau / lam
    f1 = (1 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), f1, f2])


def _sv_loadings(tau, lam1, lam2):
    x2 = np.asarray(tau, float) / lam2
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return np.column_stack([_ns_loadings(tau, lam1), f3])


@dataclass
class CurveFit:
    model: str
    betas: np.ndarray
    lams: tuple
    rmse_bp: float

    def __call__(self, tau):
        X = _ns_loadings(tau, self.lams[0]) if self.model == "NS" else _sv_loadings(tau, *self.lams)
        return X @ self.betas


def _ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, float(np.sqrt(np.mean((X @ b - y) ** 2)))


def fit_nelson_siegel(tau, y, lam_bounds=(0.1, 10.0)) -> CurveFit:
    tau, y = np.asarray(tau, float), np.asarray(y, float)
    res = minimize_scalar(lambda l: _ols(_ns_loadings(tau, l), y)[1], bounds=lam_bounds, method="bounded")
    b, rmse = _ols(_ns_loadings(tau, res.x), y)
    return CurveFit("NS", b, (float(res.x),), 100 * rmse)


def fit_svensson(tau, y, lam_bounds=(0.1, 30.0), n_starts: int = 12, seed: int = 0) -> CurveFit:
    """Multi-start over (lam1, lam2), constrained lam1 < lam2 to keep the humps identified."""
    tau, y = np.asarray(tau, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)

    def obj(p):
        l1, l2 = p
        if l1 >= l2:
            return 1e6
        return _ols(_sv_loadings(tau, l1, l2), y)[1]

    best = None
    for _ in range(n_starts):
        x0 = np.sort(rng.uniform(*lam_bounds, 2))
        r = minimize(obj, x0, method="L-BFGS-B", bounds=[lam_bounds, lam_bounds])
        if best is None or r.fun < best.fun:
            best = r
    l1, l2 = best.x
    b, rmse = _ols(_sv_loadings(tau, l1, l2), y)
    return CurveFit("NSS", b, (float(l1), float(l2)), 100 * rmse)


def fit_panel(yields, fixed_lam: float | None = None):
    """Dynamic Nelson-Siegel (Diebold-Li 2006): fit every date with a common lambda and
    return the level / slope / curvature factor time series."""
    import pandas as pd
    tau = np.asarray(yields.columns, float)
    if fixed_lam is None:
        # choose the lambda that minimises pooled RMSE
        def pooled(l):
            X = _ns_loadings(tau, l)
            B, *_ = np.linalg.lstsq(X, yields.values.T, rcond=None)
            return np.sqrt(np.mean((X @ B - yields.values.T) ** 2))
        fixed_lam = float(minimize_scalar(pooled, bounds=(0.1, 10), method="bounded").x)
    X = _ns_loadings(tau, fixed_lam)
    B, *_ = np.linalg.lstsq(X, yields.values.T, rcond=None)
    resid = X @ B - yields.values.T
    f = pd.DataFrame(B.T, index=yields.index, columns=["level", "slope", "curvature"])
    f["rmse_bp"] = 100 * np.sqrt((resid ** 2).mean(axis=0))
    return f, fixed_lam
