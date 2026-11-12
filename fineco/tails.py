"""Fat tails: Hill estimator, peaks-over-threshold (GPD) VaR/ES, and copulas for tail dependence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats, optimize, special


# ---------------------------------------------------------------- extreme value theory
def hill_estimator(losses, k: int) -> float:
    """Hill tail index xi = 1/alpha using the k largest losses."""
    x = np.sort(np.asarray(losses))[::-1]
    x = x[x > 0]
    return float(np.mean(np.log(x[:k]) - np.log(x[k])))


def fit_pot(losses, threshold_q: float = 0.95):
    """Fit a generalized Pareto distribution to exceedances over the empirical q-quantile.
    Returns dict(u, xi, beta, n, n_u)."""
    x = np.asarray(losses)
    u = np.quantile(x, threshold_q)
    exc = x[x > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    return {"u": float(u), "xi": float(xi), "beta": float(beta), "n": len(x), "n_u": len(exc)}


def pot_var_es(fit: dict, p: float = 0.99):
    """McNeil-Frey closed forms for VaR and ES at confidence p (valid for xi < 1)."""
    u, xi, beta, n, nu = fit["u"], fit["xi"], fit["beta"], fit["n"], fit["n_u"]
    var = u + beta / xi * (((n / nu) * (1 - p)) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)
    return float(var), float(es)


def var_es_table(returns: pd.Series, p: float = 0.99, threshold_q: float = 0.95) -> pd.DataFrame:
    """Compare 1-day VaR/ES (as positive % losses) from Normal, Student-t, historical and EVT."""
    L = -100 * returns.dropna().values
    mu, sd = L.mean(), L.std(ddof=1)
    z = stats.norm.ppf(p)
    rows = {"Normal": (mu + sd * z, mu + sd * stats.norm.pdf(z) / (1 - p))}
    nu, loc, sc = stats.t.fit(L)
    q = stats.t.ppf(p, nu)
    rows["Student-t"] = (loc + sc * q, loc + sc * stats.t.pdf(q, nu) / (1 - p) * (nu + q ** 2) / (nu - 1))
    hv = np.quantile(L, p)
    rows["Historical"] = (hv, L[L >= hv].mean())
    rows["EVT (POT-GPD)"] = pot_var_es(fit_pot(L, threshold_q), p)
    return pd.DataFrame(rows, index=[f"VaR {p:.0%}", f"ES {p:.0%}"]).T


# ---------------------------------------------------------------- copulas
def pseudo_obs(x: pd.DataFrame) -> np.ndarray:
    """Rank-transform to (0,1): the copula only sees dependence, not the marginals."""
    return (x.rank().values) / (len(x) + 1)


def _gauss_ll(rho, u, v):
    a, b = stats.norm.ppf(u), stats.norm.ppf(v)
    return np.sum(-0.5 * np.log(1 - rho ** 2) - (rho ** 2 * (a ** 2 + b ** 2) - 2 * rho * a * b) / (2 * (1 - rho ** 2)))


def _t_ll(rho, nu, u, v):
    a, b = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    q = (a ** 2 + b ** 2 - 2 * rho * a * b) / (1 - rho ** 2)
    logc = (special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
            - 0.5 * np.log(1 - rho ** 2) - (nu + 2) / 2 * np.log1p(q / nu)
            + (nu + 1) / 2 * (np.log1p(a ** 2 / nu) + np.log1p(b ** 2 / nu)))
    return np.sum(logc)


def _clayton_ll(th, u, v):
    return np.sum(np.log1p(th) - (1 + th) * (np.log(u) + np.log(v))
                  - (2 + 1 / th) * np.log(u ** -th + v ** -th - 1))


def _gumbel_ll(th, u, v):
    a, b = -np.log(u), -np.log(v)
    s = a ** th + b ** th
    C = np.exp(-s ** (1 / th))
    return np.sum(np.log(C) + (th - 1) * (np.log(a) + np.log(b)) - np.log(u) - np.log(v)
                  + (1 / th - 2) * np.log(s) + np.log(s ** (1 / th) + th - 1))


def fit_copulas(u: np.ndarray) -> pd.DataFrame:
    """MLE for four bivariate copulas on pseudo-observations u (n x 2); compares AIC and the
    implied lower/upper tail-dependence coefficients."""
    a, b = u[:, 0], u[:, 1]
    tau = stats.kendalltau(a, b)[0]
    out = {}
    r = optimize.minimize_scalar(lambda p: -_gauss_ll(p, a, b), bounds=(-0.99, 0.99), method="bounded")
    out["Gaussian"] = dict(param=f"rho={r.x:.3f}", loglik=-r.fun, k=1, lower_tail=0.0, upper_tail=0.0)
    r = optimize.minimize(lambda p: -_t_ll(p[0], p[1], a, b), x0=[np.sin(np.pi * tau / 2), 6],
                          bounds=[(-0.99, 0.99), (2.1, 60)], method="L-BFGS-B")
    rho, nu = r.x
    lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    out["Student-t"] = dict(param=f"rho={rho:.3f}, nu={nu:.1f}", loglik=-r.fun, k=2, lower_tail=lam, upper_tail=lam)
    if tau > 0:
        r = optimize.minimize_scalar(lambda t: -_clayton_ll(t, a, b), bounds=(1e-3, 20), method="bounded")
        out["Clayton"] = dict(param=f"theta={r.x:.3f}", loglik=-r.fun, k=1, lower_tail=2 ** (-1 / r.x), upper_tail=0.0)
        r = optimize.minimize_scalar(lambda t: -_gumbel_ll(t, a, b), bounds=(1.0001, 20), method="bounded")
        out["Gumbel"] = dict(param=f"theta={r.x:.3f}", loglik=-r.fun, k=1, lower_tail=0.0, upper_tail=2 - 2 ** (1 / r.x))
    df = pd.DataFrame(out).T
    df["AIC"] = 2 * df["k"].astype(float) - 2 * df["loglik"].astype(float)
    return df.drop(columns="k").sort_values("AIC")


def empirical_tail_dependence(u: np.ndarray, q: float = 0.05):
    """Share of joint extremes: P(V<q | U<q) and P(V>1-q | U>1-q)."""
    a, b = u[:, 0], u[:, 1]
    lower = np.mean((a < q) & (b < q)) / q
    upper = np.mean((a > 1 - q) & (b > 1 - q)) / q
    return float(lower), float(upper)


def sample_t_copula(rho, nu, n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    w = np.sqrt(nu / rng.chisquare(nu, n))[:, None]
    return stats.t.cdf(z * w, nu)
