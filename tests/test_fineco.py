import numpy as np
import pandas as pd
import pytest
from fineco import yieldcurve as yc, cointegration as co, tails, volatility as vol, forecasting as fc, data

TAU = np.array([1/12, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30])


def test_nelson_siegel_recovers_parameters():
    true = yc.CurveFit("NS", np.array([4.0, -2.0, 1.5]), (1.8,), 0.0)
    fit = yc.fit_nelson_siegel(TAU, true(TAU))
    assert fit.rmse_bp < 1e-3
    assert np.allclose(fit.betas, true.betas, atol=1e-3)
    assert fit.lams[0] == pytest.approx(1.8, rel=1e-3)


def test_svensson_at_least_as_good_as_ns():
    y = data.load_yields().iloc[-1]
    ns = yc.fit_nelson_siegel(y.index.values, y.values)
    sv = yc.fit_svensson(y.index.values, y.values)
    assert sv.rmse_bp <= ns.rmse_bp + 1e-9


def test_spurious_cointegration_on_returns():
    d = co.spurious_returns_demo(n=500, n_sims=60)
    assert d["reject_rate_levels"] < 0.2 and d["reject_rate_returns"] > 0.9


def test_johansen_finds_planted_cointegration():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1500).cumsum()
    df = pd.DataFrame({"x": x, "y": 0.5 * x + rng.standard_normal(1500)})
    assert co.johansen_rank(df) == 1


def test_pot_var_close_to_true_quantile():
    rng = np.random.default_rng(2)
    L = rng.standard_t(4, 200_000)
    var, es = tails.pot_var_es(tails.fit_pot(L, 0.95), 0.995)
    from scipy import stats
    assert var == pytest.approx(stats.t.ppf(0.995, 4), rel=0.05)
    assert es > var


def test_t_copula_recovers_parameters():
    u = tails.sample_t_copula(0.6, 5, 6000, seed=3)
    tab = tails.fit_copulas(u)
    assert tab.index[0] == "Student-t"
    rho = float(tab.loc["Student-t", "param"].split(",")[0].split("=")[1])
    assert rho == pytest.approx(0.6, abs=0.03)


def test_qlike_minimised_at_truth():
    rng = np.random.default_rng(4)
    s2 = np.exp(rng.normal(0, .5, 20000))
    rv = s2 * rng.standard_normal(20000) ** 2
    assert vol.qlike(rv, s2) < vol.qlike(rv, 1.5 * s2) and vol.qlike(rv, s2) < vol.qlike(rv, 0.7 * s2)


def test_dm_detects_better_model():
    rng = np.random.default_rng(5)
    e_good, e_bad = rng.normal(0, 1, 2000), rng.normal(0, 1.3, 2000)
    stat, p = fc.diebold_mariano(e_bad, e_good)
    assert stat > 0 and p < 0.01


def test_arma_walk_forward_has_no_lookahead():
    rng = np.random.default_rng(6)
    y = np.zeros(1200)
    for t in range(1, 1200):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal()
    s = pd.Series(y)
    out = fc.walk_forward_arma(s, 200, order=(1, 0), train_window=600)
    # forecast should be ~0.5 * previous actual, never the current actual
    prev = s.shift(1).iloc[-200:].values
    assert np.corrcoef(out["forecast"], prev)[0, 1] > 0.99
    assert fc.evaluate(out)["OOS_R2_%"] > 10
