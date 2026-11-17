"""CARR(1,1) benchmark of the paper pipeline: the filtered conditional range is causal."""

import numpy as np

from src.paper.baselines import _carr_log_lambda


def _series(n: int = 900, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lam, out = 0.02, np.empty(n)
    for t in range(n):                       # a persistent positive range process
        r = lam * rng.exponential()
        out[t] = max(r, 1e-6)
        lam = 0.002 + 0.25 * r + 0.65 * lam
    return np.log(out)


def test_carr_forecast_uses_no_later_bar():
    y = _series()
    train = np.arange(600)
    base = _carr_log_lambda(y, train)
    changed = y.copy()
    changed[700:] += 3.0                     # a shock after bar 699, outside the training block
    again = _carr_log_lambda(changed, train)
    assert np.isfinite(base).all() and base.shape == y.shape
    np.testing.assert_allclose(again[:700], base[:700])       # forecasts made up to bar 699 are unchanged
    assert not np.allclose(again[700:], base[700:])


def test_carr_tracks_the_level_of_the_range():
    y = _series(seed=1)
    lam = _carr_log_lambda(y, np.arange(600))
    # the forecast for bar t+1 is positively related to the realised log range of bar t+1
    assert np.corrcoef(lam[:-1], y[1:])[0, 1] > 0.2
