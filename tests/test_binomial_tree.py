import pytest
import numpy as np
from src.binomial_tree import crr_american_put, crr_european_put
from src.black_scholes import bs_price
from src.american_options import lsm_american_put


class TestCRREuropean:
    """European put via CRR should converge to Black-Scholes."""

    def test_atm_european_put_vs_bs(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        crr = crr_european_put(S, K, T, r, sigma, n_steps=500)
        bs = bs_price(S, K, T, r, sigma, 'put')
        assert abs(crr - bs) < 0.02

    def test_itm_european_put_vs_bs(self):
        S, K, T, r, sigma = 90, 100, 0.5, 0.05, 0.25
        crr = crr_european_put(S, K, T, r, sigma, n_steps=500)
        bs = bs_price(S, K, T, r, sigma, 'put')
        assert abs(crr - bs) < 0.05

    def test_otm_european_put_vs_bs(self):
        S, K, T, r, sigma = 110, 100, 1.0, 0.05, 0.20
        crr = crr_european_put(S, K, T, r, sigma, n_steps=500)
        bs = bs_price(S, K, T, r, sigma, 'put')
        assert abs(crr - bs) < 0.02

    def test_european_converges_with_more_steps(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        bs = bs_price(S, K, T, r, sigma, 'put')
        err_100 = abs(crr_european_put(S, K, T, r, sigma, n_steps=100) - bs)
        err_500 = abs(crr_european_put(S, K, T, r, sigma, n_steps=500) - bs)
        assert err_500 < err_100


class TestCRRAmerican:
    """American put via CRR must be >= European put and >= intrinsic."""

    def test_american_put_ge_european_put(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        american = crr_american_put(S, K, T, r, sigma)
        european = crr_european_put(S, K, T, r, sigma)
        assert american >= european - 1e-10

    def test_american_put_ge_intrinsic(self):
        S, K, T, r, sigma = 90, 100, 1.0, 0.05, 0.20
        american = crr_american_put(S, K, T, r, sigma)
        assert american >= K - S - 1e-10

    def test_early_exercise_premium_deep_itm(self):
        S, K, T, r, sigma = 70, 100, 1.0, 0.05, 0.20
        american = crr_american_put(S, K, T, r, sigma)
        european = crr_european_put(S, K, T, r, sigma)
        assert american - european > 0.5  # meaningful early exercise value

    def test_american_put_positive(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        assert crr_american_put(S, K, T, r, sigma) > 0

    def test_crr_vs_lsm_atm(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        crr = crr_american_put(S, K, T, r, sigma, n_steps=300)
        lsm = lsm_american_put(S, K, T, r, sigma, n_sims=50_000, n_steps=50)
        assert abs(crr - lsm) < 0.30  # MC has sampling noise

    def test_crr_vs_lsm_itm(self):
        S, K, T, r, sigma = 90, 100, 1.0, 0.05, 0.20
        crr = crr_american_put(S, K, T, r, sigma, n_steps=300)
        lsm = lsm_american_put(S, K, T, r, sigma, n_sims=50_000, n_steps=50)
        assert abs(crr - lsm) < 0.50
