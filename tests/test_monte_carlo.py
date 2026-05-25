import pytest
import numpy as np
from src.monte_carlo import simulate_gbm_paths, mc_european
from src.black_scholes import bs_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
BS_CALL = 10.4506
BS_PUT  = 5.5735

class TestGBMPaths:
    def test_paths_shape(self):
        paths = simulate_gbm_paths(S, T, r, sigma, n_sims=1000, n_steps=252, seed=42)
        assert paths.shape == (1000, 253)  # n_sims × (n_steps + 1) including t=0

    def test_paths_start_at_spot(self):
        paths = simulate_gbm_paths(S, T, r, sigma, n_sims=1000, n_steps=252, seed=42)
        assert np.allclose(paths[:, 0], S)

    def test_paths_all_positive(self):
        paths = simulate_gbm_paths(S, T, r, sigma, n_sims=5000, n_steps=50, seed=42)
        assert np.all(paths > 0)

    def test_terminal_mean_close_to_forward(self):
        # E[S_T] = S * exp(r*T) under risk-neutral measure
        paths = simulate_gbm_paths(S, T, r, sigma, n_sims=100_000, n_steps=1, seed=42)
        forward = S * np.exp(r * T)
        assert abs(np.mean(paths[:, -1]) / forward - 1.0) < 0.01  # within 1%

class TestEuropeanMC:
    def test_call_converges_to_bs(self):
        price = mc_european(S, K, T, r, sigma, 'call', n_sims=500_000, seed=42)
        assert abs(price - BS_CALL) < 0.15  # within 15 cents with 500k paths

    def test_put_converges_to_bs(self):
        price = mc_european(S, K, T, r, sigma, 'put', n_sims=500_000, seed=42)
        assert abs(price - BS_PUT) < 0.15

    def test_put_call_parity_mc(self):
        call = mc_european(S, K, T, r, sigma, 'call', n_sims=500_000, seed=42)
        put  = mc_european(S, K, T, r, sigma, 'put',  n_sims=500_000, seed=42)
        assert abs((call - put) - (S - K * np.exp(-r * T))) < 0.30
