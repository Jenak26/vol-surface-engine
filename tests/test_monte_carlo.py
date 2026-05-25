import pytest
import numpy as np
from src.monte_carlo import simulate_gbm_paths, mc_european, mc_antithetic, mc_control_variate, mc_asian, mc_barrier_down_out
from src.black_scholes import bs_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
BS_CALL = 10.4506
BS_PUT  = 5.5735
N_SIMS = 100_000  # same budget for fair comparison

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

class TestVarianceReduction:
    def test_antithetic_tighter_than_naive(self):
        # Run 30 trials, compare standard deviations of price estimates
        naive_prices = [mc_european(S, K, T, r, sigma, 'call', n_sims=N_SIMS, seed=i)
                        for i in range(30)]
        anti_prices  = [mc_antithetic(S, K, T, r, sigma, 'call', n_sims=N_SIMS, seed=i)
                        for i in range(30)]
        assert np.std(anti_prices) < np.std(naive_prices)

    def test_antithetic_call_close_to_bs(self):
        price = mc_antithetic(S, K, T, r, sigma, 'call', n_sims=N_SIMS, seed=42)
        assert abs(price - BS_CALL) < 0.10

    def test_control_variate_call_close_to_bs(self):
        price = mc_control_variate(S, K, T, r, sigma, 'call', n_sims=N_SIMS, seed=42)
        assert abs(price - BS_CALL) < 0.08

    def test_control_variate_tighter_than_antithetic(self):
        cv_prices   = [mc_control_variate(S, K, T, r, sigma, 'call', n_sims=N_SIMS, seed=i)
                       for i in range(30)]
        anti_prices = [mc_antithetic(S, K, T, r, sigma, 'call', n_sims=N_SIMS, seed=i)
                       for i in range(30)]
        assert np.std(cv_prices) < np.std(anti_prices)

class TestPathDependentOptions:
    def test_asian_call_cheaper_than_european_call(self):
        # Asian (average) call < European call because averaging reduces effective volatility
        asian    = mc_asian(S, K, T, r, sigma, 'call', n_sims=200_000, seed=42)
        european = mc_european(S, K, T, r, sigma, 'call', n_sims=200_000, seed=42)
        assert asian < european

    def test_asian_call_positive(self):
        assert mc_asian(S, K, T, r, sigma, 'call', n_sims=100_000, seed=42) > 0

    def test_asian_put_positive(self):
        assert mc_asian(S, K, T, r, sigma, 'put', n_sims=100_000, seed=42) > 0

    def test_barrier_knock_out_below_european(self):
        # Down-and-out call <= vanilla call (barrier can kill the option)
        barrier_price  = mc_barrier_down_out(S, K, T, r, sigma,
                                              barrier=80.0, n_sims=200_000, seed=42)
        vanilla_price  = mc_european(S, K, T, r, sigma, 'call', n_sims=200_000, seed=42)
        assert barrier_price <= vanilla_price + 0.01  # small tolerance for MC noise

    def test_barrier_below_spot_activated_lowers_price(self):
        # Barrier just below spot (90) vs barrier far below (50)
        close_barrier = mc_barrier_down_out(S, K, T, r, sigma,
                                             barrier=90.0, n_sims=200_000, seed=42)
        far_barrier   = mc_barrier_down_out(S, K, T, r, sigma,
                                             barrier=50.0, n_sims=200_000, seed=42)
        assert close_barrier < far_barrier  # closer barrier → more likely to knock out

    def test_barrier_above_spot_zero(self):
        # If barrier ≥ spot, option is immediately knocked out → worthless
        price = mc_barrier_down_out(S, K, T, r, sigma,
                                     barrier=100.0, n_sims=10_000, seed=42)
        assert price < 0.01
