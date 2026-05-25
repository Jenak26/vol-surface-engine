import pytest
import numpy as np
from src.american_options import lsm_american_put
from src.american_options import compute_exercise_boundary
from src.black_scholes import bs_price
from src.monte_carlo import mc_european

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

class TestLSMAmerican:
    def test_american_put_exceeds_european_put(self):
        european_put = mc_european(S, K, T, r, sigma, 'put', n_sims=50_000, seed=42)
        american_put = lsm_american_put(S, K, T, r, sigma, n_sims=50_000, seed=42)
        assert american_put > european_put  # early exercise premium must be positive

    def test_american_put_deep_itm_early_exercise_premium_large(self):
        # Deep ITM put: S=70, K=100 — should be exercised early, large premium
        eu_put  = mc_european(70, K, T, r, sigma, 'put', n_sims=50_000, seed=42)
        am_put  = lsm_american_put(70, K, T, r, sigma, n_sims=50_000, seed=42)
        premium = am_put - eu_put
        assert premium > 1.0  # meaningful premium on deep ITM

    def test_american_put_equals_european_when_r_zero(self):
        # When r=0, there is no benefit to early exercise for a put
        eu_put = mc_european(S, K, T, 0.0, sigma, 'put', n_sims=50_000, seed=42)
        am_put = lsm_american_put(S, K, T, 0.0, sigma, n_sims=50_000, seed=42)
        assert abs(am_put - eu_put) < 0.5  # small difference with r=0

    def test_american_put_otm_close_to_bs(self):
        # OTM American put (S=110): early exercise unlikely, close to European BS
        bs_put   = bs_price(110, K, T, r, sigma, 'put')
        am_put   = lsm_american_put(110, K, T, r, sigma, n_sims=50_000, seed=42)
        assert abs(am_put - bs_put) < 0.50  # within 50 cents

    def test_american_put_positive(self):
        assert lsm_american_put(S, K, T, r, sigma, n_sims=10_000, seed=42) > 0


class TestExerciseBoundary:
    def test_boundary_below_strike(self):
        # Exercise boundary must be below K everywhere (only exercise put when S < K)
        times, boundary = compute_exercise_boundary(K, T, r, sigma, n_steps=20)
        assert np.all(boundary <= K + 0.01)

    def test_boundary_increases_toward_maturity(self):
        # Boundary should approach K as t → T (at expiry, exercise whenever S < K)
        times, boundary = compute_exercise_boundary(K, T, r, sigma, n_steps=20)
        # Last few values should be close to K
        assert boundary[-1] > boundary[0]  # boundary rises toward K

    def test_boundary_length_matches_steps(self):
        times, boundary = compute_exercise_boundary(K, T, r, sigma, n_steps=20)
        assert len(times) == len(boundary) == 20
