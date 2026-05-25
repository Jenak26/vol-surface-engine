import pytest
import numpy as np
from src.heston import heston_price, heston_delta, heston_vega, heston_theta

S, K, T, r = 100.0, 100.0, 1.0, 0.05
PARAMS = {'kappa': 1.5, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.5, 'v0': 0.04}


class TestHestonGreeks:
    def test_delta_call_in_range(self):
        d = heston_delta(S, K, T, r, **PARAMS, option_type='call')
        assert 0.0 < d < 1.0, f"Call delta out of range: {d}"

    def test_delta_put_in_range(self):
        d = heston_delta(S, K, T, r, **PARAMS, option_type='put')
        assert -1.0 < d < 0.0, f"Put delta out of range: {d}"

    def test_delta_call_put_relationship(self):
        # Put-call parity implies delta_call - delta_put = 1
        d_call = heston_delta(S, K, T, r, **PARAMS, option_type='call')
        d_put  = heston_delta(S, K, T, r, **PARAMS, option_type='put')
        assert abs(d_call - d_put - 1.0) < 0.02

    def test_vega_positive(self):
        # Higher initial variance → higher option price (positive vega)
        v = heston_vega(S, K, T, r, **PARAMS, option_type='call')
        assert v > 0.0, f"Call vega should be positive: {v}"

    def test_theta_call_negative(self):
        # Time decay is negative (option loses value as T decreases)
        th = heston_theta(S, K, T, r, **PARAMS, option_type='call')
        assert th < 0.0, f"Call theta should be negative: {th}"

    def test_theta_zero_at_expiry(self):
        # At T ≤ dt, theta returns 0 (no time value left)
        th = heston_theta(S, K, 1e-4, r, **PARAMS, option_type='call')
        assert th == 0.0

    def test_delta_itm_call_approaches_one(self):
        # Deep ITM call delta ≈ 1
        d = heston_delta(150.0, K, T, r, **PARAMS, option_type='call')
        assert d > 0.85

    def test_delta_otm_call_small(self):
        # Deep OTM call delta ≈ 0
        d = heston_delta(60.0, K, T, r, **PARAMS, option_type='call')
        assert d < 0.15
