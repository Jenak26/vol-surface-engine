import pytest
import numpy as np
from src.black_scholes import bs_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBSPrice:
    def test_atm_call_price(self):
        assert abs(bs_price(S, K, T, r, sigma, 'call') - 10.4506) < 0.01

    def test_atm_put_price(self):
        assert abs(bs_price(S, K, T, r, sigma, 'put') - 5.5735) < 0.01

    def test_put_call_parity(self):
        call = bs_price(S, K, T, r, sigma, 'call')
        put  = bs_price(S, K, T, r, sigma, 'put')
        assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-6

    def test_call_at_expiry_itm(self):
        assert abs(bs_price(110, 100, 1e-8, 0.05, 0.20, 'call') - 10.0) < 0.01

    def test_call_at_expiry_otm(self):
        assert bs_price(90, 100, 1e-8, 0.05, 0.20, 'call') < 0.001

    def test_call_price_increases_with_spot(self):
        assert bs_price(110, K, T, r, sigma, 'call') > bs_price(90, K, T, r, sigma, 'call')

    def test_put_price_increases_as_spot_falls(self):
        assert bs_price(90, K, T, r, sigma, 'put') > bs_price(110, K, T, r, sigma, 'put')

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            bs_price(S, K, T, r, sigma, 'invalid')
