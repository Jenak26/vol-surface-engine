import pytest
import numpy as np
from src.black_scholes import bs_price, delta, gamma, vega, theta, rho, numerical_delta, numerical_gamma, numerical_vega

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


class TestAnalyticalGreeks:
    def test_delta_call_value(self):
        assert abs(delta(S, K, T, r, sigma, 'call') - 0.6368) < 0.001

    def test_delta_put_value(self):
        assert abs(delta(S, K, T, r, sigma, 'put') - (-0.3632)) < 0.001

    def test_delta_call_put_relationship(self):
        d_call = delta(S, K, T, r, sigma, 'call')
        d_put  = delta(S, K, T, r, sigma, 'put')
        assert abs(d_call - d_put - 1.0) < 1e-6

    def test_delta_call_between_0_and_1(self):
        assert 0 < delta(S, K, T, r, sigma, 'call') < 1

    def test_delta_put_between_minus1_and_0(self):
        assert -1 < delta(S, K, T, r, sigma, 'put') < 0

    def test_gamma_value(self):
        assert abs(gamma(S, K, T, r, sigma) - 0.01876) < 0.0002

    def test_gamma_same_for_call_and_put(self):
        assert abs(gamma(S, K, T, r, sigma) - gamma(S, K, T, r, sigma)) < 1e-10

    def test_gamma_positive(self):
        assert gamma(S, K, T, r, sigma) > 0

    def test_vega_value(self):
        assert abs(vega(S, K, T, r, sigma) - 0.3752) < 0.001

    def test_vega_positive(self):
        assert vega(S, K, T, r, sigma) > 0

    def test_theta_call_negative(self):
        assert theta(S, K, T, r, sigma, 'call') < 0

    def test_theta_call_value(self):
        assert abs(theta(S, K, T, r, sigma, 'call') - (-0.01757)) < 0.001

    def test_rho_call_value(self):
        assert abs(rho(S, K, T, r, sigma, 'call') - 0.5323) < 0.005

    def test_rho_put_negative(self):
        assert rho(S, K, T, r, sigma, 'put') < 0


class TestNumericalGreeks:
    def test_numerical_delta_matches_analytical(self):
        analytical = delta(S, K, T, r, sigma, 'call')
        numerical  = numerical_delta(S, K, T, r, sigma, 'call')
        assert abs(analytical - numerical) < 1e-4

    def test_numerical_gamma_matches_analytical(self):
        analytical = gamma(S, K, T, r, sigma)
        numerical  = numerical_gamma(S, K, T, r, sigma)
        assert abs(analytical - numerical) < 1e-4

    def test_numerical_vega_matches_analytical(self):
        analytical = vega(S, K, T, r, sigma)
        numerical  = numerical_vega(S, K, T, r, sigma)
        assert abs(analytical - numerical) < 1e-3

    def test_numerical_put_delta_matches_analytical(self):
        analytical = delta(S, K, T, r, sigma, 'put')
        numerical  = numerical_delta(S, K, T, r, sigma, 'put')
        assert abs(analytical - numerical) < 1e-4
